# last modified: 2024/06/26
import logging
import os
import os.path as osp
import sys
from collections import OrderedDict

import torch
import torch.nn as nn
import torchsnooper
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.nn.parameter import Parameter
from tqdm import tqdm
from yacs.config import CfgNode as CN

sys.path.append(".")
from dassl.config import clean_cfg
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.evaluation import build_evaluator
from dassl.metrics import compute_accuracy
from dassl.optim import build_lr_scheduler, build_optimizer
from dassl.utils import load_checkpoint, load_pretrained_weights
from libs.engine import (
    default_argument_parser,
    default_setup,
    get_cfg,
    launch,
    merge_from_args,
)
from libs.modeling.clip import clip
from libs.modeling.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from libs.trainers.pldg_base_trainer import (
    DGPlBaseTrainer,
    TextEncoder,
    load_clip_to_cpu,
    set_dgpl_config,
)
from libs.utils import comm

logger = logging.getLogger(
    f'fastdg.{os.path.relpath(__file__).replace(os.path.sep, ".")}'
)
_tokenizer = _Tokenizer()


# class CustomTextEncoder(TextEncoder):

#     def __init__(self, clip_model, name_lens):
#         super().__init__(clip_model)
#         self.name_lens = name_lens

#     def forward(self, prompts, tokenized_prompts):
#         x = prompts + self.positional_embedding.type(self.dtype)
#         x = x.permute(1, 0, 2)  # NLD -> LND
#         x = self.transformer(x)
#         x = x.permute(1, 0, 2)  # LND -> NLD
#         x = self.ln_final(x).type(self.dtype)

#         # x.shape = [batch_size, n_ctx, transformer.width]
#         # take features from the eot embedding (eot_token is the highest number in each sequence)
#         eos_pos = tokenized_prompts.argmax(dim=-1)
#         cls_pos = eos_pos - 1

#         x = (
#             x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
#             @ self.text_projection
#         )

#         return x


class PromptLearner(nn.Module):

    def __init__(self, cfg, clip_model, cls_names, dom_names):
        super().__init__()
        n_cls = len(cls_names)
        n_dom = len(dom_names)
        n_ctx = cfg.METHOD.N_CTX
        ctx_init = cfg.METHOD.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert (
            cfg_imsize == clip_imsize
        ), f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # random initialization
        logger.info("Initializing domain-specific contexts")
        ctx_vectors = torch.empty(n_dom, n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        ctx_placeholder = " ".join(["X"] * n_ctx)

        logger.info(f'Initial context: "{ctx_placeholder}"')
        logger.info(f"Number of context words (tokens): {n_ctx}")

        self.ctx = Parameter(ctx_vectors)  # to be optimized

        self.meta_net = nn.Linear(n_dom, n_dom)
        if cfg.TRAIN.PREC == "fp16":
            self.meta_net.half()

        cls_names = [name.replace("_", " ") for name in cls_names]
        name_lens = [len(_tokenizer.encode(name)) for name in cls_names]
        P_ctx_cls = [f"{ctx_placeholder} {name}." for name in cls_names]

        tokenized_P_ctx = clip.tokenize(ctx_placeholder).view(1, -1)  # [1, 77]
        tokenized_P_ctx_cls = torch.cat(
            [clip.tokenize(p) for p in P_ctx_cls]
        )  # [C, 77]
        with torch.no_grad():
            P_ctx_embedding = clip_model.token_embedding(tokenized_P_ctx).type(dtype)
            P_ctx_cls_embedding = clip_model.token_embedding(tokenized_P_ctx_cls).type(
                dtype
            )

        self.register_buffer(
            "ctx_token_prefix", P_ctx_embedding[:, :1, :]
        )  # [1, 1, d] SOS
        self.register_buffer(
            "ctx_token_suffix", P_ctx_embedding[:, 1 + n_ctx :, :]
        )  # [1, *, d] EOS

        self.register_buffer(
            "ctx_cls_token_prefix", P_ctx_cls_embedding[:, :1, :]
        )  # [C, 1, d] SOS
        self.register_buffer(
            "ctx_cls_token_suffix", P_ctx_cls_embedding[:, 1 + n_ctx :, :]
        )  # [C, *, d] . CLS, EOS

        self.n_cls = n_cls
        self.n_dom = self.ctx.shape[0]
        self.n_ctx = n_ctx
        self.tokenized_P_ctx = tokenized_P_ctx
        self.tokenized_P_ctx_cls = tokenized_P_ctx_cls
        self.name_lens = name_lens

    def get_P_ctx_cls(self):
        D = self.ctx.shape[0]
        C = self.n_cls
        ctx = self.ctx  # [D, n_ctx, d]
        ctx = ctx.unsqueeze(1).expand(-1, C, -1, -1)  # [D, C, n_ctx, d]

        prefix = self.ctx_cls_token_prefix  # [C, 1, d] SOS
        suffix = self.ctx_cls_token_suffix  # [C, *, d] . CLS EOS

        prompts = torch.cat(
            [
                prefix.unsqueeze(0).expand(D, -1, -1, -1),  # [D, C, 1, d]
                ctx,  # [D, C, n_ctx, d]
                suffix.unsqueeze(0).expand(D, -1, -1, -1),  # [D, C, *, d]
            ],
            dim=2,
        )

        return prompts  # [D, C, 77, d]

    def get_P_ctx(self):
        D = self.ctx.shape[0]

        prefix = self.ctx_token_prefix  # [1, 1, d] SOS
        suffix = self.ctx_token_suffix  # [1, *, d] EOS

        ctx = self.ctx  # [D, n_ctx, d]
        prompts = torch.cat(
            [
                prefix.expand(D, -1, -1),  # [D, 1, d] SOS
                ctx,  # [D, n_ctx, d]
                suffix.expand(D, -1, -1),  # [D, *, d] EOS
            ],
            dim=1,
        )
        return prompts  # [D, 77, d]

    def get_batch_P_ctx_cls(self, domains):
        B = domains.shape[0]
        C = self.n_cls

        prefix = self.ctx_cls_token_prefix  # [C, 1, d] SOS
        suffix = self.ctx_cls_token_suffix  # [C, *, d] . CLS EOS

        batch_ctx = self.ctx[domains]  # [B, n_ctx, d]
        prompts = torch.cat(
            [
                prefix.unsqueeze(0).expand(B, -1, -1, -1),  # [B, C, 1, d] SOS
                batch_ctx.unsqueeze(1).expand(-1, C, -1, -1),  # [B, C, n_ctx, d]
                suffix.unsqueeze(0).expand(B, -1, -1, -1),  # [B, C, *, d] . CLS EOS
            ],
            dim=2,
        )
        return prompts  # [B, C, 77, d]


class CustomCLIP(nn.Module):

    def __init__(self, cfg, clip_model, cls_names, dom_names):
        super().__init__()
        self.cfg = cfg

        self.clip_model = clip_model
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)

        self.prompt_learner = PromptLearner(cfg, clip_model, cls_names, dom_names)
        self.meta_net = self.prompt_learner.meta_net
        self.tokenized_P_ctx = self.prompt_learner.tokenized_P_ctx
        self.tokenized_P_ctx_cls = self.prompt_learner.tokenized_P_ctx_cls

    @property
    def device(self):
        return next(self.clip_model.parameters()).device

    @torch.no_grad()
    def get_dom_txt_feats(self, dom_names):
        dom_names = [name.replace("_", " ") for name in sorted(dom_names)]
        P_dom_texts = [f"{name}" for name in dom_names]
        tokenized_P_dom_texts = clip.tokenize(P_dom_texts).to(self.device)  # [D, 77]

        dom_txt_feats = self.clip_model.encode_text(tokenized_P_dom_texts)
        dom_txt_feats = F.normalize(dom_txt_feats, dim=-1)
        self.register_buffer(
            "dom_txt_feats", dom_txt_feats.type(self.dtype)
        )  # [D, 1024]

    @torch.no_grad()
    def get_cls_txt_feats(self, cls_names):
        cls_names = [name.replace("_", " ") for name in sorted(cls_names)]
        P_cls_texts = [f"{name}" for name in cls_names]

        tokenized_P_cls_texts = clip.tokenize(P_cls_texts).to(self.device)  # [C, 77]
        cls_txt_feats = self.clip_model.encode_text(tokenized_P_cls_texts)
        cls_txt_feats = F.normalize(cls_txt_feats, dim=-1)
        self.register_buffer(
            "cls_txt_feats", cls_txt_feats.type(self.dtype)
        )  # [C, 1024]

    # @torchsnooper.snoop()
    def forward(self, images, labels=None, domains=None):
        logit_scale = self.logit_scale.exp()
        prompt_learner = self.prompt_learner

        B = images.shape[0]
        C = self.tokenized_P_ctx_cls.shape[0]
        D = prompt_learner.ctx.shape[0]

        img_feats = self.image_encoder(images)  # [B, 1024]
        img_feats = F.normalize(img_feats, dim=-1)  # [B, 1024]

        P_ctx_cls = prompt_learner.get_P_ctx_cls()  # [D, C, 77, d]
        tokenized_P = self.tokenized_P_ctx_cls.unsqueeze(0).expand(
            D, -1, -1
        )  # [D, C, 77]
        ctx_cls_txt_feats = self.text_encoder(
            P_ctx_cls.view(D * C, 77, -1), tokenized_P.reshape(D * C, -1)
        )  # [D*C, 1024]
        ctx_cls_txt_feats = F.normalize(ctx_cls_txt_feats, dim=-1)
        ctx_cls_txt_feats = ctx_cls_txt_feats.view(D, C, -1)  # [D, C, 1024]

        with torch.no_grad():
            logits_list = []
            for i in range(D):
                logits_i = logit_scale * img_feats @ ctx_cls_txt_feats[i].t()  # [B, C]
                logits_list.append(logits_i.unsqueeze(1))

            ensemble_logits = torch.cat(logits_list, dim=1)  # [B, D, C]
            mean_logits = torch.mean(ensemble_logits, dim=1)  # [B, C]

        sims_by_dom = []
        for i in range(D):
            sim = img_feats @ ctx_cls_txt_feats[i].t()  # [B, C]
            sims_by_dom.append(sim.unsqueeze(1))  # [B, 1, C]
        sims_by_dom = torch.cat(sims_by_dom, dim=1)  # [B, D, C]
        dom_relevance = sims_by_dom.mean(dim=-1)  # [B, D]

        dom_w = self.meta_net(dom_relevance)  # [B, D]
        dom_w = F.softmax(dom_w, dim=-1)
        dom_w = dom_w.unsqueeze(-1).unsqueeze(-1)  # [B, D, 1, 1]

        weighted_txt_feats = (ctx_cls_txt_feats * dom_w).sum(dim=1)  # [B, C, 1024]
        weighted_logits = logit_scale * torch.bmm(
            img_feats.unsqueeze(1),  # [B, 1, 1024]
            weighted_txt_feats.permute(0, 2, 1),  # [B, 1024, C]
        )  # [B, 1, C]
        weighted_logits = weighted_logits.squeeze(1)  # [B, C]

        if (labels is not None) and (domains is not None):
            batch_ctx_cls_feats = ctx_cls_txt_feats[domains]  # [B, C, 1024]
            logits = logit_scale * torch.bmm(
                img_feats.unsqueeze(1),  # [B, 1, 1024]
                batch_ctx_cls_feats.permute(0, 2, 1),  # [B, 1024, C]
            )  # [B, 1, C]
            logits = logits.squeeze(1)  # [B, C]

            loss_dict = {}
            loss_dict["loss_ce"] = F.cross_entropy(logits, labels)
            loss_dict["loss_weighted_ce"] = F.cross_entropy(weighted_logits, labels)

            if self.cfg.METHOD.DECORR_CTX_CLS or self.cfg.METHOD.DECORR_CTX:
                P_ctx = prompt_learner.get_P_ctx()  # [D, 77, d]
                D = P_ctx.shape[0]
                tokenized_P_ctx = self.tokenized_P_ctx.expand(D, -1)  # [D, 77]
                ctx_feats = self.text_encoder(P_ctx, tokenized_P_ctx)
                ctx_feats = torch.div(
                    ctx_feats, ctx_feats.norm(dim=-1, keepdim=True)
                )  # [D, 1024]
                cls_txt_feats = self.cls_txt_feats  # [C, 1024]

                if self.cfg.METHOD.DECORR_CTX_CLS:
                    loss_dict["loss_DecorrCtxCls"] = (
                        F.cosine_similarity(
                            cls_txt_feats.unsqueeze(1), ctx_feats.unsqueeze(0), dim=2
                        )
                        .abs()
                        .mean()
                        * 10.0
                    )

                if self.cfg.METHOD.DECORR_CTX:
                    cos_sims = F.cosine_similarity(
                        ctx_feats.unsqueeze(1),  # [D, 1, 1024]
                        ctx_feats.unsqueeze(0),  # [1, D, 1024]
                        dim=2,
                    )
                    loss_dict["loss_DecorrCtx"] = (
                        cos_sims[~torch.eye(D, dtype=torch.bool).cuda()].abs().mean()
                        * 10.0
                    )

            return loss_dict, logits, mean_logits, weighted_logits
        else:
            return mean_logits, weighted_logits


@TRAINER_REGISTRY.register()
class METHOD(DGPlBaseTrainer):

    best_val_mean_acc = 0.0
    best_val_weighted_acc = 0.0
    best_val_mean_acc_iter = 0
    best_val_weighted_acc_iter = 0

    best_test_mean_acc = 0.0
    best_test_weighted_acc = 0.0
    best_test_mean_acc_iter = 0
    best_test_weighted_acc_iter = 0

    def __init__(self, cfg):
        super().__init__(cfg)
        self.mean_evaluator = build_evaluator(cfg, lab2cname=self.lab2cname)

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        dom_names = cfg.DATASET.SOURCE_DOMAINS

        logger.info(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAIN.PREC == "fp32" or cfg.TRAIN.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        logger.info("Building custom CLIP")
        self.model = CustomCLIP(cfg, clip_model, classnames, dom_names)

        logger.info("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        logger.info(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.model.get_dom_txt_feats(dom_names)
        self.model.get_cls_txt_feats(classnames)

        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model(
            "prompt_learner", self.model.prompt_learner, self.optim, self.sched
        )

        self.scaler = GradScaler() if cfg.TRAIN.PREC == "amp" else None

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            logger.info("wrap the model with DistributedDataParallel")
            # ref to https://github.com/pytorch/pytorch/issues/22049 to set `find_unused_parameters=True`
            # for part of the parameters is not updated.
            # self.model = DistributedDataParallel(
            #     self.model,
            #     device_ids=[comm.get_local_rank()],
            #     broadcast_buffers=False,
            # )
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[comm.get_local_rank()],
                broadcast_buffers=False,
            )

    def forward_backward(self, batch):
        images, labels, domains = self.parse_batch_train(batch)

        if self.cfg.TRAIN.PREC == "amp":
            with autocast():
                loss_dict, logits, mean_logits, weighted_logits = self.model(
                    images, labels, domains
                )
                losses = sum(loss_dict.values())
            self.optim.zero_grad()
            self.scaler.scale(losses).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            loss_dict, logits, mean_logits, weighted_logits = self.model(
                images, labels, domains
            )
            losses = sum(loss_dict.values())
            self.model_backward_and_update(losses)

        loss_summary = loss_dict
        loss_summary.update(
            {
                "acc": compute_accuracy(logits, labels)[0].item(),
                "mean_acc": compute_accuracy(mean_logits, labels)[0].item(),
                "weighted_acc": compute_accuracy(weighted_logits, labels)[0].item(),
            }
        )

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0
            else False
        )

        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            cur_mean_res, cur_weighted_res = self.test(split="val")
            is_best = cur_weighted_res > self.best_result
            if is_best:
                self.best_result = cur_weighted_res
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    val_result=cur_weighted_res,
                    model_name="model-best.pth.tar",
                )

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)

    @torch.no_grad()
    def test(self, split="test"):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.mean_evaluator.reset()
        self.evaluator.reset()

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        logger.info(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            mean_logits, weighted_logits = self.model_inference(input)
            self.mean_evaluator.process(mean_logits, label)
            self.evaluator.process(weighted_logits, label)

        mean_results = self.mean_evaluator.evaluate()
        weighted_results = self.evaluator.evaluate()

        for k, v in mean_results.items():
            tag = f"mean-{split}/{k}"
            self.write_scalar(tag, v, self.epoch)
        for k, v in weighted_results.items():
            tag = f"weighted-{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(mean_results.values())[0], list(weighted_results.values())[0]


def set_method_config(cfg):
    _C = cfg
    _C.SEED = 1

    # -----------------------------------------------------------------------------
    # METHOD
    # -----------------------------------------------------------------------------
    _C.METHOD = CN()
    _C.METHOD.N_CTX = 16  # number of context vectors
    _C.METHOD.DSC = True  # domain-specific context (False or True)
    _C.METHOD.CTX_INIT = ""  # initialization words
    _C.METHOD.DECORR_CTX_CLS = True
    _C.METHOD.DECORR_CTX = True


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    set_dgpl_config(cfg)
    set_method_config(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    merge_from_args(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    # 5. clean unused trainer configs
    clean_cfg(cfg, cfg.TRAINER.NAME)

    default_setup(cfg, args)

    cfg.freeze()
    return cfg


def main(args):
    cfg = setup(args)
    trainer = METHOD(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test(split="test")
        return

    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    logger.info("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
