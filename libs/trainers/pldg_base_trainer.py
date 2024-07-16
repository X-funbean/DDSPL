import logging
import os
import os.path as osp
from yacs.config import CfgNode as CN
import torch
import torch.nn as nn

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_checkpoint  # setup_logger,,
from libs.modeling.clip import clip
from libs.modeling.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from libs.trainers.pl_base_trainer import TextEncoder, PlBaseTrainer, load_clip_to_cpu

logger = logging.getLogger(
    f'fastdg.{os.path.relpath(__file__).replace(os.path.sep, ".")}'
)


@TRAINER_REGISTRY.register()
class DGPlBaseTrainer(PlBaseTrainer):

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        domain = batch["domain"]
        input = input.to(self.device, non_blocking=True)
        label = label.to(self.device, non_blocking=True)
        domain = domain.to(self.device, non_blocking=True)
        return input, label, domain


def set_dgpl_config(cfg):
    _C = cfg
    _C.DATASET.ROOT = "/root/xfb/datasets/DG"
    _C.DATASET.NAME = "PACS"
    _C.DATASET.SOURCE_DOMAINS = ("cartoon", "photo", "sketch")
    _C.DATASET.TARGET_DOMAINS = ("art_painting",)

    _C.TRAIN.VAL_FREQ = 0

    _C.TEST.FINAL_MODEL = "best_val"

    # -----------------------------------------------------------------------------
    # DG
    # -----------------------------------------------------------------------------
    _C.DOMAINBED = CN()
    _C.DOMAINBED.USE_FIXED_SPLIT = False
    # _C.DOMAINBED.USE_FIXED_SPLIT = True
    _C.DOMAINBED.HOLDOUT_FRACTION = 0.2

    # _C.DOMAINBED.TRAIN_ITERS = 15000
    # _C.DOMAINBED.CHECKPOINT_PERIOD = 1000 # iters
    # _C.DOMAINBED.USE_DATASET_CONFIG = True
    # _C.DOMAINBED.RESNET_DROPOUT = 0. # [0., 0.1, 0.5]
    # _C.DOMAINBED.FREEZE_BN = True
