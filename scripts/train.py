import os
import random
import string
from argparse import Namespace
import yaml
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    RichProgressBar,
    ModelCheckpoint,
    EarlyStopping,
    DeviceStatsMonitor,
    StochasticWeightAveraging,
)
from pytorch_lightning.loggers.wandb import WandbLogger
import numpy as np
from rich import print

from vitstr import Model
from vitstr import DataModule

warnings.filterwarnings(action="ignore")


def train(opt):
    dm = DataModule(opt)
    model = Model(opt)

    if opt.saved_model != "":
        try:
            model.load_from_checkpoint(opt.saved_model)
            print(f"continue to train, from {opt.saved_model}")
        except:
            pass

    trainer = pl.Trainer(
        accelerator="auto",
        devices=opt.num_gpu,
        gradient_clip_val=opt.grad_clip,
        precision=16,
        max_epochs=opt.num_epoch,
        callbacks=[
            RichProgressBar(),
            DeviceStatsMonitor(),
            ModelCheckpoint(
                dirpath=f"./saved_models/{opt.exp_name}",
                monitor="val-ned",
                mode="max",
                filename="{epoch:02d}-{val-acc:.3f}",
                verbose=True,
                save_last=True,
                save_top_k=5,
            ),
            EarlyStopping(
                monitor="val-ned",
                mode="max",
                min_delta=0.00,
                patience=30,
                verbose=True,
            ),
            StochasticWeightAveraging(swa_lrs=opt.swa_lrs),
        ],
        # logger=WandbLogger(project="ViTSTR"),
    )

    trainer.fit(model, dm)


if __name__ == "__main__":
    """load configuration"""
    with open("config.yaml", "r") as f:
        opt = yaml.safe_load(f)
        print(opt)
        opt = Namespace(**opt)

    if not opt.exp_name:
        opt.exp_name = f"ViTSTR-Seed{opt.manualSeed}"
        # print(opt.exp_name)

    os.makedirs(f"./saved_models/{opt.exp_name}", exist_ok=True)

    """ Seed and GPU setting """
    pl.seed_everything()

    cudnn.benchmark = True
    cudnn.deterministic = True

    opt.num_gpu = torch.cuda.device_count()

    train(opt)
