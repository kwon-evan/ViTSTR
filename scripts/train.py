import os
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
from rich import print

from components.datamodule import DataModule
from components.pl_wrapper import Model
from vitstr import ModelConfig


warnings.filterwarnings(action="ignore")


def train(opt):
    cfg = ModelConfig()
    # Merge config with opt
    for k, v in vars(cfg).items():
        if k not in vars(opt):
            setattr(opt, k, v)
    if opt.kor:
        kor_character = "0123456789가강거경계고관광구금기김나남너노누다대더도동두등라러로루리마머명모무문미바배버보부북사산서소수시아악안양어연영오용우울원육이인자작저전조주중지차천초추충카타파평포하허호홀히"
        opt.character = kor_character
    opt.pretrained = True
    print(opt)

    dm = DataModule(opt)
    model = Model(opt)

    if opt.saved_model:
        try:
            model = Model.load_from_checkpoint(opt.saved_model, opt=opt)
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
            StochasticWeightAveraging(
                swa_lrs=opt.swa_lrs,
                swa_epoch_start=opt.swa_epoch_start,
            ),
        ],
        # logger=WandbLogger(project="ViTSTR"),
    )

    trainer.fit(model, dm)


if __name__ == "__main__":
    """load configuration"""
    opt = Namespace(**yaml.safe_load(open("scripts/components/config.yaml", "r")))

    if not opt.exp_name:
        opt.exp_name = f"ViTSTR-Seed{opt.manualSeed}"

    os.makedirs(f"./saved_models/{opt.exp_name}", exist_ok=True)

    """ Seed and GPU setting """
    pl.seed_everything(opt.manualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    train(opt)
