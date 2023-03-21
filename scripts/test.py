import yaml
import torch
import torch.backends.cudnn as cudnn
from argparse import Namespace
import pytorch_lightning as pl

from components.datamodule import DataModule
from components.pl_wrapper import Model
from vitstr import ModelConfig


def test(opt):
    cfg = ModelConfig()
    # Merge config with opt
    for k, v in vars(cfg).items():
        if k not in vars(opt):
            setattr(opt, k, v)
    if opt.kor:
        kor_character = "0123456789가강거경계고관광구금기김나남너노누다대더도동두등라러로루리마머명모무문미바배버보부북사산서소수시아악안양어연영오용우울원육이인자작저전조주중지차천초추충카타파평포하허호홀히"
        opt.character = kor_character
    print(opt)

    dm = DataModule(opt)
    model = Model.load_from_checkpoint(opt.saved_model, opt=opt).eval()
    model.freeze()

    trainer = pl.Trainer(
        accelerator="auto",
        devices=opt.num_gpu,
        precision=16,
    )

    trainer.test(model, dm)


if __name__ == "__main__":
    """load configuration"""
    opt = Namespace(**yaml.safe_load(open("scripts/components/config.yaml", "r")))

    """ Seed and GPU setting """
    pl.seed_everything(opt.manualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    test(opt)
