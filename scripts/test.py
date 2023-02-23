import pytorch_lightning as pl

from vitstr import DataModule, load_ViTSTR


def test(model, opt):
    dm = DataModule(opt)
    print(model.hparams)

    trainer = pl.Trainer(
        accelerator="auto",
        devices=opt.num_gpu,
        precision=16,
    )

    trainer.test(model, dm)


if __name__ == "__main__":
    """load configuration"""
    model, opt = load_ViTSTR("config.yaml")
    model.eval().freeze()
    test(model, opt)
