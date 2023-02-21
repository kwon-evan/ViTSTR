from torch.utils.data import DataLoader
import pytorch_lightning as pl

from vitstr.dataset import AlignCollate, RawDataset


class DataModule(pl.LightningDataModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = RawDataset(root=self.opt.train_data, opt=self.opt)
            self.valid_dataset = RawDataset(root=self.opt.valid_data, opt=self.opt)

        if stage == "test":
            self.test_dataset = RawDataset(root=self.opt.test_data, opt=self.opt)

        if stage == "predict":
            self.predict_dataset = RawDataset(root=self.opt.image_folder, opt=self.opt)

    def train_dataloader(self) -> DataLoader:
        AlignCollate_train = AlignCollate(
            imgH=self.opt.imgH, imgW=self.opt.imgW, keep_ratio_with_pad=self.opt.PAD
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.opt.batch_size,
            shuffle=True,  # 'True' to check training progress with validation function.
            num_workers=int(self.opt.workers),
            collate_fn=AlignCollate_train,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        AlignCollate_valid = AlignCollate(
            imgH=self.opt.imgH, imgW=self.opt.imgW, keep_ratio_with_pad=self.opt.PAD
        )

        return DataLoader(
            self.valid_dataset,
            batch_size=self.opt.batch_size,
            shuffle=False,
            num_workers=int(self.opt.workers),
            collate_fn=AlignCollate_valid,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        AlignCollate_valid = AlignCollate(
            imgH=self.opt.imgH, imgW=self.opt.imgW, keep_ratio_with_pad=self.opt.PAD
        )

        return DataLoader(
            self.test_dataset,
            batch_size=self.opt.batch_size,
            shuffle=False,
            num_workers=int(self.opt.workers),
            collate_fn=AlignCollate_valid,
            pin_memory=True,
        )

    def predict_dataloader(self) -> DataLoader:
        AlignCollate_predict = AlignCollate(
            imgH=self.opt.imgH, imgW=self.opt.imgW, keep_ratio_with_pad=self.opt.PAD
        )

        return DataLoader(
            self.predict_dataset,
            batch_size=self.opt.batch_size,
            shuffle=False,
            num_workers=int(self.opt.workers),
            collate_fn=AlignCollate_predict,
            pin_memory=True,
        )
