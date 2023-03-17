import os
import math
import torch

from PIL import Image
from glob import glob
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pytorch_lightning as pl


class RawDataset(Dataset):
    def __init__(self, image_path_list, opt):
        self.opt = opt
        self.image_path_list = image_path_list
        self.labels = [
            os.path.split(path)[-1].split(".jpg")[0].split("-")[0]
            for path in self.image_path_list
        ]
        self.nSamples = len(self.image_path_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        try:
            if self.opt.rgb:  # for color image
                img = Image.open(self.image_path_list[index]).convert("RGB")
            else:
                img = Image.open(self.image_path_list[index]).convert("L")

        except IOError:
            print(f"Corrupted image for {index}")
            # make dummy image and dummy label for corrupted image.
            if self.opt.rgb:
                img = Image.new("RGB", (self.opt.imgW, self.opt.imgH))
            else:
                img = Image.new("L", (self.opt.imgW, self.opt.imgH))

        return (img, self.labels[index])


class ResizeNormalize(object):
    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class NormalizePAD(object):
    def __init__(self, max_size, PAD_type="right"):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = (
                img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)
            )

        return Pad_img


class AlignCollate(object):
    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)

        if self.keep_ratio_with_pad:  # same concept with 'Rosetta' paper
            resized_max_w = self.imgW
            input_channel = 3 if images[0].mode == "RGB" else 1
            transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

            resized_images = []
            for image in images:
                w, h = image.size
                ratio = w / float(h)
                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = math.ceil(self.imgH * ratio)

                resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
                resized_images.append(transform(resized_image))
                # resized_image.save('./image_test/%d_test.jpg' % w)

            image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

        else:
            transform = ResizeNormalize((self.imgW, self.imgH))
            image_tensors = [transform(image) for image in images]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors, labels


class DataModule(pl.LightningDataModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.opt.imgH, self.opt.imgW = self.opt.img_size

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_path_list = glob(
                os.path.join(self.opt.data_root, "*", "train", "*.jpg")
            )
            self.train_dataset = RawDataset(self.train_path_list, opt=self.opt)

            self.valid_path_list = glob(
                os.path.join(self.opt.data_root, "*", "valid", "*.jpg")
            )
            self.valid_dataset = RawDataset(self.valid_path_list, opt=self.opt)

            print(f"Train dataset size: {len(self.train_dataset)}")
            print(f"Valid dataset size: {len(self.valid_dataset)}")

        if stage == "test":
            self.test_path_list = glob(
                os.path.join(self.opt.data_root, "*", "test", "*.jpg")
            )
            self.test_dataset = RawDataset(self.test_path_list, opt=self.opt)
            print(f"Test dataset size: {len(self.test_dataset)}")

        if stage == "predict":
            self.test_path_list = glob(
                os.path.join(self.opt.data_root, "*", "test", "*.jpg")
            )
            self.predict_dataset = RawDataset(self.test_path_list, opt=self.opt)
            print(f"Predict dataset size: {len(self.predict_dataset)}")

    def train_dataloader(self) -> DataLoader:
        AlignCollate_train = AlignCollate(
            imgH=self.opt.imgH, imgW=self.opt.imgW, keep_ratio_with_pad=self.opt.pad
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
            imgH=self.opt.imgH, imgW=self.opt.imgW, keep_ratio_with_pad=self.opt.pad
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
            imgH=self.opt.imgH, imgW=self.opt.imgW, keep_ratio_with_pad=self.opt.pad
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
            imgH=self.opt.imgH, imgW=self.opt.imgW, keep_ratio_with_pad=self.opt.pad
        )

        return DataLoader(
            self.predict_dataset,
            batch_size=self.opt.batch_size,
            shuffle=False,
            num_workers=int(self.opt.workers),
            collate_fn=AlignCollate_predict,
            pin_memory=True,
        )
