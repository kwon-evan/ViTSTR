import os
import warnings

import torch
import torch.utils.data
from rich import print
from PIL import Image
import torchvision.transforms as transforms

from vitstr import load_ViTSTR

warnings.filterwarnings(action="ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SRC = "demo_images"
DST = "warped"


def TPS(model, opt):
    print("Generating warped images...")
    for img_name in sorted(os.listdir(SRC)):
        image = Image.open(f"{SRC}/{img_name}")
        image_size = image.size

        to_tensor = transforms.ToTensor()
        to_pil = transforms.ToPILImage()

        image = image.resize((opt.imgW, opt.imgH), Image.ANTIALIAS).convert("L")
        image = to_tensor(image).to(device).unsqueeze(0)
        image.sub_(0.5).div_(0.5)  # Image Normalize

        """ Transformation stage """
        warped = model.Transformation(image)

        warped.mul_(0.5).add_(0.5)  # Image UnNormalize
        warped = warped.squeeze(0)
        warped = to_pil(warped)
        warped = warped.resize(image_size, Image.ANTIALIAS)
        warped.save(f"{DST}/{img_name}", dpi=(200, 200))
    print("Done!")


if __name__ == "__main__":
    """load configuration"""
    model, opt = load_ViTSTR("config.yaml")
    model.eval().freeze()

    os.makedirs(DST, exist_ok=True)
    print("warped images will be saved in", DST)

    TPS(model, opt)
