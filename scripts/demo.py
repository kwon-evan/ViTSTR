import yaml
import torch
from argparse import Namespace
from vitstr import ViTSTR
from PIL import Image
from glob import glob

if __name__ == "__main__":
    """load configuration"""
    opt = Namespace(**yaml.safe_load(open("scripts/components/config.yaml", "r")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """load model"""
    model = ViTSTR.load_from(opt.saved_model).to(device)

    """inference"""
    images = [Image.open(path) for path in glob("demo_images/*.jpg")]
    for image in images:
        print(model.imread(image))
