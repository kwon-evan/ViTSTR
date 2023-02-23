import os
import numpy as np
import cv2
from PIL import Image

import torch
import torch.utils.data
import torchvision.transforms as transforms
import pytorch_lightning as pl

from vitstr import load_ViTSTR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rollout(attentions, discard_ratio, head_fusion):
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0 * I) / 2
            a = a / a.sum(dim=-1)

            result = torch.matmul(a, result)

    # Look at the total attention between the class token,
    # and the image patches
    mask = result[0, 0, 1:]
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1) ** 0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask


class ViTAttentionRollout:
    def __init__(
        self,
        model,
        attention_layer_name="attn_drop",
        head_fusion="mean",
        discard_ratio=0.9,
    ):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)
        self.attentions = []
        self.dummy_input = torch.LongTensor(1, opt.batch_max_length + 1).fill_(0)

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())

    def __call__(self, input_tensor):
        self.attentions = []
        with torch.no_grad():
            output = self.model(input_tensor, text=self.dummy_input)

        return rollout(self.attentions, self.discard_ratio, self.head_fusion)


def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def img2tensor(image):
    totensor = transforms.ToTensor()

    image = image.resize((opt.imgW, opt.imgH), Image.BICUBIC)
    tensor = totensor(image).to(device)
    tensor.sub_(0.5).div_(0.5)
    tensor = tensor.unsqueeze(0)

    return tensor


def img_transform(model, image):
    image_size = image.size

    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    image = image.resize((opt.imgW, opt.imgH), Image.LANCZOS).convert("L")
    image = to_tensor(image).to(device).unsqueeze(0)
    image.sub_(0.5).div_(0.5)  # Image Normalize

    """ Transformation stage """
    warped = model.Transformation(image)

    warped = warped.squeeze(0)
    warped.mul_(0.5).add_(0.5)
    warped = to_pil(warped)
    warped = warped.resize(image_size, Image.LANCZOS)
    warped = np.array(warped)
    warped = cv2.cvtColor(warped, cv2.COLOR_RGB2BGR)

    return warped


def cam(model, opt):
    images = os.listdir("demo_images")

    print("Generating CAMs...")
    for image in images:
        img_path = f"demo_images/{image}"
        image = Image.open(img_path).convert("RGB" if opt.rgb else "L")
        image_size = image.size
        input_tensor = img2tensor(image)

        grad_rollout = ViTAttentionRollout(model, discard_ratio=0.7, head_fusion="mean")
        mask = grad_rollout(input_tensor)
        mask = cv2.resize(mask, image_size)

        warped = img_transform(model, image)

        _, img_name = os.path.split(img_path)
        cv2.imwrite(f"cam/{img_name}", show_mask_on_image(warped, mask))
    print("Done!")


if __name__ == "__main__":
    """load configuration"""
    model, opt = load_ViTSTR("config.yaml")
    model.eval().freeze()

    os.makedirs("cam", exist_ok=True)
    print("CAM Images will be at 'cam' folder.")

    cam(model, opt)
