from dataclasses import dataclass
from typing import Tuple


@dataclass
class ModelConfig:
    kor: bool = False  # use Korean character or not
    rgb: bool = False  # use rgb input
    pad: bool = False  # whether to keep ratio then pad for image resize
    pretrained: bool = False  # load pretrained model
    num_fiducial: int = 20  # number of fiducial points of TPS-STN
    batch_max_length: int = 10  # maximum label length
    img_size: Tuple[int, int] = (224, 224)  # image size (H, W)
    model_name: str = "vitstr_base_patch16_224"
    character: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # character label
