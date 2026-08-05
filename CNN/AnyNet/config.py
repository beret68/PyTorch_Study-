import torch
from dataclasses import dataclass

@dataclass
class TrainingConfig:

    #System
    seed: int = 42
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.mps.is_available() else
        "cpu"
    )

    save_pth = "AnyNet.pth"

    # Data
    batch_size: int = 128
    data_dir: str = "/home/beret/Documents/moje_projekty/pytorch_study/4_Linear_classification/root"
    num_workers: int = 6
    input_shape: tuple = (96, 96)

    #Model
    lr : float = 0.01
    weight_decay: float = 5e-7
    epochs: int = 10
    num_classes: int = 10

    input_channels: int = 1
    stem_channels: int = 32
    depths: tuple = (4, 6)
    in_channels: tuple = (32, 32)
    out_channels: tuple = (32, 80)


    group_width: int = 16
    bottleneck_multiplier: float = 1
