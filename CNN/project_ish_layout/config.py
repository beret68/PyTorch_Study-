import torch
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    #Model
    lr: float = 0.1
    weight_decay: float = 5e-7
    epochs: int = 10
    num_classes: int = 10

    #Data
    batch_size: int = 256
    data_dir: str = "/home/beret/Documents/moje_projekty/pytorch_study/4_Linear_classification/root"

    #System
    seed: int = 42
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.mps.is_available()
        else "cpu"
    )

    save_path: str = "lenet_model.pth"