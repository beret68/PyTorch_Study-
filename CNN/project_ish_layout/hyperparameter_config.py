import torch
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    #---Training hyperparameters---
    lr: float = 0.099604
    weight_decay: float = 8.371515026065028e-09
    batch_size: int = 64
    num_of_workers = 6

    # --- Model Hyperparameters ---
    conv1_channels: int = 32
    conv2_channels: int = 256
    linear1_size: int = 120
    linear2_size: int = 84
    dropout_rate: float = 0.11892544  # 0.0 means no dropout
    activation: str = "tanh"  # 'tanh' or 'relu'
    epochs: int = 50

    #---Model---
    in_channels: int = 1
    num_of_classes: int = 10

    #---Data---
    data_dir: str = "/home/beret/Documents/moje_projekty/pytorch_study/4_Linear_classification/root"


    #System
    seed: int = 42
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.mps.is_available()
        else "cpu"
    )
    save_path: str = "lenet_model.pth"