import logging
import torch
from config import TrainingConfig
from data import FashionMNIST
from model import AnyNet
from trainer import Trainer

def set_up_logger() -> logging.Logger:

    logging.basicConfig(
        level = logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def main():
    config = TrainingConfig
    logger = set_up_logger()

    set_seed(config.seed)

    data_module = FashionMNIST(
        dir = config.data_dir,
        batch_size=config.batch_size,
        device=config.device,
        num_of_workers=config.num_workers,
        input_shape=config.input_shape
    )

    model = AnyNet