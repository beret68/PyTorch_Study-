import torch
import logging
from config import TrainingConfig
from data import FashionMNISTDataModule
from model import LeNet
from trainer import Trainer

def setup_logger() -> logging.Logger:
    """Configures standard output logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)

def set_seed(seed:int) -> None:
    """Sets Random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    config = TrainingConfig
    logger = setup_logger()

    # 1. Setup Reproducibility
    set_seed(config.seed)

    # 2. Setup data
    data_module = FashionMNISTDataModule(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        device=config.device
    )

    # 3. Setup model
    model = LeNet(
        in_channels=1,
        num_of_classes=config.num_classes
    )
    model.initialize_model(config.device)

    # 4. Setup optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    # 5. Train
    trainer = Trainer(
        model=model,
        train_loader=data_module.get_train_loader(),
        val_loader=data_module.get_validation_loader(),
        optimizer=optimizer,
        config=config,
        logger=logger
    )

    trainer.fit()

if __name__ == "__main__":
    main()
