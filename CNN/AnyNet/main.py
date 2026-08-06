import logging
import torch
from torch import nn
from config import TrainingConfig
from data import FashionMNIST
from model import RegNet
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

    model = RegNet(
        config.depths, config.in_channels, config.out_channels, config.group_width, config.bottleneck_multiplier,
        config.input_channels, config.stem_channels, config.num_classes
    )

    model.initialize_model(config.device)
    # Changing the optimizer as sugggested by LLM (lol)
    # optimizer = torch.optim.SGD(
    #     params=model.parameters(),
    #     lr=config.lr,
    #     weight_decay=config.weight_decay
    # )

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=1e-3,
        weight_decay=1e-2
    )

    #learning rate scheduler again added as suggested by LLM
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=1e-6
    )


    loss_fn = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        data=data_module, #TODO This part has a low cohesion. Should be split into 'train_data' and 'validation_data'
        config=config,
        logger=logger,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn
    )

    trainer.fit()

if __name__ == "__main__":
    main()