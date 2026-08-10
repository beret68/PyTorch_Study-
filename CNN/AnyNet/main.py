import logging
import torch
import wandb
from torch import nn
import hydra
from omegaconf import DictConfig, OmegaConf
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

def resolve_device(cli_device: str) -> torch.device:
    if cli_device != "auto":
        return torch.device(cli_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon
    return torch.device("cpu")

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config: DictConfig):

    # logger = set_up_logger()
    set_seed(config.experiment.seed)
    device = resolve_device(config.hardware.device)

    wandb.init(
        project=config.experiment.project_name,
        config=OmegaConf.to_container(config, resolve=True)
    )
    print(f"Running on device {device}")

    data_module = FashionMNIST(
        dir = config.data.data_dir,
        batch_size=config.training.batch_size,
        device=device,
        num_of_workers=config.data.num_workers,
        input_shape=tuple(config.data.input_shape)
    )

    model = RegNet(
        depths=config.model.depths,
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        groups_width=config.model.group_width,
        bottleneck_multiplier=config.model.bottleneck_multiplier,
        input_channels=config.model.input_channels,
        stem_channels=config.model.stem_channels,
        num_classes=config.model.num_classes
    )

    model.initialize_model(device)
    # Changing the optimizer as sugggested by LLM (lol)
    # optimizer = torch.optim.SGD(
    #     params=model.parameters(),
    #     lr=config.lr,
    #     weight_decay=config.weight_decay
    # )

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay
    )

    #learning rate scheduler again added as suggested by LLM
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.training.epochs,
        eta_min=config.training.eta_min
    )


    loss_fn = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        train_loader=data_module.get_training_data(),
        validation_loader=data_module.get_validation_data(),
        epochs=config.training.epochs,
        save_path=config.experiment.save_path,
        # logger=logger,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        config=config
    )

    trainer.fit()

    wandb.finish()

if __name__ == "__main__":
    main()