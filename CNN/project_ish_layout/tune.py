import optuna
import torch
from torch import nn
from hyperparameter_config import TrainingConfig
from data import FashionMNISTDataModule
from model import HyperParametrizationLeNET
from trainer import Trainer
import logging

def setup_logger() -> logging.Logger:
    """Configures standard output logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)

def objective(trial: optuna.Trial) -> float:
    # 1. Let Optuna suggest values for this specific run
    config = TrainingConfig(
        # Log-uniform search is great for learning rates (e.g., 0.0001 to 0.1)
        lr=trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-10, 1e-7, log=True),

        # Categorical choices for network width
        conv1_channels=trial.suggest_categorical("conv1_channels", [6, 16, 32]),
        conv2_channels=trial.suggest_categorical("conv2_channels", [16, 32, 64]),
        # linear1_size=trial.suggest_categorical("linear1_size", [6, 16, 32]),
        # linear2_size=trial.suggest_categorical("linear2_size", [16, 32, 64]),

        # Linear search for dropout
        dropout_rate=trial.suggest_float("dropout_rate", 0.0, 0.5),

        # Categorical for activation functions
        activation=trial.suggest_categorical("activation", ["relu", "tanh"]),

        epochs=10  # Keeping epochs relatively low during search to save time
    )

    # 2. Build the pipeline
    logger = setup_logger()


    data_module = FashionMNISTDataModule(data_dir=config.data_dir, batch_size=config.batch_size , device=config.device, num_workers=config.num_of_workers)

    model = HyperParametrizationLeNET(config).to(config.device)
    model.initialize_model()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    # 4. Train
    trainer = Trainer(
        model=model,
        train_loader=data_module.get_train_loader(),
        val_loader=data_module.get_validation_loader(),
        optimizer=optimizer,
        config=config,
        logger=logger
    )

    final_val_acc = trainer.train()
    # 5. Return the score for Optuna to optimize
    return final_val_acc


if __name__ == "__main__":
    # Create a study object and specify if we want to maximize or minimize the returned metric
    study = optuna.create_study(direction="maximize", study_name="LeNet_FashionMNIST")

    # Run 50 different configurations (trials)
    study.optimize(objective, n_trials=50)

    # Print the best results
    print("\n=== Best Trial ===")
    print(f"Accuracy: {study.best_value:.4f}")
    print("Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")