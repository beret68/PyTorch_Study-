import torch
import logging
from torch import nn
from torch.optim import Optimizer

class Trainer:
    def __init__(self, model: nn.Module, train_loader, val_loader,optimizer: Optimizer, config, logger: logging.Logger):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.config = config
        self.logger = logger
        self.loss_fn = nn.CrossEntropyLoss()

        self.best_acc = 0.0

    def train_epoch(self) -> None:
        self.model.train()
        for X, y in self.train_loader:
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

    def validate(self) -> float:
        self.model.eval()
        total_loss, correct = 0.0, 0

        with torch.no_grad():
            for X,y in self.val_loader:
                y_hat = self.model(X)
                total_loss += self.loss_fn(y_hat,y).item()
                correct += (y_hat.argmax(dim=1) == y).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / len(self.val_loader.dataset)

        self.logger.info(f"Val Loss: {avg_loss:.4f} | Val Acc: {accuracy * 100:.2f}%")
        return accuracy

    def fit(self):
        self.logger.info(f"Starting training on {self.config.device} for {self.config.epochs} epochs")

        for epoch in range(self.config.epochs):
            self.logger.info(f"--- Epoch {epoch + 1}/{self.config.epochs} ---")

            self.train_epoch()
            val_acc = self.validate()

            #checkpointing saving the best current model
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                torch.save(self.model.state_dict(), self.config.save_path)
                self.logger.info(f" New best model saved! (Acc: {self.best_acc * 100:.2f}%")

#Function used in hyper parametrization by Optuna.
    def train(self) -> float:
        best_acc = 0.0
        for epoch in range(self.config.epochs):

            self.train_epoch()
            val_acc = self.validate()

            #Save best_acc
            if val_acc > self.best_acc:
                best_acc = val_acc

        return best_acc