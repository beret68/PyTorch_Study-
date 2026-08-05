import torch

class Trainer:
    def __init__(self, model, data, config, logger, optimizer, loss_fn):
        self.model = model
        self.train_loader = data.get_training_data()
        self.validation_loader = data.get_validation_data()
        self.config = config
        self.logger = logger
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def fit(self):
        self.logger.info(f"Starting training on {self.config.device} for {self.config.epochs} epochs")

        for epoch in range(self.config.epochs):
            self.logger.info(f"--- Epoch {epoch + 1}/{self.config.epochs} ---")

            self.fit_epoch()
            val_acc = self.validation()

            # checkpointing saving the best current model
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                torch.save(self.model.state_dict(), self.config.save_path)
                self.logger.info(f" New best model saved! (Acc: {self.best_acc * 100:.2f}%")

    def fit_epoch(self):
        self.model.train()
        for X, y in self.train_loader:
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

    def validation(self):
        self.model.eval()
        total_loss, correct = 0.0, 0

        with torch.no_grad():
            for X, y in self.validation_loader:
                y_hat = self.model(X)
                total_loss += self.loss_fn(y_hat, y).item()
                correct += (y_hat.argmax(dim=1) == y).sum().item()

        avg_loss = total_loss / len(self.validation_loader)
        accuracy = correct / len(self.validation_loader.dataset)

        self.logger.info(f"Val Loss: {avg_loss:.4f} | Val Acc: {accuracy * 100:.2f}%")
        return accuracy
