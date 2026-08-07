import torch

class Trainer:
    def __init__(self, model, train_loader, validation_loader, epochs, save_path, logger, optimizer, scheduler, loss_fn, device):
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.epochs = epochs
        self.save_path = save_path
        self.logger = logger
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device

    def fit(self):
        self.logger.info(f"Starting training on {self.device} for {self.epochs} epochs")

        for epoch in range(self.epochs):
            self.logger.info(f"--- Epoch {epoch + 1}/{self.epochs} ---")

            self.fit_epoch()
            val_acc = self.validation()

            #Scheduler step so the learning rate can be changed with every epoch
            self.scheduler.step()
            # checkpointing saving the best current model
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                torch.save(self.model.state_dict(), self.save_path)
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
