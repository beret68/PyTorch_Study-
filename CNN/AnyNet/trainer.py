import torch
import wandb
from omegaconf import OmegaConf
from torchgen.gen_functionalization_type import return_from_mutable_noop_redispatch



class Trainer:
    def __init__(self, model, train_loader, validation_loader, epochs, save_path, optimizer, scheduler, loss_fn, device, config):
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.epochs = epochs
        self.save_path = save_path
        # self.logger = logger
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.config = config
        self.best_acc = 0.0

    def fit(self):
        print(f"Starting training on {self.device} for {self.epochs} epochs")

        for epoch in range(self.epochs):
            train_loss = self.fit_epoch()
            val_loss, val_acc = self.validation()
            current_lr = self.optimizer.param_groups[0]['lr']

            #Scheduler step so the learning rate can be changed with every epoch
            self.scheduler.step()

            wandb.log({
                "epoch" : epoch+1,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/accuracy": val_acc,
                "train/lr": current_lr
            })
            # checkpointing saving the best current model

            print(f"Epoch {epoch + 1:02d}/{self.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%")

            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self._save_checkpoint(epoch+1)
                # torch.save(self.model.state_dict(), self.save_path)
                # self.logger.info(f" New best model saved! (Acc: {self.best_acc * 100:.2f}%")



    def fit_epoch(self):
        self.model.train()
        total_loss, samples = 0.0, 0
        for X, y in self.train_loader:
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * len(y)
            samples += len(y)
        return total_loss/samples

    def validation(self):
        self.model.eval()
        total_loss, correct, samples = 0.0, 0, 0

        with torch.no_grad():
            for X, y in self.validation_loader:
                pred = self.model(X)
                total_loss += self.loss_fn(pred, y).item() * len(y)
                correct += (pred.argmax(dim=1) == y).sum().item()
                samples += len(y)

        # avg_loss = total_loss / len(self.validation_loader)
        # accuracy = correct / len(self.validation_loader.dataset)
        # self.logger.info(f"Val Loss: {avg_loss:.4f} | Val Acc: {accuracy * 100:.2f}%")
        return total_loss / samples, correct / samples

    def _save_checkpoint(self, epoch: int):

        checkpoint = {
            "epoch": epoch,
            "best_acc": self.best_acc,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "config": OmegaConf.to_container(self.config, resolve=True)
        }
        torch.save(checkpoint, self.config.experiment.save_path)
        print(f"--> Saved Best Checkpoint (Val Acc: {self.best_acc * 100:.2f}%) to {self.config.experiment.save_path}")