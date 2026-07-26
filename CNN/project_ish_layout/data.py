import torch
import torchvision
from torchvision import transforms
from typing import Iterator, Tuple

class DeviceDataLoader:
    """Wraps a dataloader into the correct device"""
    def __init__(self, dataloader: torch.utils.data.DataLoader, device: torch.device):
        self.dataloader = dataloader
        self.device = device
        self.dataset = dataloader.dataset

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        for X, y in self.dataloader:
            yield X.to(self.device), y.to(self.device)

    def __len__(self) -> int:
        return len(self.dataloader)

class FashionMNISTDataModule:
    def __init__(self, data_dir: str, batch_size: int, device: torch.device, num_workers):
                self.batch_size = batch_size
                self.device = device
                self.num_workers = num_workers
                self.transform = transforms.Compose([transforms.Resize((28,28)),
                                                     transforms.ToTensor()])
                self.validation = torchvision.datasets.FashionMNIST(root=data_dir, train=False, transform=self.transform,
                                                                    download=True)
                self.train = torchvision.datasets.FashionMNIST(root=data_dir, train=True, transform=self.transform,
                                                                    download=True)
    def get_train_loader(self) -> DeviceDataLoader:
        use_pin = self.device.type in ["cuda", "mps"]

        loader = torch.utils.data.DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=use_pin,
            drop_last=True
        )
        return DeviceDataLoader(loader, self.device)

    def get_validation_loader(self) -> DeviceDataLoader:
        use_pin = self.device.type in ["cuda", "mps"]

        loader = torch.utils.data.DataLoader(
            self.validation,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=use_pin,
            drop_last=False
        )
        return DeviceDataLoader(loader, self.device)