import torch
import torchvision
from torchvision import transforms

class DeviceDataLoader:
    def __init__(self, dataloader: torch.utils.data.DataLoader, device: torch.device):
        self.dataloader = dataloader
        self.device = device
        self.dataset = dataloader.dataset

    def __iter__(self):
        for X, y in self.dataloader:
            yield X.to(self.device), y.to(self.device)

    def __len__(self):
        return len(self.dataloader)

class FashionMNIST:
    def __init__(self, dir: str, batch_size: int, device: torch.device, num_of_workers: int, input_shape: tuple=(28,28)):
        self.batch_size = batch_size
        self.device = device
        self.num_of_workers = num_of_workers
        transform = transforms.Compose([transforms.Resize(input_shape), transforms.ToTensor()])

        self.training_data = torchvision.datasets.FashionMNIST(root=dir, transform=transform, train=True, download=True)
        self.validation_data = torchvision.datasets.FashionMNIST(root=dir, transform=transform, train=False, download=True)

    def get_training_data(self):
        use_pin = self.device.type in ["cuda", "mps"]

        loader = torch.utils.data.DataLoader(
            dataset=self.training_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_of_workers,
            pin_memory=use_pin,
            drop_last=True
        )
        return DeviceDataLoader(loader, device=self.device)

    def get_validation_data(self):
        use_pin = self.device.type in ["cuda", "mps"]

        loader = torch.utils.data.DataLoader(
            dataset=self.validation_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_of_workers,
            pin_memory=use_pin,
            drop_last=True
        )
        return DeviceDataLoader(loader, device=self.device)