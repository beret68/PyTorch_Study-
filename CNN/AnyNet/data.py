import torch
import torchvision
from torchvision.transforms import v2

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
        train_transform = v2.Compose([
            # Keep as ToImage() + ToDtype(uint8) to preserve the ultra-fast uint8 pipeline
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),

            # Efficient uint8 spatial operations
            v2.Resize(size=input_shape, interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            # v2.RandomRotation(degrees=(-12,12)),
            # v2.RandomAffine(degrees=(0,0), translate=(0.08, 0.08), scale=(0.92, 1.08)),

            # Delayed casting to float32 happens at the very end
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.2860], std=[0.3530])
        ])

        val_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=input_shape, interpolation=v2.InterpolationMode.BILINEAR, antialias=True),

            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.2860], std=[0.3530])
        ])

        self.training_data = torchvision.datasets.FashionMNIST(root=dir, transform=train_transform, train=True,
                                                               download=True)
        self.validation_data = torchvision.datasets.FashionMNIST(root=dir, transform=val_transform, train=False,
                                                                 download=True)

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