import torch
from torch import nn

def init_cnn(module: nn.Module) -> None:
    """Applies Xavier initizalization to Convolutional and Linear Layers"""
    if isinstance(module, (nn.Linear, nn.LazyLinear, nn.Conv2d, nn.LazyConv2d)):
        nn.init.xavier_uniform_(module.weight)

class LeNet(nn.Module):
    def __init__(self, in_channels: int, num_of_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, padding=2), nn.ReLU(),
            nn.AvgPool2d(kernel_size=2,stride=2),
            nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(400, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.ReLU(),
            nn.Linear(84, num_of_classes)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def initialize_model(self, device: torch.device) -> None:
        self.to(device)
        self.apply(init_cnn)
