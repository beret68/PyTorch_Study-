import typing
from typing import Any

import torch
from torch import nn
from torch.distributions import one_hot_categorical
from torch.nn import functional as F

class ResNextBLK(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, width_of_group: int, use_1x1_conv : bool=False, bottleneck_multiplier: float= 1.0, strides: int = 1) -> None:
        super().__init__()
        bottleneck_channels = int(round(out_channels * bottleneck_multiplier))

        # if not use_1x1_conv: #1x1 is only used for the first block in a stage different stages have different
        #     # in_channels but the rest of the block in a stage follow the same in_channels as the out_channel
        #     in_channels = bottleneck_channels
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3,
                                stride=strides, padding=1, groups=bottleneck_channels//width_of_group)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, stride=1)

        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        if use_1x1_conv:
            self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides)

            self.bn4 = nn.BatchNorm2d(out_channels)
        else:
            self.conv4 = None

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = F.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))

        if self.conv4:
            X = self.bn4(self.conv4(X))

        return F.relu(Y + X)

class AnyNet(nn.Module):
    def __init__(self, arch, input_channels, stem_channels, num_classes=10, lr=0.1):
        super().__init__()
        self.net = nn.Sequential(self.stem(input_channels, stem_channels), self.body(arch))
        self.net.add_module("head", nn.Sequential(
            nn.AdaptiveMaxPool2d((1,1)), nn.Flatten(),
            nn.Linear(arch[-1][2],num_classes)
        ))

    def forward(self, X):
        return self.net(X)

    def stem(self,in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                             nn.BatchNorm2d(out_channels), nn.ReLU())

    def stage(self, depth, in_channels, out_channels, width_of_groups, bot_mul: float=1.0):
        blk = []
        for i in range(depth):
            if i ==0:
                blk.append(ResNextBLK(in_channels, out_channels, width_of_groups, use_1x1_conv=True, bottleneck_multiplier=bot_mul))
            else:
                blk.append(ResNextBLK(out_channels, out_channels, width_of_groups, bottleneck_multiplier=bot_mul)) #except for the first block in a stage
                #all of the other blocks have the same number of channels
        return nn.Sequential(*blk)

    def body(self, arch):
        stage = []
        for params in arch:
            stage.append(self.stage(*params))
        return nn.Sequential(*stage)

    def initialize_model(self, device: torch.device):
        self.to(device)
        self.apply(initialize_xavier)


def initialize_xavier(module: nn.Module):
    if isinstance(module, (nn.Conv2d, nn.LazyConv2d, nn.Linear, nn.LazyLinear, nn.BatchNorm2d, nn.LazyBatchNorm2d)):
        nn.init.xavier_normal_(module.weight)


class RegNet(AnyNet):
    def __init__(self, depths: list, in_channels: list, out_channels: list, groups, bottleneck_multiplier,
                 input_channels, stem_channels, num_classes=10, lr=0.1):
        arch = []
        for depth, in_channels, out_channels in zip(depths, in_channels, out_channels):
            arch.append((depth, in_channels, out_channels, groups, bottleneck_multiplier))

        super().__init__(*arch, input_channels, stem_channels, num_classes, lr)