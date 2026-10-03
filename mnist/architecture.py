from dataclasses import dataclass

import torch
from torch import nn

from core.config import Config, require_float, require_int

IMAGE_SIDE_PX = 28
CONV_KERNEL_PX = 3
CONV_PADDING_PX = 1
POOL_KERNEL_PX = 2
POOL_COUNT = 2
FEATURE_SIDE_PX = IMAGE_SIDE_PX // POOL_KERNEL_PX**POOL_COUNT


@dataclass(frozen=True)
class CnnConfig:
    in_channels: int
    conv1_channels: int
    conv2_channels: int
    hidden_dim: int
    num_classes: int
    dropout: float

    @classmethod
    def from_config(cls, config: Config) -> 'CnnConfig':
        return cls(
            in_channels=require_int(config, 'in_channels'),
            conv1_channels=require_int(config, 'conv1_channels'),
            conv2_channels=require_int(config, 'conv2_channels'),
            hidden_dim=require_int(config, 'hidden_dim'),
            num_classes=require_int(config, 'num_classes'),
            dropout=require_float(config, 'dropout'),
        )


class MnistCNN(nn.Module):
    def __init__(self, settings: CnnConfig) -> None:
        super().__init__()
        self.num_classes = settings.num_classes
        self.features = nn.Sequential(
            nn.Conv2d(
                settings.in_channels,
                settings.conv1_channels,
                kernel_size=CONV_KERNEL_PX,
                padding=CONV_PADDING_PX,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=POOL_KERNEL_PX),
            nn.Conv2d(
                settings.conv1_channels,
                settings.conv2_channels,
                kernel_size=CONV_KERNEL_PX,
                padding=CONV_PADDING_PX,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=POOL_KERNEL_PX),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                settings.conv2_channels * FEATURE_SIDE_PX * FEATURE_SIDE_PX,
                settings.hidden_dim,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(settings.dropout),
            nn.Linear(settings.hidden_dim, settings.num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """(batch, channels, 28, 28) images to (batch, num_classes) logits."""
        logits: torch.Tensor = self.classifier(self.features(images))
        return logits
