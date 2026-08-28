"""simple convolutional classifier for MNIST digits.

a small, readable CNN of conv+relu+pool blocks followed by a linear head.
uses native ``nn.Conv2d`` / ``nn.MaxPool2d``; this is a teaching/demo model,
not a residual network.
"""

import torch
import torch.nn as nn


class MnistCNN(nn.Module):
    """two-block conv net that maps 28x28 grayscale digits to 10 class logits.

    layout: conv-relu-pool, conv-relu-pool, flatten, linear-relu-dropout, linear.
    ``forward`` returns raw logits; argmax / softmax belong in inference.
    """

    def __init__(
        self,
        in_channels: int = 1,
        conv1_channels: int = 32,
        conv2_channels: int = 64,
        hidden_dim: int = 128,
        num_classes: int = 10,
        dropout: float = 0.25,
    ) -> None:
        """initializes the MNIST CNN.

        Args:
            in_channels: number of input image channels (1 for MNIST).
            conv1_channels: output channels of the first conv layer.
            conv2_channels: output channels of the second conv layer.
            hidden_dim: width of the hidden linear layer.
            num_classes: number of output classes (10 digits).
            dropout: dropout probability applied before the output layer.
        """
        super().__init__()
        self.num_classes = num_classes

        # 28x28 -> 14x14 after the first pool, 7x7 after the second.
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, conv1_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(conv1_channels, conv2_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv2_channels * 7 * 7, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """computes class logits for a batch of images.

        Args:
            x: float tensor of shape (batch, channels, 28, 28).

        Returns:
            logits tensor of shape (batch, num_classes).
        """
        return self.classifier(self.features(x))  # (batch, num_classes)
