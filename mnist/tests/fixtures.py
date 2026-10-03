import torch
from torch.utils.data import TensorDataset

SMALL_CNN = {
    'in_channels': 1,
    'conv1_channels': 4,
    'conv2_channels': 8,
    'hidden_dim': 16,
    'num_classes': 10,
    'dropout': 0.0,
}


def digits(count: int) -> TensorDataset:
    return TensorDataset(torch.rand(count, 1, 28, 28), torch.arange(count) % 10)
