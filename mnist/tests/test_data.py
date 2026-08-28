"""tests for MNIST dataset helpers (synthetic tensors only)."""

from typing import Tuple

import pytest
import torch

from mnist.data import TensorImageDataset, create_dataloaders


def _synthetic_split(
    n_train: int = 16, n_val: int = 8
) -> Tuple[TensorImageDataset, TensorImageDataset]:
    """builds tiny in-memory MNIST-shaped splits."""
    train = TensorImageDataset(
        torch.rand(n_train, 1, 28, 28),
        torch.randint(0, 10, (n_train,)),
    )
    val = TensorImageDataset(
        torch.rand(n_val, 1, 28, 28),
        torch.randint(0, 10, (n_val,)),
    )
    return train, val


def test_tensor_dataset_length_and_item() -> None:
    images = torch.rand(5, 1, 28, 28)
    labels = torch.arange(5)
    dataset = TensorImageDataset(images, labels)
    assert len(dataset) == 5
    image, label = dataset[2]
    assert image.shape == (1, 28, 28)
    assert int(label.item()) == 2


def test_tensor_dataset_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        TensorImageDataset(torch.rand(3, 1, 28, 28), torch.arange(2))


def test_create_dataloaders_shapes() -> None:
    train, val = _synthetic_split()
    train_loader, val_loader = create_dataloaders(
        train, val, batch_size=4, shuffle=False, num_workers=0
    )
    images, labels = next(iter(train_loader))
    assert images.shape == (4, 1, 28, 28)
    assert labels.shape == (4,)
    assert val_loader is not None
    val_images, val_labels = next(iter(val_loader))
    assert val_images.shape[0] <= 4
    assert val_labels.ndim == 1


def test_create_dataloaders_without_val() -> None:
    train, _ = _synthetic_split()
    train_loader, val_loader = create_dataloaders(train, None, batch_size=4)
    assert val_loader is None
    assert len(train_loader) > 0
