"""MNIST dataset loading and DataLoader construction.

downloads MNIST via torchvision into a reusable local cache under the package
data directory. unit tests should pass a synthetic tensor dataset instead of
hitting the network.
"""

from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


class TensorImageDataset(Dataset):
    """in-memory ``(image, label)`` dataset used by tests and helpers."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        """stores image and label tensors.

        Args:
            images: float tensor of shape (n, channels, height, width).
            labels: long tensor of shape (n,).
        """
        if images.size(0) != labels.size(0):
            raise ValueError(
                f"images ({images.size(0)}) and labels ({labels.size(0)}) "
                "must have the same length."
            )
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        """returns the number of examples."""
        return int(self.labels.size(0))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """returns the ``(image, label)`` pair at ``idx``."""
        return self.images[idx], self.labels[idx]


def load_mnist_datasets(
    data_dir: str,
    val_fraction: float = 0.1,
    dataset_cache_only: bool = False,
    seed: int = 42,
) -> Tuple[Dataset, Optional[Dataset], Dataset]:
    """loads MNIST train/val/test splits from a local torchvision cache.

    Args:
        data_dir: directory passed to ``torchvision.datasets.MNIST`` as ``root``.
        val_fraction: fraction of the official train split held out for
            validation (``0.0`` disables the validation set).
        dataset_cache_only: when True, do not download; require files already
            present under ``data_dir``.
        seed: RNG seed for the train/val split.

    Returns:
        ``(train, val, test)`` datasets. ``val`` is ``None`` when
        ``val_fraction`` is 0.

    Raises:
        ValueError: if ``val_fraction`` is invalid, or cache-only mode is set
            and MNIST has not been downloaded yet.
    """
    if val_fraction < 0.0 or val_fraction >= 1.0:
        raise ValueError("val_fraction must be in [0.0, 1.0).")

    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    transform = transforms.ToTensor()
    download = not dataset_cache_only
    try:
        train_full = datasets.MNIST(
            root=str(root), train=True, download=download, transform=transform
        )
        test_set = datasets.MNIST(
            root=str(root), train=False, download=download, transform=transform
        )
    except (RuntimeError, OSError) as exc:
        if dataset_cache_only:
            raise ValueError(
                "dataset_cache_only=True but MNIST is not cached in "
                f"{data_dir}. run once with dataset_cache_only: False to "
                "download it."
            ) from exc
        raise

    if val_fraction == 0.0:
        print(f"Loaded MNIST from {root}: {len(train_full)} train / {len(test_set)} test")
        return train_full, None, test_set

    n_total = len(train_full)
    n_val = int(n_total * val_fraction)
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(
        train_full, [n_train, n_val], generator=generator
    )
    print(
        f"Loaded MNIST from {root}: {len(train_set)} train / "
        f"{len(val_set)} val / {len(test_set)} test"
    )
    return train_set, val_set, test_set


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """builds train (and optional validation) dataloaders.

    Args:
        train_dataset: training ``Dataset`` of ``(image, label)`` pairs.
        val_dataset: optional validation ``Dataset``.
        batch_size: batch size for the loaders.
        shuffle: whether to shuffle the training set.
        num_workers: number of subprocesses for data loading.

    Returns:
        a ``(train_loader, val_loader)`` tuple; ``val_loader`` is ``None`` when
        ``val_dataset`` is ``None``.
    """
    train_loader = _make_loader(train_dataset, batch_size, shuffle, num_workers)
    val_loader = (
        _make_loader(val_dataset, batch_size, False, num_workers)
        if val_dataset is not None
        else None
    )
    return train_loader, val_loader


def _make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    """creates a DataLoader with sensible cross-platform defaults."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=False,
    )
