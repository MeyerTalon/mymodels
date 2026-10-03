from pathlib import Path
from typing import TypeVar

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

from core.data import make_loader, require_val_fraction

ItemT = TypeVar('ItemT')
ImageBatch = tuple[torch.Tensor, torch.Tensor]
MnistItem = tuple[torch.Tensor, int]


def load_mnist(data_dir: Path, *, train: bool, cache_only: bool) -> datasets.MNIST:
    data_dir.mkdir(parents=True, exist_ok=True)
    try:
        return datasets.MNIST(
            root=str(data_dir),
            train=train,
            download=not cache_only,
            transform=transforms.ToTensor(),
        )
    except (RuntimeError, OSError) as error:
        if cache_only:
            raise ValueError(
                f'dataset_cache_only is set but MNIST is not cached in {data_dir}; '
                'run once with dataset_cache_only: false to download it'
            ) from error
        raise


def split_train_val(
    train_full: datasets.MNIST, *, val_fraction: float, seed: int
) -> tuple[Dataset[MnistItem], Dataset[MnistItem] | None]:
    """Seeded so a checkpoint's validation images stay out of its training set across runs."""
    require_val_fraction(val_fraction)
    total = len(train_full)
    val_count = int(total * val_fraction)
    if val_count == 0:
        return train_full, None
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(
        train_full, [total - val_count, val_count], generator=generator
    )
    return train_set, val_set


def create_dataloaders(
    train_dataset: Dataset[ItemT],
    val_dataset: Dataset[ItemT] | None,
    *,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader[ItemT], DataLoader[ItemT] | None]:
    train_loader = make_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    if val_dataset is None:
        return train_loader, None
    val_loader = make_loader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    return train_loader, val_loader
