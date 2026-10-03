from collections.abc import Callable, Sized
from typing import TypeVar

import torch
from torch.utils.data import DataLoader, Dataset

ItemT = TypeVar('ItemT')
BatchT = TypeVar('BatchT')


def make_loader(
    dataset: Dataset[ItemT],
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    drop_last: bool,
    collate_fn: Callable[[list[ItemT]], BatchT] | None = None,
) -> DataLoader[ItemT]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )


def require_val_fraction(val_fraction: float) -> None:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError('val_fraction must be in [0.0, 1.0)')


def train_split_index(item_count: int, val_fraction: float) -> int:
    """Items before the index train; items from it on validate. A `val_fraction` of 0 keeps everything for training."""
    require_val_fraction(val_fraction)
    return int(item_count * (1.0 - val_fraction))


def describe_loaders(train: Sized, val: Sized | None) -> str:
    val_batches = 0 if val is None else len(val)
    return f'DataLoaders ready: {len(train)} train / {val_batches} val batches'
