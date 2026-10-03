from collections.abc import Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from core.data import make_loader, train_split_index
from core.tokenizer import TextTokenizer

TokenBatch = tuple[torch.Tensor, torch.Tensor]


class PackedDataset(Dataset[TokenBatch]):
    """Cuts one token stream into `block_size` windows; targets are the inputs shifted by one."""

    def __init__(self, token_ids: Sequence[int], block_size: int) -> None:
        if len(token_ids) < block_size + 1:
            raise ValueError(
                f'token stream of length {len(token_ids)} is too short for '
                f'block_size={block_size}; need at least {block_size + 1} tokens'
            )
        self.block_size = block_size
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.block_count = (len(self.data) - 1) // block_size

    def __len__(self) -> int:
        return self.block_count

    def __getitem__(self, index: int) -> TokenBatch:
        start = index * self.block_size
        chunk = self.data[start : start + self.block_size + 1]
        return chunk[:-1], chunk[1:]


def build_token_stream(texts: Sequence[str], tokenizer: TextTokenizer) -> list[int]:
    """Ends every text with `eos_id` so the model learns document boundaries."""
    stream: list[int] = []
    for text in texts:
        stream.extend(tokenizer.encode(text))
        stream.append(tokenizer.eos_id)
    return stream


def create_dataloaders(
    texts: Sequence[str],
    tokenizer: TextTokenizer,
    *,
    block_size: int,
    batch_size: int,
    val_fraction: float,
    num_workers: int,
) -> tuple[DataLoader[TokenBatch], DataLoader[TokenBatch] | None]:
    """Splits at the token level so train and validation never share a block; no validation loader if the tail is too short."""
    stream = build_token_stream(texts, tokenizer)
    split = train_split_index(len(stream), val_fraction)
    train_loader = make_loader(
        PackedDataset(stream[:split], block_size),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_stream = stream[split:]
    if len(val_stream) < block_size + 1:
        return train_loader, None
    val_loader = make_loader(
        PackedDataset(val_stream, block_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    return train_loader, val_loader
