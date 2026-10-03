import pytest
import torch

from gpt.data import PackedDataset, build_token_stream, create_dataloaders
from gpt.tests.fakes import EOS_ID, CharTokenizer


def test_packed_dataset_targets_shift_by_one() -> None:
    dataset = PackedDataset(list(range(10)), block_size=3)
    assert len(dataset) == 3
    inputs, targets = dataset[1]
    assert inputs.tolist() == [3, 4, 5]
    assert targets.tolist() == [4, 5, 6]


def test_packed_dataset_rejects_short_stream() -> None:
    with pytest.raises(ValueError, match='too short'):
        PackedDataset([1, 2, 3], block_size=3)


def test_token_stream_ends_texts_with_eos() -> None:
    stream = build_token_stream(['ab', 'c'], CharTokenizer())
    assert stream.count(EOS_ID) == 2
    assert stream[2] == EOS_ID
    assert stream[-1] == EOS_ID


def test_create_dataloaders_split_and_shapes() -> None:
    texts = ['the quick brown fox jumps over the lazy dog'] * 4
    train, val = create_dataloaders(
        texts,
        CharTokenizer(),
        block_size=8,
        batch_size=2,
        val_fraction=0.25,
        num_workers=0,
    )
    inputs, targets = next(iter(train))
    assert inputs.shape == (2, 8)
    assert torch.equal(inputs[:, 1:], targets[:, :-1])
    assert val is not None


def test_create_dataloaders_without_validation() -> None:
    _, val = create_dataloaders(
        ['abcdefghij'],
        CharTokenizer(),
        block_size=4,
        batch_size=1,
        val_fraction=0.0,
        num_workers=0,
    )
    assert val is None
