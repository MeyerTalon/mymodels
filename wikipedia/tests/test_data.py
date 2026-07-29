"""tests for data loading and preprocessing."""

import json
import random
from pathlib import Path
from typing import Any, Callable, Dict, List

import pytest

import wikipedia.data as data_module
from wikipedia.data import (
    MANIFEST_FILENAME,
    SNAPSHOT_FILENAME,
    WikipediaDataset,
    build_token_stream,
    create_dataloaders,
    load_wikipedia_texts,
)


class FakeIterableDataset:
    """small in-memory stand-in for a Hugging Face iterable dataset."""

    def __init__(self, rows: List[Dict[str, str]]) -> None:
        """stores rows for filter, shuffle, and take operations."""
        self.rows = rows

    def filter(
        self, predicate: Callable[[Dict[str, str]], bool]
    ) -> "FakeIterableDataset":
        """returns rows accepted by ``predicate``."""
        return FakeIterableDataset([row for row in self.rows if predicate(row)])

    def shuffle(self, seed: int, buffer_size: int) -> "FakeIterableDataset":
        """returns a deterministically shuffled copy."""
        del buffer_size
        rows = list(self.rows)
        random.Random(seed).shuffle(rows)
        return FakeIterableDataset(rows)

    def take(self, count: int) -> List[Dict[str, str]]:
        """returns the first ``count`` rows."""
        return self.rows[:count]


def _rows(count: int) -> List[Dict[str, str]]:
    """builds synthetic Wikimedia rows."""
    return [
        {
            "id": str(index),
            "url": f"https://example.test/{index}",
            "title": f"article {index}",
            "text": f"article text {index}",
        }
        for index in range(count)
    ]


def _fake_loader(
    rows: List[Dict[str, str]],
    calls: List[Dict[str, Any]],
) -> Callable[..., FakeIterableDataset]:
    """builds a load_dataset replacement that records arguments."""

    def loader(*args: Any, **kwargs: Any) -> FakeIterableDataset:
        """records one load and returns synthetic rows."""
        calls.append({"args": args, "kwargs": kwargs})
        return FakeIterableDataset(rows)

    return loader


def test_packed_dataset_shifts_by_one(dummy_tokenizer: Any) -> None:
    stream = list(range(1, 21))  # 20 tokens
    dataset = WikipediaDataset(stream, block_size=4)
    assert len(dataset) == (20 - 1) // 4
    input_ids, target_ids = dataset[0]
    assert input_ids.tolist() == [1, 2, 3, 4]
    assert target_ids.tolist() == [2, 3, 4, 5]
    # the second block continues contiguously from the first
    next_inputs, _ = dataset[1]
    assert next_inputs.tolist() == [5, 6, 7, 8]


def test_packed_dataset_too_short_raises() -> None:
    with pytest.raises(ValueError):
        WikipediaDataset([1, 2, 3], block_size=8)


def test_build_token_stream_inserts_eos(dummy_tokenizer: Any) -> None:
    dummy_tokenizer.eos_id = 99
    stream = build_token_stream(["ab", "cd"], dummy_tokenizer)
    expected = (
        dummy_tokenizer.encode("ab")
        + [99]
        + dummy_tokenizer.encode("cd")
        + [99]
    )
    assert stream == expected


def test_streams_deterministic_nonempty_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _rows(6) + [{"id": "empty", "url": "", "title": "", "text": "  "}]
    first_calls: List[Dict[str, Any]] = []
    monkeypatch.setattr(data_module, "load_dataset", _fake_loader(rows, first_calls))
    first = load_wikipedia_texts(
        str(tmp_path / "first"),
        n_articles=4,
        dataset_seed=7,
        shuffle_buffer_size=3,
    )

    second_calls: List[Dict[str, Any]] = []
    monkeypatch.setattr(data_module, "load_dataset", _fake_loader(rows, second_calls))
    second = load_wikipedia_texts(
        str(tmp_path / "second"),
        n_articles=4,
        dataset_seed=7,
        shuffle_buffer_size=3,
    )

    assert first == second
    assert len(first) == 4
    assert all(text.strip() for text in first)
    assert first_calls[0]["args"] == ("wikimedia/wikipedia", "20231101.en")
    assert first_calls[0]["kwargs"]["streaming"] is True
    assert (tmp_path / "first" / SNAPSHOT_FILENAME).exists()
    assert (tmp_path / "first" / MANIFEST_FILENAME).exists()


def test_larger_snapshot_serves_smaller_request_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: List[Dict[str, Any]] = []
    monkeypatch.setattr(data_module, "load_dataset", _fake_loader(_rows(8), calls))
    full = load_wikipedia_texts(str(tmp_path), n_articles=6)

    def unexpected_load(*args: Any, **kwargs: Any) -> None:
        """fails if snapshot reuse attempts Hub access."""
        raise AssertionError(f"unexpected Hub load: {args}, {kwargs}")

    monkeypatch.setattr(data_module, "load_dataset", unexpected_load)
    subset = load_wikipedia_texts(
        str(tmp_path),
        n_articles=3,
        dataset_cache_only=True,
    )

    assert subset == full[:3]
    assert len(calls) == 1


def test_cache_only_rejects_missing_or_mismatched_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="no compatible Wikipedia snapshot"):
        load_wikipedia_texts(
            str(tmp_path),
            n_articles=2,
            dataset_cache_only=True,
        )

    calls: List[Dict[str, Any]] = []
    monkeypatch.setattr(data_module, "load_dataset", _fake_loader(_rows(4), calls))
    load_wikipedia_texts(str(tmp_path), n_articles=3, dataset_seed=1)
    with pytest.raises(ValueError, match="no compatible Wikipedia snapshot"):
        load_wikipedia_texts(
            str(tmp_path),
            n_articles=2,
            dataset_seed=2,
            dataset_cache_only=True,
        )


def test_corrupt_snapshot_is_not_reused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: List[Dict[str, Any]] = []
    monkeypatch.setattr(data_module, "load_dataset", _fake_loader(_rows(4), calls))
    load_wikipedia_texts(str(tmp_path), n_articles=3)
    snapshot_path = tmp_path / SNAPSHOT_FILENAME
    with snapshot_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(_rows(1)[0]) + "\n")

    with pytest.raises(ValueError, match="no compatible Wikipedia snapshot"):
        load_wikipedia_texts(
            str(tmp_path),
            n_articles=2,
            dataset_cache_only=True,
        )


def test_create_dataloaders_with_val_split(dummy_tokenizer: Any) -> None:
    train_loader, val_loader = create_dataloaders(
        texts=["hello world " * 40 for _ in range(3)],
        tokenizer=dummy_tokenizer,
        block_size=8,
        batch_size=2,
        val_fraction=0.2,
    )
    inputs, targets = next(iter(train_loader))
    assert inputs.shape == (2, 8)
    assert targets.shape == (2, 8)
    assert val_loader is not None
