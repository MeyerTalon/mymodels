"""tests for data loading and preprocessing."""

import json
from pathlib import Path
from typing import Any

import pytest

from western.data import (
    MANIFEST_FILENAME,
    SNAPSHOT_FILENAME,
    WesternDataset,
    build_token_stream,
    create_dataloaders,
    load_western_texts,
)

_NOVEL_A = (
    "The wind came down off the high plains and the rider sat his horse "
    "in the long grass until the sun dropped behind the ridge. " * 4
)
_NOVEL_B = (
    "Dust hung over the cattle trail and the men rode in silence toward "
    "the river crossing at dusk. " * 4
)
_NOVEL_C = (
    "She kept the ranch while the herd moved north, watching the empty "
    "road for a letter that never came. " * 4
)


def _write_corpus(tmp_path: Path) -> Path:
    """writes three fixture novels under ``tmp_path/corpus``."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "high_plains.txt").write_text(_NOVEL_A, encoding="utf-8")
    (corpus_dir / "cattle_trail.txt").write_text(_NOVEL_B, encoding="utf-8")
    (corpus_dir / "empty_road.txt").write_text(_NOVEL_C, encoding="utf-8")
    return corpus_dir


def test_packed_dataset_shifts_by_one(dummy_tokenizer: Any) -> None:
    stream = list(range(1, 21))  # 20 tokens
    dataset = WesternDataset(stream, block_size=4)
    assert len(dataset) == (20 - 1) // 4
    input_ids, target_ids = dataset[0]
    assert input_ids.tolist() == [1, 2, 3, 4]
    assert target_ids.tolist() == [2, 3, 4, 5]
    # the second block continues contiguously from the first
    next_inputs, _ = dataset[1]
    assert next_inputs.tolist() == [5, 6, 7, 8]


def test_packed_dataset_too_short_raises() -> None:
    with pytest.raises(ValueError):
        WesternDataset([1, 2, 3], block_size=8)


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


def test_loads_local_corpus_and_caches_snapshot(tmp_path: Path) -> None:
    corpus_dir = _write_corpus(tmp_path)
    texts = load_western_texts(str(tmp_path), corpus_dir=str(corpus_dir))

    assert len(texts) == 3
    assert all(text.strip() for text in texts)
    assert (tmp_path / SNAPSHOT_FILENAME).exists()
    assert (tmp_path / MANIFEST_FILENAME).exists()


def test_missing_corpus_raises_with_path(tmp_path: Path) -> None:
    missing = tmp_path / "corpus"
    with pytest.raises(ValueError, match="no western corpus found"):
        load_western_texts(str(tmp_path), corpus_dir=str(missing))


def test_larger_snapshot_serves_smaller_request(
    tmp_path: Path,
) -> None:
    corpus_dir = _write_corpus(tmp_path)
    full = load_western_texts(str(tmp_path), corpus_dir=str(corpus_dir))
    subset = load_western_texts(
        str(tmp_path),
        corpus_dir=str(corpus_dir),
        max_works=2,
        dataset_cache_only=True,
    )
    assert subset == full[:2]


def test_cache_only_rejects_missing_snapshot(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no compatible western snapshot"):
        load_western_texts(str(tmp_path), dataset_cache_only=True)


def test_corrupt_snapshot_is_not_reused(tmp_path: Path) -> None:
    corpus_dir = _write_corpus(tmp_path)
    load_western_texts(str(tmp_path), corpus_dir=str(corpus_dir))
    snapshot_path = tmp_path / SNAPSHOT_FILENAME
    with snapshot_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps({"id": "x", "title": "x", "text": "extra"}) + "\n")

    with pytest.raises(ValueError, match="no compatible western snapshot"):
        load_western_texts(
            str(tmp_path),
            corpus_dir=str(corpus_dir),
            max_works=1,
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
