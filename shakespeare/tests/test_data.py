"""tests for data loading and preprocessing."""

import json
from pathlib import Path
from typing import Any, List

import pytest

import shakespeare.data as data_module
from shakespeare.data import (
    MANIFEST_FILENAME,
    SNAPSHOT_FILENAME,
    ShakespeareDataset,
    build_token_stream,
    create_dataloaders,
    load_shakespeare_texts,
)

def _padded_speech(title: str, body: str) -> str:
    """builds a work chunk longer than the parser's minimum length."""
    return f"{title}\n\n{(body + ' ') * 8}".strip()


# synthetic Gutenberg-shaped corpus with start/end markers and three works.
# titles must match shakespeare.data._WORK_TITLES exactly (including curly quotes).
_FAKE_CORPUS = (
    "*** START OF THE PROJECT GUTENBERG EBOOK THE COMPLETE WORKS ***\n\n"
    + _padded_speech(
        "THE TRAGEDY OF HAMLET, PRINCE OF DENMARK",
        "To be, or not to be, that is the question: Whether 'tis nobler in the "
        "mind to suffer the slings and arrows of outrageous fortune.",
    )
    + "\n\n\n"
    + _padded_speech(
        "THE TRAGEDY OF MACBETH",
        "Tomorrow, and tomorrow, and tomorrow, creeps in this petty pace from "
        "day to day to the last syllable of recorded time.",
    )
    + "\n\n\n"
    + _padded_speech(
        "THE TRAGEDY OF ROMEO AND JULIET",
        "But soft, what light through yonder window breaks? It is the east, "
        "and Juliet is the sun. Arise, fair sun, and kill the envious moon.",
    )
    + "\n\n*** END OF THE PROJECT GUTENBERG EBOOK THE COMPLETE WORKS ***\n"
)


def test_packed_dataset_shifts_by_one(dummy_tokenizer: Any) -> None:
    stream = list(range(1, 21))  # 20 tokens
    dataset = ShakespeareDataset(stream, block_size=4)
    assert len(dataset) == (20 - 1) // 4
    input_ids, target_ids = dataset[0]
    assert input_ids.tolist() == [1, 2, 3, 4]
    assert target_ids.tolist() == [2, 3, 4, 5]
    # the second block continues contiguously from the first
    next_inputs, _ = dataset[1]
    assert next_inputs.tolist() == [5, 6, 7, 8]


def test_packed_dataset_too_short_raises() -> None:
    with pytest.raises(ValueError):
        ShakespeareDataset([1, 2, 3], block_size=8)


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


def test_downloads_and_caches_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: List[str] = []

    def fake_download(url: str) -> str:
        """returns the synthetic corpus and records the URL."""
        calls.append(url)
        return _FAKE_CORPUS

    monkeypatch.setattr(data_module, "_download_corpus", fake_download)
    texts = load_shakespeare_texts(str(tmp_path), corpus_url="https://example.test/s.txt")

    assert len(calls) == 1
    assert len(texts) >= 2
    assert all(text.strip() for text in texts)
    assert (tmp_path / SNAPSHOT_FILENAME).exists()
    assert (tmp_path / MANIFEST_FILENAME).exists()


def test_larger_snapshot_serves_smaller_request_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: List[str] = []

    def fake_download(url: str) -> str:
        """returns the synthetic corpus and records the URL."""
        calls.append(url)
        return _FAKE_CORPUS

    monkeypatch.setattr(data_module, "_download_corpus", fake_download)
    full = load_shakespeare_texts(str(tmp_path))

    def unexpected_download(url: str) -> str:
        """fails if snapshot reuse attempts network access."""
        raise AssertionError(f"unexpected download: {url}")

    monkeypatch.setattr(data_module, "_download_corpus", unexpected_download)
    subset = load_shakespeare_texts(
        str(tmp_path),
        max_works=2,
        dataset_cache_only=True,
    )

    assert subset == full[:2]
    assert len(calls) == 1


def test_cache_only_rejects_missing_or_mismatched_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="no compatible Shakespeare snapshot"):
        load_shakespeare_texts(
            str(tmp_path),
            dataset_cache_only=True,
        )

    monkeypatch.setattr(data_module, "_download_corpus", lambda url: _FAKE_CORPUS)
    load_shakespeare_texts(str(tmp_path), corpus_url="https://example.test/a.txt")
    with pytest.raises(ValueError, match="no compatible Shakespeare snapshot"):
        load_shakespeare_texts(
            str(tmp_path),
            corpus_url="https://example.test/b.txt",
            dataset_cache_only=True,
        )


def test_corrupt_snapshot_is_not_reused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(data_module, "_download_corpus", lambda url: _FAKE_CORPUS)
    load_shakespeare_texts(str(tmp_path))
    snapshot_path = tmp_path / SNAPSHOT_FILENAME
    with snapshot_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps({"id": "x", "title": "x", "text": "extra"}) + "\n")

    with pytest.raises(ValueError, match="no compatible Shakespeare snapshot"):
        load_shakespeare_texts(
            str(tmp_path),
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
