"""tests for the WikipediaBPETokenizer wrapper."""

import pytest

from wikipedia.tokenizer import WikipediaBPETokenizer

TINY_VOCAB_SIZE = 300  # must cover the 256-byte alphabet plus special tokens


def _write_corpus(data_dir) -> None:
    """writes a tiny training corpus of .txt files."""
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "a.txt").write_text(
        "the quick brown fox jumps over the lazy dog. " * 20, encoding="utf-8"
    )


def test_train_or_load_roundtrip(tmp_path):
    data_dir = tmp_path / "data"
    tok_dir = tmp_path / "tok"
    _write_corpus(data_dir)

    tokenizer = WikipediaBPETokenizer.train_or_load(
        str(data_dir), str(tok_dir), vocab_size=TINY_VOCAB_SIZE
    )
    ids = tokenizer.encode("the quick brown fox")
    assert ids and all(isinstance(i, int) for i in ids)
    assert tokenizer.decode(ids) == "the quick brown fox"
    assert tokenizer.vocab_size > 0

    # a second call must load the saved files and produce identical encodings
    reloaded = WikipediaBPETokenizer.train_or_load(
        str(data_dir), str(tok_dir), vocab_size=TINY_VOCAB_SIZE
    )
    assert reloaded.encode("fox") == tokenizer.encode("fox")


def test_train_or_load_without_data_raises(tmp_path):
    with pytest.raises(ValueError):
        WikipediaBPETokenizer.train_or_load(
            str(tmp_path / "empty"), str(tmp_path / "tok")
        )


def test_load_missing_files_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        WikipediaBPETokenizer.load(str(tmp_path))
