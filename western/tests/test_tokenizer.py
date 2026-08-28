"""tests for the WesternBPETokenizer wrapper."""

from pathlib import Path

import pytest

from western.tokenizer import WesternBPETokenizer

TINY_VOCAB_SIZE = 300  # must cover the 256-byte alphabet plus special tokens
TINY_CORPUS = ["the wind came down off the high plains at dusk. " * 20]


def test_train_or_load_roundtrip(tmp_path: Path) -> None:
    tok_dir = tmp_path / "tok"

    tokenizer = WesternBPETokenizer.train_or_load(
        TINY_CORPUS, str(tok_dir), vocab_size=TINY_VOCAB_SIZE
    )
    ids = tokenizer.encode("the wind came down")
    assert ids and all(isinstance(i, int) for i in ids)
    assert tokenizer.decode(ids) == "the wind came down"
    assert tokenizer.vocab_size > 0

    # a second call must load the saved files and produce identical encodings
    reloaded = WesternBPETokenizer.train_or_load(
        [], str(tok_dir), vocab_size=TINY_VOCAB_SIZE
    )
    assert reloaded.encode("plains") == tokenizer.encode("plains")


def test_special_token_ids(tmp_path: Path) -> None:
    tok_dir = tmp_path / "tok"

    tokenizer = WesternBPETokenizer.train_or_load(
        TINY_CORPUS, str(tok_dir), vocab_size=TINY_VOCAB_SIZE
    )
    # special tokens occupy the conventional leading ids
    assert tokenizer.pad_id == 0
    assert tokenizer.bos_id == 1
    assert tokenizer.eos_id == 2
    assert tokenizer.unk_id == 3


def test_train_or_load_without_data_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        WesternBPETokenizer.train_or_load([], str(tmp_path / "tok"))


def test_load_missing_files_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        WesternBPETokenizer.load(str(tmp_path))
