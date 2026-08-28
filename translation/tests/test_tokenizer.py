"""tests for the TranslationBPETokenizer wrapper."""

from pathlib import Path

import pytest

from translation.tokenizer import TranslationBPETokenizer, lang_token

TINY_VOCAB_SIZE = 300  # must cover the 256-byte alphabet plus special tokens
TINY_CORPUS = [
    "hello world, how are you today? " * 10,
    "hola mundo, como estas hoy? " * 10,
    "bonjour le monde, comment ca va? " * 10,
]


def test_train_or_load_roundtrip(tmp_path: Path) -> None:
    tok_dir = tmp_path / "tok"
    tokenizer = TranslationBPETokenizer.train_or_load(
        TINY_CORPUS, str(tok_dir), vocab_size=TINY_VOCAB_SIZE
    )
    ids = tokenizer.encode("hello world")
    assert ids and all(isinstance(i, int) for i in ids)
    assert tokenizer.decode(ids) == "hello world"
    assert tokenizer.vocab_size > 0

    reloaded = TranslationBPETokenizer.train_or_load(
        [], str(tok_dir), vocab_size=TINY_VOCAB_SIZE
    )
    assert reloaded.encode("hola") == tokenizer.encode("hola")


def test_special_token_ids_and_lang_prefix(tmp_path: Path) -> None:
    tok_dir = tmp_path / "tok"
    tokenizer = TranslationBPETokenizer.train_or_load(
        TINY_CORPUS, str(tok_dir), vocab_size=TINY_VOCAB_SIZE
    )
    assert tokenizer.pad_id == 0
    assert tokenizer.bos_id == 1
    assert tokenizer.eos_id == 2
    assert tokenizer.unk_id == 3
    assert lang_token("es") == "<2es>"
    src_ids = tokenizer.encode_source("hello", "es")
    assert src_ids[0] == tokenizer.lang_id("es")
    assert src_ids[1:] == tokenizer.encode("hello")


def test_unknown_language_raises(tmp_path: Path) -> None:
    tok_dir = tmp_path / "tok"
    tokenizer = TranslationBPETokenizer.train_or_load(
        TINY_CORPUS, str(tok_dir), vocab_size=TINY_VOCAB_SIZE
    )
    with pytest.raises(ValueError, match="unknown language"):
        tokenizer.lang_id("xx")


def test_train_or_load_without_data_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        TranslationBPETokenizer.train_or_load([], str(tmp_path / "tok"))


def test_load_missing_files_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        TranslationBPETokenizer.load(str(tmp_path))
