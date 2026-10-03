from pathlib import Path

import pytest

from core.tokenizer import BPETokenizer

TEXTS = ['the quick brown fox jumps over the lazy dog'] * 20
VOCAB_SIZE = 300


def test_train_or_load_reserves_special_ids_and_reloads(tmp_path: Path) -> None:
    tokenizer = BPETokenizer.train_or_load(
        TEXTS, tmp_path, vocab_size=VOCAB_SIZE, min_frequency=1
    )
    assert (tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id, tokenizer.unk_id) == (
        0,
        1,
        2,
        3,
    )
    reloaded = BPETokenizer.train_or_load(
        [], tmp_path, vocab_size=VOCAB_SIZE, min_frequency=1
    )
    assert reloaded.encode('the fox') == tokenizer.encode('the fox')


def test_decode_drops_special_tokens(tmp_path: Path) -> None:
    tokenizer = BPETokenizer.train_or_load(
        TEXTS, tmp_path, vocab_size=VOCAB_SIZE, min_frequency=1
    )
    token_ids = [tokenizer.bos_id, *tokenizer.encode('lazy dog'), tokenizer.eos_id]
    assert tokenizer.decode(token_ids) == 'lazy dog'


def test_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        BPETokenizer.load(tmp_path)
    with pytest.raises(ValueError, match='at least one text'):
        BPETokenizer.train_or_load(
            ['', ''], tmp_path, vocab_size=VOCAB_SIZE, min_frequency=1
        )
