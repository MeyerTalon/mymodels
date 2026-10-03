from pathlib import Path

import pytest

from translation.tokenizer import TranslationTokenizer, lang_token

LANGUAGES = ['en', 'es']
TEXTS = ['hello world', 'hola mundo', 'good morning', 'buenos dias'] * 10


def _tokenizer(tmp_path: Path) -> TranslationTokenizer:
    return TranslationTokenizer.train_or_load(
        TEXTS, tmp_path, vocab_size=300, min_frequency=1, languages=LANGUAGES
    )


def test_language_tokens_follow_base_specials(tmp_path: Path) -> None:
    tokenizer = _tokenizer(tmp_path)
    assert lang_token('es') == '<2es>'
    assert [tokenizer.lang_id(lang) for lang in LANGUAGES] == [4, 5]
    with pytest.raises(ValueError, match='unknown language'):
        tokenizer.lang_id('fr')


def test_encode_source_prefixes_target_language(tmp_path: Path) -> None:
    tokenizer = _tokenizer(tmp_path)
    source = tokenizer.encode_source('hello world', 'es')
    assert source[0] == tokenizer.lang_id('es')
    assert source[1:] == tokenizer.encode('hello world')


def test_decode_drops_language_and_special_tokens(tmp_path: Path) -> None:
    tokenizer = _tokenizer(tmp_path)
    token_ids = [
        tokenizer.bos_id,
        *tokenizer.encode_source('hola mundo', 'en'),
        tokenizer.eos_id,
    ]
    assert tokenizer.decode(token_ids) == 'hola mundo'
    reloaded = TranslationTokenizer.load(tmp_path, LANGUAGES)
    assert reloaded.encode('hola') == tokenizer.encode('hola')
