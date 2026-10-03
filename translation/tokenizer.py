from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Self

from core.tokenizer import BASE_SPECIAL_TOKENS, BPETokenizer


def lang_token(lang: str) -> str:
    return f'<2{lang}>'


def special_tokens_for(languages: Sequence[str]) -> list[str]:
    return [*BASE_SPECIAL_TOKENS, *map(lang_token, languages)]


class TranslationTokenizer:
    """Prefixes the source with a `<2xx>` target-language token so one model translates into every language."""

    def __init__(self, bpe: BPETokenizer, languages: Sequence[str]) -> None:
        self.bpe = bpe
        self.languages = list(languages)
        self.vocab_size = bpe.vocab_size
        self.pad_id = bpe.pad_id
        self.bos_id = bpe.bos_id
        self.eos_id = bpe.eos_id
        self._lang_ids = {lang: bpe.token_id(lang_token(lang)) for lang in languages}

    @classmethod
    def train_or_load(
        cls,
        texts: Iterable[str],
        tokenizer_dir: Path,
        *,
        vocab_size: int,
        min_frequency: int,
        languages: Sequence[str],
    ) -> Self:
        bpe = BPETokenizer.train_or_load(
            texts,
            tokenizer_dir,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens_for(languages),
        )
        return cls(bpe, languages)

    @classmethod
    def load(cls, tokenizer_dir: Path, languages: Sequence[str]) -> Self:
        bpe = BPETokenizer.load(tokenizer_dir, special_tokens_for(languages))
        return cls(bpe, languages)

    def lang_id(self, lang: str) -> int:
        if lang not in self._lang_ids:
            raise ValueError(
                f'unknown language {lang!r}; expected one of {self.languages}'
            )
        return self._lang_ids[lang]

    def encode(self, text: str) -> list[int]:
        return self.bpe.encode(text)

    def encode_source(self, text: str, target_lang: str) -> list[int]:
        return [self.lang_id(target_lang), *self.encode(text)]

    def decode(self, token_ids: list[int]) -> str:
        return self.bpe.decode(token_ids)
