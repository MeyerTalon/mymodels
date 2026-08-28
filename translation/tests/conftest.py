"""shared fixtures for the translation package tests."""

from typing import List

import pytest


class DummyTokenizer:
    """minimal deterministic tokenizer for tests.

    maps each character to an id in [8, vocab_size), reserving 0-7 for
    special tokens (pad, bos, eos, unk, and four language codes).
    """

    def __init__(self, vocab_size: int = 32) -> None:
        self.vocab_size = vocab_size
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3
        self.languages = ["en", "es", "fr", "de"]
        self._lang_ids = {"en": 4, "es": 5, "fr": 6, "de": 7}

    def lang_id(self, lang: str) -> int:
        """returns the control-token id for ``lang``."""
        return self._lang_ids[lang]

    def encode(self, text: str) -> List[int]:
        """encodes each character to a non-special token id."""
        return [(ord(c) % (self.vocab_size - 8)) + 8 for c in text]

    def encode_source(self, text: str, target_lang: str) -> List[int]:
        """prepends the target-language control token to the source ids."""
        return [self.lang_id(target_lang)] + self.encode(text)

    def encode_target_input(self, text: str) -> List[int]:
        """returns bos + encoded target tokens."""
        return [self.bos_id] + self.encode(text)

    def encode_target_labels(self, text: str) -> List[int]:
        """returns encoded target tokens + eos."""
        return self.encode(text) + [self.eos_id]

    def decode(self, token_ids: List[int]) -> str:
        """decodes token ids to a placeholder string."""
        skip = {self.pad_id, self.bos_id, self.eos_id, *self._lang_ids.values()}
        return "".join("a" for token_id in token_ids if token_id not in skip)


@pytest.fixture
def dummy_tokenizer() -> DummyTokenizer:
    """provides a small deterministic tokenizer."""
    return DummyTokenizer()
