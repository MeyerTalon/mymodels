"""shared fixtures for the wikipedia package tests."""

from typing import List

import pytest


class DummyTokenizer:
    """minimal deterministic tokenizer for tests.

    maps each character to an id in [1, vocab_size), reserving 0 for padding.
    ``decode`` returns a placeholder string of the same length, which is
    sufficient for shape and length assertions.
    """

    def __init__(self, vocab_size: int = 32) -> None:
        self.vocab_size = vocab_size

    def encode(self, text: str) -> List[int]:
        """encodes each character to a non-zero token id."""
        return [(ord(c) % (self.vocab_size - 1)) + 1 for c in text]

    def decode(self, token_ids: List[int]) -> str:
        """decodes token ids to a placeholder string (one char per non-pad id)."""
        return "".join("a" for i in token_ids if i != 0)


@pytest.fixture
def dummy_tokenizer() -> DummyTokenizer:
    """provides a small deterministic tokenizer."""
    return DummyTokenizer()
