from collections.abc import Iterable, Sequence
from itertools import chain
from pathlib import Path
from typing import Protocol, Self

from tokenizers import ByteLevelBPETokenizer

PAD_TOKEN = '<pad>'
BOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'
UNK_TOKEN = '<unk>'
BASE_SPECIAL_TOKENS = (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN)
VOCAB_FILENAME = 'vocab.json'
MERGES_FILENAME = 'merges.txt'


class TextTokenizer(Protocol):
    vocab_size: int
    pad_id: int
    bos_id: int
    eos_id: int

    def encode(self, text: str) -> list[int]: ...

    def decode(self, token_ids: list[int]) -> str: ...


class BPETokenizer:
    def __init__(
        self, tokenizer: ByteLevelBPETokenizer, special_tokens: Sequence[str]
    ) -> None:
        self._tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()
        self.pad_id = self.token_id(PAD_TOKEN)
        self.bos_id = self.token_id(BOS_TOKEN)
        self.eos_id = self.token_id(EOS_TOKEN)
        self.unk_id = self.token_id(UNK_TOKEN)
        self.special_ids = frozenset(map(self.token_id, special_tokens))

    def token_id(self, token: str) -> int:
        token_id: int | None = self._tokenizer.token_to_id(token)
        if token_id is None:
            raise ValueError(f'{token!r} is not in the tokenizer vocabulary')
        return token_id

    @classmethod
    def train_or_load(
        cls,
        texts: Iterable[str],
        tokenizer_dir: Path,
        *,
        vocab_size: int,
        min_frequency: int,
        special_tokens: Sequence[str] = BASE_SPECIAL_TOKENS,
    ) -> Self:
        if _has_tokenizer_files(tokenizer_dir):
            return cls.load(tokenizer_dir, special_tokens)
        training_texts = (text for text in texts if text)
        first_text = next(training_texts, None)
        if first_text is None:
            raise ValueError('at least one text is required to train the tokenizer')
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train_from_iterator(
            chain([first_text], training_texts),
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=list(special_tokens),
        )
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_model(str(tokenizer_dir))
        return cls.load(tokenizer_dir, special_tokens)

    @classmethod
    def load(
        cls,
        tokenizer_dir: Path,
        special_tokens: Sequence[str] = BASE_SPECIAL_TOKENS,
    ) -> Self:
        if not _has_tokenizer_files(tokenizer_dir):
            raise FileNotFoundError(
                f'tokenizer files not found in {tokenizer_dir}; train a model first'
            )
        tokenizer = ByteLevelBPETokenizer(
            str(tokenizer_dir / VOCAB_FILENAME),
            str(tokenizer_dir / MERGES_FILENAME),
        )
        return cls(tokenizer, special_tokens)

    def encode(self, text: str) -> list[int]:
        token_ids: list[int] = self._tokenizer.encode(text).ids
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """Filters `special_ids` itself: a tokenizer loaded from vocab and merges files no longer marks them special."""
        kept = [token_id for token_id in token_ids if token_id not in self.special_ids]
        text: str = self._tokenizer.decode(kept)
        return text


def _has_tokenizer_files(tokenizer_dir: Path) -> bool:
    return (tokenizer_dir / VOCAB_FILENAME).is_file() and (
        tokenizer_dir / MERGES_FILENAME
    ).is_file()
