"""tokenizer module using Hugging Face's ByteLevelBPETokenizer.

provides a WikipediaBPETokenizer wrapper that trains or loads a BPE tokenizer from
Wikipedia texts. exposes a simple encode/decode interface compatible with the codebase.
"""

from __future__ import annotations

from itertools import chain
from pathlib import Path
from typing import Iterable, List

from tokenizers import ByteLevelBPETokenizer


class WikipediaBPETokenizer:
    """wrapper around Hugging Face's ByteLevelBPETokenizer for this project.

    this tokenizer can be trained from an iterable of article texts or loaded
    from previously saved vocab/merges files. it exposes a minimal interface
    compatible with the rest of the codebase.
    """

    #: special tokens reserved at the start of the vocabulary, in id order.
    SPECIAL_TOKENS: List[str] = ["<pad>", "<s>", "</s>", "<unk>"]

    def __init__(self, tokenizer: ByteLevelBPETokenizer) -> None:
        """initializes the wrapper.

        Args:
            tokenizer: an instance of ByteLevelBPETokenizer.
        """
        self._tokenizer = tokenizer
        self.vocab_size: int = tokenizer.get_vocab_size()

        # resolve special-token ids once; fall back to conventional positions
        # (<pad>=0, <s>=1, </s>=2, <unk>=3) if the token is not in the vocab.
        self.pad_id: int = self._special_id("<pad>", 0)
        self.bos_id: int = self._special_id("<s>", 1)
        self.eos_id: int = self._special_id("</s>", 2)
        self.unk_id: int = self._special_id("<unk>", 3)

    def _special_id(self, token: str, default: int) -> int:
        """returns the id of a special token, or ``default`` if it is absent."""
        token_id = self._tokenizer.token_to_id(token)
        return token_id if token_id is not None else default

    @classmethod
    def train_or_load(
        cls,
        texts: Iterable[str],
        tokenizer_dir: str,
        vocab_size: int = 8000,
        min_frequency: int = 2,
    ) -> "WikipediaBPETokenizer":
        """trains a new tokenizer or loads an existing one from disk.

        if `vocab.json` and `merges.txt` exist in `tokenizer_dir`, they are
        loaded. otherwise, a new ByteLevel BPE tokenizer is trained from
        ``texts`` and saved to `tokenizer_dir`.

        Args:
            texts: article texts used when tokenizer files do not exist.
            tokenizer_dir: directory to store/load vocab and merges files.
            vocab_size: target vocabulary size.
            min_frequency: minimum token frequency to be included in the vocab.

        Returns:
            an initialized WikipediaBPETokenizer instance.
        """
        tok_dir = Path(tokenizer_dir)
        if (tok_dir / "vocab.json").exists() and (tok_dir / "merges.txt").exists():
            return cls.load(tokenizer_dir)

        tok_dir.mkdir(parents=True, exist_ok=True)
        training_texts = (text for text in texts if text)
        try:
            first_text = next(training_texts)
        except StopIteration as exc:
            raise ValueError(
                "at least one text is required to train the tokenizer."
            ) from exc

        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train_from_iterator(
            chain([first_text], training_texts),
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=cls.SPECIAL_TOKENS,
        )

        # save model files (vocab.json, merges.txt), then reload for consistency
        tokenizer.save_model(str(tok_dir))
        return cls.load(tokenizer_dir)

    @classmethod
    def load(cls, tokenizer_dir: str) -> "WikipediaBPETokenizer":
        """loads an existing tokenizer from `tokenizer_dir`.

        Args:
            tokenizer_dir: directory containing `vocab.json` and `merges.txt`.

        Returns:
            an initialized WikipediaBPETokenizer instance.

        Raises:
            FileNotFoundError: if vocab/merges files are missing.
        """
        tok_dir = Path(tokenizer_dir)
        vocab_file = tok_dir / "vocab.json"
        merges_file = tok_dir / "merges.txt"

        if not vocab_file.exists() or not merges_file.exists():
            raise FileNotFoundError(
                f"Tokenizer files not found in {tokenizer_dir}. "
                "Make sure you've trained the tokenizer first."
            )

        tokenizer = ByteLevelBPETokenizer(str(vocab_file), str(merges_file))
        return cls(tokenizer)

    def encode(self, text: str) -> List[int]:
        """converts text to a list of token IDs.

        Args:
            text: input string.

        Returns:
            list of integer token IDs.
        """
        return self._tokenizer.encode(text).ids

    def decode(self, token_ids: List[int]) -> str:
        """converts a list of token IDs back to text.

        Args:
            token_ids: list of integer token IDs.

        Returns:
            decoded string.
        """
        return self._tokenizer.decode(token_ids)
