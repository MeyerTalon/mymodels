"""tokenizer module using Hugging Face's ByteLevelBPETokenizer.

provides a WikipediaBPETokenizer wrapper that trains or loads a BPE tokenizer from
Wikipedia texts. exposes a simple encode/decode interface compatible with the codebase.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from tokenizers import ByteLevelBPETokenizer


class WikipediaBPETokenizer:
    """wrapper around Hugging Face's ByteLevelBPETokenizer for this project.

    this tokenizer can be trained from a directory of `.txt` files or loaded
    from previously saved vocab/merges files. it exposes a minimal interface
    compatible with the rest of the codebase: `encode`, `decode`, and
    `vocab_size`.
    """

    def __init__(self, tokenizer: ByteLevelBPETokenizer) -> None:
        """initializes the wrapper.

        Args:
            tokenizer: an instance of ByteLevelBPETokenizer.
        """
        self._tokenizer = tokenizer
        self.vocab_size: int = tokenizer.get_vocab_size()

    @classmethod
    def train_or_load(
        cls,
        data_dir: str,
        tokenizer_dir: str,
        vocab_size: int = 8000,
        min_frequency: int = 2,
    ) -> "WikipediaBPETokenizer":
        """trains a new tokenizer or loads an existing one from disk.

        if `vocab.json` and `merges.txt` exist in `tokenizer_dir`, they are
        loaded. otherwise, a new ByteLevel BPE tokenizer is trained on all
        `.txt` files found in `data_dir` and saved to `tokenizer_dir`.

        Args:
            data_dir: directory containing `.txt` article files.
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
        files = [str(p) for p in Path(data_dir).glob("*.txt")]
        if not files:
            raise ValueError(
                f"No .txt files found in {data_dir} to train the tokenizer."
            )

        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(
            files=files,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=["<pad>", "<s>", "</s>", "<unk>"],
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
