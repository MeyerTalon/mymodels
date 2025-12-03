"""Tokenizer module using Hugging Face's ByteLevelBPETokenizer.

Provides a WikipediaBPETokenizer wrapper that trains or loads a BPE tokenizer from
Wikipedia texts. Exposes a simple encode/decode interface compatible with the codebase.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from tokenizers import ByteLevelBPETokenizer


class WikipediaBPETokenizer:
    """Wrapper around Hugging Face's ByteLevelBPETokenizer for this project.

    This tokenizer can be trained from a directory of `.txt` files or loaded
    from previously saved vocab/merges files. It exposes a minimal interface
    compatible with the rest of the codebase: `encode`, `decode`, and
    `vocab_size`.
    """

    def __init__(self, tokenizer: ByteLevelBPETokenizer) -> None:
        """Initializes the wrapper.

        Args:
            tokenizer: An instance of ByteLevelBPETokenizer.
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
        """Trains a new tokenizer or loads an existing one from disk.

        If `vocab.json` and `merges.txt` exist in `tokenizer_dir`, they are
        loaded. Otherwise, a new ByteLevel BPE tokenizer is trained on all
        `.txt` files found in `data_dir` and saved to `tokenizer_dir`.

        Args:
            data_dir: Directory containing `.txt` article files.
            tokenizer_dir: Directory to store/load vocab and merges files.
            vocab_size: Target vocabulary size.
            min_frequency: Minimum token frequency to be included in the vocab.

        Returns:
            An initialized WikipediaBPETokenizer instance.
        """
        tok_dir = Path(tokenizer_dir)
        vocab_file = tok_dir / "vocab.json"
        merges_file = tok_dir / "merges.txt"

        if vocab_file.exists() and merges_file.exists():
            tokenizer = ByteLevelBPETokenizer(str(vocab_file), str(merges_file))
            return cls(tokenizer)

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

        # Save model files (vocab.json, merges.txt)
        tokenizer.save_model(str(tok_dir))

        # Reload to ensure consistency
        tokenizer = ByteLevelBPETokenizer(str(vocab_file), str(merges_file))
        return cls(tokenizer)

    @classmethod
    def load(cls, tokenizer_dir: str) -> "WikipediaBPETokenizer":
        """Loads an existing tokenizer from `tokenizer_dir`.

        Args:
            tokenizer_dir: Directory containing `vocab.json` and `merges.txt`.

        Returns:
            An initialized WikipediaBPETokenizer instance.

        Raises:
            FileNotFoundError: If vocab/merges files are missing.
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
        """Converts text to a list of token IDs.

        Args:
            text: Input string.

        Returns:
            List of integer token IDs.
        """
        return self._tokenizer.encode(text).ids

    def decode(self, token_ids: List[int]) -> str:
        """Converts a list of token IDs back to text.

        Args:
            token_ids: List of integer token IDs.

        Returns:
            Decoded string.
        """
        return self._tokenizer.decode(token_ids)


