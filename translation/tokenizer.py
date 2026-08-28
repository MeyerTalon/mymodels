"""joint byte-level BPE tokenizer with multilingual language tokens.

trains or loads a shared source/target vocabulary. special tokens include
padding, bos/eos/unk, and ``<2xx>`` target-language control tokens used as a
source prefix (the standard multilingual MT trick).
"""

from __future__ import annotations

from itertools import chain
from pathlib import Path
from typing import Iterable, List, Sequence

from tokenizers import ByteLevelBPETokenizer

DEFAULT_LANGUAGES: List[str] = ["en", "es", "fr", "de"]


def lang_token(lang: str) -> str:
    """returns the control token for a target language code.

    Args:
        lang: two-letter language code (e.g. ``"es"``).

    Returns:
        the special token string (e.g. ``"<2es>"``).
    """
    return f"<2{lang}>"


def special_tokens_for(languages: Sequence[str]) -> List[str]:
    """builds the special-token list: pad/bos/eos/unk then language tokens.

    Args:
        languages: language codes to reserve control tokens for.

    Returns:
        special tokens in vocabulary-id order.
    """
    return ["<pad>", "<s>", "</s>", "<unk>"] + [lang_token(lang) for lang in languages]


class TranslationBPETokenizer:
    """wrapper around Hugging Face's ByteLevelBPETokenizer for translation.

    encodes source and target with a joint vocab and prepends a ``<2xx>``
    token to the source so one model can translate into several languages.
    """

    def __init__(
        self,
        tokenizer: ByteLevelBPETokenizer,
        languages: Sequence[str] = DEFAULT_LANGUAGES,
    ) -> None:
        """initializes the wrapper.

        Args:
            tokenizer: an instance of ByteLevelBPETokenizer.
            languages: language codes whose ``<2xx>`` tokens are in the vocab.
        """
        self._tokenizer = tokenizer
        self.languages: List[str] = list(languages)
        self.vocab_size: int = tokenizer.get_vocab_size()

        self.pad_id: int = self._special_id("<pad>", 0)
        self.bos_id: int = self._special_id("<s>", 1)
        self.eos_id: int = self._special_id("</s>", 2)
        self.unk_id: int = self._special_id("<unk>", 3)
        self._lang_ids = {
            lang: self._special_id(lang_token(lang), 4 + index)
            for index, lang in enumerate(self.languages)
        }

    def _special_id(self, token: str, default: int) -> int:
        """returns the id of a special token, or ``default`` if it is absent."""
        token_id = self._tokenizer.token_to_id(token)
        return token_id if token_id is not None else default

    def lang_id(self, lang: str) -> int:
        """returns the token id of the ``<2xx>`` control token for ``lang``.

        Args:
            lang: two-letter language code.

        Returns:
            integer token id.

        Raises:
            ValueError: if ``lang`` is not in the tokenizer's language list.
        """
        if lang not in self._lang_ids:
            raise ValueError(
                f"unknown language {lang!r}; expected one of {self.languages}."
            )
        return self._lang_ids[lang]

    @classmethod
    def train_or_load(
        cls,
        texts: Iterable[str],
        tokenizer_dir: str,
        vocab_size: int = 8000,
        min_frequency: int = 2,
        languages: Sequence[str] = DEFAULT_LANGUAGES,
    ) -> "TranslationBPETokenizer":
        """trains a new tokenizer or loads an existing one from disk.

        if `vocab.json` and `merges.txt` exist in `tokenizer_dir`, they are
        loaded. otherwise, a new ByteLevel BPE tokenizer is trained from
        ``texts`` (source and target concatenated) and saved to `tokenizer_dir`.

        Args:
            texts: source and target strings used when tokenizer files do not
                exist.
            tokenizer_dir: directory to store/load vocab and merges files.
            vocab_size: target vocabulary size.
            min_frequency: minimum token frequency to be included in the vocab.
            languages: language codes to reserve ``<2xx>`` tokens for.

        Returns:
            an initialized TranslationBPETokenizer instance.
        """
        tok_dir = Path(tokenizer_dir)
        if (tok_dir / "vocab.json").exists() and (tok_dir / "merges.txt").exists():
            return cls.load(tokenizer_dir, languages=languages)

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
            special_tokens=special_tokens_for(languages),
        )
        tokenizer.save_model(str(tok_dir))
        return cls.load(tokenizer_dir, languages=languages)

    @classmethod
    def load(
        cls,
        tokenizer_dir: str,
        languages: Sequence[str] = DEFAULT_LANGUAGES,
    ) -> "TranslationBPETokenizer":
        """loads an existing tokenizer from `tokenizer_dir`.

        Args:
            tokenizer_dir: directory containing `vocab.json` and `merges.txt`.
            languages: language codes expected in the vocabulary.

        Returns:
            an initialized TranslationBPETokenizer instance.

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
        return cls(tokenizer, languages=languages)

    def encode(self, text: str) -> List[int]:
        """converts text to a list of token IDs.

        Args:
            text: input string.

        Returns:
            list of integer token IDs.
        """
        return self._tokenizer.encode(text).ids

    def encode_source(self, text: str, target_lang: str) -> List[int]:
        """encodes source text with a target-language control-token prefix.

        Args:
            text: source sentence.
            target_lang: language code to translate into.

        Returns:
            token ids starting with ``<2{target_lang}>``.
        """
        return [self.lang_id(target_lang)] + self.encode(text)

    def encode_target_input(self, text: str) -> List[int]:
        """encodes a target sentence as decoder input (bos + tokens).

        Args:
            text: target sentence.

        Returns:
            token ids starting with bos.
        """
        return [self.bos_id] + self.encode(text)

    def encode_target_labels(self, text: str) -> List[int]:
        """encodes a target sentence as training labels (tokens + eos).

        Args:
            text: target sentence.

        Returns:
            token ids ending with eos.
        """
        return self.encode(text) + [self.eos_id]

    def decode(self, token_ids: List[int]) -> str:
        """converts a list of token IDs back to text.

        Args:
            token_ids: list of integer token IDs.

        Returns:
            decoded string.
        """
        skip = {self.pad_id, self.bos_id, self.eos_id, *self._lang_ids.values()}
        filtered = [token_id for token_id in token_ids if token_id not in skip]
        return self._tokenizer.decode(filtered)
