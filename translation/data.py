"""local parallel-corpus loading and padded seq2seq data loading.

reads TSV or JSONL translation pairs from a local corpus directory into a
reusable snapshot. nothing is downloaded: if the corpus is missing the caller
gets a clear path and format message.
"""

import hashlib
import json
import os
import tempfile
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypedDict

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

SNAPSHOT_FILENAME = "translation_pairs.jsonl"
MANIFEST_FILENAME = "translation_pairs.manifest.json"
DEFAULT_CORPUS_SUBDIR = "corpus"


class TranslationPair(TypedDict):
    """one parallel sentence pair with language codes."""

    src: str
    tgt: str
    src_lang: str
    tgt_lang: str


class TranslationDataset(Dataset):
    """padded-at-collate dataset of encoded parallel pairs.

    each item is ``(src_ids, tgt_input_ids, tgt_label_ids)`` where the source
    already includes the ``<2xx>`` target-language prefix.
    """

    def __init__(
        self,
        pairs: Sequence[TranslationPair],
        tokenizer: Any,
        max_seq_len: int = 128,
    ) -> None:
        """encodes pairs up front.

        Args:
            pairs: parallel examples.
            tokenizer: tokenizer exposing ``encode_source``,
                ``encode_target_input``, and ``encode_target_labels``.
            max_seq_len: maximum source/target length after special tokens.

        Raises:
            ValueError: if ``pairs`` is empty.
        """
        if not pairs:
            raise ValueError("at least one translation pair is required.")
        self.examples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for pair in pairs:
            src_ids = tokenizer.encode_source(pair["src"], pair["tgt_lang"])
            tgt_tokens = tokenizer.encode(pair["tgt"])
            # leave room for the bos/eos token on the target side
            tgt_tokens = tgt_tokens[: max(max_seq_len - 1, 1)]
            src_ids = src_ids[:max_seq_len]
            tgt_input = [tokenizer.bos_id] + tgt_tokens
            tgt_labels = tgt_tokens + [tokenizer.eos_id]
            self.examples.append(
                (
                    torch.tensor(src_ids, dtype=torch.long),
                    torch.tensor(tgt_input, dtype=torch.long),
                    torch.tensor(tgt_labels, dtype=torch.long),
                )
            )

    def __len__(self) -> int:
        """returns the number of pairs."""
        return len(self.examples)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """returns encoded ``(src, tgt_input, tgt_labels)`` for pair ``idx``."""
        return self.examples[idx]


def load_translation_pairs(
    data_dir: str,
    corpus_dir: Optional[str] = None,
    max_pairs: Optional[int] = None,
    dataset_cache_only: bool = False,
) -> List[TranslationPair]:
    """loads pairs from a compatible snapshot or a local corpus directory.

    Args:
        data_dir: directory containing the application-owned corpus snapshot.
        corpus_dir: directory of ``*.tsv`` / ``*.jsonl`` files. defaults to
            ``<data_dir>/corpus``.
        max_pairs: if set, keep only the first ``max_pairs`` examples.
        dataset_cache_only: when True, require a compatible local snapshot.

    Returns:
        selected translation pairs.

    Raises:
        ValueError: if settings are invalid, a cache-only snapshot is
            unavailable, or no local corpus files are present.
    """
    if max_pairs is not None and max_pairs <= 0:
        raise ValueError("max_pairs must be greater than zero when set.")

    snapshot_dir = Path(data_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    resolved_corpus = (
        Path(corpus_dir) if corpus_dir else snapshot_dir / DEFAULT_CORPUS_SUBDIR
    )
    expected = _snapshot_metadata(str(resolved_corpus))
    cached = _load_snapshot(snapshot_dir, expected, max_pairs)
    if cached is not None:
        print(f"Loaded {len(cached)} pairs from {snapshot_dir / SNAPSHOT_FILENAME}")
        return cached

    if dataset_cache_only:
        raise ValueError(
            "dataset_cache_only=True but no compatible translation snapshot "
            f"exists in {data_dir}."
        )

    pairs = _load_corpus_files(resolved_corpus)
    if not pairs:
        raise ValueError(
            "no translation corpus found. place parallel files in "
            f"{resolved_corpus} as *.tsv (src<TAB>tgt<TAB>src_lang<TAB>tgt_lang) "
            "or *.jsonl ({\"src\", \"tgt\", \"src_lang\", \"tgt_lang\"}) and "
            f"re-run training. example: {resolved_corpus / 'pairs.tsv'}"
        )
    if max_pairs is not None and len(pairs) < max_pairs:
        raise ValueError(
            f"requested {max_pairs} pairs but only found {len(pairs)} "
            "in the corpus directory."
        )

    _write_snapshot(snapshot_dir, pairs, expected)
    print(f"Cached {len(pairs)} pairs in {snapshot_dir / SNAPSHOT_FILENAME}")
    return pairs if max_pairs is None else pairs[:max_pairs]


def _load_corpus_files(corpus_dir: Path) -> List[TranslationPair]:
    """reads ``*.tsv`` and ``*.jsonl`` files from ``corpus_dir``."""
    if not corpus_dir.is_dir():
        return []

    pairs: List[TranslationPair] = []
    paths = sorted(
        list(corpus_dir.glob("*.tsv")) + list(corpus_dir.glob("*.jsonl"))
    )
    for path in paths:
        if path.suffix.lower() == ".tsv":
            pairs.extend(_read_tsv(path))
        else:
            pairs.extend(_read_jsonl(path))
    return pairs


def _read_tsv(path: Path) -> List[TranslationPair]:
    """parses a four-column TSV file of parallel pairs."""
    pairs: List[TranslationPair] = []
    text = path.read_text(encoding="utf-8")
    for line_number, raw in enumerate(text.splitlines()):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        columns = line.split("\t")
        if len(columns) < 4:
            continue
        if line_number == 0 and columns[0].lower() in {"src", "source"}:
            continue
        src, tgt, src_lang, tgt_lang = [column.strip() for column in columns[:4]]
        if not src or not tgt:
            continue
        pairs.append(
            {
                "src": src,
                "tgt": tgt,
                "src_lang": src_lang.lower(),
                "tgt_lang": tgt_lang.lower(),
            }
        )
    return pairs


def _read_jsonl(path: Path) -> List[TranslationPair]:
    """parses a JSONL file of parallel pairs."""
    pairs: List[TranslationPair] = []
    with path.open("r", encoding="utf-8") as file:
        for raw in file:
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            src = str(row.get("src", "")).strip()
            tgt = str(row.get("tgt", "")).strip()
            if not src or not tgt:
                continue
            pairs.append(
                {
                    "src": src,
                    "tgt": tgt,
                    "src_lang": str(row.get("src_lang", "")).strip().lower(),
                    "tgt_lang": str(row.get("tgt_lang", "")).strip().lower(),
                }
            )
    return pairs


def _snapshot_metadata(corpus_dir: str) -> Dict[str, Any]:
    """builds the settings that identify a reproducible corpus snapshot."""
    return {"source": "local_corpus", "corpus_dir": corpus_dir}


def _load_snapshot(
    snapshot_dir: Path,
    expected: Dict[str, Any],
    n_pairs: Optional[int],
) -> Optional[List[TranslationPair]]:
    """loads a compatible, integrity-checked local snapshot."""
    snapshot_path = snapshot_dir / SNAPSHOT_FILENAME
    manifest_path = snapshot_dir / MANIFEST_FILENAME
    if not snapshot_path.exists() or not manifest_path.exists():
        return None

    try:
        with manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
        if any(manifest.get(key) != value for key, value in expected.items()):
            return None
        stored_count = int(manifest.get("pair_count", 0))
        if n_pairs is not None and stored_count < n_pairs:
            return None
        if _file_sha256(snapshot_path) != manifest.get("sha256"):
            return None

        pairs: List[TranslationPair] = []
        with snapshot_path.open("r", encoding="utf-8") as file:
            for line in file:
                if n_pairs is not None and len(pairs) == n_pairs:
                    break
                pairs.append(_normalize_pair(json.loads(line)))
        if n_pairs is not None and len(pairs) != n_pairs:
            return None
        return pairs
    except (
        AttributeError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return None


def _normalize_pair(row: Dict[str, Any]) -> TranslationPair:
    """normalizes one pair row for stable JSON serialization."""
    return {
        "src": str(row["src"]).strip(),
        "tgt": str(row["tgt"]).strip(),
        "src_lang": str(row.get("src_lang", "")).strip().lower(),
        "tgt_lang": str(row.get("tgt_lang", "")).strip().lower(),
    }


def _write_snapshot(
    snapshot_dir: Path,
    pairs: List[TranslationPair],
    metadata: Dict[str, Any],
) -> None:
    """atomically writes an integrity-checked JSONL snapshot and manifest."""
    snapshot_path = snapshot_dir / SNAPSHOT_FILENAME
    manifest_path = snapshot_dir / MANIFEST_FILENAME
    snapshot_tmp = _temporary_path(snapshot_dir, SNAPSHOT_FILENAME)
    manifest_tmp = _temporary_path(snapshot_dir, MANIFEST_FILENAME)
    try:
        with snapshot_tmp.open("w", encoding="utf-8") as file:
            for pair in pairs:
                file.write(json.dumps(pair, ensure_ascii=False) + "\n")

        manifest = {
            **metadata,
            "pair_count": len(pairs),
            "sha256": _file_sha256(snapshot_tmp),
        }
        with manifest_tmp.open("w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2, sort_keys=True)
            file.write("\n")

        os.replace(snapshot_tmp, snapshot_path)
        os.replace(manifest_tmp, manifest_path)
    finally:
        snapshot_tmp.unlink(missing_ok=True)
        manifest_tmp.unlink(missing_ok=True)


def _temporary_path(directory: Path, filename: str) -> Path:
    """creates a closed temporary file path in the snapshot directory."""
    descriptor, path = tempfile.mkstemp(prefix=f".{filename}.", dir=directory)
    os.close(descriptor)
    return Path(path)


def _file_sha256(path: Path) -> str:
    """returns the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collate_pairs(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    pad_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """pads a batch of encoded pairs.

    Args:
        batch: list of ``(src, tgt_input, tgt_labels)`` tensors.
        pad_id: padding token id.

    Returns:
        ``(src, tgt_input, tgt_labels, src_pad_mask, tgt_pad_mask)`` where
        masks are ``True`` at padded positions.
    """
    src_list, tgt_in_list, tgt_lab_list = zip(*batch)
    src = pad_sequence(list(src_list), batch_first=True, padding_value=pad_id)
    tgt_input = pad_sequence(
        list(tgt_in_list), batch_first=True, padding_value=pad_id
    )
    tgt_labels = pad_sequence(
        list(tgt_lab_list), batch_first=True, padding_value=pad_id
    )
    src_pad_mask = src == pad_id
    tgt_pad_mask = tgt_input == pad_id
    return src, tgt_input, tgt_labels, src_pad_mask, tgt_pad_mask


def create_dataloaders(
    pairs: Sequence[TranslationPair],
    tokenizer: Any,
    max_seq_len: int = 128,
    batch_size: int = 16,
    val_fraction: float = 0.0,
    shuffle: bool = True,
    num_workers: int = 0,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """builds padded train (and optional validation) dataloaders.

    pairs are split at the example level so train and val never share sentences.

    Args:
        pairs: parallel examples.
        tokenizer: tokenizer used to encode the pairs.
        max_seq_len: maximum source/target length.
        batch_size: batch size for the loaders.
        val_fraction: fraction of pairs reserved for validation.
        shuffle: whether to shuffle the training pairs.
        num_workers: number of subprocesses for data loading.

    Returns:
        a ``(train_loader, val_loader)`` tuple; ``val_loader`` is ``None`` when
        ``val_fraction`` is 0 or there are too few pairs to split.
    """
    split = (
        int(len(pairs) * (1.0 - val_fraction)) if val_fraction > 0 else len(pairs)
    )
    train_pairs = list(pairs[: max(split, 1)])
    val_pairs = list(pairs[split:]) if val_fraction > 0 else []

    pad_id = getattr(tokenizer, "pad_id", 0)
    train_dataset = TranslationDataset(train_pairs, tokenizer, max_seq_len)
    train_loader = _make_loader(
        train_dataset, batch_size, shuffle, num_workers, pad_id
    )

    val_loader: Optional[DataLoader] = None
    if val_pairs:
        val_dataset = TranslationDataset(val_pairs, tokenizer, max_seq_len)
        val_loader = _make_loader(val_dataset, batch_size, False, num_workers, pad_id)

    return train_loader, val_loader


def _make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pad_id: int,
) -> DataLoader:
    """creates a DataLoader that pads variable-length pairs."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=False,
        collate_fn=partial(collate_pairs, pad_id=pad_id),
    )
