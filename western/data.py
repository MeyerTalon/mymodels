"""local western-novel corpus loading and packed data loading.

reads plain-text novels from a local corpus directory into a reusable snapshot,
then builds the packed token stream used for causal language modeling. nothing
is downloaded: if the corpus is missing the caller gets a clear path message.
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import torch
from torch.utils.data import DataLoader, Dataset

SNAPSHOT_FILENAME = "western_works.jsonl"
MANIFEST_FILENAME = "western_works.manifest.json"
DEFAULT_CORPUS_SUBDIR = "corpus"


class WesternWork(TypedDict):
    """fields retained for one novel or other work."""

    id: str
    title: str
    text: str


class WesternDataset(Dataset):
    """packed fixed-length blocks over a single token stream.

    the full corpus is tokenized into one contiguous stream and cut into
    ``block_size``-length windows. each item is a ``(input_ids, target_ids)``
    pair where ``target_ids`` is ``input_ids`` shifted by one, the standard
    next-token-prediction setup with no padding.
    """

    def __init__(self, token_ids: List[int], block_size: int = 512) -> None:
        """initializes the dataset from a flat token stream.

        Args:
            token_ids: contiguous list of token ids spanning the whole corpus.
            block_size: sequence length of each training block.

        Raises:
            ValueError: if the stream is too short to form a single block.
        """
        if len(token_ids) < block_size + 1:
            raise ValueError(
                f"token stream of length {len(token_ids)} is too short for "
                f"block_size={block_size}; need at least {block_size + 1} tokens."
            )
        self.block_size = block_size
        self.data = torch.tensor(token_ids, dtype=torch.long)
        # number of full (block_size + 1)-length windows available
        self.n_blocks = (len(self.data) - 1) // block_size

    def __len__(self) -> int:
        """returns the number of packed blocks."""
        return self.n_blocks

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """returns the ``(input_ids, target_ids)`` pair for block ``idx``.

        Args:
            idx: block index.

        Returns:
            a tuple of:

            * input_ids: tensor of token ids of shape (block_size,).
            * target_ids: next-token labels of shape (block_size,).
        """
        start = idx * self.block_size
        chunk = self.data[start : start + self.block_size + 1]
        return chunk[:-1], chunk[1:]


def load_western_texts(
    data_dir: str,
    corpus_dir: Optional[str] = None,
    max_works: Optional[int] = None,
    dataset_cache_only: bool = False,
) -> List[str]:
    """loads work texts from a compatible snapshot or a local corpus directory.

    Args:
        data_dir: directory containing the application-owned corpus snapshot.
        corpus_dir: directory of ``*.txt`` novels (one work per file). defaults
            to ``<data_dir>/corpus``.
        max_works: if set, keep only the first ``max_works`` non-empty works
            (useful for smoke tests); ``None`` keeps the entire corpus.
        dataset_cache_only: when True, require a compatible local snapshot and
            do not re-read the corpus directory.

    Returns:
        selected work texts.

    Raises:
        ValueError: if settings are invalid, a cache-only snapshot is
            unavailable, or no local corpus files are present.
    """
    if max_works is not None and max_works <= 0:
        raise ValueError("max_works must be greater than zero when set.")

    snapshot_dir = Path(data_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    resolved_corpus = Path(corpus_dir) if corpus_dir else snapshot_dir / DEFAULT_CORPUS_SUBDIR
    expected = _snapshot_metadata(str(resolved_corpus))
    cached = _load_snapshot(snapshot_dir, expected, max_works)
    if cached is not None:
        print(f"Loaded {len(cached)} works from {snapshot_dir / SNAPSHOT_FILENAME}")
        return [work["text"] for work in cached]

    if dataset_cache_only:
        raise ValueError(
            "dataset_cache_only=True but no compatible western snapshot exists "
            f"in {data_dir}."
        )

    works = _load_corpus_files(resolved_corpus)
    if not works:
        raise ValueError(
            "no western corpus found. place plain-text novels (*.txt) in "
            f"{resolved_corpus} (one work per file) and re-run training. "
            f"example: {resolved_corpus / 'lonesome_dove.txt'}"
        )
    if max_works is not None and len(works) < max_works:
        raise ValueError(
            f"requested {max_works} works but only found {len(works)} "
            "non-empty .txt files in the corpus directory."
        )

    # always persist the full corpus so a later full-corpus run can reuse it;
    # ``max_works`` only limits what this call returns.
    _write_snapshot(snapshot_dir, works, expected)
    print(f"Cached {len(works)} works in {snapshot_dir / SNAPSHOT_FILENAME}")
    selected = works if max_works is None else works[:max_works]
    return [work["text"] for work in selected]


def _load_corpus_files(corpus_dir: Path) -> List[WesternWork]:
    """reads ``*.txt`` files from ``corpus_dir`` as individual works.

    Args:
        corpus_dir: directory that should contain one plain-text novel per file.

    Returns:
        works sorted by filename; empty files are skipped.
    """
    if not corpus_dir.is_dir():
        return []

    works: List[WesternWork] = []
    for path in sorted(corpus_dir.glob("*.txt")):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError:
            text = path.read_text(encoding="latin-1").strip()
        if not text:
            continue
        works.append({"id": str(len(works)), "title": path.stem, "text": text})
    return works


def _snapshot_metadata(corpus_dir: str) -> Dict[str, Any]:
    """builds the settings that identify a reproducible corpus snapshot."""
    return {"source": "local_corpus", "corpus_dir": corpus_dir}


def _load_snapshot(
    snapshot_dir: Path,
    expected: Dict[str, Any],
    n_works: Optional[int],
) -> Optional[List[WesternWork]]:
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
        stored_count = int(manifest.get("work_count", 0))
        if n_works is not None and stored_count < n_works:
            return None
        if _file_sha256(snapshot_path) != manifest.get("sha256"):
            return None

        works: List[WesternWork] = []
        with snapshot_path.open("r", encoding="utf-8") as file:
            for line in file:
                if n_works is not None and len(works) == n_works:
                    break
                works.append(_normalize_work(json.loads(line)))
        if n_works is not None and len(works) != n_works:
            return None
        return works
    except (
        AttributeError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return None


def _normalize_work(row: Dict[str, Any]) -> WesternWork:
    """normalizes one work row for stable JSON serialization."""
    return {
        "id": str(row.get("id", "")),
        "title": str(row.get("title", "")),
        "text": str(row["text"]).strip(),
    }


def _write_snapshot(
    snapshot_dir: Path,
    works: List[WesternWork],
    metadata: Dict[str, Any],
) -> None:
    """atomically writes an integrity-checked JSONL snapshot and manifest."""
    snapshot_path = snapshot_dir / SNAPSHOT_FILENAME
    manifest_path = snapshot_dir / MANIFEST_FILENAME
    snapshot_tmp = _temporary_path(snapshot_dir, SNAPSHOT_FILENAME)
    manifest_tmp = _temporary_path(snapshot_dir, MANIFEST_FILENAME)
    try:
        with snapshot_tmp.open("w", encoding="utf-8") as file:
            for work in works:
                file.write(json.dumps(work, ensure_ascii=False) + "\n")

        manifest = {
            **metadata,
            "work_count": len(works),
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


def build_token_stream(texts: List[str], tokenizer: Any) -> List[int]:
    """tokenizes and concatenates texts into one stream separated by eos.

    each work is followed by the tokenizer's end-of-sequence id so the model
    learns document boundaries. the id falls back to 0 when the tokenizer does
    not expose ``eos_id``.

    Args:
        texts: cleaned work bodies.
        tokenizer: tokenizer with an ``encode(str) -> List[int]`` method.

    Returns:
        a flat list of token ids spanning the whole corpus.
    """
    eos_id = getattr(tokenizer, "eos_id", 0)
    stream: List[int] = []
    for text in texts:
        stream.extend(tokenizer.encode(text))
        stream.append(eos_id)
    return stream


def create_dataloaders(
    texts: List[str],
    tokenizer: Any,
    block_size: int = 512,
    batch_size: int = 16,
    val_fraction: float = 0.0,
    shuffle: bool = True,
    num_workers: int = 0,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """builds packed train (and optional validation) dataloaders.

    the corpus is tokenized into a single stream and split *at the token level*
    into a training and validation region, so the two never share blocks.

    Args:
        texts: work bodies to tokenize and pack.
        tokenizer: tokenizer instance used to encode the text.
        block_size: sequence length of each packed block.
        batch_size: batch size for the loaders.
        val_fraction: fraction of the token stream reserved for validation
            (``0.0`` disables the validation loader).
        shuffle: whether to shuffle the training blocks.
        num_workers: number of subprocesses for data loading.

    Returns:
        a ``(train_loader, val_loader)`` tuple; ``val_loader`` is ``None`` when
        ``val_fraction`` is 0 or the stream is too short to split.
    """
    stream = build_token_stream(texts, tokenizer)

    val_loader: Optional[DataLoader] = None
    split = int(len(stream) * (1.0 - val_fraction)) if val_fraction > 0 else len(stream)

    train_dataset = WesternDataset(stream[:split], block_size=block_size)
    train_loader = _make_loader(train_dataset, batch_size, shuffle, num_workers)

    if val_fraction > 0 and len(stream) - split >= block_size + 1:
        val_dataset = WesternDataset(stream[split:], block_size=block_size)
        val_loader = _make_loader(val_dataset, batch_size, False, num_workers)

    return train_loader, val_loader


def _make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    """creates a DataLoader with sensible cross-platform defaults."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=shuffle,
    )
