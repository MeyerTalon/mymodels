"""Hugging Face Wikipedia acquisition and packed data loading.

streams a bounded, deterministic article sample into a reusable local snapshot,
then builds the packed token stream used for causal language modeling.
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

SNAPSHOT_FILENAME = "wikipedia_articles.jsonl"
MANIFEST_FILENAME = "wikipedia_articles.manifest.json"


class WikipediaArticle(TypedDict):
    """article fields retained from the Wikimedia dataset."""

    id: str
    url: str
    title: str
    text: str


class WikipediaDataset(Dataset):
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


def load_wikipedia_texts(
    data_dir: str,
    n_articles: int,
    dataset_name: str = "wikimedia/wikipedia",
    dataset_config: str = "20231101.en",
    dataset_split: str = "train",
    dataset_revision: Optional[str] = None,
    dataset_seed: int = 42,
    shuffle_buffer_size: int = 10_000,
    dataset_cache_only: bool = False,
) -> List[str]:
    """loads article texts from a compatible snapshot or the Hugging Face Hub.

    Args:
        data_dir: directory containing the application-owned corpus snapshot.
        n_articles: number of non-empty articles to load.
        dataset_name: Hugging Face dataset repository.
        dataset_config: dated language configuration.
        dataset_split: dataset split to stream.
        dataset_revision: pinned Hub revision.
        dataset_seed: seed for deterministic streaming shuffle.
        shuffle_buffer_size: number of rows used for approximate shuffle.
        dataset_cache_only: when True, prohibit Hub access.

    Returns:
        selected article texts.

    Raises:
        ValueError: if settings are invalid, a cache-only snapshot is unavailable,
            or the streamed dataset does not contain enough non-empty articles.
    """
    if n_articles <= 0:
        raise ValueError("n_articles must be greater than zero.")
    if shuffle_buffer_size <= 0:
        raise ValueError("shuffle_buffer_size must be greater than zero.")

    snapshot_dir = Path(data_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    expected = _snapshot_metadata(
        dataset_name,
        dataset_config,
        dataset_split,
        dataset_revision,
        dataset_seed,
        shuffle_buffer_size,
    )
    cached = _load_snapshot(snapshot_dir, expected, n_articles)
    if cached is not None:
        print(f"Loaded {len(cached)} articles from {snapshot_dir / SNAPSHOT_FILENAME}")
        return [article["text"] for article in cached]

    if dataset_cache_only:
        raise ValueError(
            "dataset_cache_only=True but no compatible Wikipedia snapshot exists "
            f"in {data_dir}."
        )

    print(
        f"Streaming {n_articles} articles from "
        f"{dataset_name}/{dataset_config}:{dataset_split}..."
    )
    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split=dataset_split,
        revision=dataset_revision,
        streaming=True,
    )
    shuffled = dataset.filter(_has_text).shuffle(
        seed=dataset_seed,
        buffer_size=shuffle_buffer_size,
    )
    articles = [_normalize_article(row) for row in shuffled.take(n_articles)]
    if len(articles) != n_articles:
        raise ValueError(
            f"requested {n_articles} articles but only found {len(articles)} "
            "non-empty rows."
        )

    _write_snapshot(snapshot_dir, articles, expected)
    print(f"Cached {len(articles)} articles in {snapshot_dir / SNAPSHOT_FILENAME}")
    return [article["text"] for article in articles]


def _has_text(row: Dict[str, Any]) -> bool:
    """returns whether a dataset row contains non-empty article text."""
    return bool(str(row.get("text", "")).strip())


def _normalize_article(row: Dict[str, Any]) -> WikipediaArticle:
    """normalizes one Hugging Face row for stable JSON serialization."""
    return {
        "id": str(row.get("id", "")),
        "url": str(row.get("url", "")),
        "title": str(row.get("title", "")),
        "text": str(row["text"]).strip(),
    }


def _snapshot_metadata(
    dataset_name: str,
    dataset_config: str,
    dataset_split: str,
    dataset_revision: Optional[str],
    dataset_seed: int,
    shuffle_buffer_size: int,
) -> Dict[str, Any]:
    """builds the settings that identify a reproducible corpus snapshot."""
    return {
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "dataset_split": dataset_split,
        "dataset_revision": dataset_revision,
        "dataset_seed": dataset_seed,
        "shuffle_buffer_size": shuffle_buffer_size,
    }


def _load_snapshot(
    snapshot_dir: Path,
    expected: Dict[str, Any],
    n_articles: int,
) -> Optional[List[WikipediaArticle]]:
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
        if int(manifest.get("article_count", 0)) < n_articles:
            return None
        if _file_sha256(snapshot_path) != manifest.get("sha256"):
            return None

        articles: List[WikipediaArticle] = []
        with snapshot_path.open("r", encoding="utf-8") as file:
            for line in file:
                if len(articles) == n_articles:
                    break
                articles.append(_normalize_article(json.loads(line)))
        return articles if len(articles) == n_articles else None
    except (
        AttributeError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return None


def _write_snapshot(
    snapshot_dir: Path,
    articles: List[WikipediaArticle],
    metadata: Dict[str, Any],
) -> None:
    """atomically writes an integrity-checked JSONL snapshot and manifest."""
    snapshot_path = snapshot_dir / SNAPSHOT_FILENAME
    manifest_path = snapshot_dir / MANIFEST_FILENAME
    snapshot_tmp = _temporary_path(snapshot_dir, SNAPSHOT_FILENAME)
    manifest_tmp = _temporary_path(snapshot_dir, MANIFEST_FILENAME)
    try:
        with snapshot_tmp.open("w", encoding="utf-8") as file:
            for article in articles:
                file.write(json.dumps(article, ensure_ascii=False) + "\n")

        manifest = {
            **metadata,
            "article_count": len(articles),
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

    each article is followed by the tokenizer's end-of-sequence id so the model
    learns document boundaries. the id falls back to 0 when the tokenizer does
    not expose ``eos_id``.

    Args:
        texts: cleaned article bodies.
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
        texts: article bodies to tokenize and pack.
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

    train_dataset = WikipediaDataset(stream[:split], block_size=block_size)
    train_loader = _make_loader(train_dataset, batch_size, shuffle, num_workers)

    if val_fraction > 0 and len(stream) - split >= block_size + 1:
        val_dataset = WikipediaDataset(stream[split:], block_size=block_size)
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
