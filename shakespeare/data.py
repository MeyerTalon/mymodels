"""Shakespeare corpus acquisition and packed data loading.

downloads the Project Gutenberg complete works into a reusable local snapshot,
splits them into individual works, then builds the packed token stream used for
causal language modeling.
"""

import hashlib
import json
import os
import re
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import torch
from torch.utils.data import DataLoader, Dataset

SNAPSHOT_FILENAME = "shakespeare_works.jsonl"
MANIFEST_FILENAME = "shakespeare_works.manifest.json"
RAW_FILENAME = "shakespeare_complete.txt"

# Project Gutenberg ebook #100 — the complete works of William Shakespeare.
DEFAULT_CORPUS_URL = "https://www.gutenberg.org/files/100/100-0.txt"

# minimum characters for a split chunk to count as a work (filters front matter).
_MIN_WORK_CHARS = 500

# canonical work titles as they appear as standalone lines in Gutenberg ebook #100.
# used to split the complete works into plays/poems rather than scenes.
_WORK_TITLES = frozenset(
    {
        "THE SONNETS",
        "ALL’S WELL THAT ENDS WELL",
        "THE TRAGEDY OF ANTONY AND CLEOPATRA",
        "AS YOU LIKE IT",
        "THE COMEDY OF ERRORS",
        "THE TRAGEDY OF CORIOLANUS",
        "CYMBELINE",
        "THE TRAGEDY OF HAMLET, PRINCE OF DENMARK",
        "THE FIRST PART OF KING HENRY THE FOURTH",
        "THE SECOND PART OF KING HENRY THE FOURTH",
        "THE LIFE OF KING HENRY THE FIFTH",
        "THE FIRST PART OF HENRY THE SIXTH",
        "THE SECOND PART OF KING HENRY THE SIXTH",
        "THE THIRD PART OF KING HENRY THE SIXTH",
        "KING HENRY THE EIGHTH",
        "THE LIFE AND DEATH OF KING JOHN",
        "THE TRAGEDY OF JULIUS CAESAR",
        "THE TRAGEDY OF KING LEAR",
        "LOVE’S LABOUR’S LOST",
        "THE TRAGEDY OF MACBETH",
        "MEASURE FOR MEASURE",
        "THE MERCHANT OF VENICE",
        "THE MERRY WIVES OF WINDSOR",
        "A MIDSUMMER NIGHT’S DREAM",
        "MUCH ADO ABOUT NOTHING",
        "THE TRAGEDY OF OTHELLO, THE MOOR OF VENICE",
        "PERICLES, PRINCE OF TYRE",
        "KING RICHARD THE SECOND",
        "KING RICHARD THE THIRD",
        "THE TRAGEDY OF ROMEO AND JULIET",
        "THE TAMING OF THE SHREW",
        "THE TEMPEST",
        "THE LIFE OF TIMON OF ATHENS",
        "THE TRAGEDY OF TITUS ANDRONICUS",
        "TROILUS AND CRESSIDA",
        "TWELFTH NIGHT; OR, WHAT YOU WILL",
        "THE TWO GENTLEMEN OF VERONA",
        "THE TWO NOBLE KINSMEN",
        "THE WINTER’S TALE",
        "A LOVER’S COMPLAINT",
        "THE PASSIONATE PILGRIM",
        "THE PHOENIX AND THE TURTLE",
        "THE RAPE OF LUCRECE",
        "VENUS AND ADONIS",
    }
)


class ShakespeareWork(TypedDict):
    """fields retained for one play, poem, or other work."""

    id: str
    title: str
    text: str


class ShakespeareDataset(Dataset):
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


def load_shakespeare_texts(
    data_dir: str,
    corpus_url: str = DEFAULT_CORPUS_URL,
    max_works: Optional[int] = None,
    dataset_cache_only: bool = False,
) -> List[str]:
    """loads work texts from a compatible snapshot or downloads the corpus.

    Args:
        data_dir: directory containing the application-owned corpus snapshot.
        corpus_url: URL of the complete-works plain-text file.
        max_works: if set, keep only the first ``max_works`` non-empty works
            (useful for smoke tests); ``None`` keeps the entire corpus.
        dataset_cache_only: when True, prohibit network access.

    Returns:
        selected work texts.

    Raises:
        ValueError: if settings are invalid, a cache-only snapshot is unavailable,
            or the corpus does not contain enough works.
    """
    if max_works is not None and max_works <= 0:
        raise ValueError("max_works must be greater than zero when set.")

    snapshot_dir = Path(data_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    expected = _snapshot_metadata(corpus_url)
    cached = _load_snapshot(snapshot_dir, expected, max_works)
    if cached is not None:
        print(f"Loaded {len(cached)} works from {snapshot_dir / SNAPSHOT_FILENAME}")
        return [work["text"] for work in cached]

    if dataset_cache_only:
        raise ValueError(
            "dataset_cache_only=True but no compatible Shakespeare snapshot exists "
            f"in {data_dir}."
        )

    print(f"Downloading Shakespeare complete works from {corpus_url}...")
    raw_text = _download_corpus(corpus_url)
    raw_path = snapshot_dir / RAW_FILENAME
    raw_path.write_text(raw_text, encoding="utf-8")

    works = _parse_works(raw_text)
    if not works:
        raise ValueError("no Shakespeare works found after parsing the corpus.")
    if max_works is not None and len(works) < max_works:
        raise ValueError(
            f"requested {max_works} works but only found {len(works)} "
            "non-empty works in the corpus."
        )

    # always persist the full corpus so a later full-corpus run can reuse it;
    # ``max_works`` only limits what this call returns.
    _write_snapshot(snapshot_dir, works, expected)
    print(f"Cached {len(works)} works in {snapshot_dir / SNAPSHOT_FILENAME}")
    selected = works if max_works is None else works[:max_works]
    return [work["text"] for work in selected]


def _download_corpus(url: str) -> str:
    """downloads the complete-works text from ``url``.

    Args:
        url: remote plain-text corpus URL.

    Returns:
        the downloaded text decoded as UTF-8 (with Latin-1 fallback).

    Raises:
        ValueError: if the download fails.
    """
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "mymodels-shakespeare/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise ValueError(
            f"failed to download Shakespeare corpus from {url}: {exc}"
        ) from exc

    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        return payload.decode("latin-1")


def _strip_gutenberg_boilerplate(text: str) -> str:
    """removes the Project Gutenberg header and footer when present."""
    start_match = re.search(r"\*\*\*\s*START OF .+?\*\*\*", text, flags=re.IGNORECASE)
    if start_match is not None:
        text = text[start_match.end() :]
    end_match = re.search(r"\*\*\*\s*END OF .+?\*\*\*", text, flags=re.IGNORECASE)
    if end_match is not None:
        text = text[: end_match.start()]
    return text.strip()


def _parse_works(raw_text: str) -> List[ShakespeareWork]:
    """splits the cleaned complete works into individual play/poem texts.

    splits on standalone lines that match known Shakespeare work titles from
    Project Gutenberg ebook #100. chunks shorter than ``_MIN_WORK_CHARS`` are
    dropped. if title-based splitting finds nothing usable, falls back to
    blank-line splits, then finally to the whole cleaned text as one work.
    """
    body = _strip_gutenberg_boilerplate(raw_text)
    works = _split_on_titles(body)
    if len(works) < 2:
        works = _split_on_blank_lines(body)
    if not works and body:
        works.append({"id": "0", "title": "Complete Works", "text": body})
    return works


def _split_on_titles(body: str) -> List[ShakespeareWork]:
    """splits ``body`` whenever a line exactly matches a known work title.

    when a title appears more than once (contents listing vs body), the longest
    chunk for that title is kept.
    """
    lines = body.splitlines()
    starts: List[Tuple[int, str]] = []
    for index, line in enumerate(lines):
        title = line.strip()
        if title in _WORK_TITLES:
            starts.append((index, title))

    if not starts:
        return []

    best_by_title: Dict[str, str] = {}
    for i, (start, title) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(lines)
        text = "\n".join(lines[start:end]).strip()
        if len(text) < _MIN_WORK_CHARS:
            continue
        previous = best_by_title.get(title)
        if previous is None or len(text) > len(previous):
            best_by_title[title] = text

    # preserve first-seen title order from the corpus body
    ordered_titles: List[str] = []
    for _, title in starts:
        if title in best_by_title and title not in ordered_titles:
            ordered_titles.append(title)

    return [
        {"id": str(index), "title": title, "text": best_by_title[title]}
        for index, title in enumerate(ordered_titles)
    ]

def _split_on_blank_lines(body: str) -> List[ShakespeareWork]:
    """fallback splitter using runs of blank lines between sections."""
    chunks = re.split(r"\n\s*\n\s*\n+", body)
    works: List[ShakespeareWork] = []
    for index, chunk in enumerate(chunks):
        text = chunk.strip()
        if len(text) < _MIN_WORK_CHARS:
            continue
        title = _infer_title(text, index)
        works.append({"id": str(len(works)), "title": title, "text": text})
    return works


def _infer_title(text: str, index: int) -> str:
    """picks a short title from the first non-empty line of a work chunk."""
    for line in text.splitlines():
        candidate = line.strip()
        if candidate:
            return candidate[:120]
    return f"work_{index}"


def _snapshot_metadata(corpus_url: str) -> Dict[str, Any]:
    """builds the settings that identify a reproducible corpus snapshot."""
    return {"corpus_url": corpus_url}


def _load_snapshot(
    snapshot_dir: Path,
    expected: Dict[str, Any],
    n_works: Optional[int],
) -> Optional[List[ShakespeareWork]]:
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

        works: List[ShakespeareWork] = []
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


def _normalize_work(row: Dict[str, Any]) -> ShakespeareWork:
    """normalizes one work row for stable JSON serialization."""
    return {
        "id": str(row.get("id", "")),
        "title": str(row.get("title", "")),
        "text": str(row["text"]).strip(),
    }


def _write_snapshot(
    snapshot_dir: Path,
    works: List[ShakespeareWork],
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

    train_dataset = ShakespeareDataset(stream[:split], block_size=block_size)
    train_loader = _make_loader(train_dataset, batch_size, shuffle, num_workers)

    if val_fraction > 0 and len(stream) - split >= block_size + 1:
        val_dataset = ShakespeareDataset(stream[split:], block_size=block_size)
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
