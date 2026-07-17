"""data loading and Wikipedia article downloading module.

handles downloading random Wikipedia articles via the HTTP API and building the
packed, contiguous token stream used for causal language modeling. articles are
tokenized once, joined with an end-of-text separator, and split into fixed-length
blocks so every token contributes a training signal (no padding waste).
"""

import os
import random
import re
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, List, Optional, Tuple

import requests
import torch
from torch.utils.data import DataLoader, Dataset

# separator between the metadata header and the article body written by the
# downloader (a line of '=' characters).
_HEADER_SEPARATOR = "=" * 80


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


def sanitize_filename(title: str) -> str:
    """converts a Wikipedia article title to a safe filename.

    the result is lowercase, uses underscores between words, and strips all
    non-alphanumeric characters before use.

    Args:
        title: original article title.

    Returns:
        sanitized filename (without extension).
    """
    title = title.lower()
    title = re.sub(r"[^a-z0-9]+", "_", title)
    title = title.strip("_")
    return title


def extract_article_text(raw: str) -> str:
    """strips the downloader metadata header from a raw article file.

    downloaded files start with ``Title:``/``URL:`` lines followed by a line of
    ``=`` characters and a blank line; only the body after that separator is
    useful for language modeling. files without the header are returned as-is.

    Args:
        raw: full contents of an article ``.txt`` file.

    Returns:
        the article body with surrounding whitespace stripped.
    """
    marker_index = raw.find(_HEADER_SEPARATOR)
    if marker_index != -1:
        newline_index = raw.find("\n", marker_index)
        if newline_index != -1:
            return raw[newline_index + 1 :].strip()
    return raw.strip()


def download_wikipedia_articles(
    n: int,
    save_dir: str = "wikipedia/data",
    delay: float = 0.0,
) -> List[Tuple[str, str]]:
    """downloads ``n`` random Wikipedia articles via the public HTTP API.

    articles are fetched using the official Wikipedia API, saved as text files,
    and basic metadata (title and URL) is written at the top of each file.
    downloading is parallelized with multiprocessing for maximum throughput.

    Args:
        n: number of articles to download.
        save_dir: directory to which article text files will be written.
        delay: delay in seconds between requests per worker, to be polite.

    Returns:
        list of ``(title, filepath)`` tuples for successfully downloaded articles.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    api_url = "https://en.wikipedia.org/w/api.php"
    downloaded: List[Tuple[str, str]] = []

    # prepare arguments for worker processes
    args = [(i, api_url, save_dir, delay) for i in range(n)]
    num_procs = min(cpu_count() or 2, 8)

    print(f"Fetching {n} random Wikipedia articles using {num_procs} processes...")

    with Pool(processes=num_procs) as pool:
        for result in pool.imap_unordered(_download_single_article, args):
            if result is not None:
                downloaded.append(result)

    print(f"Successfully downloaded {len(downloaded)} articles")
    return downloaded


def _download_single_article(
    args: Tuple[int, str, str, float],
) -> Optional[Tuple[str, str]]:
    """worker function to download a single random Wikipedia article.

    Args:
        args: tuple of (index, api_url, save_dir, delay_seconds).

    Returns:
        (title, filepath) tuple if successful, otherwise None.
    """
    index, api_url, save_dir, delay = args

    params = {
        "action": "query",
        "format": "json",
        "list": "random",
        "rnnamespace": 0,  # main namespace only
        "rnlimit": 1,
    }

    headers = {
        "User-Agent": "WikipediaScraperBot/1.0 (Educational/Research purposes)"
    }

    try:
        # request a random article
        response = requests.get(api_url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        page = data["query"]["random"][0]
        page_id = page["id"]
        page_title = page["title"]

        # fetch the article content
        content_params = {
            "action": "query",
            "format": "json",
            "pageids": page_id,
            "prop": "extracts",
            "explaintext": True,  # plain text without HTML
        }

        content_response = requests.get(
            api_url, params=content_params, headers=headers, timeout=10
        )
        content_response.raise_for_status()
        content_data = content_response.json()

        article_text = content_data["query"]["pages"][str(page_id)].get("extract", "")

        # sanitize filename (limit length to avoid OS issues)
        base_name = sanitize_filename(page_title) or f"article_{page_id}"
        base_name = base_name[:100]
        filename = f"{base_name}.txt"
        filepath = os.path.join(save_dir, filename)

        # save to file with simple metadata header
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Title: {page_title}\n")
            f.write(f"URL: https://en.wikipedia.org/?curid={page_id}\n")
            f.write(_HEADER_SEPARATOR + "\n\n")
            f.write(article_text)

        print(f"[{index + 1}] Downloaded: {page_title}")

        # optional per-worker delay
        if delay > 0:
            time.sleep(delay)

        return page_title, filepath

    except requests.exceptions.RequestException as exc:
        print(f"[{index + 1}] Error fetching article: {exc}")
        return None
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[{index + 1}] Unexpected error: {exc}")
        return None


def load_articles_from_dir(data_dir: str) -> List[str]:
    """loads and cleans all article bodies from a directory.

    the metadata header is stripped from each file so only the article text is
    returned.

    Args:
        data_dir: directory containing ``.txt`` article files.

    Returns:
        list of cleaned article bodies as strings.
    """
    texts: List[str] = []
    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                body = extract_article_text(f.read())
            if body:
                texts.append(body)
    return texts


def gather_texts(
    data_dir: str,
    n_articles: int,
    use_local_articles: bool,
) -> List[str]:
    """collects cleaned article texts, downloading them if requested.

    Args:
        data_dir: directory where article text files are stored or read from.
        n_articles: number of articles to download or sample.
        use_local_articles: when True, sample from already-downloaded articles
            instead of hitting the network.

    Returns:
        list of cleaned article bodies.

    Raises:
        ValueError: if no readable article texts are available.
    """
    os.makedirs(data_dir, exist_ok=True)

    if use_local_articles:
        all_texts = load_articles_from_dir(data_dir)
        if not all_texts:
            raise ValueError(
                "use_local_articles=True but no .txt files were found in "
                f"{data_dir}. Please download articles first."
            )
        if len(all_texts) > n_articles:
            return random.sample(all_texts, n_articles)
        return all_texts

    # download a fresh set; fall back to any local files if the network fails
    download_wikipedia_articles(n_articles, save_dir=data_dir)
    texts = load_articles_from_dir(data_dir)
    if not texts:
        raise ValueError(
            "No readable article texts found in "
            f"{data_dir}. Ensure you have network access or pre-populated .txt files."
        )
    return texts


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
    data_dir: str,
    tokenizer: Any,
    n_articles: int,
    block_size: int = 512,
    batch_size: int = 16,
    val_fraction: float = 0.0,
    shuffle: bool = True,
    num_workers: int = 0,
    use_local_articles: bool = False,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """builds packed train (and optional validation) dataloaders.

    the corpus is tokenized into a single stream and split *at the token level*
    into a training and validation region, so the two never share blocks.

    Args:
        data_dir: directory where article text files are stored or read from.
        tokenizer: tokenizer instance used to encode the text.
        n_articles: number of articles to download or sample.
        block_size: sequence length of each packed block.
        batch_size: batch size for the loaders.
        val_fraction: fraction of the token stream reserved for validation
            (``0.0`` disables the validation loader).
        shuffle: whether to shuffle the training blocks.
        num_workers: number of subprocesses for data loading.
        use_local_articles: when True, only use already-downloaded articles.

    Returns:
        a ``(train_loader, val_loader)`` tuple; ``val_loader`` is ``None`` when
        ``val_fraction`` is 0 or the stream is too short to split.
    """
    texts = gather_texts(data_dir, n_articles, use_local_articles)
    stream = build_token_stream(texts, tokenizer)

    val_loader: Optional[DataLoader] = None
    split = int(len(stream) * (1.0 - val_fraction)) if val_fraction > 0 else len(stream)

    train_dataset = WikipediaDataset(stream[:split], block_size=block_size)
    train_loader = _make_loader(
        train_dataset, batch_size, shuffle, num_workers
    )

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
