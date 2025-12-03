"""Data loading and Wikipedia article downloading module.

Handles downloading random Wikipedia articles via the HTTP API and creating PyTorch
Dataset and DataLoader instances for training. Includes utilities for filename
sanitization and text extraction.
"""

import os
import re
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, List, Optional, Tuple

import requests
import torch
from torch.utils.data import DataLoader, Dataset


class WikipediaDataset(Dataset):
    """Dataset of tokenized Wikipedia articles for language modeling."""

    def __init__(self, texts: List[str], tokenizer: Any, max_length: int = 512) -> None:
        """Initializes the dataset.

        Args:
            texts: List of raw article texts.
            tokenizer: Tokenizer with an ``encode(str) -> List[int]`` method.
            max_length: Maximum sequence length (including target token).
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Returns the number of articles."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns a single (input_ids, target_ids) pair.

        Args:
            idx: Index of the article.

        Returns:
            A tuple of:

            * input_ids: Tensor of token ids of shape (seq_len - 1,).
            * target_ids: Tensor of next-token labels of shape (seq_len - 1,).
        """
        text = self.texts[idx]
        tokens = self.tokenizer.encode(text)

        # Truncate or pad to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[: self.max_length]
        else:
            tokens = tokens + [0] * (self.max_length - len(tokens))

        # Create input and target (shifted by 1 for next token prediction)
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)

        return input_ids, target_ids


def sanitize_filename(title: str) -> str:
    """Converts a Wikipedia article title to a safe filename.

    The result is lowercase, uses underscores between words, and strips all
    non-alphanumeric characters before use.

    Args:
        title: Original article title.

    Returns:
        Sanitized filename (without extension).
    """
    title = title.lower()
    title = re.sub(r"[^a-z0-9]+", "_", title)
    title = title.strip("_")
    return title


def download_wikipedia_articles(
    n: int,
    save_dir: str = "wikipedia/data",
    delay: float = 0.0,
) -> List[Tuple[str, str]]:
    """Downloads ``n`` random Wikipedia articles via the public HTTP API.

    Articles are fetched using the official Wikipedia API, saved as text files,
    and basic metadata (title and URL) is written at the top of each file.
    Downloading is parallelized with multiprocessing for maximum throughput.

    Args:
        n: Number of articles to download.
        save_dir: Directory to which article text files will be written.
        delay: Delay in seconds between requests per worker, to be polite.

    Returns:
        List of ``(title, filepath)`` tuples for successfully downloaded articles.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    api_url = "https://en.wikipedia.org/w/api.php"
    downloaded: List[Tuple[str, str]] = []

    # Prepare arguments for worker processes
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
    """Worker function to download a single random Wikipedia article.

    Args:
        args: Tuple of (index, api_url, save_dir, delay_seconds).

    Returns:
        (title, filepath) tuple if successful, otherwise None.
    """
    index, api_url, save_dir, delay = args

    params = {
        "action": "query",
        "format": "json",
        "list": "random",
        "rnnamespace": 0,  # Main namespace only
        "rnlimit": 1,
    }

    headers = {
        "User-Agent": "WikipediaScraperBot/1.0 (Educational/Research purposes)"
    }

    try:
        # Request a random article
        response = requests.get(api_url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        page = data["query"]["random"][0]
        page_id = page["id"]
        page_title = page["title"]

        # Fetch the article content
        content_params = {
            "action": "query",
            "format": "json",
            "pageids": page_id,
            "prop": "extracts",
            "explaintext": True,  # Plain text without HTML
        }

        content_response = requests.get(
            api_url, params=content_params, headers=headers, timeout=10
        )
        content_response.raise_for_status()
        content_data = content_response.json()

        article_text = content_data["query"]["pages"][str(page_id)].get("extract", "")

        # Sanitize filename (limit length to avoid OS issues)
        base_name = sanitize_filename(page_title) or f"article_{page_id}"
        base_name = base_name[:100]
        filename = f"{base_name}.txt"
        filepath = os.path.join(save_dir, filename)

        # Save to file with simple metadata header
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Title: {page_title}\n")
            f.write(f"URL: https://en.wikipedia.org/?curid={page_id}\n")
            f.write("=" * 80 + "\n\n")
            f.write(article_text)

        print(f"[{index + 1}] Downloaded: {page_title}")
        print(f"         Saved to: {filepath}")

        # Optional per-worker delay
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
    """Loads all article texts from a directory.

    Args:
        data_dir: Directory containing ``.txt`` article files.

    Returns:
        List of article contents as strings.
    """
    texts: List[str] = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                texts.append(f.read())
    return texts


def create_dataloader(
    data_dir: str,
    tokenizer: Any,
    n_articles: int,
    batch_size: int = 32,
    max_length: int = 512,
    shuffle: bool = True,
    num_workers: int = 0,
    use_local_articles: bool = False,
) -> DataLoader:
    """Creates a DataLoader over Wikipedia articles.

    This function will download ``n_articles`` random Wikipedia pages into the
    specified directory and then build a ``WikipediaDataset`` and
    ``DataLoader`` from the downloaded texts.

    Args:
        data_dir: Directory where article text files will be stored or read from.
        tokenizer: Tokenizer instance used to encode the text.
        n_articles: Number of articles to download or sample.
        batch_size: Batch size for the loader.
        max_length: Maximum token sequence length.
        shuffle: Whether to shuffle the dataset.
        num_workers: Number of subprocesses for data loading.
        use_local_articles: When True, only use already downloaded local
            articles and randomly sample up to ``n_articles`` of them.

    Returns:
        A configured ``torch.utils.data.DataLoader`` instance.
    """
    # Ensure directory exists
    os.makedirs(data_dir, exist_ok=True)

    texts: List[str] = []

    if use_local_articles:
        # Use only locally available articles
        all_texts = load_articles_from_dir(data_dir)
        if not all_texts:
            raise ValueError(
                "use_local_articles=True but no .txt files were found in "
                f"{data_dir}. Please download articles first."
            )
        if len(all_texts) > n_articles:
            import random

            texts = random.sample(all_texts, n_articles)
        else:
            texts = all_texts
    else:
        # Download fresh set of articles. In some environments (e.g., no network),
        # this may fail and return an empty list.
        downloaded = download_wikipedia_articles(n_articles, save_dir=data_dir)

        if downloaded:
            # Prefer texts from the files we just downloaded
            for _, filepath in downloaded:
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        texts.append(f.read())
                except OSError:
                    continue

        if not texts:
            # Fallback: use whatever .txt files already exist in the directory
            print(
                "Warning: failed to download new Wikipedia articles, "
                f"falling back to any existing .txt files in {data_dir}"
            )
            texts = load_articles_from_dir(data_dir)

        if not texts:
            raise ValueError(
                "No readable article texts found in "
                f"{data_dir}. Ensure you have network access or pre-populated .txt files."
            )

    dataset = WikipediaDataset(texts, tokenizer, max_length=max_length)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return dataloader

