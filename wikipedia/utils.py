"""shared utilities for the wikipedia package.

provides device selection and repo-root-relative path resolution used by the
training and inference entry points.
"""

import os
from pathlib import Path

import torch

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
TOKENIZER_DIR: str = str(Path(__file__).resolve().parent / "tokenizer_files")


def select_device() -> torch.device:
    """selects the best available device: MPS, then CUDA, then CPU.

    Returns:
        the preferred ``torch.device`` for this machine.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_repo_path(path: str) -> str:
    """resolves a repo-root-relative path to an absolute path.

    Args:
        path: absolute path, or path relative to the repository root
            (e.g. ``wikipedia/weights``).

    Returns:
        the absolute path as a string.
    """
    if os.path.isabs(path):
        return path
    return str(REPO_ROOT / path)
