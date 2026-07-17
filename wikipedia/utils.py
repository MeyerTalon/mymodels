"""shared utilities for the wikipedia package.

provides device selection and repo-root-relative path resolution used by the
training and inference entry points.
"""

import os
from pathlib import Path
from typing import Optional

import torch

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
TOKENIZER_DIR: str = str(Path(__file__).resolve().parent / "tokenizer_files")

# supported mixed-precision settings mapped to their autocast dtype (``None``
# means full float32, i.e. autocast disabled).
_PRECISION_DTYPES = {
    "fp32": None,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


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


def select_autocast_dtype(
    device: torch.device,
    precision: str,
) -> Optional[torch.dtype]:
    """resolves the autocast dtype for a device and precision setting.

    mixed precision is only enabled on accelerators (MPS, CUDA); on CPU it is
    disabled regardless of the requested precision, since CPU autocast offers
    little benefit here and float16 is poorly supported.

    Args:
        device: the device training/inference runs on.
        precision: one of ``"fp32"``, ``"fp16"``, or ``"bf16"``.

    Returns:
        the ``torch.dtype`` to autocast to, or ``None`` to run in float32.

    Raises:
        ValueError: if ``precision`` is not a recognized value.
    """
    if precision not in _PRECISION_DTYPES:
        raise ValueError(
            f"unknown precision {precision!r}; expected one of "
            f"{sorted(_PRECISION_DTYPES)}."
        )
    if device.type == "cpu":
        return None
    return _PRECISION_DTYPES[precision]


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
