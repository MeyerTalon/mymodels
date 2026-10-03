from dataclasses import dataclass
from pathlib import Path

import torch

CHECKPOINT_EXTENSION = '.pt'
BEST_SUFFIX = '_best'
LATEST_SUFFIX = '_latest'
EPOCH_SUFFIX = '_epoch_'


def checkpoint_path(weights_dir: Path, model_name: str, suffix: str = '') -> Path:
    return weights_dir / f'{model_name}{suffix}{CHECKPOINT_EXTENSION}'


def find_checkpoint(weights_dir: Path, model_name: str) -> Path:
    """Prefers the best checkpoint, then the latest, then one saved under the bare model name."""
    candidates = [
        checkpoint_path(weights_dir, model_name, suffix)
        for suffix in (BEST_SUFFIX, LATEST_SUFFIX, '')
    ]
    found = next((path for path in candidates if path.is_file()), None)
    if found is None:
        raise FileNotFoundError(
            f'no checkpoint found; tried {[str(path) for path in candidates]}'
        )
    return found


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    config: dict[str, object]
    model_state_dict: dict[str, torch.Tensor]
    tokenizer_vocab_size: int | None

    def vocab_size(self) -> int:
        if self.tokenizer_vocab_size is None:
            raise ValueError(f'{self.path} has no tokenizer_vocab_size')
        return self.tokenizer_vocab_size


def load_checkpoint(path: Path, device: torch.device) -> Checkpoint:
    raw: object = torch.load(path, map_location=device)
    if not isinstance(raw, dict):
        raise TypeError(f'{path} is not a training checkpoint')
    config = raw.get('config')
    state = raw.get('model_state_dict')
    vocab_size = raw.get('tokenizer_vocab_size')
    if not isinstance(config, dict) or not isinstance(state, dict):
        raise TypeError(f'{path} is missing config or model_state_dict')
    return Checkpoint(
        path=path,
        config={str(key): value for key, value in config.items()},
        model_state_dict=state,
        tokenizer_vocab_size=vocab_size if isinstance(vocab_size, int) else None,
    )
