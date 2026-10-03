from pathlib import Path

import pytest
import torch

from core.checkpoint import find_checkpoint, load_checkpoint


def test_find_checkpoint_prefers_best_then_latest(tmp_path: Path) -> None:
    (tmp_path / 'model.pt').touch()
    assert find_checkpoint(tmp_path, 'model') == tmp_path / 'model.pt'
    (tmp_path / 'model_latest.pt').touch()
    assert find_checkpoint(tmp_path, 'model') == tmp_path / 'model_latest.pt'
    (tmp_path / 'model_best.pt').touch()
    assert find_checkpoint(tmp_path, 'model') == tmp_path / 'model_best.pt'


def test_find_checkpoint_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        find_checkpoint(tmp_path, 'model')


def test_load_checkpoint_reads_config_and_vocab(tmp_path: Path) -> None:
    path = tmp_path / 'model_best.pt'
    torch.save({'config': {'d_model': 8}, 'model_state_dict': {}}, path)
    checkpoint = load_checkpoint(path, torch.device('cpu'))
    assert checkpoint.config == {'d_model': 8}
    with pytest.raises(ValueError, match='tokenizer_vocab_size'):
        checkpoint.vocab_size()
