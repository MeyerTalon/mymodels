import math
from pathlib import Path

import pytest
import torch
from torch import nn

from core.tests.factories import training_config, training_settings
from core.training import TrainingConfig, decay_param_groups, warmup_cosine_schedule


def _lr_multipliers(
    *, warmup_epochs: int, num_epochs: int, min_lr_ratio: float
) -> list[float]:
    optimizer = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=1.0)
    scheduler = warmup_cosine_schedule(
        optimizer,
        warmup_epochs=warmup_epochs,
        num_epochs=num_epochs,
        min_lr_ratio=min_lr_ratio,
    )
    multipliers = []
    for _ in range(num_epochs):
        multipliers.append(optimizer.param_groups[0]['lr'])
        optimizer.step()
        scheduler.step()
    return multipliers


def test_warmup_then_cosine_to_floor() -> None:
    multipliers = _lr_multipliers(warmup_epochs=2, num_epochs=6, min_lr_ratio=0.1)
    assert multipliers[:2] == pytest.approx([1 / 3, 2 / 3])
    assert multipliers[2] == pytest.approx(1.0)
    assert multipliers == sorted(multipliers[:3]) + sorted(
        multipliers[3:], reverse=True
    )
    assert multipliers[-1] > 0.1


def test_no_warmup_starts_at_peak() -> None:
    multipliers = _lr_multipliers(warmup_epochs=0, num_epochs=4, min_lr_ratio=0.0)
    assert multipliers[0] == pytest.approx(1.0)
    assert multipliers[2] == pytest.approx((1 + math.cos(math.pi / 2)) / 2)


def test_decay_param_groups_skip_vectors() -> None:
    model = nn.Sequential(nn.Linear(4, 3), nn.LayerNorm(3))
    decayed, undecayed = decay_param_groups(model, 0.1)
    assert decayed['weight_decay'] == 0.1
    assert undecayed['weight_decay'] == 0.0
    decayed_params = decayed['params']
    undecayed_params = undecayed['params']
    assert isinstance(decayed_params, list)
    assert isinstance(undecayed_params, list)
    assert [p.dim() for p in decayed_params] == [2]
    assert [p.dim() for p in undecayed_params] == [1, 1, 1]


def test_training_config_resolves_paths_and_validates(tmp_path: Path) -> None:
    settings = training_settings(tmp_path)
    assert settings.weights_dir == tmp_path / 'weights'
    assert settings.min_lr_ratio == pytest.approx(0.1)
    with pytest.raises(ValueError, match='learning_rate'):
        TrainingConfig.from_config(training_config(tmp_path, learning_rate=0.0))
    with pytest.raises(ValueError, match='grad_accum_steps'):
        TrainingConfig.from_config(training_config(tmp_path, grad_accum_steps=0))
    with pytest.raises(ValueError):
        TrainingConfig.from_config(training_config(tmp_path, precision='fp8'))
