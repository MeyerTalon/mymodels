from pathlib import Path

import pytest
import torch

from core.config import load_config, require_int
from core.paths import REPO_ROOT
from core.training import TrainingConfig, count_parameters
from gpt.architecture import DecoderConfig, DecoderOnlyTransformer

CONFIG_PATHS = sorted(
    path
    for package in ('shakespeare', 'western', 'wikipedia')
    for path in (REPO_ROOT / package / 'configs').glob('*.yaml')
)


@pytest.mark.parametrize('path', CONFIG_PATHS, ids=lambda path: path.stem)
def test_config_matches_expected_parameters(path: Path) -> None:
    config = load_config(path)
    TrainingConfig.from_config(config)
    with torch.device('meta'):
        model = DecoderOnlyTransformer(
            require_int(config, 'vocab_size'), DecoderConfig.from_config(config)
        )
    assert count_parameters(model) == require_int(config, 'expected_parameters')
