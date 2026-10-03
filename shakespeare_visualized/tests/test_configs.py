from pathlib import Path

import pytest
import torch

from core.config import load_config, require_int
from core.training import TrainingConfig, count_parameters
from shakespeare_visualized.architecture import DecoderConfig, DecoderOnlyTransformer

CONFIG_PATHS = sorted(
    Path(__file__).resolve().parents[1].joinpath('configs').glob('*.yaml')
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
