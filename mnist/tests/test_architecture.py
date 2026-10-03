from pathlib import Path

import pytest
import torch

from core.config import load_config, require_int
from core.paths import REPO_ROOT
from core.training import count_parameters
from mnist.architecture import CnnConfig, MnistCNN
from mnist.tests.fixtures import SMALL_CNN

CONFIG_PATHS = sorted((REPO_ROOT / 'mnist' / 'configs').glob('*.yaml'))


def test_forward_shape() -> None:
    model = MnistCNN(CnnConfig.from_config(SMALL_CNN))
    assert model(torch.rand(3, 1, 28, 28)).shape == (3, 10)


@pytest.mark.parametrize('path', CONFIG_PATHS, ids=lambda path: path.stem)
def test_config_matches_expected_parameters(path: Path) -> None:
    config = load_config(path)
    with torch.device('meta'):
        model = MnistCNN(CnnConfig.from_config(config))
    assert count_parameters(model) == require_int(config, 'expected_parameters')
