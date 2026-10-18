from pathlib import Path

import pytest

from core.config import load_config
from core.paths import REPO_ROOT
from stellaris.policy import PolicyConfig

CONFIG_PATHS = sorted((REPO_ROOT / 'stellaris' / 'configs').glob('*.yaml'))


@pytest.mark.parametrize('config_path', CONFIG_PATHS, ids=lambda path: path.stem)
def test_configs_parse(config_path: Path) -> None:
    PolicyConfig.from_config(load_config(config_path))


def test_config_rejects_zero_new_tokens() -> None:
    config = load_config(CONFIG_PATHS[0])
    with pytest.raises(ValueError, match='max_new_tokens'):
        PolicyConfig.from_config({**config, 'max_new_tokens': 0})
