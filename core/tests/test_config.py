from pathlib import Path

import pytest

from core.config import (
    load_config,
    optional_int,
    require_bool,
    require_float,
    require_int,
    require_str_list,
)


def test_require_int_rejects_bool_and_missing() -> None:
    with pytest.raises(TypeError):
        require_int({'n': True}, 'n')
    with pytest.raises(ValueError, match='missing'):
        require_int({}, 'n')
    assert require_int({'n': 3}, 'n') == 3


def test_require_float_accepts_int() -> None:
    assert require_float({'lr': 1}, 'lr') == 1.0
    with pytest.raises(TypeError):
        require_float({'lr': '0.1'}, 'lr')


def test_optional_int_allows_null_but_not_absence() -> None:
    assert optional_int({'max_works': None}, 'max_works') is None
    with pytest.raises(ValueError, match='missing'):
        optional_int({}, 'max_works')


def test_require_bool_and_str_list() -> None:
    assert require_bool({'flag': False}, 'flag') is False
    assert require_str_list({'langs': ['en', 'es']}, 'langs') == ['en', 'es']
    with pytest.raises(TypeError):
        require_str_list({'langs': ['en', 1]}, 'langs')


def test_load_config_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / 'config.yaml'
    path.write_text('- a\n- b\n', encoding='utf-8')
    with pytest.raises(TypeError):
        load_config(path)
