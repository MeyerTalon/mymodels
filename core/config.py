from collections.abc import Mapping
from pathlib import Path

import yaml

Config = Mapping[str, object]


def load_config(path: Path) -> dict[str, object]:
    with path.open(encoding='utf-8') as file:
        loaded: object = yaml.safe_load(file)
    if not isinstance(loaded, dict):
        raise TypeError(f'{path} must contain a YAML mapping')
    return {str(key): value for key, value in loaded.items()}


def _value(config: Config, key: str) -> object:
    if key not in config:
        raise ValueError(f'config is missing {key!r}')
    return config[key]


def _type_error(key: str, expected: str, value: object) -> TypeError:
    return TypeError(f'config {key!r} must be {expected}, got {value!r}')


def require_int(config: Config, key: str) -> int:
    value = _value(config, key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise _type_error(key, 'an integer', value)
    return value


def optional_int(config: Config, key: str) -> int | None:
    return None if _value(config, key) is None else require_int(config, key)


def require_float(config: Config, key: str) -> float:
    value = _value(config, key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _type_error(key, 'a number', value)
    return float(value)


def require_bool(config: Config, key: str) -> bool:
    value = _value(config, key)
    if not isinstance(value, bool):
        raise _type_error(key, 'a boolean', value)
    return value


def require_str(config: Config, key: str) -> str:
    value = _value(config, key)
    if not isinstance(value, str):
        raise _type_error(key, 'a string', value)
    return value


def optional_str(config: Config, key: str) -> str | None:
    return None if _value(config, key) is None else require_str(config, key)


def require_str_list(config: Config, key: str) -> list[str]:
    value = _value(config, key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise _type_error(key, 'a list of strings', value)
    return [str(item) for item in value]
