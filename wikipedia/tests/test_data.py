import json
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pytest

from core.snapshot import manifest_path
from core.tests.factories import training_config, training_settings
from wikipedia import data
from wikipedia.data import SNAPSHOT_NAME, WikipediaSource, load_texts, normalize_article

Row = dict[str, object]
SOURCE_CONFIG = {
    'dataset_name': 'wikimedia/wikipedia',
    'dataset_config': '20231101.en',
    'dataset_split': 'train',
    'dataset_revision': 'abc123',
    'dataset_seed': 42,
    'shuffle_buffer_size': 100,
}


@dataclass
class FakeStream:
    rows: list[Row]
    calls: list[str] = field(default_factory=list)

    def filter(self, function: Callable[[Row], bool]) -> 'FakeStream':
        self.calls.append('filter')
        return FakeStream([row for row in self.rows if function(row)], self.calls)

    def shuffle(self, *, seed: int, buffer_size: int) -> 'FakeStream':
        self.calls.append(f'shuffle:{seed}:{buffer_size}')
        return self

    def take(self, n: int) -> 'FakeStream':
        return FakeStream(self.rows[:n], self.calls)

    def __iter__(self) -> Iterator[Row]:
        return iter(self.rows)


def _config(tmp_path: Path, **overrides: object) -> dict[str, object]:
    return training_config(tmp_path, **SOURCE_CONFIG, number_of_articles=2, **overrides)


def test_source_fields_match_manifest_keys() -> None:
    assert set(asdict(WikipediaSource.from_config(SOURCE_CONFIG))) == set(SOURCE_CONFIG)


def test_normalize_article_strips_text() -> None:
    article = normalize_article({'id': 7, 'title': 'T', 'text': '  body \n'})
    assert article == {'id': '7', 'url': '', 'title': 'T', 'text': 'body'}


def test_load_texts_streams_once_then_reads_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stream = FakeStream(
        [
            {'id': 1, 'url': 'u1', 'title': 'A', 'text': 'alpha'},
            {'id': 2, 'url': 'u2', 'title': 'B', 'text': '   '},
            {'id': 3, 'url': 'u3', 'title': 'C', 'text': 'gamma'},
        ]
    )
    monkeypatch.setattr(data, 'load_dataset', lambda *args, **kwargs: stream)
    config = _config(tmp_path)
    assert load_texts(config, training_settings(tmp_path)) == ['alpha', 'gamma']
    assert stream.calls == ['filter', 'shuffle:42:100']
    manifest = json.loads(manifest_path(tmp_path / 'data', SNAPSHOT_NAME).read_text())
    assert manifest['dataset_revision'] == 'abc123'

    monkeypatch.setattr(data, 'load_dataset', None)
    cached = load_texts(config, training_settings(tmp_path, dataset_cache_only=True))
    assert cached == ['alpha', 'gamma']


def test_load_texts_rejects_bad_settings(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='shuffle_buffer_size'):
        load_texts(
            _config(tmp_path) | {'shuffle_buffer_size': 0}, training_settings(tmp_path)
        )
    with pytest.raises(ValueError, match='no compatible'):
        load_texts(
            _config(tmp_path), training_settings(tmp_path, dataset_cache_only=True)
        )
