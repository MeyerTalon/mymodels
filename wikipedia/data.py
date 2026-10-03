from collections.abc import Mapping
from dataclasses import asdict, dataclass

from datasets import load_dataset

from core.config import (
    Config,
    optional_str,
    require_int,
    require_str,
)
from core.snapshot import Record, load_or_build_snapshot
from core.training import TrainingConfig

SNAPSHOT_NAME = 'wikipedia_articles'
ARTICLE_FIELDS = ('id', 'url', 'title', 'text')
TEXT_FIELD = 'text'


@dataclass(frozen=True)
class WikipediaSource:
    """Field names are the snapshot manifest keys; changing one invalidates existing snapshots."""

    dataset_name: str
    dataset_config: str
    dataset_split: str
    dataset_revision: str | None
    dataset_seed: int
    shuffle_buffer_size: int

    @classmethod
    def from_config(cls, config: Config) -> 'WikipediaSource':
        source = cls(
            dataset_name=require_str(config, 'dataset_name'),
            dataset_config=require_str(config, 'dataset_config'),
            dataset_split=require_str(config, 'dataset_split'),
            dataset_revision=optional_str(config, 'dataset_revision'),
            dataset_seed=require_int(config, 'dataset_seed'),
            shuffle_buffer_size=require_int(config, 'shuffle_buffer_size'),
        )
        if source.shuffle_buffer_size <= 0:
            raise ValueError('shuffle_buffer_size must be positive')
        return source


def load_texts(config: Config, settings: TrainingConfig) -> list[str]:
    source = WikipediaSource.from_config(config)
    article_count = require_int(config, 'number_of_articles')
    articles = load_or_build_snapshot(
        settings.data_dir,
        SNAPSHOT_NAME,
        description='Wikipedia',
        metadata=asdict(source),
        fields=ARTICLE_FIELDS,
        record_limit=article_count,
        cache_only=settings.dataset_cache_only,
        build=lambda: stream_articles(source, article_count),
    )
    return [article[TEXT_FIELD] for article in articles]


def stream_articles(source: WikipediaSource, article_count: int) -> list[Record]:
    print(
        f'Streaming {article_count} articles from '
        f'{source.dataset_name}/{source.dataset_config}:{source.dataset_split}...'
    )
    dataset = load_dataset(
        source.dataset_name,
        source.dataset_config,
        split=source.dataset_split,
        revision=source.dataset_revision,
        streaming=True,
    )
    shuffled = dataset.filter(has_text).shuffle(
        seed=source.dataset_seed, buffer_size=source.shuffle_buffer_size
    )
    return [normalize_article(row) for row in shuffled.take(article_count)]


def has_text(row: Mapping[str, object]) -> bool:
    return bool(str(row.get(TEXT_FIELD, '')).strip())


def normalize_article(row: Mapping[str, object]) -> Record:
    article = {field: str(row.get(field, '')) for field in ARTICLE_FIELDS}
    article[TEXT_FIELD] = article[TEXT_FIELD].strip()
    return article
