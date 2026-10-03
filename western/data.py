from pathlib import Path

from core.config import Config, optional_int, require_str
from core.paths import resolve_repo_path
from core.snapshot import Record, load_or_build_snapshot
from core.training import TrainingConfig

SNAPSHOT_NAME = 'western_works'
WORK_FIELDS = ('id', 'title', 'text')
TEXT_FIELD = 'text'
LOCAL_CORPUS_SOURCE = 'local_corpus'
NOVEL_GLOB = '*.txt'


def load_texts(config: Config, settings: TrainingConfig) -> list[str]:
    corpus_dir = resolve_repo_path(require_str(config, 'corpus_dir'))
    works = load_or_build_snapshot(
        settings.data_dir,
        SNAPSHOT_NAME,
        description='western',
        metadata={'source': LOCAL_CORPUS_SOURCE, 'corpus_dir': str(corpus_dir)},
        fields=WORK_FIELDS,
        record_limit=optional_int(config, 'max_works'),
        cache_only=settings.dataset_cache_only,
        build=lambda: read_corpus(corpus_dir),
    )
    return [work[TEXT_FIELD] for work in works]


def read_corpus(corpus_dir: Path) -> list[Record]:
    """One work per non-empty `*.txt` file, in filename order."""
    paths = sorted(corpus_dir.glob(NOVEL_GLOB)) if corpus_dir.is_dir() else []
    titled_texts = [
        (path.stem, text)
        for path in paths
        if path.is_file() and (text := read_text(path))
    ]
    works = [
        {'id': str(index), 'title': title, 'text': text}
        for index, (title, text) in enumerate(titled_texts)
    ]
    if not works:
        raise ValueError(
            f'no western corpus found. place plain-text novels ({NOVEL_GLOB}) in '
            f'{corpus_dir} (one work per file) and re-run training'
        )
    return works


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding='utf-8').strip()
    except UnicodeDecodeError:
        return path.read_text(encoding='latin-1').strip()
