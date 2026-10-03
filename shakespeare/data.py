import re
import urllib.error
import urllib.request
from pathlib import Path

from core.config import Config, optional_int, require_str
from core.snapshot import Record, load_or_build_snapshot
from core.training import TrainingConfig

SNAPSHOT_NAME = 'shakespeare_works'
RAW_FILENAME = 'shakespeare_complete.txt'
WORK_FIELDS = ('id', 'title', 'text')
TEXT_FIELD = 'text'
USER_AGENT = 'mymodels-shakespeare/1.0'
DOWNLOAD_TIMEOUT_S = 120
MIN_WORK_CHARS = 500
MAX_TITLE_CHARS = 120
MIN_TITLE_SPLIT_WORKS = 2
WHOLE_CORPUS_TITLE = 'Complete Works'
GUTENBERG_START = re.compile(r'\*\*\*\s*START OF .+?\*\*\*', flags=re.IGNORECASE)
GUTENBERG_END = re.compile(r'\*\*\*\s*END OF .+?\*\*\*', flags=re.IGNORECASE)
SECTION_BREAK = re.compile(r'\n\s*\n\s*\n+')

WORK_TITLES = frozenset(
    {
        'THE SONNETS',
        'ALL’S WELL THAT ENDS WELL',
        'THE TRAGEDY OF ANTONY AND CLEOPATRA',
        'AS YOU LIKE IT',
        'THE COMEDY OF ERRORS',
        'THE TRAGEDY OF CORIOLANUS',
        'CYMBELINE',
        'THE TRAGEDY OF HAMLET, PRINCE OF DENMARK',
        'THE FIRST PART OF KING HENRY THE FOURTH',
        'THE SECOND PART OF KING HENRY THE FOURTH',
        'THE LIFE OF KING HENRY THE FIFTH',
        'THE FIRST PART OF HENRY THE SIXTH',
        'THE SECOND PART OF KING HENRY THE SIXTH',
        'THE THIRD PART OF KING HENRY THE SIXTH',
        'KING HENRY THE EIGHTH',
        'THE LIFE AND DEATH OF KING JOHN',
        'THE TRAGEDY OF JULIUS CAESAR',
        'THE TRAGEDY OF KING LEAR',
        'LOVE’S LABOUR’S LOST',
        'THE TRAGEDY OF MACBETH',
        'MEASURE FOR MEASURE',
        'THE MERCHANT OF VENICE',
        'THE MERRY WIVES OF WINDSOR',
        'A MIDSUMMER NIGHT’S DREAM',
        'MUCH ADO ABOUT NOTHING',
        'THE TRAGEDY OF OTHELLO, THE MOOR OF VENICE',
        'PERICLES, PRINCE OF TYRE',
        'KING RICHARD THE SECOND',
        'KING RICHARD THE THIRD',
        'THE TRAGEDY OF ROMEO AND JULIET',
        'THE TAMING OF THE SHREW',
        'THE TEMPEST',
        'THE LIFE OF TIMON OF ATHENS',
        'THE TRAGEDY OF TITUS ANDRONICUS',
        'TROILUS AND CRESSIDA',
        'TWELFTH NIGHT; OR, WHAT YOU WILL',
        'THE TWO GENTLEMEN OF VERONA',
        'THE TWO NOBLE KINSMEN',
        'THE WINTER’S TALE',
        'A LOVER’S COMPLAINT',
        'THE PASSIONATE PILGRIM',
        'THE PHOENIX AND THE TURTLE',
        'THE RAPE OF LUCRECE',
        'VENUS AND ADONIS',
    }
)


def load_texts(config: Config, settings: TrainingConfig) -> list[str]:
    corpus_url = require_str(config, 'corpus_url')
    works = load_or_build_snapshot(
        settings.data_dir,
        SNAPSHOT_NAME,
        description='Shakespeare',
        metadata={'corpus_url': corpus_url},
        fields=WORK_FIELDS,
        record_limit=optional_int(config, 'max_works'),
        cache_only=settings.dataset_cache_only,
        build=lambda: download_works(corpus_url, settings.data_dir),
    )
    return [work[TEXT_FIELD] for work in works]


def download_works(corpus_url: str, data_dir: Path) -> list[Record]:
    print(f'Downloading Shakespeare complete works from {corpus_url}...')
    raw_text = download_corpus(corpus_url)
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / RAW_FILENAME).write_text(raw_text, encoding='utf-8')
    works = parse_works(raw_text)
    if not works:
        raise ValueError('no Shakespeare works found after parsing the corpus')
    return works


def download_corpus(url: str) -> str:
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_S) as response:
            payload: bytes = response.read()
    except (urllib.error.URLError, TimeoutError) as error:
        raise ValueError(
            f'failed to download Shakespeare corpus from {url}: {error}'
        ) from error
    try:
        return payload.decode('utf-8')
    except UnicodeDecodeError:
        return payload.decode('latin-1')


def strip_gutenberg_boilerplate(text: str) -> str:
    start = GUTENBERG_START.search(text)
    if start is not None:
        text = text[start.end() :]
    end = GUTENBERG_END.search(text)
    if end is not None:
        text = text[: end.start()]
    return text.strip()


def parse_works(raw_text: str) -> list[Record]:
    """Splits on known title lines, falling back to blank-line runs, then to the whole text as one work."""
    body = strip_gutenberg_boilerplate(raw_text)
    works = split_on_titles(body)
    if len(works) < MIN_TITLE_SPLIT_WORKS:
        works = split_on_blank_lines(body)
    if not works and body:
        works = [_work(0, WHOLE_CORPUS_TITLE, body)]
    return works


def split_on_titles(body: str) -> list[Record]:
    """A title that appears twice (contents listing and body) keeps its longest chunk."""
    lines = body.splitlines()
    starts = [
        (index, line.strip())
        for index, line in enumerate(lines)
        if line.strip() in WORK_TITLES
    ]
    if not starts:
        return []
    ends = [start for start, _ in starts[1:]] + [len(lines)]
    longest_by_title: dict[str, str] = {}
    for (start, title), end in zip(starts, ends, strict=True):
        text = '\n'.join(lines[start:end]).strip()
        if len(text) >= MIN_WORK_CHARS and len(text) > len(
            longest_by_title.get(title, '')
        ):
            longest_by_title[title] = text
    ordered_titles = list(
        dict.fromkeys(title for _, title in starts if title in longest_by_title)
    )
    return [
        _work(index, title, longest_by_title[title])
        for index, title in enumerate(ordered_titles)
    ]


def split_on_blank_lines(body: str) -> list[Record]:
    chunks = [chunk.strip() for chunk in SECTION_BREAK.split(body)]
    return [
        _work(index, infer_title(text), text)
        for index, text in enumerate(
            chunk for chunk in chunks if len(chunk) >= MIN_WORK_CHARS
        )
    ]


def infer_title(text: str) -> str:
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), '')
    return first_line[:MAX_TITLE_CHARS]


def _work(index: int, title: str, text: str) -> Record:
    return {'id': str(index), 'title': title, 'text': text}
