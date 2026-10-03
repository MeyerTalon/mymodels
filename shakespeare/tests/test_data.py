from pathlib import Path

import pytest

from core.tests.factories import training_config, training_settings
from shakespeare import data
from shakespeare.data import (
    MIN_WORK_CHARS,
    WHOLE_CORPUS_TITLE,
    load_texts,
    parse_works,
    strip_gutenberg_boilerplate,
)

PLAY_BODY = 'ACT I. Scene one.\n' + 'To be, or not to be.\n' * (MIN_WORK_CHARS // 10)
CORPUS = (
    'Project Gutenberg header\n'
    '*** START OF THE PROJECT GUTENBERG EBOOK ***\n'
    'Contents\nTHE TEMPEST\nTHE TRAGEDY OF MACBETH\n\n'
    f'THE TEMPEST\n{PLAY_BODY}\n'
    f'THE TRAGEDY OF MACBETH\n{PLAY_BODY}\n'
    '*** END OF THE PROJECT GUTENBERG EBOOK ***\n'
    'license text'
)
CORPUS_URL = 'https://example.com/shakespeare.txt'


def test_strip_gutenberg_boilerplate() -> None:
    body = strip_gutenberg_boilerplate(CORPUS)
    assert 'header' not in body
    assert 'license' not in body


def test_parse_works_splits_on_titles_and_keeps_longest_chunk() -> None:
    works = parse_works(CORPUS)
    assert [work['title'] for work in works] == [
        'THE TEMPEST',
        'THE TRAGEDY OF MACBETH',
    ]
    assert all(len(work['text']) >= MIN_WORK_CHARS for work in works)
    assert [work['id'] for work in works] == ['0', '1']


def test_parse_works_falls_back_to_whole_text() -> None:
    works = parse_works('a short untitled text')
    assert works == [
        {'id': '0', 'title': WHOLE_CORPUS_TITLE, 'text': 'a short untitled text'}
    ]


def test_load_texts_downloads_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    downloads: list[str] = []

    def fake_download(url: str) -> str:
        downloads.append(url)
        return CORPUS

    monkeypatch.setattr(data, 'download_corpus', fake_download)
    config = training_config(tmp_path, corpus_url=CORPUS_URL, max_works=1)
    assert len(load_texts(config, training_settings(tmp_path))) == 1
    full = load_texts(config | {'max_works': None}, training_settings(tmp_path))
    assert len(full) == 2
    assert downloads == [CORPUS_URL]
