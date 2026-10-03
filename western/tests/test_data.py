from pathlib import Path

import pytest

from core.tests.factories import training_config, training_settings
from western.data import load_texts, read_corpus


def test_read_corpus_skips_empty_files_and_sorts(tmp_path: Path) -> None:
    (tmp_path / 'b_novel.txt').write_text('second novel\n', encoding='utf-8')
    (tmp_path / 'a_novel.txt').write_text('  first novel ', encoding='utf-8')
    (tmp_path / 'empty.txt').write_text('   ', encoding='utf-8')
    (tmp_path / 'notes.md').write_text('ignored', encoding='utf-8')
    works = read_corpus(tmp_path)
    assert works == [
        {'id': '0', 'title': 'a_novel', 'text': 'first novel'},
        {'id': '1', 'title': 'b_novel', 'text': 'second novel'},
    ]


def test_read_corpus_falls_back_to_latin1(tmp_path: Path) -> None:
    (tmp_path / 'novel.txt').write_bytes('caf\xe9'.encode('latin-1'))
    assert read_corpus(tmp_path)[0]['text'] == 'café'


def test_read_corpus_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='no western corpus'):
        read_corpus(tmp_path / 'missing')


def test_load_texts_snapshot_survives_corpus_removal(tmp_path: Path) -> None:
    corpus_dir = tmp_path / 'corpus'
    corpus_dir.mkdir()
    novel = corpus_dir / 'novel.txt'
    novel.write_text('a lonesome dove', encoding='utf-8')
    config = training_config(tmp_path, corpus_dir=str(corpus_dir), max_works=None)
    assert load_texts(config, training_settings(tmp_path)) == ['a lonesome dove']
    novel.unlink()
    cached = load_texts(config, training_settings(tmp_path, dataset_cache_only=True))
    assert cached == ['a lonesome dove']
