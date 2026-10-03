import json
from pathlib import Path

import pytest
import torch

from core.tests.factories import training_config, training_settings
from translation.data import collate_pairs, load_pairs, make_pair, read_corpus

PAIR = {'src': 'hello', 'tgt': 'hola', 'src_lang': 'en', 'tgt_lang': 'es'}


def test_read_corpus_parses_tsv_and_jsonl(tmp_path: Path) -> None:
    (tmp_path / 'a.tsv').write_text(
        'src\ttgt\tsrc_lang\ttgt_lang\n'
        '# a comment\n'
        'hello\thola\tEN\tES\n'
        'too\tfew\n'
        '\tempty\ten\tes\n',
        encoding='utf-8',
    )
    (tmp_path / 'b.jsonl').write_text(
        json.dumps({'src': 'cat', 'tgt': 'chat', 'src_lang': 'en', 'tgt_lang': 'fr'})
        + '\n\n',
        encoding='utf-8',
    )
    assert read_corpus(tmp_path) == [
        PAIR,
        {'src': 'cat', 'tgt': 'chat', 'src_lang': 'en', 'tgt_lang': 'fr'},
    ]


def test_read_corpus_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='no translation corpus'):
        read_corpus(tmp_path)


def test_make_pair_rejects_empty_text() -> None:
    assert make_pair({'src': ' ', 'tgt': 'hola'}) is None


def test_collate_pads_and_masks() -> None:
    batch = collate_pairs(
        [
            (torch.tensor([4, 7]), torch.tensor([1, 8]), torch.tensor([8, 2])),
            (torch.tensor([5]), torch.tensor([1, 8, 9]), torch.tensor([8, 9, 2])),
        ],
        pad_id=0,
    )
    assert batch.src.tolist() == [[4, 7], [5, 0]]
    assert batch.src_pad_mask.tolist() == [[False, False], [False, True]]
    assert batch.tgt_input.shape == (2, 3)
    assert batch.tgt_pad_mask[0].tolist() == [False, False, True]


def test_load_pairs_caches_snapshot(tmp_path: Path) -> None:
    corpus_dir = tmp_path / 'corpus'
    corpus_dir.mkdir()
    (corpus_dir / 'pairs.tsv').write_text('hello\thola\ten\tes\n', encoding='utf-8')
    config = training_config(tmp_path, corpus_dir=str(corpus_dir), max_pairs=None)
    assert load_pairs(config, training_settings(tmp_path)) == [PAIR]
    (corpus_dir / 'pairs.tsv').unlink()
    assert load_pairs(config, training_settings(tmp_path, dataset_cache_only=True)) == [
        PAIR
    ]
