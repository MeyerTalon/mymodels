"""tests for parallel-corpus loading and packing."""

import json
from pathlib import Path
from typing import Any, List

import pytest

from translation.data import (
    MANIFEST_FILENAME,
    SNAPSHOT_FILENAME,
    TranslationDataset,
    TranslationPair,
    create_dataloaders,
    load_translation_pairs,
)

_FIXTURE_PAIRS: List[TranslationPair] = [
    {"src": "hello world", "tgt": "hola mundo", "src_lang": "en", "tgt_lang": "es"},
    {"src": "good morning", "tgt": "bonjour", "src_lang": "en", "tgt_lang": "fr"},
    {"src": "thank you", "tgt": "danke", "src_lang": "en", "tgt_lang": "de"},
    {"src": "see you later", "tgt": "hasta luego", "src_lang": "en", "tgt_lang": "es"},
]


def _write_tsv_corpus(tmp_path: Path) -> Path:
    """writes fixture pairs as a TSV under ``tmp_path/corpus``."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    lines = ["src\ttgt\tsrc_lang\ttgt_lang"]
    for pair in _FIXTURE_PAIRS:
        lines.append(
            f"{pair['src']}\t{pair['tgt']}\t{pair['src_lang']}\t{pair['tgt_lang']}"
        )
    (corpus_dir / "pairs.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return corpus_dir


def test_dataset_prepends_lang_token(dummy_tokenizer: Any) -> None:
    dataset = TranslationDataset(_FIXTURE_PAIRS[:1], dummy_tokenizer, max_seq_len=32)
    src, tgt_input, tgt_labels = dataset[0]
    assert src[0].item() == dummy_tokenizer.lang_id("es")
    assert tgt_input[0].item() == dummy_tokenizer.bos_id
    assert tgt_labels[-1].item() == dummy_tokenizer.eos_id
    assert tgt_input.tolist()[1:] == tgt_labels.tolist()[:-1]


def test_empty_dataset_raises(dummy_tokenizer: Any) -> None:
    with pytest.raises(ValueError):
        TranslationDataset([], dummy_tokenizer)


def test_loads_tsv_corpus_and_caches_snapshot(tmp_path: Path) -> None:
    corpus_dir = _write_tsv_corpus(tmp_path)
    pairs = load_translation_pairs(str(tmp_path), corpus_dir=str(corpus_dir))
    assert len(pairs) == 4
    assert pairs[0]["tgt_lang"] == "es"
    assert (tmp_path / SNAPSHOT_FILENAME).exists()
    assert (tmp_path / MANIFEST_FILENAME).exists()


def test_loads_jsonl_corpus(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    with (corpus_dir / "pairs.jsonl").open("w", encoding="utf-8") as file:
        for pair in _FIXTURE_PAIRS:
            file.write(json.dumps(pair) + "\n")
    pairs = load_translation_pairs(str(tmp_path), corpus_dir=str(corpus_dir))
    assert len(pairs) == 4
    assert pairs[1]["tgt"] == "bonjour"


def test_missing_corpus_raises_with_path(tmp_path: Path) -> None:
    missing = tmp_path / "corpus"
    with pytest.raises(ValueError, match="no translation corpus found"):
        load_translation_pairs(str(tmp_path), corpus_dir=str(missing))


def test_snapshot_reuse_and_max_pairs(tmp_path: Path) -> None:
    corpus_dir = _write_tsv_corpus(tmp_path)
    full = load_translation_pairs(str(tmp_path), corpus_dir=str(corpus_dir))
    subset = load_translation_pairs(
        str(tmp_path),
        corpus_dir=str(corpus_dir),
        max_pairs=2,
        dataset_cache_only=True,
    )
    assert subset == full[:2]


def test_cache_only_rejects_missing_snapshot(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no compatible translation snapshot"):
        load_translation_pairs(str(tmp_path), dataset_cache_only=True)


def test_create_dataloaders_pads_and_masks(dummy_tokenizer: Any) -> None:
    train_loader, val_loader = create_dataloaders(
        _FIXTURE_PAIRS,
        dummy_tokenizer,
        max_seq_len=32,
        batch_size=2,
        val_fraction=0.25,
        shuffle=False,
        num_workers=0,
    )
    src, tgt_input, tgt_labels, src_pad, tgt_pad = next(iter(train_loader))
    assert src.dim() == 2
    assert tgt_input.shape == tgt_labels.shape
    assert src_pad.shape == src.shape
    assert tgt_pad.shape == tgt_input.shape
    assert src[0, 0].item() == dummy_tokenizer.lang_id("es")
    assert val_loader is not None
