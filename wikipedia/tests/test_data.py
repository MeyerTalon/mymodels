"""tests for data loading and preprocessing."""

import pytest

from wikipedia.data import (
    WikipediaDataset,
    create_dataloader,
    load_articles_from_dir,
    sanitize_filename,
)


def test_sanitize_filename():
    assert sanitize_filename("Hello, World!") == "hello_world"
    assert sanitize_filename("  --Weird__Title--  ") == "weird_title"
    assert sanitize_filename("café & résumé") == "caf_r_sum"


def test_dataset_pads_and_shifts(dummy_tokenizer):
    dataset = WikipediaDataset(["abc"], dummy_tokenizer, max_length=8)
    input_ids, target_ids = dataset[0]
    tokens = dummy_tokenizer.encode("abc") + [0] * 5
    assert input_ids.tolist() == tokens[:-1]
    assert target_ids.tolist() == tokens[1:]


def test_dataset_truncates_long_text(dummy_tokenizer):
    dataset = WikipediaDataset(["x" * 50], dummy_tokenizer, max_length=8)
    input_ids, target_ids = dataset[0]
    assert input_ids.shape == (7,)
    assert target_ids.shape == (7,)


def test_load_articles_from_dir(tmp_path):
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta", encoding="utf-8")
    (tmp_path / "c.md").write_text("ignored", encoding="utf-8")
    texts = load_articles_from_dir(str(tmp_path))
    assert sorted(texts) == ["alpha", "beta"]


def test_create_dataloader_with_local_articles(tmp_path, dummy_tokenizer):
    for i in range(3):
        (tmp_path / f"{i}.txt").write_text("hello world " * 3, encoding="utf-8")
    dataloader = create_dataloader(
        data_dir=str(tmp_path),
        tokenizer=dummy_tokenizer,
        n_articles=2,
        batch_size=2,
        max_length=8,
        use_local_articles=True,
    )
    batch_inputs, batch_targets = next(iter(dataloader))
    assert batch_inputs.shape == (2, 7)
    assert batch_targets.shape == (2, 7)


def test_create_dataloader_local_without_articles_raises(tmp_path, dummy_tokenizer):
    with pytest.raises(ValueError):
        create_dataloader(
            data_dir=str(tmp_path),
            tokenizer=dummy_tokenizer,
            n_articles=2,
            use_local_articles=True,
        )
