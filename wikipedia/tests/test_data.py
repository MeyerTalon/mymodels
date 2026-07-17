"""tests for data loading and preprocessing."""

import pytest

from wikipedia.data import (
    WikipediaDataset,
    build_token_stream,
    create_dataloaders,
    extract_article_text,
    gather_texts,
    load_articles_from_dir,
    sanitize_filename,
)


def test_sanitize_filename():
    assert sanitize_filename("Hello, World!") == "hello_world"
    assert sanitize_filename("  --Weird__Title--  ") == "weird_title"
    assert sanitize_filename("café & résumé") == "caf_r_sum"


def test_extract_article_text_strips_header():
    raw = (
        "Title: Foo\nURL: https://en.wikipedia.org/?curid=1\n"
        + "=" * 80
        + "\n\nActual body text here."
    )
    assert extract_article_text(raw) == "Actual body text here."


def test_extract_article_text_without_header():
    assert extract_article_text("no header here") == "no header here"


def test_packed_dataset_shifts_by_one(dummy_tokenizer):
    stream = list(range(1, 21))  # 20 tokens
    dataset = WikipediaDataset(stream, block_size=4)
    assert len(dataset) == (20 - 1) // 4
    input_ids, target_ids = dataset[0]
    assert input_ids.tolist() == [1, 2, 3, 4]
    assert target_ids.tolist() == [2, 3, 4, 5]
    # the second block continues contiguously from the first
    next_inputs, _ = dataset[1]
    assert next_inputs.tolist() == [5, 6, 7, 8]


def test_packed_dataset_too_short_raises():
    with pytest.raises(ValueError):
        WikipediaDataset([1, 2, 3], block_size=8)


def test_build_token_stream_inserts_eos(dummy_tokenizer):
    dummy_tokenizer.eos_id = 99
    stream = build_token_stream(["ab", "cd"], dummy_tokenizer)
    expected = (
        dummy_tokenizer.encode("ab")
        + [99]
        + dummy_tokenizer.encode("cd")
        + [99]
    )
    assert stream == expected


def test_load_articles_from_dir_strips_headers(tmp_path):
    (tmp_path / "a.txt").write_text(
        "Title: A\nURL: u\n" + "=" * 80 + "\n\nalpha", encoding="utf-8"
    )
    (tmp_path / "b.txt").write_text("beta", encoding="utf-8")
    (tmp_path / "c.md").write_text("ignored", encoding="utf-8")
    assert sorted(load_articles_from_dir(str(tmp_path))) == ["alpha", "beta"]


def test_gather_texts_local_without_articles_raises(tmp_path, dummy_tokenizer):
    with pytest.raises(ValueError):
        gather_texts(str(tmp_path), n_articles=2, use_local_articles=True)


def test_create_dataloaders_with_val_split(tmp_path, dummy_tokenizer):
    for i in range(3):
        (tmp_path / f"{i}.txt").write_text("hello world " * 40, encoding="utf-8")
    train_loader, val_loader = create_dataloaders(
        data_dir=str(tmp_path),
        tokenizer=dummy_tokenizer,
        n_articles=3,
        block_size=8,
        batch_size=2,
        val_fraction=0.2,
        use_local_articles=True,
    )
    inputs, targets = next(iter(train_loader))
    assert inputs.shape == (2, 8)
    assert targets.shape == (2, 8)
    assert val_loader is not None
