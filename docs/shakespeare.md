# Shakespeare Language Model

A decoder-only transformer trained from scratch on the Project Gutenberg complete works of William Shakespeare. The package mirrors the `wikipedia/` layout: byte-level BPE tokenization, packed causal language modeling, mixed-precision training, and CLI inference.

## Overview

Training downloads [Project Gutenberg ebook #100](https://www.gutenberg.org/ebooks/100) (the complete works), strips Gutenberg boilerplate, splits the text into individual plays/poems, caches a reusable local snapshot, trains a BPE tokenizer on that corpus, and packs tokens into contiguous blocks for next-token prediction.

## Model sizes

| Config | Parameters | Key dims |
|--------|------------|----------|
| `shakespeare_small` | ~5.3M (5,273,088) | `d_model=256`, `n_layers=4`, `vocab_size=8000`, `max_seq_len=256` |
| `shakespeare_medium` | ~33.7M (33,674,240) | `d_model=512`, `n_layers=8`, `vocab_size=16000`, `max_seq_len=512` |
| `shakespeare_large` | ~110.4M (110,418,432) | `d_model=768`, `n_layers=12`, `vocab_size=32000`, `max_seq_len=1024` |

Counts include weight-tied token embeddings and output projection. See the first line of each config in `shakespeare/configs/` for the canonical number.

## Folder structure

```
docs/
└── shakespeare.md         # This file

shakespeare/
├── __init__.py
├── architecture.py        # Decoder-only transformer
├── data.py                # Gutenberg download, snapshot cache, packed Dataset
├── tokenizer.py           # Byte-level BPE wrapper
├── training.py            # Trainer + CLI
├── inference.py           # Generation CLI
├── reporting.py           # Per-epoch loss charts
├── utils.py               # Device selection and path helpers
├── configs/
├── tests/
├── data/                  # Cached works snapshot (gitignored)
├── tokenizer_files/       # Trained tokenizer (gitignored)
├── weights/               # Checkpoints (gitignored)
└── reports/               # Loss charts (gitignored)
```

## Usage

### Training

From the repo root:

```bash
uv run python -m shakespeare.training shakespeare/configs/shakespeare_small.yaml
```

On first run this downloads the complete works (unless a compatible snapshot already exists under `shakespeare/data/`). Set `dataset_cache_only: True` to require the local snapshot and skip the network. Set `max_works` to an integer for a smaller smoke-test corpus.

### Inference

```bash
uv run python -m shakespeare.inference \
    --model_name shakespeare_small \
    --prompt "To be, or not to be" \
    --max_length 200 \
    --temperature 0.8 \
    --top_k 50
```

### Tests

```bash
uv run pytest shakespeare/tests
```

## Data format

`shakespeare/data/` contains:

- `shakespeare_complete.txt`: Raw downloaded Gutenberg text (written on first download)
- `shakespeare_works.jsonl`: One JSON object per work with `id`, `title`, and `text`
- `shakespeare_works.manifest.json`: Corpus URL, work count, and snapshot SHA-256

Works are split on known play/poem title lines from Gutenberg ebook #100 (44 works). Each work is tokenized, concatenated into one stream (separated by an end-of-sequence token), and cut into contiguous `max_seq_len`-length blocks.

Project Gutenberg texts are public domain in the United States; see [ebook #100](https://www.gutenberg.org/ebooks/100) for license details.

## Architecture notes

Same GPT-style stack as `wikipedia/`: native `nn.TransformerEncoder` with causal masking, pre-norm + GELU, weight-tied embeddings, mixed precision (`bf16` recommended on MPS), gradient accumulation, and self-describing checkpoints (`_best.pt` / `_latest.pt` / `_epoch_N.pt`).
