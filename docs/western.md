# Western Language Model

A decoder-only transformer trained from scratch on a local corpus of western novels. The package mirrors the `shakespeare/` layout: byte-level BPE tokenization, packed causal language modeling, mixed-precision training, and CLI inference. Training data is not downloaded — drop novels on disk before training.

## Overview

Put plain-text novels in `western/data/corpus/` (one `.txt` file per work). Training reads those files into a reusable local snapshot, trains a BPE tokenizer on the corpus, and packs tokens into contiguous blocks for next-token prediction.

If the corpus directory is empty or missing, training fails with a message pointing at `western/data/corpus/`.

## Model sizes

| Config | Parameters | Key dims |
|--------|------------|----------|
| `western_small` | ~5.3M (5,273,088) | `d_model=256`, `n_layers=4`, `vocab_size=8000`, `max_seq_len=256` |

Counts include weight-tied token embeddings and output projection. See the first line of `western/configs/western_small.yaml` for the canonical number.

## Folder structure

```
docs/
└── western.md             # This file

western/
├── __init__.py
├── architecture.py        # Decoder-only transformer
├── data.py                # Local corpus + snapshot cache, packed Dataset
├── tokenizer.py           # Byte-level BPE wrapper
├── training.py            # Trainer + CLI
├── inference.py           # Generation CLI
├── reporting.py           # Per-epoch loss charts
├── utils.py               # Device selection and path helpers
├── configs/
├── tests/
├── data/                  # Corpus + cached snapshot (gitignored)
│   └── corpus/            # Drop *.txt novels here
├── tokenizer_files/       # Trained tokenizer (gitignored)
├── weights/               # Checkpoints (gitignored)
└── reports/               # Loss charts (gitignored)
```

## Usage

### Prepare data

Place novels as UTF-8 (or Latin-1) `.txt` files:

```
western/data/corpus/lonesome_dove.txt
western/data/corpus/true_grit.txt
```

Empty files are skipped. After the first successful load, a JSONL snapshot and manifest are written under `western/data/` so later runs can reuse them. Set `dataset_cache_only: True` to require that snapshot (and skip re-reading the corpus directory). To pick up new novels, delete the snapshot and re-run training.

### Training

From the repo root:

```bash
uv run python -m western.training western/configs/western_small.yaml
```

Set `max_works` to an integer for a smaller smoke-test corpus.

### Inference

```bash
uv run python -m western.inference \
    --model_name western_small \
    --prompt "The wind came down off the high plains" \
    --max_length 200 \
    --temperature 0.8 \
    --top_k 50
```

### Tests

```bash
uv run pytest western/tests
```

Tests use tiny fixture novels in a temp directory — they never need the real corpus.

## Data format

`western/data/` contains:

- `corpus/*.txt`: Source novels you provide (one work per file)
- `western_works.jsonl`: One JSON object per work with `id`, `title`, and `text` (written on first successful load)
- `western_works.manifest.json`: Corpus path, work count, and snapshot SHA-256

Each work is tokenized, concatenated into one stream (separated by an end-of-sequence token), and cut into contiguous `max_seq_len`-length blocks.

## Architecture notes

Same GPT-style stack as `wikipedia/` and `shakespeare/`: native `nn.TransformerEncoder` with causal masking, pre-norm + GELU, weight-tied embeddings, mixed precision (`bf16` recommended on MPS), gradient accumulation, and self-describing checkpoints (`_best.pt` / `_latest.pt` / `_epoch_N.pt`). Architecture is duplicated into this package (the existing packages do not share a library).
