# Multilingual Seq2Seq Translator

An encoder-decoder transformer for translating among multiple languages (English, Spanish, French, and German by default). The package mirrors the `wikipedia/` layout: config-driven Trainer, byte-level BPE, mixed-precision training, self-describing checkpoints, and CLI inference. Training data is not downloaded — drop a local parallel corpus on disk before training.

## Overview

One model handles several target languages via a source-side control token (`<2es>`, `<2fr>`, …). Source and target share a joint BPE vocabulary; the decoder output projection is weight-tied to that embedding.

If the corpus directory is empty or missing, training fails with a message pointing at `translation/data/corpus/` and describing the file format.

## Model sizes

| Config | Parameters | Key dims |
|--------|------------|----------|
| `translation_small` | ~7.6M (7,610,880) | `d_model=256`, 3 encoder + 3 decoder layers, `vocab_size=8000`, `max_seq_len=128` |

Counts include weight-tied shared embeddings and output projection. See the first line of `translation/configs/translation_small.yaml` for the canonical number.

## Folder structure

```
docs/
└── translation.md         # This file

translation/
├── __init__.py
├── architecture.py        # Encoder-decoder transformer
├── data.py                # Local parallel corpus + padded Dataset
├── tokenizer.py           # Joint BPE + <2xx> language tokens
├── training.py            # Trainer + CLI
├── inference.py           # Translation CLI
├── reporting.py           # Per-epoch loss charts
├── utils.py               # Device selection and path helpers
├── configs/
├── tests/
├── data/                  # Corpus + cached snapshot (gitignored)
│   └── corpus/            # Drop *.tsv / *.jsonl here
├── tokenizer_files/       # Trained tokenizer (gitignored)
├── weights/               # Checkpoints (gitignored)
└── reports/               # Loss charts (gitignored)
```

## Usage

### Prepare data

Place parallel files in `translation/data/corpus/`. Two formats are accepted (files may be mixed):

**TSV** (`*.tsv`), four columns, optional header:

```
src	tgt	src_lang	tgt_lang
hello	hola	en	es
good morning	bonjour	en	fr
```

**JSONL** (`*.jsonl`), one object per line:

```json
{"src": "hello", "tgt": "hola", "src_lang": "en", "tgt_lang": "es"}
```

Language codes should match the `languages` list in the config (`en`, `es`, `fr`, `de` by default). After the first successful load, a JSONL snapshot and manifest are written under `translation/data/`. Set `dataset_cache_only: True` to require that snapshot. To pick up new pairs, delete the snapshot and re-run training.

### Training

From the repo root:

```bash
uv run python -m translation.training translation/configs/translation_small.yaml
```

Set `max_pairs` to an integer for a smaller smoke-test corpus.

### Inference

```bash
uv run python -m translation.inference \
    --model_name translation_small \
    --source_lang en \
    --target_lang es \
    --prompt "hello world" \
    --max_length 64 \
    --temperature 0.8 \
    --top_k 50
```

`--source_lang` is validated against the tokenizer; steering uses the `<2{target_lang}>` prefix on the source (the usual multilingual MT trick). Use `--top_k 1` for greedy decoding.

### Tests

```bash
uv run pytest translation/tests
```

Tests use tiny fixture pairs in a temp directory — they never need a real corpus.

## Architecture notes

Native `nn.TransformerEncoder` + `nn.TransformerDecoder`, pre-norm (`norm_first=True`), GELU, learned positional embeddings, shared source/target embedding, weight-tied output head. Training is teacher forcing with cross-entropy on target tokens (`ignore_index` = pad). Mixed precision (`bf16` recommended on MPS), gradient accumulation, and self-describing checkpoints (`_best.pt` / `_latest.pt` / `_epoch_N.pt`) match the other packages.
