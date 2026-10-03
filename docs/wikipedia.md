# Wikipedia language model

A GPT-style decoder-only transformer (the shared `gpt/` stack) trained from scratch on a bounded sample of English Wikipedia. The package itself is only the data source in `wikipedia/data.py`.

## Configs

| Config | Parameters | Articles | Key dims |
|---|---|---|---|
| `wikipedia_small` | 5,273,088 | 200 | `d_model=256`, `n_layers=4`, `vocab_size=8000`, `max_seq_len=256` |
| `wikipedia_medium` | 33,674,240 | 10,000 | `d_model=512`, `n_layers=8`, `vocab_size=16000`, `max_seq_len=512` |
| `wikipedia_large` | 110,418,432 | 50,000 | `d_model=768`, `n_layers=12`, `vocab_size=32000`, `max_seq_len=1024` |

Each config's `expected_parameters` is the canonical count (weight-tied embeddings included).

## Usage

```bash
mise run train:wikipedia wikipedia/configs/wikipedia_small.yaml
mise run infer:wikipedia --model_name wikipedia_small --prompt "The history of" --temperature 0.8
```

## Data

The source is the cleaned English `20231101.en` subset of [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia), pinned to the revision in the config. The full subset is about 6.4 million articles (20 GB prepared), so training streams it and keeps only `number_of_articles` non-empty articles.

`wikipedia/data/` holds `wikipedia_articles.jsonl` (`id`, `url`, `title`, `text` per article) and `wikipedia_articles.manifest.json` (dataset identity, revision, sampling settings, record count, SHA-256). A larger compatible snapshot serves a smaller request from its prefix. Changing the source, revision, seed, or shuffle buffer builds a new snapshot. The streaming shuffle is deterministic for a pinned revision and seed, but it is an approximate buffer shuffle, not a uniform permutation.

Every article is tokenized, joined into one stream with an end-of-sequence token after each, and cut into contiguous `max_seq_len` blocks: no truncation and no padding.

## Troubleshooting

- "no compatible Wikipedia snapshot": set `dataset_cache_only: false` for one networked run, or restore a snapshot made with the same dataset settings.
- Out of memory: lower `batch_size` and raise `grad_accum_steps`, or shrink `max_seq_len` or the model.

## License

Wikimedia article text is distributed under the GNU Free Documentation License and Creative Commons Attribution-ShareAlike 3.0; see the [dataset card](https://huggingface.co/datasets/wikimedia/wikipedia).
