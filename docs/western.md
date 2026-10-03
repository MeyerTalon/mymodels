# Western language model

The shared `gpt/` decoder-only transformer trained from scratch on a local corpus of western novels. Nothing is downloaded. The package itself is only the data source in `western/data.py`.

## Configs

| Config | Parameters | Key dims |
|---|---|---|
| `western_small` | 5,273,088 | `d_model=256`, `n_layers=4`, `vocab_size=8000`, `max_seq_len=256` |

## Usage

Put plain-text novels in `western/data/corpus/`, one work per `*.txt` file (UTF-8, with a Latin-1 fallback). Empty files are skipped; works are ordered by filename.

```bash
mise run train:western western/configs/western_small.yaml
mise run infer:western --model_name western_small --prompt "The wind came down off the high plains"
```

The first load writes `western/data/western_works.jsonl` and its manifest. Set `dataset_cache_only: true` to train from that snapshot alone. To pick up new novels, delete the snapshot and retrain.
