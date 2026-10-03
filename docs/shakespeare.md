# Shakespeare language model

The shared `gpt/` decoder-only transformer trained from scratch on the Project Gutenberg complete works of Shakespeare. The package is the data source in `shakespeare/data.py`, plus an optional activation view in `shakespeare/visualization.py`.

## Configs

| Config | Parameters | Key dims |
|---|---|---|
| `shakespeare_small` | 5,273,088 | `d_model=256`, `n_layers=4`, `vocab_size=8000`, `max_seq_len=256` |
| `shakespeare_medium` | 33,674,240 | `d_model=512`, `n_layers=8`, `vocab_size=16000`, `max_seq_len=512` |
| `shakespeare_large` | 110,418,432 | `d_model=768`, `n_layers=12`, `vocab_size=32000`, `max_seq_len=1024` |

The corpus is far smaller than a Wikipedia sample, so configs train for more epochs.

## Usage

```bash
mise run train:shakespeare shakespeare/configs/shakespeare_small.yaml
mise run infer:shakespeare --model_name shakespeare_small --prompt "To be, or not to be" --max_length 200 --temperature 0.8
mise run infer:shakespeare --model_name shakespeare_small --prompt "To be, or not to be" --show_activations
```

`--show_activations` opens a live activation window for that run. Omit it and inference does not attach hooks.

Set `max_works` to an integer for a smoke-test corpus, and `dataset_cache_only: true` to skip the network.

## Data

Training downloads [ebook #100](https://www.gutenberg.org/ebooks/100), strips the Gutenberg header and footer, and splits the text on the 44 known play and poem title lines. When a title appears twice (contents listing and body), the longest chunk wins. `shakespeare/data/` holds the raw `shakespeare_complete.txt`, `shakespeare_works.jsonl` (`id`, `title`, `text` per work), and its manifest (corpus URL, record count, SHA-256). The full corpus is always cached, even when `max_works` limits a run.

Project Gutenberg texts are public domain in the United States; see the ebook page for license details.
