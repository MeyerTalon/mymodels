# Multilingual translation model

An encoder-decoder transformer (native `nn.TransformerEncoder` and `nn.TransformerDecoder`, pre-norm GELU) trained from scratch on a local parallel corpus. Source and target share one byte-level BPE vocabulary and one embedding, tied to the output head. A `<2xx>` token before the source picks the target language, so one model translates into every language in `languages`.

## Configs

| Config | Parameters | Key dims |
|---|---|---|
| `translation_small` | 7,610,880 | `d_model=256`, 3 encoder and 3 decoder layers, `vocab_size=8000`, `max_seq_len=128` |

## Usage

Put parallel files in `translation/data/corpus/`, in either format (files may be mixed):

- `*.tsv`: four columns `src`, `tgt`, `src_lang`, `tgt_lang`, with an optional header row; lines starting with `#` are skipped.
- `*.jsonl`: one `{"src": ..., "tgt": ..., "src_lang": ..., "tgt_lang": ...}` object per line.

Language codes must be in the config's `languages` list (`en`, `es`, `fr`, `de`).

```bash
mise run train:translation translation/configs/translation_small.yaml
mise run infer:translation --model_name translation_small --source_lang en --target_lang es --prompt "hello world" --top_k 1
```

The first load writes `translation/data/translation_pairs.jsonl` and its manifest. Set `dataset_cache_only: true` to train from that snapshot alone. To pick up new pairs, delete the snapshot and retrain. Train and validation split at the pair level, so they never share a sentence.
