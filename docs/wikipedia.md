# Wikipedia Sentence Completion Model

A decoder-only transformer model trained on Wikipedia articles for sentence completion tasks. This implementation uses PyTorch's native transformer modules and includes a complete pipeline for data downloading, training, and inference.

## Overview

This project implements a GPT-style decoder-only transformer that learns to complete sentences based on patterns learned from Wikipedia articles. The model uses a byte-level BPE tokenizer and is designed to be lightweight and easy to train on consumer hardware.

## Model sizes

| Config | Parameters | Key dims |
|--------|------------|----------|
| `wikipedia_small` | ~5.3M (5,273,088) | `d_model=256`, `n_layers=4`, `vocab_size=8000`, `max_seq_len=256` |
| `wikipedia_medium` | ~33.7M (33,674,240) | `d_model=512`, `n_layers=8`, `vocab_size=16000`, `max_seq_len=512` |

Counts include weight-tied token embeddings and output projection. See the first line of each config in `wikipedia/configs/` for the canonical number.

## Folder structure

This README describes the `wikipedia/` model package. Per-model docs live in the repo-root `docs/` directory (`<pkg>.md`).

```
docs/
└── wikipedia.md           # This file

wikipedia/
├── __init__.py            # Marks this directory as a Python package
├── architecture.py        # Model architecture definition
├── data.py                # Data downloading and preprocessing
├── tokenizer.py           # Byte-level BPE tokenizer wrapper
├── training.py            # Training script and Trainer class
├── inference.py           # Inference script for text generation
├── reporting.py           # Per-epoch train/val loss chart generation
├── utils.py               # Shared device selection and path resolution
├── configs/               # Configuration YAML files
├── tests/                 # Lightweight pytest tests
├── data/                  # Downloaded Wikipedia articles (gitignored, created automatically)
├── tokenizer_files/       # Trained tokenizer vocab/merges (gitignored, created automatically)
├── weights/               # Saved model checkpoints (gitignored, created automatically)
└── reports/               # Per-run train/val loss charts (gitignored, created automatically)
```

## Files description

### `wikipedia/architecture.py`

Contains the model architecture:

- **`DecoderOnlyTransformer`**: Main model class implementing a decoder-only transformer using PyTorch's native modules

**Key Features:**
- Uses PyTorch's built-in transformer components (`nn.TransformerEncoderLayer`, `nn.TransformerEncoder`) with a causal mask, configured GPT-style: pre-layer-norm (`norm_first=True`), GELU feed-forward activations, and the fused scaled-dot-product-attention causal fast path (`is_causal=True`)
- Uses native `nn.Embedding` for both token and positional embeddings (no custom modules), with the token embedding weight-tied to the output projection
- GPT-style normal (std 0.02) weight initialization
- Causal masking for autoregressive generation (`generate` method with temperature and top-k sampling, stopping on the tokenizer's end-of-sequence id)
- Configurable model size (embedding dimension, number of heads, layers, etc.)

### `wikipedia/tokenizer.py`

Byte-level BPE tokenization:

- **`WikipediaBPETokenizer`**: Wrapper around Hugging Face's `ByteLevelBPETokenizer` exposing `encode`, `decode`, `vocab_size`, and the special-token ids (`pad_id`, `bos_id`, `eos_id`, `unk_id`). Trained on the downloaded articles the first time you train (`train_or_load`, with `vocab_size`/`min_frequency` from config) and saved to `wikipedia/tokenizer_files/`.

### `wikipedia/data.py`

Handles Wikipedia article downloading and data preprocessing:

- **`WikipediaDataset`**: PyTorch Dataset over a single packed token stream, cut into contiguous `block_size`-length blocks (`input_ids`, next-token `target_ids`) — no padding
- **`sanitize_filename()`**: Converts article titles to valid filenames (lowercase, underscores, alphanumeric only)
- **`extract_article_text()`**: Strips the metadata header from a downloaded article file, keeping only the body
- **`download_wikipedia_articles()`**: Downloads n randomly selected Wikipedia articles and saves them to disk
- **`load_articles_from_dir()`**: Loads and cleans article bodies from saved files
- **`gather_texts()`**: Collects cleaned article texts, downloading them or sampling local ones
- **`build_token_stream()`**: Tokenizes and concatenates texts into one stream, separated by the end-of-sequence id
- **`create_dataloaders()`**: Builds packed train (and optional validation) DataLoaders, splitting the token stream at the token level

**Features:**
- Automatically downloads random Wikipedia articles (or reuses local ones offline)
- Strips metadata headers so only article text is trained on
- **Packs the whole corpus into contiguous fixed-length blocks** so every token contributes a training signal (no wasted padding), the standard efficient setup for causal language modeling
- Inserts an end-of-sequence separator between articles so the model learns document boundaries

### `wikipedia/training.py`

Training pipeline and Trainer class:

- **`Trainer`**: Main training class that handles:
  - Configuration loading from YAML
  - Data setup and downloading
  - Model initialization
  - Mixed-precision training loop (fp16 autocast on MPS/CUDA) with gradient accumulation and progress tracking
  - Validation on a held-out token region and perplexity reporting
  - Checkpoint saving (best, latest, and epoch-specific), best selected on validation loss when available
  - Learning rate scheduling (per-epoch linear warmup then cosine decay)

**Usage (from repo root):**
```bash
uv run python -m wikipedia.training wikipedia/configs/wikipedia_small.yaml
```

**Features:**
- Automatic data downloading if directory is empty (or offline reuse of local articles)
- Mixed precision (fp16 autocast) and gradient accumulation, tuned for Apple silicon on ~24GB of unified memory
- `AdamW` with decoupled weight decay applied only to weight matrices (not biases, norms, or embeddings)
- Gradient clipping for training stability
- Linear-warmup then cosine-decay learning rate schedule
- Saves multiple checkpoint types (best model, latest, per-epoch)
- Progress bars with loss and learning-rate tracking
- Writes a per-run train/val loss chart after every epoch via `TrainingReporter` (see `wikipedia/reporting.py`)

### `wikipedia/inference.py`

Command-line interface for text generation:

- **`load_model()`**: Loads a trained model from checkpoint
- **`generate_text()`**: Generates text continuation from a prompt
- **`main()`**: CLI entry point

**Usage (from repo root):**
```bash
uv run python -m wikipedia.inference --model_name wikipedia_small --prompt "The capital of Mexico is"
```

**Options:**
- `--model_name`: Name of the model to load (required)
- `--prompt`: Beginning of sentence to complete (required)
- `--max_length`: Maximum number of tokens to generate (default: 100)
- `--temperature`: Sampling temperature, higher = more random (default: 1.0)
- `--top_k`: Top-k sampling, 0 to disable (default: 50)
- `--weights_dir`: Directory containing model weights (default: wikipedia/weights)

### `wikipedia/utils.py`

Shared helpers used by the training and inference entry points:

- **`select_device()`**: Picks the best available device (MPS, then CUDA, then CPU)
- **`select_autocast_dtype()`**: Resolves the autocast dtype for a device and precision setting (returns `None` on CPU or for `fp32`)
- **`resolve_repo_path()`**: Resolves repo-root-relative paths (e.g. `wikipedia/weights`) to absolute paths

### `wikipedia/reporting.py`

Per-run training report generation:

- **`TrainingReporter`**: Accumulates per-epoch train and validation loss and renders them as a single line chart. It is re-rendered after every epoch, so a partially trained (or interrupted) run always has an up-to-date chart.

**Key Features:**
- Train and validation loss are drawn on the **same axes** for direct comparison (train as blue circles, validation as red squares)
- One integer x-tick and vertical gridline per epoch, so it is clear where each epoch starts and ends
- The best (lowest) validation-loss epoch is annotated on the chart
- Uses the headless `Agg` backend, so it works during background/headless training with no display
- Each run writes to its own timestamped directory: `<reports_dir>/<model_name>_<date>_<time>/report.png`

### `wikipedia/tests/`

Lightweight `pytest` tests for the critical functions (tiny model dims, no network, no real weights). Run from the repo root:

```bash
uv run pytest wikipedia/tests
```

### `wikipedia/configs/wikipedia_small.yaml`, `wikipedia/configs/wikipedia_medium.yaml`

Configuration files defining model and training parameters:

**Tokenizer:**
- `vocab_size`: Target BPE vocabulary size (default: 8000)
- `min_frequency`: Minimum token frequency to enter the vocab (default: 2)

**Model Architecture:**
- `d_model`: Embedding dimension (default: 512)
- `n_heads`: Number of attention heads (default: 8)
- `n_layers`: Number of transformer layers (default: 6)
- `d_ff`: Feed-forward dimension (default: 2048)
- `max_seq_len`: Maximum sequence length, also the packed block size (default: 512)
- `dropout`: Dropout rate (default: 0.1)

**Training:**
- `num_epochs`: Number of training epochs (default: 10)
- `batch_size`: Batch size (default: 16)
- `grad_accum_steps`: Micro-batches accumulated per optimizer step; effective batch = `batch_size * grad_accum_steps` (default: 1)
- `learning_rate`: Peak learning rate after warmup (default: 0.0003)
- `min_lr`: Floor learning rate the cosine schedule decays to (default: 0.00003)
- `warmup_epochs`: Linear-warmup epochs before cosine decay (default: 0)
- `weight_decay`: Decoupled weight decay for weight matrices (default: 0.1)
- `max_grad_norm`: Gradient clipping threshold (default: 1.0)
- `precision`: `fp16`, `bf16`, or `fp32`; autocast is used on MPS/CUDA and disabled on CPU (default: `fp16` on accelerators, `fp32` on CPU)
- `val_fraction`: Fraction of the token stream held out for validation, 0 to disable (default: 0.0)
- `save_every`: Save checkpoint every N epochs (default: 1)
- `log_interval`: Progress-bar update interval in steps (default: 50)

**Data:**
- `number_of_articles`: Number of Wikipedia articles to download or sample (default: 5)
- `use_local_articles`: When True, reuse already-downloaded articles instead of downloading (no network needed)
- `data_dir`: Directory to save/load articles (default: wikipedia/data)
- `num_workers`: Number of DataLoader workers (default: 0)

**Paths:**
- `model_name`: Name for saved checkpoints
- `weights_dir`: Directory to save model weights (default: wikipedia/weights)
- `reports_dir`: Base directory for per-run loss charts, written to `<reports_dir>/<model_name>_<date>_<time>/report.png` (default: wikipedia/reports)

## Setup

### Installation

1. Install uv, then create the project environment from the repo root:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

2. Run all commands from the project root. `uv run` automatically uses and
   synchronizes the project `.venv`; activation is not required.

## Usage

### Training a Model

1. Create or modify a configuration file in `wikipedia/configs/`:
```yaml
model_name: my_model
number_of_articles: 10
d_model: 256
n_heads: 4
n_layers: 4
num_epochs: 20
batch_size: 16
```

2. Run training (from repo root):
```bash
uv run python -m wikipedia.training wikipedia/configs/wikipedia_small.yaml
```

The script will:
- Download Wikipedia articles (or reuse local ones when `use_local_articles` is True)
- Train or load the BPE tokenizer in `wikipedia/tokenizer_files/`
- Create the model according to config
- Train for the specified number of epochs
- Save checkpoints to `wikipedia/weights/`
- Write a train/val loss chart to `wikipedia/reports/{model_name}_{date}_{time}/report.png`, refreshed after every epoch

**Checkpoint Files:**
- `{model_name}_best.pt`: Best model (lowest loss)
- `{model_name}_latest.pt`: Most recent checkpoint
- `{model_name}_epoch_{N}.pt`: Checkpoint at epoch N

**Report Files:**
- `{model_name}_{date}_{time}/report.png`: Train and validation loss on a single chart, one point per epoch

### Running Inference

After training, generate text completions:

```bash
uv run python -m wikipedia.inference \
    --model_name wikipedia_small \
    --prompt "The history of artificial intelligence" \
    --max_length 200 \
    --temperature 0.8 \
    --top_k 50
```

**Example Output:**
```
Prompt: The history of artificial intelligence
Generating...

Generated text:
The history of artificial intelligence began in the 1950s when researchers...
```

### Customizing Generation

- **Temperature** (0.1 - 2.0): Lower = more deterministic, higher = more creative
- **Top-k** (1-100): Limits sampling to top k most likely tokens. Set to 0 to disable
- **Max length**: Maximum number of tokens to generate

## Model architecture details

The model uses a standard GPT-style decoder-only transformer architecture:

1. **Token Embedding**: BPE token embeddings using `nn.Embedding`
2. **Positional Encoding**: Learned positional embeddings using native `nn.Embedding`
3. **Transformer Stack**: Pre-norm stack of self-attention layers with:
   - Multi-head self-attention (causal masking, fused SDPA fast path)
   - GELU feed-forward networks
   - Layer normalization applied before each sub-layer (`norm_first=True`)
   - Residual connections
4. **Output Head**: Linear projection to vocabulary size, weight-tied to the token embedding

The implementation leverages PyTorch's native `nn.TransformerEncoder` with a causal attention mask (equivalent to a decoder-only/GPT-style stack) for efficiency and maintainability.

## Data format

Wikipedia articles are downloaded and saved as plain text files in `wikipedia/data/`:
- Filenames are sanitized article titles (lowercase, underscores, alphanumeric only)
- Each file starts with a short metadata header (title and URL) followed by the full article text
- For training, the header is stripped and every article is tokenized, concatenated into one stream (separated by an end-of-sequence token), and cut into contiguous `max_seq_len`-length blocks — no truncation to a single window and no padding

## Tips and best practices

1. **Start Small**: Begin with a small model (e.g., `d_model=256`, `n_layers=4`) and few articles to test the pipeline
2. **Monitor Training**: Watch the loss decrease - if it plateaus, try adjusting learning rate or model size
3. **Data Quality**: More articles generally lead to better results, but require more training time
4. **Memory**: Larger models and longer sequences require more memory. On Apple silicon, prefer `precision: bf16` (fp16 has no gradient scaler on MPS and can diverge) and raise the effective batch with `grad_accum_steps` rather than a larger `batch_size` if you hit the unified-memory ceiling
5. **Checkpoints**: The training script saves multiple checkpoint types - use `_best.pt` for inference

## Troubleshooting

**Issue**: "No text files found in data_dir"
- **Solution**: The data directory is empty. The training script should auto-download, but you can manually trigger it by deleting the data directory.

**Issue**: "Model checkpoint not found"
- **Solution**: Ensure the model name matches the checkpoint filename. Check `wikipedia/weights/` for available checkpoints.

**Issue**: Out of memory errors
- **Solution**: Reduce `batch_size`, `max_seq_len`, or model size (`d_model`, `n_layers`).

**Issue**: Poor generation quality
- **Solution**: Train for more epochs, use more articles, or increase model size. Also try adjusting temperature during inference.

## License

See the LICENSE file in the project root.

## Contributing

This is a personal project, but suggestions and improvements are welcome!
