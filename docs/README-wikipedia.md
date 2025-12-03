# Wikipedia Sentence Completion Model

A decoder-only transformer model trained on Wikipedia articles for sentence completion tasks. This implementation uses PyTorch's native transformer modules and includes a complete pipeline for data downloading, training, and inference.

## Overview

This project implements a GPT-style decoder-only transformer that learns to complete sentences based on patterns learned from Wikipedia articles. The model uses character-level tokenization and is designed to be lightweight and easy to train on consumer hardware.

## Folder Structure

This README describes the contents of the `wikipedia/` package in this repo.
Similar READMEs will be added under `docs/` for other fun example models.

```
wikipedia/
├── __init__.py            # Marks this directory as a Python package
├── architecture.py        # Model architecture definition
├── data.py                # Data downloading and preprocessing
├── training.py            # Training script and Trainer class
├── inference.py           # Inference script for text generation
├── configs/               # Configuration YAML files
├── data/                  # Downloaded Wikipedia articles (created automatically)
└── weights/               # Saved model checkpoints (created automatically)
```

## Files Description

### `wikipedia/architecture.py`

Contains the model architecture:

- **`DecoderOnlyTransformer`**: Main model class implementing a decoder-only transformer using PyTorch's native modules

**Key Features:**
- Uses PyTorch's built-in transformer components (`nn.TransformerDecoderLayer`, `nn.TransformerDecoder`)
- Uses native `nn.Embedding` for both token and positional embeddings (no custom modules)
- Character-level tokenization support
- Causal masking for autoregressive generation
- Configurable model size (embedding dimension, number of heads, layers, etc.)

### `wikipedia/data.py`

Handles Wikipedia article downloading and data preprocessing:

- **`SimpleTokenizer`**: Character-level tokenizer that maps characters to indices
- **`WikipediaDataset`**: PyTorch Dataset class for loading and preprocessing Wikipedia articles
- **`sanitize_filename()`**: Converts article titles to valid filenames (lowercase, underscores, alphanumeric only)
- **`download_wikipedia_articles()`**: Downloads n randomly selected Wikipedia articles and saves them to disk
- **`load_articles_from_dir()`**: Loads article texts from saved files
- **`create_dataloader()`**: Creates a PyTorch DataLoader from downloaded articles

**Features:**
- Automatically downloads random Wikipedia articles
- Sanitizes article titles for use as filenames
- Handles disambiguation pages gracefully
- Creates trainable batches with proper padding and tokenization

### `wikipedia/training.py`

Training pipeline and Trainer class:

- **`Trainer`**: Main training class that handles:
  - Configuration loading from YAML
  - Data setup and downloading
  - Model initialization
  - Training loop with progress tracking
  - Checkpoint saving (best, latest, and epoch-specific)
  - Learning rate scheduling

**Usage (from repo root):**
```bash
python wikipedia/training.py wikipedia/configs/wikipedia_small.yaml
```

**Features:**
- Automatic data downloading if directory is empty
- Gradient clipping for training stability
- Cosine annealing learning rate schedule
- Saves multiple checkpoint types (best model, latest, per-epoch)
- Progress bars with loss tracking

### `wikipedia/inference.py`

Command-line interface for text generation:

- **`load_model()`**: Loads a trained model from checkpoint
- **`generate_text()`**: Generates text continuation from a prompt
- **`main()`**: CLI entry point

**Usage (from repo root):**
```bash
python wikipedia/inference.py --model_name wikipedia_small --prompt "The capital of Mexico is"
```

**Options:**
- `--model_name`: Name of the model to load (required)
- `--prompt`: Beginning of sentence to complete (required)
- `--max_length`: Maximum number of tokens to generate (default: 100)
- `--temperature`: Sampling temperature, higher = more random (default: 1.0)
- `--top_k`: Top-k sampling, 0 to disable (default: 50)
- `--weights_dir`: Directory containing model weights (default: wikipedia/weights)

### `configs/wikipedia_small.yaml`

Configuration file defining model and training parameters:

**Model Architecture:**
- `d_model`: Embedding dimension (default: 512)
- `n_heads`: Number of attention heads (default: 8)
- `n_layers`: Number of transformer layers (default: 6)
- `d_ff`: Feed-forward dimension (default: 2048)
- `max_seq_len`: Maximum sequence length (default: 512)
- `dropout`: Dropout rate (default: 0.1)

**Training:**
- `num_epochs`: Number of training epochs (default: 10)
- `batch_size`: Batch size (default: 32)
- `learning_rate`: Initial learning rate (default: 0.0001)
- `weight_decay`: Weight decay for regularization (default: 0.01)
- `min_lr`: Minimum learning rate for scheduler (default: 0.000001)
- `max_grad_norm`: Gradient clipping threshold (default: 1.0)
- `save_every`: Save checkpoint every N epochs (default: 1)

**Data:**
- `number_of_articles`: Number of Wikipedia articles to download (default: 5)
- `data_dir`: Directory to save/load articles (default: wikipedia/data)
- `num_workers`: Number of DataLoader workers (default: 0)

**Paths:**
- `model_name`: Name for saved checkpoints
- `weights_dir`: Directory to save model weights (default: wikipedia/weights)

## Setup

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- Required packages (see `requirements.txt` in project root)

### Installation

1. Install dependencies:
```bash
pip install torch wikipedia pyyaml tqdm
```

Or from the project root:
```bash
pip install -r requirements.txt
```

2. Ensure you're in the project root directory when running scripts.

## Usage

### Training a Model

1. Create or modify a configuration file in `configs/`:
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
python wikipedia/training.py wikipedia/configs/wikipedia_small.yaml
```

The script will:
- Download Wikipedia articles if `data/` is empty
- Create the model according to config
- Train for the specified number of epochs
- Save checkpoints to `wikipedia/weights/`

**Checkpoint Files:**
- `{model_name}_best.pt`: Best model (lowest loss)
- `{model_name}_latest.pt`: Most recent checkpoint
- `{model_name}_epoch_{N}.pt`: Checkpoint at epoch N

### Running Inference

After training, generate text completions:

```bash
python wikipedia/inference.py \
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

## Model Architecture Details

The model uses a standard decoder-only transformer architecture:

1. **Token Embedding**: Character-level embeddings using `nn.Embedding`
2. **Positional Encoding**: Learned positional embeddings using native `nn.Embedding`
3. **Transformer Decoder**: Stack of decoder layers with:
   - Multi-head self-attention (causal masking)
   - Feed-forward networks
   - Layer normalization
   - Residual connections
4. **Output Head**: Linear projection to vocabulary size

The implementation leverages PyTorch's native `nn.TransformerDecoder` for efficiency and maintainability.

## Data Format

Wikipedia articles are downloaded and saved as plain text files in `wikipedia/data/`:
- Filenames are sanitized article titles (lowercase, underscores, alphanumeric only)
- Each file contains the full article text
- Articles are automatically split into sequences for training

## Tips and Best Practices

1. **Start Small**: Begin with a small model (e.g., `d_model=256`, `n_layers=4`) and few articles to test the pipeline
2. **Monitor Training**: Watch the loss decrease - if it plateaus, try adjusting learning rate or model size
3. **Data Quality**: More articles generally lead to better results, but require more training time
4. **Memory**: Larger models and longer sequences require more GPU memory. Adjust `batch_size` and `max_seq_len` accordingly
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

