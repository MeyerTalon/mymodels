# MNIST Digit Classifier

A small convolutional network that classifies MNIST handwritten digits (0–9). The package mirrors the `wikipedia/` layout where it applies (config-driven Trainer, self-describing checkpoints, reporting, CLI inference) but has no tokenizer: images are not tokens. `tokenizer.py` is omitted on purpose.

## Overview

Training downloads MNIST via torchvision into `mnist/data/` on first use and reuses that cache afterward. The model is a two-block CNN (conv → ReLU → max-pool, twice) plus a linear head — a teaching/demo architecture, not a residual network.

## Model sizes

| Config | Parameters | Key dims |
|--------|------------|----------|
| `mnist_small` | ~0.42M (421,642) | `conv1=32`, `conv2=64`, `hidden_dim=128` |

See the first line of `mnist/configs/mnist_small.yaml` for the canonical number.

## Folder structure

```
docs/
└── mnist.md               # This file

mnist/
├── __init__.py
├── architecture.py        # MnistCNN
├── data.py                # torchvision MNIST + DataLoaders
├── training.py            # Trainer + CLI
├── inference.py           # Classify an image or a test-set index
├── reporting.py           # Per-epoch loss and accuracy charts
├── utils.py               # Device selection and path helpers
├── configs/
├── tests/
├── data/                  # torchvision MNIST cache (gitignored)
├── weights/               # Checkpoints (gitignored)
└── reports/               # Charts (gitignored)
```

There is no `tokenizer.py` and no `tokenizer_files/` directory.

## Usage

### Training

From the repo root:

```bash
uv run python -m mnist.training mnist/configs/mnist_small.yaml
```

On first run this downloads MNIST into `mnist/data/`. Set `dataset_cache_only: True` to require the local cache and skip the network.

Checkpoints are written to `mnist/weights/`:

- `{model_name}_best.pt`
- `{model_name}_latest.pt`
- `{model_name}_epoch_{N}.pt`

### Inference

Classify a local image (converted to 28×28 grayscale):

```bash
uv run python -m mnist.inference --model_name mnist_small --image path/to/digit.png --show_probs
```

Or classify an example from the MNIST test set (requires a cached download):

```bash
uv run python -m mnist.inference --model_name mnist_small --index 0 --show_probs
```

### Tests

```bash
uv run pytest mnist/tests
```

Tests use synthetic `(1, 28, 28)` tensors and never download MNIST.

## Architecture notes

Native `nn.Conv2d` / `nn.MaxPool2d` / `nn.Linear`. `forward` returns raw class logits of shape `(batch, 10)`. Mixed precision (`bf16` recommended on MPS), gradient accumulation, and self-describing checkpoints match the language-model packages. Reports plot both cross-entropy loss and accuracy.
