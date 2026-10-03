# MNIST digit classifier

A small CNN (two conv, ReLU, and max-pool blocks, then a two-layer classifier) for 28×28 grayscale digits. It trains through the shared `core.training.Trainer`, which tracks accuracy alongside loss.

## Configs

| Config | Parameters | Key dims |
|---|---|---|
| `mnist_small` | 421,642 | conv channels 32 and 64, `hidden_dim=128` |

## Usage

```bash
mise run train:mnist mnist/configs/mnist_small.yaml
mise run infer:mnist --model_name mnist_small --image path/to/digit.png --show_probs
mise run infer:mnist --model_name mnist_small --index 0 --show_probs
```

Training downloads MNIST through torchvision into `mnist/data/` on first use; `dataset_cache_only: true` requires that cache. Validation is a seeded `val_fraction` slice of the official training split, so it is stable across runs with the same `dataset_seed`. `--index` reads the official test split. `--image` accepts any image and converts it to 28×28 grayscale.
