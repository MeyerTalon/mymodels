"""tests for the MnistCNN architecture."""

import torch

from mnist.architecture import MnistCNN


def _tiny_model() -> MnistCNN:
    """builds a tiny CNN that runs in milliseconds on CPU."""
    return MnistCNN(
        in_channels=1,
        conv1_channels=4,
        conv2_channels=8,
        hidden_dim=16,
        num_classes=10,
        dropout=0.0,
    )


def test_forward_output_shape() -> None:
    model = _tiny_model()
    x = torch.randn(2, 1, 28, 28)
    logits = model(x)
    assert logits.shape == (2, 10)


def test_forward_accepts_single_image() -> None:
    model = _tiny_model()
    x = torch.randn(1, 1, 28, 28)
    logits = model(x)
    assert logits.shape == (1, 10)


def test_default_small_config_param_count() -> None:
    model = MnistCNN()
    total = sum(p.numel() for p in model.parameters())
    assert total == 421642
