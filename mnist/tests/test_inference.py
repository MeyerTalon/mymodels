"""tests for checkpoint loading and digit prediction."""

from pathlib import Path

import pytest
import torch

from mnist.architecture import MnistCNN
from mnist.inference import load_model, predict

TINY_CONFIG = {
    "in_channels": 1,
    "conv1_channels": 4,
    "conv2_channels": 8,
    "hidden_dim": 16,
    "num_classes": 10,
    "dropout": 0.0,
}


def test_load_model_and_predict(tmp_path: Path) -> None:
    model = MnistCNN(**TINY_CONFIG)
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": TINY_CONFIG,
    }
    torch.save(checkpoint, weights_dir / "tiny_best.pt")

    loaded, device = load_model(
        "tiny",
        weights_dir=str(weights_dir),
        device=torch.device("cpu"),
    )
    assert not loaded.training
    assert device.type == "cpu"

    images = torch.rand(2, 1, 28, 28)
    classes, probs = predict(loaded, images)
    assert classes.shape == (2,)
    assert probs.shape == (2, 10)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)


def test_load_model_missing_checkpoint_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_model(
            "missing",
            weights_dir=str(tmp_path),
            device=torch.device("cpu"),
        )
