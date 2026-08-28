"""tests for the MNIST training loop."""

import math

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from mnist.architecture import MnistCNN
from mnist.data import TensorImageDataset
from mnist.training import Trainer


def _cpu_trainer() -> Trainer:
    """builds a minimal cpu Trainer without touching data or config files."""
    trainer = Trainer.__new__(Trainer)
    trainer.config = {"max_grad_norm": 1.0}
    trainer.device = torch.device("cpu")
    trainer.current_epoch = 0
    trainer.grad_accum_steps = 1
    trainer.log_interval = 50
    trainer.use_amp = False
    trainer.amp_dtype = None
    trainer.scaler = torch.amp.GradScaler(enabled=False)
    trainer.model = MnistCNN(
        conv1_channels=4, conv2_channels=8, hidden_dim=16, dropout=0.0
    )
    trainer.optimizer = Adam(trainer.model.parameters(), lr=1e-3)
    trainer.criterion = nn.CrossEntropyLoss()
    return trainer


def test_train_epoch_returns_finite_loss_and_accuracy() -> None:
    trainer = _cpu_trainer()
    dataset = TensorImageDataset(
        torch.rand(8, 1, 28, 28), torch.randint(0, 10, (8,))
    )
    trainer.train_loader = DataLoader(dataset, batch_size=4)

    loss, acc = trainer.train_epoch()
    assert math.isfinite(loss)
    assert loss > 0
    assert 0.0 <= acc <= 1.0


def test_evaluate_returns_none_without_val_loader() -> None:
    trainer = _cpu_trainer()
    trainer.val_loader = None
    assert trainer.evaluate() == (None, None)


def test_overfits_tiny_batch() -> None:
    torch.manual_seed(0)
    model = MnistCNN(
        conv1_channels=8, conv2_channels=16, hidden_dim=32, dropout=0.0
    )
    images = torch.rand(8, 1, 28, 28)
    labels = torch.arange(8) % 10
    optimizer = Adam(model.parameters(), lr=0.05)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(40):
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        preds = model(images).argmax(dim=-1)
    acc = float((preds == labels).float().mean().item())
    assert acc >= 0.9
