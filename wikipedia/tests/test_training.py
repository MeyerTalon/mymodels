"""tests for the training loop."""

import math

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from wikipedia.architecture import DecoderOnlyTransformer
from wikipedia.training import Trainer


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
    trainer.model = DecoderOnlyTransformer(
        vocab_size=32,
        d_model=16,
        n_heads=2,
        n_layers=1,
        d_ff=32,
        max_seq_len=16,
        dropout=0.0,
    )
    trainer.optimizer = Adam(trainer.model.parameters(), lr=1e-3)
    trainer.criterion = nn.CrossEntropyLoss(ignore_index=0)
    return trainer


def test_train_epoch_returns_finite_loss():
    trainer = _cpu_trainer()
    inputs = torch.randint(1, 32, (4, 8))
    targets = torch.randint(1, 32, (4, 8))
    trainer.train_loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)

    loss = trainer.train_epoch()
    assert math.isfinite(loss)
    assert loss > 0


def test_evaluate_returns_none_without_val_loader():
    trainer = _cpu_trainer()
    trainer.val_loader = None
    assert trainer.evaluate() is None
