"""tests for the training loop."""

import math

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from wikipedia.architecture import DecoderOnlyTransformer
from wikipedia.training import Trainer


def test_train_epoch_returns_finite_loss():
    trainer = Trainer.__new__(Trainer)
    trainer.config = {"max_grad_norm": 1.0}
    trainer.device = torch.device("cpu")
    trainer.current_epoch = 0
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

    inputs = torch.randint(1, 32, (4, 8))
    targets = torch.randint(1, 32, (4, 8))
    trainer.dataloader = DataLoader(TensorDataset(inputs, targets), batch_size=2)

    loss = trainer.train_epoch()
    assert math.isfinite(loss)
    assert loss > 0
