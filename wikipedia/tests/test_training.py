"""tests for the training loop."""

import math
from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

import wikipedia.training as training_module
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


def test_train_epoch_returns_finite_loss() -> None:
    trainer = _cpu_trainer()
    inputs = torch.randint(1, 32, (4, 8))
    targets = torch.randint(1, 32, (4, 8))
    trainer.train_loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)

    loss = trainer.train_epoch()
    assert math.isfinite(loss)
    assert loss > 0


def test_evaluate_returns_none_without_val_loader() -> None:
    trainer = _cpu_trainer()
    trainer.val_loader = None
    assert trainer.evaluate() is None


def test_setup_data_reuses_one_text_corpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    texts = ["alpha", "beta"]
    captured: Dict[str, Dict[str, Any]] = {}

    def fake_load_wikipedia_texts(**kwargs: Any) -> List[str]:
        """returns one synthetic corpus and records acquisition settings."""
        captured["load"] = kwargs
        return texts

    def fake_train_or_load(**kwargs: Any) -> object:
        """returns a tokenizer stand-in and records its corpus."""
        captured["tokenizer"] = kwargs
        return object()

    def fake_create_dataloaders(
        **kwargs: Any,
    ) -> Tuple[List[int], Optional[object]]:
        """returns a loader stand-in and records its corpus."""
        captured["loaders"] = kwargs
        return [1], None

    monkeypatch.setattr(
        training_module, "load_wikipedia_texts", fake_load_wikipedia_texts
    )
    monkeypatch.setattr(
        training_module.WikipediaBPETokenizer,
        "train_or_load",
        staticmethod(fake_train_or_load),
    )
    monkeypatch.setattr(
        training_module, "create_dataloaders", fake_create_dataloaders
    )

    trainer = Trainer.__new__(Trainer)
    trainer.config = {"number_of_articles": 2}
    trainer._setup_data()

    assert captured["tokenizer"]["texts"] is texts
    assert captured["loaders"]["texts"] is texts
