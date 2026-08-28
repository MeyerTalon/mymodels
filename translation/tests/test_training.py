"""tests for the translation training loop."""

import math
from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from translation.architecture import EncoderDecoderTransformer
from translation.data import collate_pairs
from translation.training import Trainer
import translation.training as training_module


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
    trainer.model = EncoderDecoderTransformer(
        vocab_size=32,
        d_model=16,
        n_heads=2,
        n_encoder_layers=1,
        n_decoder_layers=1,
        d_ff=32,
        max_seq_len=16,
        dropout=0.0,
    )
    trainer.optimizer = Adam(trainer.model.parameters(), lr=1e-3)
    trainer.criterion = nn.CrossEntropyLoss(ignore_index=0)
    return trainer


def test_train_epoch_returns_finite_loss() -> None:
    trainer = _cpu_trainer()
    batch = [
        (
            torch.tensor([4, 9, 10]),
            torch.tensor([1, 11, 12]),
            torch.tensor([11, 12, 2]),
        ),
        (
            torch.tensor([4, 13]),
            torch.tensor([1, 14]),
            torch.tensor([14, 2]),
        ),
    ]
    trainer.train_loader = DataLoader(
        batch,
        batch_size=2,
        collate_fn=lambda items: collate_pairs(items, pad_id=0),
    )

    loss = trainer.train_epoch()
    assert math.isfinite(loss)
    assert loss > 0


def test_evaluate_returns_none_without_val_loader() -> None:
    trainer = _cpu_trainer()
    trainer.val_loader = None
    assert trainer.evaluate() is None


def test_setup_data_reuses_one_pair_corpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pairs = [
        {"src": "hello", "tgt": "hola", "src_lang": "en", "tgt_lang": "es"},
    ]
    captured: Dict[str, Dict[str, Any]] = {}

    def fake_load(**kwargs: Any) -> List[Dict[str, str]]:
        """returns one synthetic corpus and records acquisition settings."""
        captured["load"] = kwargs
        return pairs

    def fake_train_or_load(**kwargs: Any) -> object:
        """returns a tokenizer stand-in and records its corpus."""
        captured["tokenizer"] = kwargs

        class _Tok:
            pad_id = 0
            vocab_size = 32

        return _Tok()

    def fake_create_dataloaders(
        **kwargs: Any,
    ) -> Tuple[List[int], Optional[object]]:
        """returns a loader stand-in and records its corpus."""
        captured["loaders"] = kwargs
        return [1], None

    monkeypatch.setattr(training_module, "load_translation_pairs", fake_load)
    monkeypatch.setattr(
        training_module.TranslationBPETokenizer,
        "train_or_load",
        staticmethod(fake_train_or_load),
    )
    monkeypatch.setattr(
        training_module, "create_dataloaders", fake_create_dataloaders
    )

    trainer = Trainer.__new__(Trainer)
    trainer.config = {"max_pairs": 1}
    trainer._setup_data()

    assert captured["loaders"]["pairs"] is pairs
    assert "hello" in captured["tokenizer"]["texts"]
    assert "hola" in captured["tokenizer"]["texts"]
