import math
from pathlib import Path

import torch

from core.checkpoint import load_checkpoint
from core.tests.factories import training_config
from mnist.tests.fixtures import SMALL_CNN, digits
from mnist.training import build_trainer


def test_training_tracks_accuracy_in_checkpoint(tmp_path: Path) -> None:
    torch.manual_seed(0)
    config = training_config(tmp_path, **SMALL_CNN)
    trainer = build_trainer(config, digits(8), digits(4), torch.device('cpu'))
    trainer.train()
    assert math.isfinite(trainer.best_loss)
    best_path = tmp_path / 'weights' / 'test_model_best.pt'
    assert 0.0 <= torch.load(best_path)['accuracy'] <= 1.0
    assert load_checkpoint(best_path, torch.device('cpu')).tokenizer_vocab_size is None
