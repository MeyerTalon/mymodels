import math
from pathlib import Path

import torch
from torch import nn

from core.checkpoint import load_checkpoint
from core.tests.factories import training_config
from core.training import TrainingConfig
from gpt.data import create_dataloaders
from gpt.tests.fakes import CharTokenizer
from shakespeare_visualized.architecture import DecoderConfig, DecoderOnlyTransformer
from shakespeare_visualized.training import LanguageModelTrainer

TEXTS = ['the quick brown fox jumps over the lazy dog.'] * 6
DECODER = DecoderConfig(
    d_model=16, n_heads=2, n_layers=1, d_ff=32, max_seq_len=8, dropout=0.0
)


def _trainer(tmp_path: Path) -> LanguageModelTrainer:
    torch.manual_seed(0)
    tokenizer = CharTokenizer()
    config = training_config(tmp_path, val_fraction=0.2)
    settings = TrainingConfig.from_config(config)
    train_loader, val_loader = create_dataloaders(
        TEXTS,
        tokenizer,
        block_size=DECODER.max_seq_len,
        batch_size=settings.batch_size,
        val_fraction=settings.val_fraction,
        num_workers=settings.num_workers,
    )
    return LanguageModelTrainer(
        settings=settings,
        config=config,
        model=DecoderOnlyTransformer(tokenizer.vocab_size, DECODER),
        criterion=nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id),
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device('cpu'),
        checkpoint_extras={'tokenizer_vocab_size': tokenizer.vocab_size},
    )


def test_train_saves_compatible_checkpoints(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path)
    trainer.train()
    assert math.isfinite(trainer.best_loss)
    weights_dir = tmp_path / 'weights'
    names = {path.name for path in weights_dir.iterdir()}
    assert {
        'test_model_latest.pt',
        'test_model_best.pt',
        'test_model_epoch_2.pt',
    } <= names
    checkpoint = load_checkpoint(
        weights_dir / 'test_model_best.pt', torch.device('cpu')
    )
    assert checkpoint.vocab_size() == CharTokenizer.vocab_size
    assert checkpoint.config['model_name'] == 'test_model'
    raw = torch.load(weights_dir / 'test_model_best.pt')
    assert {'epoch', 'optimizer_state_dict', 'scheduler_state_dict', 'loss'} <= set(raw)
    assert trainer.reporter.report_path.is_file()
