import math
from pathlib import Path

import torch
from torch import nn

from core.tests.factories import training_config
from core.training import TrainingConfig
from translation.architecture import EncoderDecoderConfig, EncoderDecoderTransformer
from translation.data import create_dataloaders
from translation.inference import translate
from translation.tokenizer import TranslationTokenizer
from translation.training import TranslationTrainer

LANGUAGES = ['en', 'es']
PAIRS = [
    {'src': 'hello world', 'tgt': 'hola mundo', 'src_lang': 'en', 'tgt_lang': 'es'},
    {'src': 'good morning', 'tgt': 'buenos dias', 'src_lang': 'en', 'tgt_lang': 'es'},
] * 4
ARCHITECTURE = EncoderDecoderConfig(
    d_model=16,
    n_heads=2,
    n_encoder_layers=1,
    n_decoder_layers=1,
    d_ff=32,
    max_seq_len=16,
    dropout=0.0,
)


def test_train_and_translate(tmp_path: Path) -> None:
    torch.manual_seed(0)
    tokenizer = TranslationTokenizer.train_or_load(
        [pair['src'] for pair in PAIRS] + [pair['tgt'] for pair in PAIRS],
        tmp_path / 'tokenizer',
        vocab_size=300,
        min_frequency=1,
        languages=LANGUAGES,
    )
    config = training_config(tmp_path, val_fraction=0.25)
    settings = TrainingConfig.from_config(config)
    train_loader, val_loader = create_dataloaders(
        PAIRS,
        tokenizer,
        max_seq_len=ARCHITECTURE.max_seq_len,
        batch_size=settings.batch_size,
        val_fraction=settings.val_fraction,
        num_workers=settings.num_workers,
    )
    assert val_loader is not None
    model = EncoderDecoderTransformer(tokenizer.vocab_size, ARCHITECTURE)
    trainer = TranslationTrainer(
        settings=settings,
        config=config,
        model=model,
        criterion=nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id),
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device('cpu'),
        checkpoint_extras={'tokenizer_vocab_size': tokenizer.vocab_size},
    )
    trainer.train()
    assert math.isfinite(trainer.best_loss)
    assert (tmp_path / 'weights' / 'test_model_best.pt').is_file()
    translation = translate(
        model,
        tokenizer,
        'hello world',
        target_lang='es',
        max_new_tokens=4,
        temperature=1.0,
        top_k=1,
    )
    assert '<' not in translation
