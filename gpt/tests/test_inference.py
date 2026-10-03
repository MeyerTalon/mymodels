from dataclasses import asdict
from pathlib import Path

import torch

from core.checkpoint import BEST_SUFFIX, checkpoint_path
from core.tokenizer import BPETokenizer
from gpt.architecture import DecoderConfig, DecoderOnlyTransformer
from gpt.inference import generate_text, load_model
from gpt.tests.fakes import CharTokenizer

DECODER = DecoderConfig(
    d_model=16, n_heads=2, n_layers=1, d_ff=32, max_seq_len=8, dropout=0.5
)
PROMPT = 'the'


def test_generate_text_returns_only_continuation() -> None:
    torch.manual_seed(0)
    model = DecoderOnlyTransformer(CharTokenizer.vocab_size, DECODER)
    continuation = generate_text(
        model, CharTokenizer(), PROMPT, max_new_tokens=5, temperature=1.0, top_k=1
    )
    assert len(continuation) <= 5


def test_load_model_rebuilds_from_checkpoint_without_dropout(tmp_path: Path) -> None:
    tokenizer_dir = tmp_path / 'tokenizer'
    tokenizer = BPETokenizer.train_or_load(
        ['the quick brown fox'] * 5, tokenizer_dir, vocab_size=300, min_frequency=1
    )
    model = DecoderOnlyTransformer(tokenizer.vocab_size, DECODER)
    torch.save(
        {
            'config': asdict(DECODER),
            'model_state_dict': model.state_dict(),
            'tokenizer_vocab_size': tokenizer.vocab_size,
        },
        checkpoint_path(tmp_path, 'tiny', BEST_SUFFIX),
    )
    loaded, loaded_tokenizer = load_model(
        'tiny', tmp_path, tokenizer_dir, torch.device('cpu')
    )
    assert loaded.dropout.p == 0.0
    assert not loaded.training
    assert loaded_tokenizer.vocab_size == tokenizer.vocab_size
    assert torch.equal(loaded.head.weight, model.head.weight)
