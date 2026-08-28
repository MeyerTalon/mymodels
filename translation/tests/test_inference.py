"""tests for checkpoint loading and translation."""

from pathlib import Path

import pytest
import torch

from translation.architecture import EncoderDecoderTransformer
from translation.inference import load_model, translate
from translation.tokenizer import TranslationBPETokenizer

TINY_CONFIG = {
    "d_model": 16,
    "n_heads": 2,
    "n_encoder_layers": 1,
    "n_decoder_layers": 1,
    "d_ff": 32,
    "max_seq_len": 16,
    "languages": ["en", "es", "fr", "de"],
}

TINY_CORPUS = [
    "hello world, how are you today? " * 10,
    "hola mundo, como estas hoy? " * 10,
]


def test_load_model_and_translate(tmp_path: Path) -> None:
    tok_dir = tmp_path / "tok"
    tokenizer = TranslationBPETokenizer.train_or_load(
        TINY_CORPUS, str(tok_dir), vocab_size=300
    )
    model = EncoderDecoderTransformer(
        vocab_size=tokenizer.vocab_size, dropout=0.0, **{
            k: v for k, v in TINY_CONFIG.items() if k != "languages"
        }
    )
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": TINY_CONFIG,
        "tokenizer_vocab_size": tokenizer.vocab_size,
    }
    torch.save(checkpoint, weights_dir / "tiny_best.pt")

    loaded_model, loaded_tokenizer, device = load_model(
        "tiny",
        weights_dir=str(weights_dir),
        device=torch.device("cpu"),
        tokenizer_dir=str(tok_dir),
    )
    assert not loaded_model.training
    assert device.type == "cpu"

    out = translate(loaded_model, loaded_tokenizer, "hello", "es", max_length=4)
    assert isinstance(out, str)


def test_load_model_missing_checkpoint_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_model(
            "missing",
            weights_dir=str(tmp_path),
            device=torch.device("cpu"),
            tokenizer_dir=str(tmp_path),
        )
