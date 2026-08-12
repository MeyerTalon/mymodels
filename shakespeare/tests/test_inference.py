"""tests for checkpoint loading and text generation."""

from pathlib import Path

import pytest
import torch

from shakespeare.architecture import DecoderOnlyTransformer
from shakespeare.inference import generate_text, load_model
from shakespeare.tokenizer import ShakespeareBPETokenizer

TINY_CONFIG = {
    "d_model": 16,
    "n_heads": 2,
    "n_layers": 1,
    "d_ff": 32,
    "max_seq_len": 16,
}


def test_load_model_and_generate(tmp_path: Path) -> None:
    tok_dir = tmp_path / "tok"
    tokenizer = ShakespeareBPETokenizer.train_or_load(
        ["to be or not to be that is the question. " * 20],
        str(tok_dir),
        vocab_size=300,
    )

    model = DecoderOnlyTransformer(
        vocab_size=tokenizer.vocab_size, dropout=0.0, **TINY_CONFIG
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

    prompt = "to be"
    continuation = generate_text(
        loaded_model, loaded_tokenizer, prompt, max_length=3
    )
    assert isinstance(continuation, str)
    assert not continuation.startswith(prompt)


def test_load_model_missing_checkpoint_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_model(
            "missing",
            weights_dir=str(tmp_path),
            device=torch.device("cpu"),
            tokenizer_dir=str(tmp_path),
        )
