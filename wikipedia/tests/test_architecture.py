"""tests for the DecoderOnlyTransformer architecture."""

import torch

from wikipedia.architecture import DecoderOnlyTransformer


def _tiny_model(vocab_size: int = 32, max_seq_len: int = 16) -> DecoderOnlyTransformer:
    """builds a tiny model that runs in milliseconds on CPU."""
    return DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=16,
        n_heads=2,
        n_layers=1,
        d_ff=32,
        max_seq_len=max_seq_len,
        dropout=0.0,
    )


def test_forward_output_shape():
    model = _tiny_model()
    x = torch.randint(1, 32, (2, 8))
    logits = model(x)
    assert logits.shape == (2, 8, 32)


def test_causal_mask_blocks_future_positions():
    model = _tiny_model()
    mask = model._causal_mask(4, torch.device("cpu"))
    assert mask.shape == (4, 4)
    assert torch.all(torch.tril(mask) == 0)
    above_diagonal = mask[torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)]
    assert torch.all(above_diagonal == float("-inf"))


def test_generate_returns_string(dummy_tokenizer):
    model = _tiny_model()
    out = model.generate(dummy_tokenizer, "abc", max_length=5, top_k=4)
    assert isinstance(out, str)


def test_generate_handles_prompt_longer_than_max_seq_len(dummy_tokenizer):
    model = _tiny_model(max_seq_len=8)
    prompt_ids = list(range(1, 13))  # 12 tokens > max_seq_len
    out = model.generate(dummy_tokenizer, prompt_ids, max_length=2)
    assert isinstance(out, str)


def test_generate_with_top_k_disabled(dummy_tokenizer):
    model = _tiny_model()
    out = model.generate(dummy_tokenizer, "abc", max_length=3, top_k=0)
    assert isinstance(out, str)
