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


def test_output_head_is_weight_tied():
    model = _tiny_model()
    # the head must share storage with the token embedding (weight tying)
    assert model.head.weight is model.token_embedding.weight


def test_forward_is_causal():
    # changing a future token must not affect an earlier position's logits
    torch.manual_seed(0)
    model = _tiny_model()
    model.eval()
    x = torch.randint(1, 32, (1, 6))
    base = model(x)
    x_mod = x.clone()
    x_mod[0, -1] = (x[0, -1] + 1) % 32
    modified = model(x_mod)
    assert torch.allclose(base[0, :-1], modified[0, :-1], atol=1e-5)


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
