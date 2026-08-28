"""tests for the EncoderDecoderTransformer architecture."""

import torch

from translation.architecture import EncoderDecoderTransformer


def _tiny_model(
    vocab_size: int = 32, max_seq_len: int = 16
) -> EncoderDecoderTransformer:
    """builds a tiny model that runs in milliseconds on CPU."""
    return EncoderDecoderTransformer(
        vocab_size=vocab_size,
        d_model=16,
        n_heads=2,
        n_encoder_layers=1,
        n_decoder_layers=1,
        d_ff=32,
        max_seq_len=max_seq_len,
        dropout=0.0,
    )


def test_forward_output_shape() -> None:
    model = _tiny_model()
    src = torch.randint(1, 32, (2, 5))
    tgt = torch.randint(1, 32, (2, 7))
    logits = model(src, tgt)
    assert logits.shape == (2, 7, 32)


def test_output_head_is_weight_tied() -> None:
    model = _tiny_model()
    assert model.head.weight is model.token_embedding.weight


def test_decoder_is_causal() -> None:
    torch.manual_seed(0)
    model = _tiny_model()
    model.eval()
    src = torch.randint(1, 32, (1, 4))
    tgt = torch.randint(1, 32, (1, 6))
    base = model(src, tgt)
    tgt_mod = tgt.clone()
    tgt_mod[0, -1] = (tgt[0, -1] + 1) % 32
    modified = model(src, tgt_mod)
    assert torch.allclose(base[0, :-1], modified[0, :-1], atol=1e-5)


def test_generate_returns_string(dummy_tokenizer) -> None:
    model = _tiny_model()
    src_ids = dummy_tokenizer.encode_source("hello", "es")
    out = model.generate(dummy_tokenizer, src_ids, max_length=5, top_k=4)
    assert isinstance(out, str)


def test_generate_with_top_k_disabled(dummy_tokenizer) -> None:
    model = _tiny_model()
    src_ids = dummy_tokenizer.encode_source("hello", "es")
    out = model.generate(dummy_tokenizer, src_ids, max_length=3, top_k=0)
    assert isinstance(out, str)


def test_padding_mask_does_not_crash() -> None:
    model = _tiny_model()
    src = torch.tensor([[4, 9, 10, 0], [4, 11, 0, 0]])
    tgt = torch.tensor([[1, 12, 13, 0], [1, 14, 0, 0]])
    src_pad = src == 0
    tgt_pad = tgt == 0
    logits = model(src, tgt, src_pad, tgt_pad)
    assert logits.shape == (2, 4, 32)
