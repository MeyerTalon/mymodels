import torch

from gpt.architecture import DecoderConfig, DecoderOnlyTransformer

VOCAB_SIZE = 32
NEVER_SAMPLED_ID = -1
SETTINGS = DecoderConfig(
    d_model=16, n_heads=2, n_layers=2, d_ff=32, max_seq_len=8, dropout=0.0
)


def _model() -> DecoderOnlyTransformer:
    torch.manual_seed(0)
    return DecoderOnlyTransformer(VOCAB_SIZE, SETTINGS).eval()


def test_forward_shape_and_tied_head() -> None:
    model = _model()
    logits = model(torch.randint(0, VOCAB_SIZE, (3, 5)))
    assert logits.shape == (3, 5, VOCAB_SIZE)
    assert model.head.weight is model.token_embedding.weight


def test_future_tokens_do_not_affect_past_logits() -> None:
    model = _model()
    tokens = torch.randint(0, VOCAB_SIZE, (1, 6))
    changed = tokens.clone()
    changed[0, -1] = (changed[0, -1] + 1) % VOCAB_SIZE
    with torch.no_grad():
        assert torch.allclose(model(tokens)[0, :-1], model(changed)[0, :-1], atol=1e-5)


def test_generate_keeps_prompt_and_respects_budget() -> None:
    model = _model()
    prompt = [5, 6, 7]
    token_ids = model.generate(
        prompt, max_new_tokens=12, temperature=1.0, top_k=0, eos_id=NEVER_SAMPLED_ID
    )
    assert token_ids[:3] == prompt
    assert len(token_ids) == len(prompt) + 12


def test_generate_stops_after_eos() -> None:
    model = _model()
    greedy = model.generate(
        [5], max_new_tokens=3, temperature=1.0, top_k=1, eos_id=NEVER_SAMPLED_ID
    )
    first_new_token = greedy[1]
    stopped = model.generate(
        [5], max_new_tokens=3, temperature=1.0, top_k=1, eos_id=first_new_token
    )
    assert stopped == [5, first_new_token]
