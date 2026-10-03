import torch

from gpt.tests.fakes import CharTokenizer
from shakespeare_visualized.architecture import DecoderConfig, DecoderOnlyTransformer
from shakespeare_visualized.inference import generate_text
from shakespeare_visualized.visualization import ActivationView

SETTINGS = DecoderConfig(
    d_model=16, n_heads=2, n_layers=2, d_ff=32, max_seq_len=8, dropout=0.0
)
PROMPT = 'ab'


def _model() -> DecoderOnlyTransformer:
    torch.manual_seed(0)
    return DecoderOnlyTransformer(CharTokenizer.vocab_size, SETTINGS).eval()


def test_activation_view_records_layers_and_then_unhooks() -> None:
    model = _model()
    tokenizer = CharTokenizer()
    attention = model.encoder.layers[0].self_attn
    hidden = torch.randn(1, 3, SETTINGS.d_model)
    view = ActivationView(model, tokenizer, live=False)
    with view:
        _output, weights = model.encoder.layers[0].self_attn(
            hidden, hidden, hidden, need_weights=False
        )
        assert weights is not None
        model(torch.tensor([tokenizer.encode(PROMPT)]))
    assert model.encoder.layers[0].self_attn is attention
    assert len(view.lens_history) == 1
    assert len(view.lens_history[0]) == SETTINGS.n_layers
    assert view.latest_attention is not None
    assert view.latest_attention.shape[0] == SETTINGS.n_heads
    assert len(model._forward_hooks) == 0
    assert len(model.encoder.layers[0]._forward_hooks) == 0
    _output, weights = attention(hidden, hidden, hidden, need_weights=False)
    assert weights is None


def test_generate_without_a_view_does_not_hook_the_model() -> None:
    model = _model()
    generate_text(
        model,
        CharTokenizer(),
        PROMPT,
        max_new_tokens=1,
        temperature=1.0,
        top_k=1,
    )
    assert len(model._forward_hooks) == 0
    assert all(len(layer._forward_hooks) == 0 for layer in model.encoder.layers)
