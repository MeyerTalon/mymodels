from pathlib import Path

import pytest
import torch

from core.config import load_config, require_int
from core.paths import REPO_ROOT
from core.training import TrainingConfig, count_parameters
from translation.architecture import EncoderDecoderConfig, EncoderDecoderTransformer

VOCAB_SIZE = 40
BOS_ID = 1
NEVER_SAMPLED_ID = -1
ARCHITECTURE = EncoderDecoderConfig(
    d_model=16,
    n_heads=2,
    n_encoder_layers=1,
    n_decoder_layers=1,
    d_ff=32,
    max_seq_len=16,
    dropout=0.0,
)
CONFIG_PATHS = sorted((REPO_ROOT / 'translation' / 'configs').glob('*.yaml'))


def _model() -> EncoderDecoderTransformer:
    torch.manual_seed(0)
    return EncoderDecoderTransformer(VOCAB_SIZE, ARCHITECTURE).eval()


def test_forward_shape_and_tied_head() -> None:
    model = _model()
    logits = model(
        torch.randint(0, VOCAB_SIZE, (2, 5)), torch.randint(0, VOCAB_SIZE, (2, 4))
    )
    assert logits.shape == (2, 4, VOCAB_SIZE)
    assert model.head.weight is model.token_embedding.weight


def test_forward_rejects_sequences_past_max_seq_len() -> None:
    too_long = ARCHITECTURE.max_seq_len + 1
    with pytest.raises(ValueError, match='max_seq_len'):
        _model()(
            torch.zeros(1, too_long, dtype=torch.long),
            torch.zeros(1, 2, dtype=torch.long),
        )


def test_generate_returns_target_ids_without_bos() -> None:
    target_ids = _model().generate(
        [4, 9, 10],
        max_new_tokens=6,
        temperature=1.0,
        top_k=1,
        bos_id=BOS_ID,
        eos_id=NEVER_SAMPLED_ID,
    )
    assert len(target_ids) == 6


def test_generate_stops_before_eos() -> None:
    model = _model()
    greedy = model.generate(
        [4, 9],
        max_new_tokens=3,
        temperature=1.0,
        top_k=1,
        bos_id=BOS_ID,
        eos_id=NEVER_SAMPLED_ID,
    )
    stopped = model.generate(
        [4, 9],
        max_new_tokens=3,
        temperature=1.0,
        top_k=1,
        bos_id=BOS_ID,
        eos_id=greedy[0],
    )
    assert stopped == []


@pytest.mark.parametrize('path', CONFIG_PATHS, ids=lambda path: path.stem)
def test_config_matches_expected_parameters(path: Path) -> None:
    config = load_config(path)
    TrainingConfig.from_config(config)
    with torch.device('meta'):
        model = EncoderDecoderTransformer(
            require_int(config, 'vocab_size'), EncoderDecoderConfig.from_config(config)
        )
    assert count_parameters(model) == require_int(config, 'expected_parameters')
