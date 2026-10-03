import torch

from core.sampling import sample_next_token


def test_top_k_one_is_greedy() -> None:
    logits = torch.tensor([0.1, 3.0, 0.5, 2.9])
    for _ in range(10):
        token = sample_next_token(logits, temperature=1.0, top_k=1)
        assert token.tolist() == [1]


def test_full_vocabulary_sampling_returns_valid_id() -> None:
    logits = torch.zeros(5)
    token = sample_next_token(logits, temperature=0.0, top_k=0)
    assert token.shape == (1,)
    assert 0 <= int(token.item()) < 5
