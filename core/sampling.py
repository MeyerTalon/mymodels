import torch
import torch.nn.functional as F

MIN_TEMPERATURE = 1e-6


def sample_next_token(
    logits: torch.Tensor, *, temperature: float, top_k: int
) -> torch.Tensor:
    """`logits` has shape (vocab,); returns a shape-(1,) token id. `top_k` of 0 samples the whole vocabulary."""
    scaled = logits / max(temperature, MIN_TEMPERATURE)
    if top_k > 0:
        top_logits, top_indices = torch.topk(scaled, min(top_k, scaled.size(-1)))
        return top_indices[torch.multinomial(F.softmax(top_logits, dim=-1), 1)]
    return torch.multinomial(F.softmax(scaled, dim=-1), 1)
