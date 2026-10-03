from dataclasses import dataclass

import torch
from torch import nn

from core.config import Config, require_float, require_int
from core.sampling import sample_next_token
from core.weights import init_gpt_weights


@dataclass(frozen=True)
class DecoderConfig:
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    max_seq_len: int
    dropout: float

    @classmethod
    def from_config(cls, config: Config) -> 'DecoderConfig':
        return cls(
            d_model=require_int(config, 'd_model'),
            n_heads=require_int(config, 'n_heads'),
            n_layers=require_int(config, 'n_layers'),
            d_ff=require_int(config, 'd_ff'),
            max_seq_len=require_int(config, 'max_seq_len'),
            dropout=require_float(config, 'dropout'),
        )


class DecoderOnlyTransformer(nn.Module):
    """GPT-style: pre-norm GELU layers, learned positions, output head tied to the token embedding."""

    def __init__(self, vocab_size: int, settings: DecoderConfig) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = settings.max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, settings.d_model)
        self.position_embedding = nn.Embedding(settings.max_seq_len, settings.d_model)
        self.dropout = nn.Dropout(settings.dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=settings.d_model,
            nhead=settings.n_heads,
            dim_feedforward=settings.d_ff,
            dropout=settings.dropout,
            activation='gelu',
            norm_first=True,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=settings.n_layers, enable_nested_tensor=False
        )
        self.ln_f = nn.LayerNorm(settings.d_model)
        self.head = nn.Linear(settings.d_model, vocab_size, bias=False)
        self.apply(init_gpt_weights)
        self.head.weight = self.token_embedding.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """(batch, seq) token ids to (batch, seq, vocab) logits; position i sees only positions up to i."""
        seq_len = token_ids.size(1)
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
        hidden = self.dropout(
            self.token_embedding(token_ids) + self.position_embedding(positions)
        )
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=token_ids.device
        )
        hidden = self.encoder(hidden, mask=causal_mask, is_causal=True)
        logits: torch.Tensor = self.head(self.ln_f(hidden))
        return logits

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: list[int],
        *,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        eos_id: int,
    ) -> list[int]:
        """Returns the prompt plus new tokens, stopping after `eos_id`; conditions on the last `max_seq_len` tokens."""
        self.eval()
        device = self.token_embedding.weight.device
        tokens = torch.tensor([prompt_ids], device=device)
        for _ in range(max_new_tokens):
            logits = self(tokens[:, -self.max_seq_len :])[0, -1]
            next_token = sample_next_token(logits, temperature=temperature, top_k=top_k)
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
            if next_token.item() == eos_id:
                break
        token_ids: list[int] = tokens[0].tolist()
        return token_ids
