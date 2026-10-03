from dataclasses import dataclass

import torch
from torch import nn

from core.config import Config, require_float, require_int
from core.sampling import sample_next_token
from core.weights import init_gpt_weights


@dataclass(frozen=True)
class EncoderDecoderConfig:
    d_model: int
    n_heads: int
    n_encoder_layers: int
    n_decoder_layers: int
    d_ff: int
    max_seq_len: int
    dropout: float

    @classmethod
    def from_config(cls, config: Config) -> 'EncoderDecoderConfig':
        return cls(
            d_model=require_int(config, 'd_model'),
            n_heads=require_int(config, 'n_heads'),
            n_encoder_layers=require_int(config, 'n_encoder_layers'),
            n_decoder_layers=require_int(config, 'n_decoder_layers'),
            d_ff=require_int(config, 'd_ff'),
            max_seq_len=require_int(config, 'max_seq_len'),
            dropout=require_float(config, 'dropout'),
        )


class EncoderDecoderTransformer(nn.Module):
    """Source and target share one embedding (joint vocabulary), and the output head is tied to it."""

    def __init__(self, vocab_size: int, settings: EncoderDecoderConfig) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = settings.max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, settings.d_model)
        self.position_embedding = nn.Embedding(settings.max_seq_len, settings.d_model)
        self.dropout = nn.Dropout(settings.dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=settings.d_model,
            nhead=settings.n_heads,
            dim_feedforward=settings.d_ff,
            dropout=settings.dropout,
            activation='gelu',
            norm_first=True,
            batch_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=settings.d_model,
            nhead=settings.n_heads,
            dim_feedforward=settings.d_ff,
            dropout=settings.dropout,
            activation='gelu',
            norm_first=True,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=settings.n_encoder_layers,
            enable_nested_tensor=False,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=settings.n_decoder_layers
        )
        self.ln_f = nn.LayerNorm(settings.d_model)
        self.head = nn.Linear(settings.d_model, vocab_size, bias=False)
        self.apply(init_gpt_weights)
        self.head.weight = self.token_embedding.weight

    def _embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        seq_len = token_ids.size(1)
        if seq_len > self.max_seq_len:
            raise ValueError(
                f'sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}'
            )
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
        embedded: torch.Tensor = self.dropout(
            self.token_embedding(token_ids) + self.position_embedding(positions)
        )
        return embedded

    def _decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor | None,
        memory_key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt.size(1), device=tgt.device, dtype=torch.bool
        )
        decoded = self.decoder(
            tgt=self._embed(tgt),
            memory=memory,
            tgt_mask=causal_mask,
            tgt_is_causal=True,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        logits: torch.Tensor = self.head(self.ln_f(decoded))
        return logits

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Teacher forcing: (batch, src_len) and (batch, tgt_len) ids to (batch, tgt_len, vocab) logits. Padding masks are `True` at pads."""
        memory = self.encoder(
            src=self._embed(src), src_key_padding_mask=src_key_padding_mask
        )
        return self._decode(tgt, memory, tgt_key_padding_mask, src_key_padding_mask)

    @torch.no_grad()
    def generate(
        self,
        src_ids: list[int],
        *,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        bos_id: int,
        eos_id: int,
    ) -> list[int]:
        """`src_ids` must already carry the target-language prefix. Returns target ids without bos or eos."""
        self.eval()
        device = self.token_embedding.weight.device
        src = torch.tensor([src_ids[: self.max_seq_len]], device=device)
        memory = self.encoder(src=self._embed(src))
        tokens = torch.tensor([[bos_id]], device=device)
        for _ in range(max_new_tokens):
            logits = self._decode(tokens[:, -self.max_seq_len :], memory, None, None)
            next_token = sample_next_token(
                logits[0, -1], temperature=temperature, top_k=top_k
            )
            if next_token.item() == eos_id:
                break
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
        target_ids: list[int] = tokens[0, 1:].tolist()
        return target_ids
