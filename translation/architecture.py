"""encoder-decoder transformer for multilingual translation.

uses native ``nn.TransformerEncoder`` and ``nn.TransformerDecoder`` with
pre-norm GELU layers, a shared token embedding (joint source/target vocab),
learned positional embeddings, and a weight-tied output projection. the target
language is selected by a ``<2xx>`` prefix on the source sequence.
"""

from typing import Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderDecoderTransformer(nn.Module):
    """seq2seq transformer for multilingual machine translation.

    source and target share one embedding matrix (joint vocabulary). the
    decoder output projection is weight-tied to that embedding. ``forward``
    returns teacher-forcing logits; sampling lives in ``generate``.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_encoder_layers: int = 3,
        n_decoder_layers: int = 3,
        d_ff: int = 1024,
        max_seq_len: int = 128,
        dropout: float = 0.1,
    ) -> None:
        """initializes the encoder-decoder transformer.

        Args:
            vocab_size: size of the shared token vocabulary.
            d_model: dimensionality of the model / embeddings.
            n_heads: number of attention heads per layer.
            n_encoder_layers: number of encoder layers.
            n_decoder_layers: number of decoder layers.
            d_ff: dimensionality of the feed-forward sub-layer.
            max_seq_len: maximum supported sequence length.
            dropout: dropout probability.
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            norm_first=True,
            batch_first=True,  # (batch, seq, feature)
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            norm_first=True,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_encoder_layers,
            enable_nested_tensor=False,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        self.apply(self._init_weights)
        # weight tying: share the token-embedding matrix with the output head.
        self.head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """applies GPT-style normal initialization to linear and embedding layers."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """adds token and positional embeddings.

        Args:
            token_ids: long tensor of shape (batch, seq).

        Returns:
            float tensor of shape (batch, seq, d_model).
        """
        batch_size, seq_len = token_ids.size()
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}."
            )
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(
            batch_size, -1
        )
        return self.dropout(
            self.token_embedding(token_ids) + self.position_embedding(positions)
        )

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """computes decoder logits under teacher forcing.

        Args:
            src: source token ids of shape (batch, src_len), including a
                ``<2xx>`` language-control prefix.
            tgt: decoder input ids of shape (batch, tgt_len) (bos + target).
            src_key_padding_mask: bool mask of shape (batch, src_len); ``True``
                marks padding positions to ignore.
            tgt_key_padding_mask: bool mask of shape (batch, tgt_len); ``True``
                marks padding positions to ignore.

        Returns:
            logits tensor of shape (batch, tgt_len, vocab_size).
        """
        src_emb = self._embed(src)  # (batch, src_len, d_model)
        tgt_emb = self._embed(tgt)  # (batch, tgt_len, d_model)
        memory = self.encoder(
            src=src_emb, src_key_padding_mask=src_key_padding_mask
        )
        tgt_len = tgt.size(1)
        # bool subsequent mask (True = masked) so it matches bool padding masks.
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt_len, device=tgt.device, dtype=torch.bool
        )
        decoded = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_is_causal=True,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.head(self.ln_f(decoded))  # (batch, tgt_len, vocab_size)

    @torch.no_grad()
    def generate(
        self,
        tokenizer: Any,
        src_ids: List[int],
        max_length: int = 64,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> str:
        """autoregressively decodes a translation from source token ids.

        Args:
            tokenizer: object with ``decode``, ``bos_id``, and ``eos_id``.
            src_ids: source token ids already prefixed with a language token.
            max_length: maximum number of target tokens to generate.
            temperature: softmax temperature; higher values increase randomness.
            top_k: if > 0, restrict sampling to the top-k tokens by logit.

        Returns:
            decoded target text (special tokens stripped by the tokenizer).
        """
        self.eval()
        device = next(self.parameters()).device
        eos_id = getattr(tokenizer, "eos_id", 0)
        bos_id = getattr(tokenizer, "bos_id", 1)

        src = torch.tensor([src_ids[: self.max_seq_len]], device=device)
        src_emb = self._embed(src)
        memory = self.encoder(src=src_emb)

        tokens = torch.tensor([[bos_id]], device=device)
        for _ in range(max_length):
            tgt_emb = self._embed(tokens[:, -self.max_seq_len :])
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                tgt_emb.size(1), device=device, dtype=torch.bool
            )
            decoded = self.decoder(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=causal_mask,
                tgt_is_causal=True,
            )
            logits = self.head(self.ln_f(decoded))[0, -1, :] / max(temperature, 1e-6)

            if top_k > 0:
                k = min(top_k, logits.size(-1))
                top_k_logits, top_k_indices = torch.topk(logits, k)
                probs = F.softmax(top_k_logits, dim=-1)
                next_token = top_k_indices[torch.multinomial(probs, 1)]
            else:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)

            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
            if next_token.item() == eos_id:
                break

        generated = tokens[0, 1:].cpu().tolist()
        if generated and generated[-1] == eos_id:
            generated = generated[:-1]
        return tokenizer.decode(generated)
