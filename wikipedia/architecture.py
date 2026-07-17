"""model architecture for decoder-only transformer.

defines the DecoderOnlyTransformer class using PyTorch's native nn.TransformerEncoder
with causal masking for autoregressive text generation. the stack is configured
GPT-style: pre-layer-norm, GELU activations, weight-tied input/output embeddings,
and the fused scaled-dot-product-attention causal fast path.
"""

from typing import Any, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderOnlyTransformer(nn.Module):
    """decoder-only transformer for sentence completion.

    this model uses PyTorch's native ``nn.TransformerEncoder`` with a causal
    mask to implement a GPT-style decoder-only transformer for next-token
    prediction. it is pre-norm (``norm_first=True``), uses GELU feed-forward
    activations, and ties the token-embedding and output-projection weights.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ) -> None:
        """initializes the decoder-only transformer.

        Args:
            vocab_size: size of the token vocabulary.
            d_model: dimensionality of the model / embeddings.
            n_heads: number of attention heads per layer.
            n_layers: number of encoder (decoder-style) layers.
            d_ff: dimensionality of the feed-forward sub-layer.
            max_seq_len: maximum supported sequence length.
            dropout: dropout probability.
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # token and position embeddings (learned, native nn.Embedding)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # native pre-norm transformer stack with GELU feed-forward. pre-norm
        # (norm_first=True) trains more stably than the post-norm default.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            norm_first=True,
            batch_first=True,  # (batch, seq, feature)
        )
        # enable_nested_tensor is incompatible with norm_first; disable it to
        # keep the causal path active and silence the runtime warning.
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )

        # final norm and vocabulary projection
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        self.apply(self._init_weights)

        # weight tying: share the token-embedding matrix with the output head.
        # done after init so both refer to the same initialized tensor.
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

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """computes logits over the vocabulary for each position.

        Args:
            x: long tensor of token ids with shape (batch_size, seq_len).
            mask: optional causal attention mask of shape (seq_len, seq_len). if
                ``None``, a standard causal mask is created internally.

        Returns:
            logits tensor of shape (batch_size, seq_len, vocab_size).
        """
        device = x.device
        batch_size, seq_len = x.size()

        # token + learned positional embeddings  # (batch, seq, d_model)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(
            batch_size, -1
        )
        x = self.token_embedding(x) + self.position_embedding(positions)
        x = self.dropout(x)

        # causal self-attention. passing is_causal=True lets nn.TransformerEncoder
        # take the fused scaled-dot-product-attention path.
        if mask is None:
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)
        x = self.encoder(src=x, mask=mask, is_causal=True)  # (batch, seq, d_model)

        x = self.ln_f(x)
        logits = self.head(x)  # (batch, seq, vocab_size)
        return logits

    @torch.no_grad()
    def generate(
        self,
        tokenizer: Any,
        prompt: Union[str, List[int]],
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> str:
        """generates a text continuation from a prompt.

        Args:
            tokenizer: object implementing ``encode(str) -> List[int]`` and
                ``decode(List[int]) -> str``; an ``eos_id`` attribute is used as
                the stop token when present.
            prompt: input prompt as a string or list of token ids.
            max_length: maximum number of tokens to generate.
            temperature: softmax temperature; higher values increase randomness.
            top_k: if > 0, restrict sampling to the top-k tokens by logit.

        Returns:
            generated text as a string (including the original prompt).
        """
        self.eval()
        device = next(self.parameters()).device

        if isinstance(prompt, str):
            token_ids = tokenizer.encode(prompt)
        else:
            token_ids = list(prompt)

        # stop token: prefer the tokenizer's eos, fall back to padding id 0
        eos_id = getattr(tokenizer, "eos_id", 0)

        tokens = torch.tensor([token_ids], device=device)

        for _ in range(max_length):
            # condition on at most the last max_seq_len tokens
            logits = self.forward(tokens[:, -self.max_seq_len :])
            logits = logits[0, -1, :] / max(temperature, 1e-6)

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

        return tokenizer.decode(tokens[0].cpu().tolist())
