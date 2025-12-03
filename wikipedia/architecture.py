"""Model architecture for decoder-only transformer.

Defines the DecoderOnlyTransformer class using PyTorch's native nn.TransformerEncoder
with causal masking for autoregressive text generation.
"""

import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderOnlyTransformer(nn.Module):
    """Decoder-only transformer for sentence completion.

    This model uses PyTorch's native ``nn.TransformerEncoder`` with a causal
    mask to implement a GPT-style decoder-only transformer for next-token
    prediction.
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
        """Initializes the decoder-only transformer.

        Args:
            vocab_size: Size of the token vocabulary.
            d_model: Dimensionality of the model / embeddings.
            n_heads: Number of attention heads per layer.
            n_layers: Number of encoder (decoder-style) layers.
            d_ff: Dimensionality of the feed-forward sub-layer.
            max_seq_len: Maximum supported sequence length.
            dropout: Dropout probability.
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # Token and position embeddings (using native nn.Embedding)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # Use PyTorch's built-in TransformerEncoder with a causal mask
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,  # (batch, seq, feature)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output projection
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Builds an upper-triangular causal attention mask.

        The resulting mask is suitable for ``nn.TransformerEncoder`` as
        ``mask``.

        Args:
            seq_len: Sequence length for which to build the mask.
            device: Device on which to allocate the mask.

        Returns:
            A tensor of shape (seq_len, seq_len) with zeros on and below
            the diagonal and ``-inf`` above the diagonal.
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        mask = mask.masked_fill(mask == 0, 0.0)
        return mask

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes logits over the vocabulary for each position.

        Args:
            x: Long tensor of token ids with shape (batch_size, seq_len).
            mask: Optional causal attention mask. If ``None``, a standard
                causal mask is created internally.

        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size).
        """
        device = x.device
        batch_size, seq_len = x.size()

        # Token embeddings
        x = self.token_embedding(x) * math.sqrt(self.d_model)  # (batch, seq, d_model)

        # Position embeddings (using native nn.Embedding)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(
            batch_size, -1
        )
        x = x + self.position_embedding(positions)  # (batch, seq, d_model)

        x = self.dropout(x)

        # Causal mask for self-attention
        if mask is None:
            mask = self._causal_mask(seq_len, device)  # (seq, seq)

        # nn.TransformerEncoder API: (src, mask)
        x = self.encoder(src=x, mask=mask)

        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def generate(
        self,
        tokenizer: Any,
        prompt: str | list[int],
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> str:
        """Generates a text continuation from a prompt.

        Args:
            tokenizer: Object implementing ``encode(str) -> List[int]`` and
                ``decode(List[int]) -> str``.
            prompt: Input prompt as a string or list of token ids.
            max_length: Maximum number of tokens to generate.
            temperature: Softmax temperature; higher values increase randomness.
            top_k: If > 0, restrict sampling to the top-k tokens by logit.

        Returns:
            Generated text as a string (including the original prompt).
        """
        self.eval()
        device = next(self.parameters()).device

        # Tokenize prompt
        if isinstance(prompt, str):
            tokens = tokenizer.encode(prompt)
        else:
            tokens = prompt

        tokens = torch.tensor([tokens], device=device)

        with torch.no_grad():
            for _ in range(max_length):
                # Get predictions
                logits = self.forward(tokens)
                logits = logits[0, -1, :] / temperature

                # Top-k sampling
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k)
                    probs = F.softmax(top_k_logits, dim=-1)
                    next_token = top_k_indices[torch.multinomial(probs, 1)]
                else:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)

                tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)

                # Stop if we hit end token (assuming 0 is padding, adjust as needed)
                if next_token.item() == 0:
                    break

        return tokenizer.decode(tokens[0].cpu().tolist())

