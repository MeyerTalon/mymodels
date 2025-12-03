"""Inference script for generating text from trained models.

Provides a command-line interface for loading trained checkpoints and generating
text continuations from prompts. Supports configurable sampling parameters (temperature, top-k).
"""

import argparse
import os
import sys
from typing import Tuple, Optional

import torch

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from architecture import DecoderOnlyTransformer
from tokenizer import WikipediaBPETokenizer


def load_model(
    model_name: str,
    weights_dir: str = "wikipedia/weights",
    device: Optional[torch.device] = None,
) -> Tuple[DecoderOnlyTransformer, WikipediaBPETokenizer, torch.device]:
    """Loads a trained model and tokenizer from a checkpoint.

    Args:
        model_name: Base name of the model checkpoint files.
        weights_dir: Directory containing model weights.
        device: Explicit device to load the model on. If ``None``, chooses
            CUDA if available, otherwise CPU.

    Returns:
        A tuple ``(model, tokenizer, device)`` where:

        * ``model`` is a ``DecoderOnlyTransformer`` in evaluation mode.
        * ``tokenizer`` is the associated ``SimpleTokenizer``.
        * ``device`` is the torch device used.

    Raises:
        FileNotFoundError: If no checkpoint file is found.
    """
    if device is None:
        # Prefer Apple Silicon (MPS), then CUDA, then CPU
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    # Make path absolute if relative
    if not os.path.isabs(weights_dir):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        weights_dir = os.path.join(base_dir, weights_dir)

    # Try to load best model first, then latest
    checkpoint_paths = [
        os.path.join(weights_dir, f"{model_name}_best.pt"),
        os.path.join(weights_dir, f"{model_name}_latest.pt"),
        os.path.join(weights_dir, f"{model_name}.pt"),
    ]

    checkpoint_path: Optional[str] = None
    for path in checkpoint_paths:
        if os.path.exists(path):
            checkpoint_path = path
            break

    if checkpoint_path is None:
        raise FileNotFoundError(
            f"Model checkpoint not found. Tried: {checkpoint_paths}"
        )

    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config from checkpoint
    config = checkpoint.get("config", {})
    vocab_size = checkpoint.get("tokenizer_vocab_size", 100)
    
    # Initialize tokenizer (load pre-trained ByteLevel BPE)
    tokenizer_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "tokenizer_files"
    )
    tokenizer = WikipediaBPETokenizer.load(tokenizer_dir)

    # Initialize model
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=config.get("d_model", 512),
        n_heads=config.get("n_heads", 8),
        n_layers=config.get("n_layers", 6),
        d_ff=config.get("d_ff", 2048),
        max_seq_len=config.get("max_seq_len", 512),
        dropout=0.0,  # No dropout during inference
    ).to(device)

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, tokenizer, device


def generate_text(
    model: DecoderOnlyTransformer,
    tokenizer: WikipediaBPETokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    device: Optional[torch.device] = None,
) -> str:
    """Generates a text continuation from a prompt.

    Args:
        model: Trained ``DecoderOnlyTransformer`` in eval mode.
        tokenizer: Corresponding ``SimpleTokenizer``.
        prompt: Prompt string to be continued.
        max_length: Maximum number of tokens to generate.
        temperature: Sampling temperature (higher is more random).
        top_k: Top-k sampling cutoff (0 to disable).
        device: Device to use. If ``None``, inferred from model parameters.

    Returns:
        The generated continuation text (excluding the original prompt).
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Tokenize prompt
    tokens = tokenizer.encode(prompt)
    tokens_tensor = torch.tensor([tokens], device=device)

    generated_tokens = tokens_tensor.clone()

    with torch.no_grad():
        for _ in range(max_length):
            # Get predictions
            logits = model(generated_tokens)
            logits = logits[0, -1, :] / temperature

            # Top-k sampling
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(
                    logits, min(top_k, logits.size(-1))
                )
                probs = torch.nn.functional.softmax(top_k_logits, dim=-1)
                next_token = top_k_indices[torch.multinomial(probs, 1)]
            else:
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)

            # Append new token (keep tensor 2D: [batch, seq_len])
            generated_tokens = torch.cat(
                [generated_tokens, next_token.unsqueeze(0)], dim=1
            )

            # Stop if we hit padding token (0) or if sequence gets too long
            if next_token.item() == 0:
                break

    # Decode generated tokens
    generated_text = tokenizer.decode(generated_tokens[0].cpu().tolist())

    # Return only the generated part (after the prompt)
    return generated_text[len(prompt) :] if len(generated_text) > len(prompt) else ""


def main() -> None:
    """CLI entry point for generating text from a trained model."""
    parser = argparse.ArgumentParser(description="Generate text using a trained model")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model to load (used as checkpoint prefix)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Beginning of sentence to complete",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=100,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (higher = more random)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling cutoff (0 to disable)",
    )
    parser.add_argument(
        "--weights_dir",
        type=str,
        default="wikipedia/weights",
        help="Directory containing model weights",
    )

    args = parser.parse_args()

    try:
        model, tokenizer, device = load_model(args.model_name, args.weights_dir)

        print(f"Prompt: {args.prompt}")
        print("Generating...")

        generated = generate_text(
            model,
            tokenizer,
            args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device,
        )

        print(f"\nGenerated text:\n{args.prompt}{generated}")

    except Exception as exc:  # pragma: no cover - CLI guard
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

