"""inference script for generating text from trained models.

provides a command-line interface for loading trained checkpoints and generating
text continuations from prompts. supports configurable sampling parameters (temperature, top-k).
"""

import argparse
import os
import sys
from typing import Optional, Tuple

import torch

# allow running as a script from the repo root: python wikipedia/inference.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wikipedia.architecture import DecoderOnlyTransformer
from wikipedia.tokenizer import WikipediaBPETokenizer
from wikipedia.utils import TOKENIZER_DIR, resolve_repo_path, select_device


def load_model(
    model_name: str,
    weights_dir: str = "wikipedia/weights",
    device: Optional[torch.device] = None,
    tokenizer_dir: str = TOKENIZER_DIR,
) -> Tuple[DecoderOnlyTransformer, WikipediaBPETokenizer, torch.device]:
    """loads a trained model and tokenizer from a checkpoint.

    Args:
        model_name: base name of the model checkpoint files.
        weights_dir: directory containing model weights.
        device: explicit device to load the model on. if ``None``, the best
            available device is selected (MPS, then CUDA, then CPU).
        tokenizer_dir: directory containing the trained tokenizer files.

    Returns:
        a tuple ``(model, tokenizer, device)`` where:

        * ``model`` is a ``DecoderOnlyTransformer`` in evaluation mode.
        * ``tokenizer`` is the associated ``WikipediaBPETokenizer``.
        * ``device`` is the torch device used.

    Raises:
        FileNotFoundError: if no checkpoint file is found.
    """
    if device is None:
        device = select_device()

    weights_dir = resolve_repo_path(weights_dir)

    # try to load best model first, then latest
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

    # get config from checkpoint
    config = checkpoint.get("config", {})
    vocab_size = checkpoint.get("tokenizer_vocab_size", 100)

    # load the pre-trained ByteLevel BPE tokenizer
    tokenizer = WikipediaBPETokenizer.load(tokenizer_dir)

    # initialize model
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=config.get("d_model", 512),
        n_heads=config.get("n_heads", 8),
        n_layers=config.get("n_layers", 6),
        d_ff=config.get("d_ff", 2048),
        max_seq_len=config.get("max_seq_len", 512),
        dropout=0.0,  # no dropout during inference
    ).to(device)

    # load weights
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
) -> str:
    """generates a text continuation from a prompt.

    thin wrapper around ``DecoderOnlyTransformer.generate`` that returns only
    the continuation, without the original prompt.

    Args:
        model: trained ``DecoderOnlyTransformer`` in eval mode.
        tokenizer: corresponding ``WikipediaBPETokenizer``.
        prompt: prompt string to be continued.
        max_length: maximum number of tokens to generate.
        temperature: sampling temperature (higher is more random).
        top_k: top-k sampling cutoff (0 to disable).

    Returns:
        the generated continuation text (excluding the original prompt).
    """
    generated_text = model.generate(
        tokenizer,
        prompt,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
    )
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
        model, tokenizer, _ = load_model(args.model_name, args.weights_dir)

        print(f"Prompt: {args.prompt}")
        print("Generating...")

        generated = generate_text(
            model,
            tokenizer,
            args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
        )

        print(f"\nGenerated text:\n{args.prompt}{generated}")

    except Exception as exc:  # pragma: no cover - CLI guard
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

