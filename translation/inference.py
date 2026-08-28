"""inference script for translating text with a trained seq2seq model.

loads a checkpoint and generates a translation for a prompt given source and
target language codes. sampling matches the language-model packages
(temperature / top-k); greedy decoding is temperature→0-ish via a tiny
temperature or top_k=1.
"""

import argparse
import contextlib
import os
import sys
from typing import Optional, Tuple

import torch

from translation.architecture import EncoderDecoderTransformer
from translation.tokenizer import DEFAULT_LANGUAGES, TranslationBPETokenizer
from translation.utils import (
    TOKENIZER_DIR,
    resolve_repo_path,
    select_autocast_dtype,
    select_device,
)


def load_model(
    model_name: str,
    weights_dir: str = "translation/weights",
    device: Optional[torch.device] = None,
    tokenizer_dir: str = TOKENIZER_DIR,
) -> Tuple[EncoderDecoderTransformer, TranslationBPETokenizer, torch.device]:
    """loads a trained model and tokenizer from a checkpoint.

    Args:
        model_name: base name of the model checkpoint files.
        weights_dir: directory containing model weights.
        device: explicit device to load the model on. if ``None``, the best
            available device is selected (MPS, then CUDA, then CPU).
        tokenizer_dir: directory containing the trained tokenizer files.

    Returns:
        a tuple ``(model, tokenizer, device)``.

    Raises:
        FileNotFoundError: if no checkpoint file is found.
    """
    if device is None:
        device = select_device()

    weights_dir = resolve_repo_path(weights_dir)

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
    config = checkpoint.get("config", {})
    vocab_size = checkpoint.get("tokenizer_vocab_size", 100)
    languages = config.get("languages", DEFAULT_LANGUAGES)

    tokenizer = TranslationBPETokenizer.load(tokenizer_dir, languages=languages)

    model = EncoderDecoderTransformer(
        vocab_size=vocab_size,
        d_model=config.get("d_model", 256),
        n_heads=config.get("n_heads", 4),
        n_encoder_layers=config.get("n_encoder_layers", 3),
        n_decoder_layers=config.get("n_decoder_layers", 3),
        d_ff=config.get("d_ff", 1024),
        max_seq_len=config.get("max_seq_len", 128),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, tokenizer, device


def translate(
    model: EncoderDecoderTransformer,
    tokenizer: TranslationBPETokenizer,
    prompt: str,
    target_lang: str,
    max_length: int = 64,
    temperature: float = 1.0,
    top_k: int = 50,
) -> str:
    """translates ``prompt`` into ``target_lang``.

    Args:
        model: trained encoder-decoder in eval mode.
        tokenizer: corresponding joint BPE tokenizer.
        prompt: source sentence.
        target_lang: language code to translate into (e.g. ``"es"``).
        max_length: maximum number of target tokens to generate.
        temperature: sampling temperature (higher is more random).
        top_k: top-k sampling cutoff (0 to disable; 1 is greedy).

    Returns:
        generated translation text.
    """
    src_ids = tokenizer.encode_source(prompt, target_lang)
    device = next(model.parameters()).device
    amp_dtype = select_autocast_dtype(device, "fp16")
    autocast = (
        torch.autocast(device_type=device.type, dtype=amp_dtype)
        if amp_dtype is not None
        else contextlib.nullcontext()
    )
    with autocast:
        return model.generate(
            tokenizer,
            src_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
        )


def main() -> None:
    """CLI entry point for translating text with a trained model.

    Command-line arguments (``argparse``):
        --model_name: checkpoint prefix in the weights dir to load (required).
        --prompt: source sentence to translate (required).
        --source_lang: source language code, e.g. ``en`` (required; recorded
            for the caller, the model is steered by ``--target_lang``).
        --target_lang: target language code, e.g. ``es`` (required).
        --max_length: maximum number of tokens to generate (default 64).
        --temperature: sampling temperature, higher is more random (default 1.0).
        --top_k: top-k sampling cutoff, 0 to disable, 1 for greedy (default 50).
        --weights_dir: directory containing model weights
            (default ``translation/weights``).
    """
    parser = argparse.ArgumentParser(
        description="Translate text using a trained multilingual model"
    )
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
        help="Source sentence to translate",
    )
    parser.add_argument(
        "--source_lang",
        type=str,
        required=True,
        help="Source language code (e.g. en)",
    )
    parser.add_argument(
        "--target_lang",
        type=str,
        required=True,
        help="Target language code (e.g. es)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=64,
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
        help="Top-k sampling cutoff (0 to disable, 1 for greedy)",
    )
    parser.add_argument(
        "--weights_dir",
        type=str,
        default="translation/weights",
        help="Directory containing model weights",
    )

    args = parser.parse_args()

    try:
        model, tokenizer, _ = load_model(args.model_name, args.weights_dir)
        # validate both language codes against the tokenizer even though only
        # the target-language prefix is prepended to the source.
        tokenizer.lang_id(args.source_lang)
        tokenizer.lang_id(args.target_lang)

        print(f"Source ({args.source_lang}): {args.prompt}")
        print(f"Translating to {args.target_lang}...")

        generated = translate(
            model,
            tokenizer,
            args.prompt,
            target_lang=args.target_lang,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        print(f"\nTranslation ({args.target_lang}):\n{generated}")

    except Exception as exc:  # pragma: no cover - CLI guard
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
