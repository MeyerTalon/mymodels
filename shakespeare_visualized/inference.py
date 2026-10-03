import argparse
import sys
from contextlib import nullcontext
from pathlib import Path

import torch

from core.checkpoint import find_checkpoint, load_checkpoint
from core.device import Precision, autocast, select_device
from core.paths import resolve_repo_path, tokenizer_dir
from core.tokenizer import BPETokenizer, TextTokenizer
from shakespeare_visualized.architecture import DecoderConfig, DecoderOnlyTransformer
from shakespeare_visualized.visualization import ActivationView

PACKAGE = 'shakespeare'
INFERENCE_PRECISION = Precision.FP16
INFERENCE_DROPOUT = 0.0
DEFAULT_MAX_NEW_TOKENS = 100
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_K = 50


def load_model(
    model_name: str, weights_dir: Path, tokenizer_path: Path, device: torch.device
) -> tuple[DecoderOnlyTransformer, BPETokenizer]:
    path = find_checkpoint(weights_dir, model_name)
    print(f'Loading model from {path}')
    checkpoint = load_checkpoint(path, device)
    settings = DecoderConfig.from_config(
        {**checkpoint.config, 'dropout': INFERENCE_DROPOUT}
    )
    model = DecoderOnlyTransformer(checkpoint.vocab_size(), settings).to(device)
    model.load_state_dict(checkpoint.model_state_dict)
    model.eval()
    return model, BPETokenizer.load(tokenizer_path)


def generate_text(
    model: DecoderOnlyTransformer,
    tokenizer: TextTokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    activation_view: ActivationView | None = None,
) -> str:
    """Returns only the continuation; the prompt's characters are sliced off the decoded text."""
    device = model.token_embedding.weight.device
    context = nullcontext() if activation_view is None else activation_view
    with context, autocast(device, INFERENCE_PRECISION):
        token_ids = model.generate(
            tokenizer.encode(prompt),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_id=tokenizer.eos_id,
        )
    return tokenizer.decode(token_ids)[len(prompt) :]


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate text with a trained shakespeare model'
    )
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--max_length', type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument('--top_k', type=int, default=DEFAULT_TOP_K)
    parser.add_argument('--weights_dir', default=f'{PACKAGE}/weights')
    parser.add_argument('--show_activations', action='store_true')
    args = parser.parse_args()
    prompt: str = args.prompt
    try:
        model, tokenizer = load_model(
            args.model_name,
            resolve_repo_path(args.weights_dir),
            tokenizer_dir(PACKAGE),
            select_device(),
        )
    except (FileNotFoundError, TypeError, ValueError) as error:
        sys.exit(f'Error: {error}')
    print(f'Prompt: {prompt}')
    print('Generating...')
    view = (
        ActivationView(model, tokenizer, live=True) if args.show_activations else None
    )
    generated = generate_text(
        model,
        tokenizer,
        prompt,
        max_new_tokens=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        activation_view=view,
    )
    print(f'\nGenerated text:\n{prompt}{generated}')
    if view is not None:
        view.block()


if __name__ == '__main__':
    main()
