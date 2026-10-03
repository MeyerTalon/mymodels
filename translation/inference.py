import argparse
import sys
from pathlib import Path

import torch

from core.checkpoint import find_checkpoint, load_checkpoint
from core.config import require_str_list
from core.device import Precision, autocast, select_device
from core.paths import resolve_repo_path, tokenizer_dir
from translation.architecture import EncoderDecoderConfig, EncoderDecoderTransformer
from translation.tokenizer import TranslationTokenizer

PACKAGE = 'translation'
INFERENCE_PRECISION = Precision.FP16
INFERENCE_DROPOUT = 0.0
DEFAULT_MAX_NEW_TOKENS = 64
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_K = 50


def load_model(
    model_name: str, weights_dir: Path, tokenizer_path: Path, device: torch.device
) -> tuple[EncoderDecoderTransformer, TranslationTokenizer]:
    path = find_checkpoint(weights_dir, model_name)
    print(f'Loading model from {path}')
    checkpoint = load_checkpoint(path, device)
    settings = EncoderDecoderConfig.from_config(
        {**checkpoint.config, 'dropout': INFERENCE_DROPOUT}
    )
    model = EncoderDecoderTransformer(checkpoint.vocab_size(), settings).to(device)
    model.load_state_dict(checkpoint.model_state_dict)
    model.eval()
    languages = require_str_list(checkpoint.config, 'languages')
    return model, TranslationTokenizer.load(tokenizer_path, languages)


def translate(
    model: EncoderDecoderTransformer,
    tokenizer: TranslationTokenizer,
    prompt: str,
    *,
    target_lang: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> str:
    device = model.token_embedding.weight.device
    with autocast(device, INFERENCE_PRECISION):
        target_ids = model.generate(
            tokenizer.encode_source(prompt, target_lang),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            bos_id=tokenizer.bos_id,
            eos_id=tokenizer.eos_id,
        )
    return tokenizer.decode(target_ids)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Translate text using a trained multilingual model'
    )
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--source_lang', required=True)
    parser.add_argument('--target_lang', required=True)
    parser.add_argument('--max_length', type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument('--top_k', type=int, default=DEFAULT_TOP_K)
    parser.add_argument('--weights_dir', default=f'{PACKAGE}/weights')
    args = parser.parse_args()
    try:
        model, tokenizer = load_model(
            args.model_name,
            resolve_repo_path(args.weights_dir),
            tokenizer_dir(PACKAGE),
            select_device(),
        )
        tokenizer.lang_id(args.source_lang)
        tokenizer.lang_id(args.target_lang)
    except (FileNotFoundError, TypeError, ValueError) as error:
        sys.exit(f'Error: {error}')
    print(f'Source ({args.source_lang}): {args.prompt}')
    print(f'Translating to {args.target_lang}...')
    translation = translate(
        model,
        tokenizer,
        args.prompt,
        target_lang=args.target_lang,
        max_new_tokens=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(f'\nTranslation ({args.target_lang}):\n{translation}')


if __name__ == '__main__':
    main()
