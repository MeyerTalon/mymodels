from collections.abc import Sequence

import torch
from torch import nn

from core.config import Config, load_config, require_int
from core.data import describe_loaders
from core.device import select_device
from core.paths import tokenizer_dir
from core.tokenizer import BPETokenizer
from core.training import (
    BatchResult,
    Trainer,
    TrainingConfig,
    parse_config_path,
)
from gpt.data import TokenBatch, create_dataloaders
from shakespeare.data import load_texts
from shakespeare_visualized.architecture import DecoderConfig, DecoderOnlyTransformer

PACKAGE = 'shakespeare'


class LanguageModelTrainer(Trainer[TokenBatch]):
    REPORTS_PERPLEXITY = True

    def compute_batch(self, batch: TokenBatch) -> BatchResult:
        input_ids, target_ids = (self.to_device(tensor) for tensor in batch)
        logits = self.model(input_ids)
        loss = self.criterion(
            logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1)
        )
        return BatchResult(loss=loss, example_count=input_ids.size(0))


def build_trainer(
    config: Config, *, package: str, texts: Sequence[str], device: torch.device
) -> LanguageModelTrainer:
    settings = TrainingConfig.from_config(config)
    decoder = DecoderConfig.from_config(config)
    tokenizer = BPETokenizer.train_or_load(
        texts,
        tokenizer_dir(package),
        vocab_size=require_int(config, 'vocab_size'),
        min_frequency=require_int(config, 'min_frequency'),
    )
    train_loader, val_loader = create_dataloaders(
        texts,
        tokenizer,
        block_size=decoder.max_seq_len,
        batch_size=settings.batch_size,
        val_fraction=settings.val_fraction,
        num_workers=settings.num_workers,
    )
    print(describe_loaders(train_loader, val_loader))
    return LanguageModelTrainer(
        settings=settings,
        config=config,
        model=DecoderOnlyTransformer(tokenizer.vocab_size, decoder),
        criterion=nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id),
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_extras={'tokenizer_vocab_size': tokenizer.vocab_size},
    )


def main() -> None:
    config = load_config(parse_config_path('Train a shakespeare language model'))
    texts = load_texts(config, TrainingConfig.from_config(config))
    build_trainer(config, package=PACKAGE, texts=texts, device=select_device()).train()


if __name__ == '__main__':
    main()
