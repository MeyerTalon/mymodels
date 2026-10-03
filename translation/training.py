import torch
from torch import nn

from core.config import Config, load_config, require_int, require_str_list
from core.data import describe_loaders
from core.device import select_device
from core.paths import tokenizer_dir
from core.snapshot import Record
from core.training import BatchResult, Trainer, TrainingConfig, parse_config_path
from translation.architecture import EncoderDecoderConfig, EncoderDecoderTransformer
from translation.data import PairBatch, create_dataloaders, load_pairs
from translation.tokenizer import TranslationTokenizer

PACKAGE = 'translation'


class TranslationTrainer(Trainer[PairBatch]):
    def compute_batch(self, batch: PairBatch) -> BatchResult:
        src, tgt_input, tgt_labels, src_pad_mask, tgt_pad_mask = (
            self.to_device(tensor) for tensor in batch
        )
        logits = self.model(
            src,
            tgt_input,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
        )
        loss = self.criterion(
            logits.reshape(-1, logits.size(-1)), tgt_labels.reshape(-1)
        )
        return BatchResult(loss=loss, example_count=src.size(0))


def build_trainer(
    config: Config, pairs: list[Record], device: torch.device
) -> TranslationTrainer:
    settings = TrainingConfig.from_config(config)
    architecture = EncoderDecoderConfig.from_config(config)
    tokenizer = TranslationTokenizer.train_or_load(
        [pair['src'] for pair in pairs] + [pair['tgt'] for pair in pairs],
        tokenizer_dir(PACKAGE),
        vocab_size=require_int(config, 'vocab_size'),
        min_frequency=require_int(config, 'min_frequency'),
        languages=require_str_list(config, 'languages'),
    )
    train_loader, val_loader = create_dataloaders(
        pairs,
        tokenizer,
        max_seq_len=architecture.max_seq_len,
        batch_size=settings.batch_size,
        val_fraction=settings.val_fraction,
        num_workers=settings.num_workers,
    )
    print(describe_loaders(train_loader, val_loader))
    return TranslationTrainer(
        settings=settings,
        config=config,
        model=EncoderDecoderTransformer(tokenizer.vocab_size, architecture),
        criterion=nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id),
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_extras={'tokenizer_vocab_size': tokenizer.vocab_size},
    )


def main() -> None:
    config = load_config(parse_config_path('Train a translation model'))
    pairs = load_pairs(config, TrainingConfig.from_config(config))
    build_trainer(config, pairs, select_device()).train()


if __name__ == '__main__':
    main()
