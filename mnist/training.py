from typing import TypeVar

import torch
from torch import nn
from torch.utils.data import Dataset

from core.config import Config, load_config, require_int
from core.data import describe_loaders
from core.device import select_device
from core.training import BatchResult, Trainer, TrainingConfig, parse_config_path
from mnist.architecture import CnnConfig, MnistCNN
from mnist.data import ImageBatch, create_dataloaders, load_mnist, split_train_val

ItemT = TypeVar('ItemT')


class ClassifierTrainer(Trainer[ImageBatch]):
    def compute_batch(self, batch: ImageBatch) -> BatchResult:
        images, labels = (self.to_device(tensor) for tensor in batch)
        logits = self.model(images)
        return BatchResult(
            loss=self.criterion(logits, labels),
            example_count=labels.size(0),
            correct_count=int((logits.argmax(dim=-1) == labels).sum().item()),
        )


def build_trainer(
    config: Config,
    train_dataset: Dataset[ItemT],
    val_dataset: Dataset[ItemT] | None,
    device: torch.device,
) -> ClassifierTrainer:
    settings = TrainingConfig.from_config(config)
    train_loader, val_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=settings.batch_size,
        num_workers=settings.num_workers,
    )
    print(describe_loaders(train_loader, val_loader))
    return ClassifierTrainer(
        settings=settings,
        config=config,
        model=MnistCNN(CnnConfig.from_config(config)),
        criterion=nn.CrossEntropyLoss(),
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_extras={},
    )


def main() -> None:
    config = load_config(parse_config_path('Train an MNIST classifier'))
    settings = TrainingConfig.from_config(config)
    train_full = load_mnist(
        settings.data_dir, train=True, cache_only=settings.dataset_cache_only
    )
    train_dataset, val_dataset = split_train_val(
        train_full,
        val_fraction=settings.val_fraction,
        seed=require_int(config, 'dataset_seed'),
    )
    print(f'Loaded MNIST from {settings.data_dir}: {len(train_full)} examples')
    build_trainer(config, train_dataset, val_dataset, select_device()).train()


if __name__ == '__main__':
    main()
