import argparse
import math
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Generic, Protocol, TypeVar

import torch
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from core.checkpoint import BEST_SUFFIX, EPOCH_SUFFIX, LATEST_SUFFIX, checkpoint_path
from core.config import (
    Config,
    require_bool,
    require_float,
    require_int,
    require_str,
)
from core.device import Precision, autocast, autocast_dtype
from core.paths import resolve_repo_path
from core.reporting import EpochMetrics, TrainingReporter

ADAM_BETAS = (0.9, 0.95)
MIN_DECAYED_PARAM_DIMS = 2
MAX_PERPLEXITY_LOSS = 20.0

BatchT = TypeVar('BatchT')
BatchT_co = TypeVar('BatchT_co', covariant=True)


class BatchLoader(Protocol[BatchT_co]):
    def __iter__(self) -> Iterator[BatchT_co]: ...

    def __len__(self) -> int: ...


@dataclass(frozen=True)
class TrainingConfig:
    model_name: str
    num_epochs: int
    batch_size: int
    grad_accum_steps: int
    learning_rate: float
    min_lr: float
    weight_decay: float
    warmup_epochs: int
    max_grad_norm: float
    precision: Precision
    val_fraction: float
    save_every: int
    log_interval: int
    num_workers: int
    dataset_cache_only: bool
    data_dir: Path
    weights_dir: Path
    reports_dir: Path

    @classmethod
    def from_config(cls, config: Config) -> 'TrainingConfig':
        settings = cls(
            model_name=require_str(config, 'model_name'),
            num_epochs=require_int(config, 'num_epochs'),
            batch_size=require_int(config, 'batch_size'),
            grad_accum_steps=require_int(config, 'grad_accum_steps'),
            learning_rate=require_float(config, 'learning_rate'),
            min_lr=require_float(config, 'min_lr'),
            weight_decay=require_float(config, 'weight_decay'),
            warmup_epochs=require_int(config, 'warmup_epochs'),
            max_grad_norm=require_float(config, 'max_grad_norm'),
            precision=Precision(require_str(config, 'precision')),
            val_fraction=require_float(config, 'val_fraction'),
            save_every=require_int(config, 'save_every'),
            log_interval=require_int(config, 'log_interval'),
            num_workers=require_int(config, 'num_workers'),
            dataset_cache_only=require_bool(config, 'dataset_cache_only'),
            data_dir=resolve_repo_path(require_str(config, 'data_dir')),
            weights_dir=resolve_repo_path(require_str(config, 'weights_dir')),
            reports_dir=resolve_repo_path(require_str(config, 'reports_dir')),
        )
        if settings.learning_rate <= 0:
            raise ValueError('learning_rate must be positive')
        if settings.grad_accum_steps < 1:
            raise ValueError('grad_accum_steps must be at least 1')
        return settings

    @property
    def min_lr_ratio(self) -> float:
        return self.min_lr / self.learning_rate


@dataclass(frozen=True)
class BatchResult:
    """`correct_count` is `None` for models that don't track accuracy."""

    loss: torch.Tensor
    example_count: int
    correct_count: int | None = None


class _EpochTally:
    def __init__(self) -> None:
        self.loss_sum = 0.0
        self.batch_count = 0
        self.example_count = 0
        self.correct_count: int | None = None

    def add(self, result: BatchResult) -> None:
        self.loss_sum += result.loss.item()
        self.batch_count += 1
        self.example_count += result.example_count
        if result.correct_count is not None:
            self.correct_count = (self.correct_count or 0) + result.correct_count

    def metrics(self) -> EpochMetrics:
        loss = self.loss_sum / max(self.batch_count, 1)
        if self.correct_count is None:
            return EpochMetrics(loss)
        return EpochMetrics(loss, self.correct_count / max(self.example_count, 1))


class Trainer(ABC, Generic[BatchT]):
    REPORTS_PERPLEXITY: ClassVar[bool] = False

    def __init__(
        self,
        *,
        settings: TrainingConfig,
        config: Config,
        model: nn.Module,
        criterion: nn.Module,
        train_loader: BatchLoader[BatchT],
        val_loader: BatchLoader[BatchT] | None,
        device: torch.device,
        checkpoint_extras: Mapping[str, object],
    ) -> None:
        self.settings = settings
        self.config = dict(config)
        self.device = device
        self.model = model.to(device)
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_extras = dict(checkpoint_extras)
        self.optimizer = AdamW(
            decay_param_groups(self.model, settings.weight_decay),
            lr=settings.learning_rate,
            betas=ADAM_BETAS,
        )
        self.scheduler = warmup_cosine_schedule(
            self.optimizer,
            warmup_epochs=settings.warmup_epochs,
            num_epochs=settings.num_epochs,
            min_lr_ratio=settings.min_lr_ratio,
        )
        autocast_on = autocast_dtype(device, settings.precision) is not None
        self.scaler = torch.amp.GradScaler(
            enabled=autocast_on and device.type == 'cuda'
        )
        self.reporter = TrainingReporter(settings.model_name, settings.reports_dir)
        self.best_loss = math.inf
        print(f'Using device: {device}')
        print(
            f'Precision: {settings.precision} '
            f'(autocast={"on" if autocast_on else "off"})'
        )
        print(f'Model initialized with {count_parameters(self.model):,} parameters')

    @abstractmethod
    def compute_batch(self, batch: BatchT) -> BatchResult:
        """Runs under autocast; move tensors with `to_device` first."""

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self.device, non_blocking=True)

    def train_epoch(self, epoch: int) -> EpochMetrics:
        self.model.train()
        tally = _EpochTally()
        self.optimizer.zero_grad(set_to_none=True)
        progress = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for step, batch in enumerate(progress, start=1):
            with autocast(self.device, self.settings.precision):
                result = self.compute_batch(batch)
            torch.autograd.backward(
                self.scaler.scale(result.loss / self.settings.grad_accum_steps)
            )
            if step % self.settings.grad_accum_steps == 0:
                self._optimizer_step()
            tally.add(result)
            if (step - 1) % self.settings.log_interval == 0:
                progress.set_postfix(self._progress_fields(result, tally))
        return tally.metrics()

    def _optimizer_step(self) -> None:
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), self.settings.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

    def _progress_fields(
        self, result: BatchResult, tally: _EpochTally
    ) -> dict[str, str]:
        fields = {'loss': f'{result.loss.item():.4f}'}
        accuracy = tally.metrics().accuracy
        if accuracy is not None:
            fields['acc'] = f'{accuracy:.3f}'
        fields['lr'] = f'{self.optimizer.param_groups[0]["lr"]:.2e}'
        return fields

    @torch.no_grad()
    def evaluate(self) -> EpochMetrics | None:
        if self.val_loader is None:
            return None
        self.model.eval()
        tally = _EpochTally()
        for batch in self.val_loader:
            with autocast(self.device, self.settings.precision):
                tally.add(self.compute_batch(batch))
        return tally.metrics()

    def save_checkpoint(
        self, epoch: int, monitored: EpochMetrics, *, is_best: bool
    ) -> None:
        weights_dir = self.settings.weights_dir
        weights_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': monitored.loss,
            'accuracy': monitored.accuracy,
            'config': self.config,
            **self.checkpoint_extras,
        }
        name = self.settings.model_name
        torch.save(checkpoint, checkpoint_path(weights_dir, name, LATEST_SUFFIX))
        if is_best:
            best_path = checkpoint_path(weights_dir, name, BEST_SUFFIX)
            torch.save(checkpoint, best_path)
            print(f'Saved best model to {best_path}')
        torch.save(
            checkpoint, checkpoint_path(weights_dir, name, f'{EPOCH_SUFFIX}{epoch}')
        )

    def train(self) -> None:
        num_epochs = self.settings.num_epochs
        print(f'Starting training for {num_epochs} epochs...')
        for epoch in range(1, num_epochs + 1):
            train = self.train_epoch(epoch)
            self.scheduler.step()
            val = self.evaluate()
            monitored = val if val is not None else train
            print(f'Epoch {epoch}/{num_epochs} - {self._summary(train, val)}')
            self.reporter.log_epoch(epoch, train, val)
            is_best = monitored.loss < self.best_loss
            if is_best:
                self.best_loss = monitored.loss
            if epoch % self.settings.save_every == 0 or is_best:
                self.save_checkpoint(epoch, monitored, is_best=is_best)
        print('Training completed!')
        print(f'Best loss: {self.best_loss:.4f}')
        print(f'Loss report saved to {self.reporter.report_path}')

    def _summary(self, train: EpochMetrics, val: EpochMetrics | None) -> str:
        parts = [f'Train: {_format_metrics(train)}']
        if val is not None:
            parts.append(f'Val: {_format_metrics(val)}')
        if self.REPORTS_PERPLEXITY:
            monitored_loss = (val if val is not None else train).loss
            perplexity = math.exp(min(monitored_loss, MAX_PERPLEXITY_LOSS))
            parts.append(f'Perplexity: {perplexity:.2f}')
        return ' - '.join(parts)


def _format_metrics(metrics: EpochMetrics) -> str:
    if metrics.accuracy is None:
        return f'{metrics.loss:.4f}'
    return f'{metrics.loss:.4f} ({metrics.accuracy:.3f} acc)'


def decay_param_groups(
    model: nn.Module, weight_decay: float
) -> list[dict[str, object]]:
    """Only weight matrices decay; biases, norm gains, and 1-d params don't."""
    trainable = [param for param in model.parameters() if param.requires_grad]
    return [
        {
            'params': [p for p in trainable if p.dim() >= MIN_DECAYED_PARAM_DIMS],
            'weight_decay': weight_decay,
        },
        {
            'params': [p for p in trainable if p.dim() < MIN_DECAYED_PARAM_DIMS],
            'weight_decay': 0.0,
        },
    ]


def warmup_cosine_schedule(
    optimizer: Optimizer,
    *,
    warmup_epochs: int,
    num_epochs: int,
    min_lr_ratio: float,
) -> LambdaLR:
    """Stepped once per epoch: linear warmup, then cosine decay to `min_lr_ratio`."""

    def lr_multiplier(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / (warmup_epochs + 1)
        progress = (epoch - warmup_epochs) / max(num_epochs - warmup_epochs, 1)
        cosine = (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0))) / 2
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_multiplier)


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def parse_config_path(description: str) -> Path:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('config_path', type=Path)
    args = parser.parse_args()
    config_path: Path = args.config_path
    return resolve_repo_path(config_path)
