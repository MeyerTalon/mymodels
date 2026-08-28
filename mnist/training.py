"""training pipeline for the MNIST convolutional classifier.

provides the Trainer class which handles the complete training workflow: loading
configuration, setting up dataloaders, running mixed-precision training loops
with gradient accumulation, validating (loss + accuracy), and saving
checkpoints. can be run as a standalone script with a config file path.
"""

import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from mnist.architecture import MnistCNN
from mnist.data import create_dataloaders, load_mnist_datasets
from mnist.reporting import TrainingReporter
from mnist.utils import resolve_repo_path, select_autocast_dtype, select_device


class Trainer:
    """training class for the MNIST CNN classifier."""

    def __init__(self, config_path: str) -> None:
        """initializes the trainer from a YAML configuration file.

        Args:
            config_path: path to the YAML configuration file.
        """
        with open(config_path, "r", encoding="utf-8") as f:
            self.config: Dict[str, Any] = yaml.safe_load(f)

        self.device = select_device()
        print(f"Using device: {self.device}")

        precision = self.config.get(
            "precision", "fp16" if self.device.type != "cpu" else "fp32"
        )
        self.amp_dtype = select_autocast_dtype(self.device, precision)
        self.use_amp = self.amp_dtype is not None
        self.scaler = torch.amp.GradScaler(
            enabled=self.use_amp and self.device.type == "cuda"
        )
        print(f"Precision: {precision} (autocast={'on' if self.use_amp else 'off'})")

        self.grad_accum_steps = max(int(self.config.get("grad_accum_steps", 1)), 1)
        self.log_interval = int(self.config.get("log_interval", 50))

        self._setup_data()
        self._setup_model()
        self._setup_optimizer()

        self.current_epoch: int = 0
        self.best_loss: float = float("inf")

        reports_dir = resolve_repo_path(self.config.get("reports_dir", "mnist/reports"))
        self.reporter = TrainingReporter(
            model_name=self.config.get("model_name", "model"),
            reports_dir=reports_dir,
        )

    def _setup_data(self) -> None:
        """sets up train/validation dataloaders from the MNIST cache."""
        data_dir = resolve_repo_path(self.config.get("data_dir", "mnist/data"))
        train_set, val_set, _test_set = load_mnist_datasets(
            data_dir=data_dir,
            val_fraction=self.config.get("val_fraction", 0.1),
            dataset_cache_only=self.config.get("dataset_cache_only", False),
            seed=self.config.get("dataset_seed", 42),
        )
        self.train_loader, self.val_loader = create_dataloaders(
            train_dataset=train_set,
            val_dataset=val_set,
            batch_size=self.config.get("batch_size", 64),
            shuffle=True,
            num_workers=self.config.get("num_workers", 0),
        )
        val_batches = len(self.val_loader) if self.val_loader is not None else 0
        print(
            f"DataLoaders ready: {len(self.train_loader)} train / "
            f"{val_batches} val batches"
        )

    def _setup_model(self) -> None:
        """initializes the CNN based on configuration."""
        self.model = MnistCNN(
            in_channels=self.config.get("in_channels", 1),
            conv1_channels=self.config.get("conv1_channels", 32),
            conv2_channels=self.config.get("conv2_channels", 64),
            hidden_dim=self.config.get("hidden_dim", 128),
            num_classes=self.config.get("num_classes", 10),
            dropout=self.config.get("dropout", 0.25),
        ).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model initialized with {total_params:,} parameters")

    def _setup_optimizer(self) -> None:
        """sets up the optimizer, LR scheduler, and loss function."""
        num_epochs = self.config.get("num_epochs", 5)
        self.optimizer = AdamW(
            self._param_groups(self.config.get("weight_decay", 0.01)),
            lr=self.config.get("learning_rate", 1e-3),
            betas=(0.9, 0.95),
        )
        self.scheduler = self._build_scheduler(
            warmup_epochs=self.config.get("warmup_epochs", 0),
            num_epochs=num_epochs,
            min_lr_ratio=self.config.get("min_lr", 1e-4)
            / max(self.config.get("learning_rate", 1e-3), 1e-12),
        )
        self.criterion = nn.CrossEntropyLoss()

    def _param_groups(self, weight_decay: float) -> List[Dict[str, Any]]:
        """splits parameters so norms/biases skip weight decay."""
        decay, no_decay = [], []
        for param in self.model.parameters():
            if not param.requires_grad:
                continue
            (decay if param.dim() >= 2 else no_decay).append(param)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    def _build_scheduler(
        self,
        warmup_epochs: int,
        num_epochs: int,
        min_lr_ratio: float,
    ) -> LambdaLR:
        """builds a per-epoch linear-warmup then cosine-decay LR scheduler."""

        def lr_lambda(epoch: int) -> float:
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return (epoch + 1) / (warmup_epochs + 1)
            progress = (epoch - warmup_epochs) / max(num_epochs - warmup_epochs, 1)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        return LambdaLR(self.optimizer, lr_lambda)

    def _forward_loss(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """runs a forward pass under autocast and returns loss plus logits."""
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.use_amp,
        ):
            logits = self.model(images)
            loss = self.criterion(logits, labels)
        return loss, logits

    def train_epoch(self) -> Tuple[float, float]:
        """trains the model for a single epoch.

        Returns:
            a tuple of ``(average_loss, accuracy)`` over the epoch.
        """
        self.model.train()
        total_loss: float = 0.0
        correct: int = 0
        total: int = 0
        num_batches: int = 0

        self.optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        for step, (images, labels) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            loss, logits = self._forward_loss(images, labels)
            self.scaler.scale(loss / self.grad_accum_steps).backward()

            if (step + 1) % self.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.get("max_grad_norm", 1.0)
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()
            num_batches += 1
            preds = logits.argmax(dim=-1)
            correct += int((preds == labels).sum().item())
            total += int(labels.size(0))
            if step % self.log_interval == 0:
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "acc": f"{correct / max(total, 1):.3f}",
                        "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                    }
                )

        return total_loss / max(num_batches, 1), correct / max(total, 1)

    @torch.no_grad()
    def evaluate(self) -> Tuple[Optional[float], Optional[float]]:
        """evaluates the model on the validation loader.

        Returns:
            ``(val_loss, val_accuracy)``, or ``(None, None)`` if there is no
            validation set.
        """
        if self.val_loader is None:
            return None, None

        self.model.eval()
        total_loss: float = 0.0
        correct: int = 0
        total: int = 0
        num_batches: int = 0
        for images, labels in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            loss, logits = self._forward_loss(images, labels)
            total_loss += loss.item()
            num_batches += 1
            preds = logits.argmax(dim=-1)
            correct += int((preds == labels).sum().item())
            total += int(labels.size(0))

        return total_loss / max(num_batches, 1), correct / max(total, 1)

    def save_checkpoint(
        self,
        epoch: int,
        loss: float,
        accuracy: Optional[float] = None,
        is_best: bool = False,
    ) -> None:
        """saves a model checkpoint to disk.

        Args:
            epoch: current epoch number (1-based).
            loss: representative loss for the epoch (validation if available).
            accuracy: representative accuracy for the epoch, if computed.
            is_best: whether this checkpoint is the best so far.
        """
        weights_dir = resolve_repo_path(self.config.get("weights_dir", "mnist/weights"))
        os.makedirs(weights_dir, exist_ok=True)

        model_name = self.config.get("model_name", "model")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "loss": loss,
            "accuracy": accuracy,
            "config": self.config,
        }

        torch.save(checkpoint, os.path.join(weights_dir, f"{model_name}_latest.pt"))
        if is_best:
            best_path = os.path.join(weights_dir, f"{model_name}_best.pt")
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
        torch.save(
            checkpoint, os.path.join(weights_dir, f"{model_name}_epoch_{epoch}.pt")
        )

    def train(self) -> None:
        """runs the full training loop over all epochs."""
        num_epochs = self.config.get("num_epochs", 5)
        save_every = self.config.get("save_every", 1)

        print(f"Starting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            train_loss, train_acc = self.train_epoch()
            self.scheduler.step()
            val_loss, val_acc = self.evaluate()

            monitored = val_loss if val_loss is not None else train_loss
            monitored_acc = val_acc if val_acc is not None else train_acc
            val_str = ""
            if val_loss is not None:
                val_str = f" - Val: {val_loss:.4f} ({val_acc:.3f} acc)"
            print(
                f"Epoch {epoch + 1}/{num_epochs} - Train: {train_loss:.4f} "
                f"({train_acc:.3f} acc){val_str}"
            )

            self.reporter.log_epoch(
                epoch + 1, train_loss, val_loss, train_acc, val_acc
            )

            is_best = monitored < self.best_loss
            if is_best:
                self.best_loss = monitored

            if (epoch + 1) % save_every == 0 or is_best:
                self.save_checkpoint(epoch + 1, monitored, monitored_acc, is_best)

        print("Training completed!")
        print(f"Best loss: {self.best_loss:.4f}")
        print(f"Loss report saved to {self.reporter.report_path}")


def main() -> None:
    """CLI entry point for training with a YAML configuration file.

    Command-line arguments (``sys.argv``):
        config_path: path to the YAML training config (required, positional).
            e.g. ``uv run python -m mnist.training mnist/configs/mnist_small.yaml``.
    """
    if len(sys.argv) < 2:
        print("usage: uv run python -m mnist.training <config_path>")
        sys.exit(1)

    trainer = Trainer(sys.argv[1])
    trainer.train()


if __name__ == "__main__":
    main()
