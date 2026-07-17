"""training pipeline for the decoder-only transformer model.

provides the Trainer class which handles the complete training workflow: loading
configuration, setting up packed dataloaders, running mixed-precision training
loops with gradient accumulation, validating, and saving checkpoints. can be run
as a standalone script with a config file path. defaults are tuned for Apple
silicon (MPS) on consumer memory budgets.
"""

import math
import os
import sys
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from wikipedia.architecture import DecoderOnlyTransformer
from wikipedia.data import create_dataloaders
from wikipedia.tokenizer import WikipediaBPETokenizer
from wikipedia.utils import (
    TOKENIZER_DIR,
    resolve_repo_path,
    select_autocast_dtype,
    select_device,
)


class Trainer:
    """training class for the decoder-only transformer."""

    def __init__(self, config_path: str) -> None:
        """initializes the trainer from a YAML configuration file.

        Args:
            config_path: path to the YAML configuration file.
        """
        with open(config_path, "r", encoding="utf-8") as f:
            self.config: Dict[str, Any] = yaml.safe_load(f)

        self.device = select_device()
        print(f"Using device: {self.device}")

        # mixed precision: fp16 on accelerators speeds up MPS/CUDA and halves
        # activation memory; disabled automatically on CPU.
        precision = self.config.get(
            "precision", "fp16" if self.device.type != "cpu" else "fp32"
        )
        self.amp_dtype = select_autocast_dtype(self.device, precision)
        self.use_amp = self.amp_dtype is not None
        # gradient scaling is only needed (and only supported) for CUDA float16.
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

    def _setup_data(self) -> None:
        """sets up the tokenizer and packed train/validation dataloaders."""
        data_dir = resolve_repo_path(self.config.get("data_dir", "wikipedia/data"))
        n_articles = self.config.get("number_of_articles", 5)
        use_local = self.config.get("use_local_articles", False)

        # train or load the ByteLevel BPE tokenizer over the article corpus
        self.tokenizer = WikipediaBPETokenizer.train_or_load(
            data_dir=data_dir,
            tokenizer_dir=TOKENIZER_DIR,
            vocab_size=self.config.get("vocab_size", 8000),
            min_frequency=self.config.get("min_frequency", 2),
        )

        self.train_loader, self.val_loader = create_dataloaders(
            data_dir=data_dir,
            tokenizer=self.tokenizer,
            n_articles=n_articles,
            block_size=self.config.get("max_seq_len", 512),
            batch_size=self.config.get("batch_size", 16),
            val_fraction=self.config.get("val_fraction", 0.0),
            shuffle=True,
            num_workers=self.config.get("num_workers", 0),
            use_local_articles=use_local,
        )

        val_batches = len(self.val_loader) if self.val_loader is not None else 0
        print(
            f"DataLoaders ready: {len(self.train_loader)} train / "
            f"{val_batches} val batches"
        )

    def _setup_model(self) -> None:
        """initializes the transformer model based on configuration."""
        self.model = DecoderOnlyTransformer(
            vocab_size=self.tokenizer.vocab_size,
            d_model=self.config.get("d_model", 512),
            n_heads=self.config.get("n_heads", 8),
            n_layers=self.config.get("n_layers", 6),
            d_ff=self.config.get("d_ff", 2048),
            max_seq_len=self.config.get("max_seq_len", 512),
            dropout=self.config.get("dropout", 0.1),
        ).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model initialized with {total_params:,} parameters")

    def _setup_optimizer(self) -> None:
        """sets up the optimizer, LR scheduler, and loss function."""
        num_epochs = self.config.get("num_epochs", 10)
        self.optimizer = AdamW(
            self._param_groups(self.config.get("weight_decay", 0.1)),
            lr=self.config.get("learning_rate", 3e-4),
            betas=(0.9, 0.95),
        )
        self.scheduler = self._build_scheduler(
            warmup_epochs=self.config.get("warmup_epochs", 0),
            num_epochs=num_epochs,
            min_lr_ratio=self.config.get("min_lr", 3e-5)
            / max(self.config.get("learning_rate", 3e-4), 1e-12),
        )

        # ignore padding tokens in the loss; harmless with packed data (no pads).
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=getattr(self.tokenizer, "pad_id", 0)
        )

    def _param_groups(self, weight_decay: float) -> List[Dict[str, Any]]:
        """splits parameters so norms/biases/embeddings skip weight decay."""
        decay, no_decay = [], []
        for param in self.model.parameters():
            if not param.requires_grad:
                continue
            # only weight-matrix params (2d+) get decayed; biases, LayerNorm
            # gains, and embeddings do not.
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
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """runs a forward pass under autocast and returns the token loss."""
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.use_amp,
        ):
            logits = self.model(input_ids)
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1)
            )
        return loss

    def train_epoch(self) -> float:
        """trains the model for a single epoch.

        Returns:
            average training loss over the epoch.
        """
        self.model.train()
        total_loss: float = 0.0
        num_batches: int = 0

        self.optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        for step, (input_ids, target_ids) in enumerate(pbar):
            input_ids = input_ids.to(self.device, non_blocking=True)
            target_ids = target_ids.to(self.device, non_blocking=True)

            loss = self._forward_loss(input_ids, target_ids)
            # scale so accumulated gradients average over the accumulation window
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
            if step % self.log_interval == 0:
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                    }
                )

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def evaluate(self) -> Optional[float]:
        """evaluates the model on the validation loader.

        Returns:
            average validation loss, or ``None`` if there is no validation set.
        """
        if self.val_loader is None:
            return None

        self.model.eval()
        total_loss: float = 0.0
        num_batches: int = 0
        for input_ids, target_ids in self.val_loader:
            input_ids = input_ids.to(self.device, non_blocking=True)
            target_ids = target_ids.to(self.device, non_blocking=True)
            total_loss += self._forward_loss(input_ids, target_ids).item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False) -> None:
        """saves a model checkpoint to disk.

        Args:
            epoch: current epoch number (1-based).
            loss: representative loss for the epoch (validation if available).
            is_best: whether this checkpoint is the best so far.
        """
        weights_dir = resolve_repo_path(
            self.config.get("weights_dir", "wikipedia/weights")
        )
        os.makedirs(weights_dir, exist_ok=True)

        model_name = self.config.get("model_name", "model")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "loss": loss,
            "config": self.config,
            "tokenizer_vocab_size": self.tokenizer.vocab_size,
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
        num_epochs = self.config.get("num_epochs", 10)
        save_every = self.config.get("save_every", 1)

        print(f"Starting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            train_loss = self.train_epoch()
            self.scheduler.step()
            val_loss = self.evaluate()

            monitored = val_loss if val_loss is not None else train_loss
            val_str = f" - Val: {val_loss:.4f}" if val_loss is not None else ""
            print(
                f"Epoch {epoch + 1}/{num_epochs} - Train: {train_loss:.4f}{val_str} "
                f"- Perplexity: {math.exp(min(monitored, 20)):.2f}"
            )

            is_best = monitored < self.best_loss
            if is_best:
                self.best_loss = monitored

            if (epoch + 1) % save_every == 0 or is_best:
                self.save_checkpoint(epoch + 1, monitored, is_best)

        print("Training completed!")
        print(f"Best loss: {self.best_loss:.4f}")


def main() -> None:
    """CLI entry point for training with a YAML configuration file."""
    if len(sys.argv) < 2:
        print("usage: uv run python -m wikipedia.training <config_path>")
        sys.exit(1)

    trainer = Trainer(sys.argv[1])
    trainer.train()


if __name__ == "__main__":
    main()
