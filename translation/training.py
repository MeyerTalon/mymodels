"""training pipeline for the encoder-decoder translation model.

provides the Trainer class which handles the complete training workflow: loading
configuration, setting up padded dataloaders, running mixed-precision teacher-
forcing training with gradient accumulation, validating, and saving checkpoints.
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

from translation.architecture import EncoderDecoderTransformer
from translation.data import create_dataloaders, load_translation_pairs
from translation.reporting import TrainingReporter
from translation.tokenizer import DEFAULT_LANGUAGES, TranslationBPETokenizer
from translation.utils import (
    TOKENIZER_DIR,
    resolve_repo_path,
    select_autocast_dtype,
    select_device,
)


class Trainer:
    """training class for the encoder-decoder translator."""

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

        reports_dir = resolve_repo_path(
            self.config.get("reports_dir", "translation/reports")
        )
        self.reporter = TrainingReporter(
            model_name=self.config.get("model_name", "model"),
            reports_dir=reports_dir,
        )

    def _setup_data(self) -> None:
        """sets up the tokenizer and padded train/validation dataloaders."""
        data_dir = resolve_repo_path(self.config.get("data_dir", "translation/data"))
        corpus_dir = self.config.get("corpus_dir")
        if corpus_dir:
            corpus_dir = resolve_repo_path(corpus_dir)
        pairs = load_translation_pairs(
            data_dir=data_dir,
            corpus_dir=corpus_dir,
            max_pairs=self.config.get("max_pairs"),
            dataset_cache_only=self.config.get("dataset_cache_only", False),
        )

        languages = self.config.get("languages", DEFAULT_LANGUAGES)
        texts = [pair["src"] for pair in pairs] + [pair["tgt"] for pair in pairs]
        self.tokenizer = TranslationBPETokenizer.train_or_load(
            texts=texts,
            tokenizer_dir=TOKENIZER_DIR,
            vocab_size=self.config.get("vocab_size", 8000),
            min_frequency=self.config.get("min_frequency", 2),
            languages=languages,
        )

        self.train_loader, self.val_loader = create_dataloaders(
            pairs=pairs,
            tokenizer=self.tokenizer,
            max_seq_len=self.config.get("max_seq_len", 128),
            batch_size=self.config.get("batch_size", 16),
            val_fraction=self.config.get("val_fraction", 0.0),
            shuffle=True,
            num_workers=self.config.get("num_workers", 0),
        )

        val_batches = len(self.val_loader) if self.val_loader is not None else 0
        print(
            f"DataLoaders ready: {len(self.train_loader)} train / "
            f"{val_batches} val batches"
        )

    def _setup_model(self) -> None:
        """initializes the encoder-decoder transformer based on configuration."""
        self.model = EncoderDecoderTransformer(
            vocab_size=self.tokenizer.vocab_size,
            d_model=self.config.get("d_model", 256),
            n_heads=self.config.get("n_heads", 4),
            n_encoder_layers=self.config.get("n_encoder_layers", 3),
            n_decoder_layers=self.config.get("n_decoder_layers", 3),
            d_ff=self.config.get("d_ff", 1024),
            max_seq_len=self.config.get("max_seq_len", 128),
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
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_id)

    def _param_groups(self, weight_decay: float) -> List[Dict[str, Any]]:
        """splits parameters so norms/biases/embeddings skip weight decay."""
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
        src: torch.Tensor,
        tgt_input: torch.Tensor,
        tgt_labels: torch.Tensor,
        src_pad_mask: torch.Tensor,
        tgt_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """runs a teacher-forcing forward pass under autocast."""
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.use_amp,
        ):
            logits = self.model(
                src,
                tgt_input,
                src_key_padding_mask=src_pad_mask,
                tgt_key_padding_mask=tgt_pad_mask,
            )
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)), tgt_labels.reshape(-1)
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
        for step, batch in enumerate(pbar):
            src, tgt_input, tgt_labels, src_pad, tgt_pad = batch
            src = src.to(self.device, non_blocking=True)
            tgt_input = tgt_input.to(self.device, non_blocking=True)
            tgt_labels = tgt_labels.to(self.device, non_blocking=True)
            src_pad = src_pad.to(self.device, non_blocking=True)
            tgt_pad = tgt_pad.to(self.device, non_blocking=True)

            loss = self._forward_loss(src, tgt_input, tgt_labels, src_pad, tgt_pad)
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
        for src, tgt_input, tgt_labels, src_pad, tgt_pad in self.val_loader:
            src = src.to(self.device, non_blocking=True)
            tgt_input = tgt_input.to(self.device, non_blocking=True)
            tgt_labels = tgt_labels.to(self.device, non_blocking=True)
            src_pad = src_pad.to(self.device, non_blocking=True)
            tgt_pad = tgt_pad.to(self.device, non_blocking=True)
            total_loss += self._forward_loss(
                src, tgt_input, tgt_labels, src_pad, tgt_pad
            ).item()
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
            self.config.get("weights_dir", "translation/weights")
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

            self.reporter.log_epoch(epoch + 1, train_loss, val_loss)

            is_best = monitored < self.best_loss
            if is_best:
                self.best_loss = monitored

            if (epoch + 1) % save_every == 0 or is_best:
                self.save_checkpoint(epoch + 1, monitored, is_best)

        print("Training completed!")
        print(f"Best loss: {self.best_loss:.4f}")
        print(f"Loss report saved to {self.reporter.report_path}")


def main() -> None:
    """CLI entry point for training with a YAML configuration file.

    Command-line arguments (``sys.argv``):
        config_path: path to the YAML training config (required, positional).
            e.g. ``uv run python -m translation.training translation/configs/translation_small.yaml``.
    """
    if len(sys.argv) < 2:
        print("usage: uv run python -m translation.training <config_path>")
        sys.exit(1)

    trainer = Trainer(sys.argv[1])
    trainer.train()


if __name__ == "__main__":
    main()
