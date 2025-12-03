"""Training pipeline for the decoder-only transformer model.

Provides the Trainer class which handles the complete training workflow: loading
configuration, setting up data loaders, running training loops, and saving checkpoints.
Can be run as a standalone script with a config file path.
"""

import os
import sys
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from architecture import DecoderOnlyTransformer
from data import create_dataloader
from tokenizer import WikipediaBPETokenizer


class Trainer:
    """Training class for the decoder-only transformer."""

    def __init__(self, config_path: str) -> None:
        """Initializes the trainer from a YAML configuration file.

        Args:
            config_path: Path to the YAML configuration file.
        """
        # Load configuration
        with open(config_path, "r", encoding="utf-8") as f:
            self.config: Dict[str, Any] = yaml.safe_load(f)

        # Set device: prefer Apple Silicon (MPS), then CUDA, then CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

        # Setup pipeline (data + tokenizer)
        self._setup_data()
        self._setup_model()
        self._setup_optimizer()

        # Training state
        self.current_epoch: int = 0
        self.best_loss: float = float("inf")

    def _setup_data(self) -> None:
        """Sets up the data loader and tokenizer."""
        data_dir = self.config.get("data_dir", "wikipedia/data")
        n_articles = self.config.get("number_of_articles", 5)
        use_local = self.config.get("use_local_articles", False)

        # Make path absolute if relative
        if not os.path.isabs(data_dir):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(base_dir, data_dir)

        # Train or load ByteLevel BPE tokenizer based on the article texts
        tokenizer_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "tokenizer_files"
        )
        self.tokenizer = WikipediaBPETokenizer.train_or_load(
            data_dir=data_dir,
            tokenizer_dir=tokenizer_dir,
        )

        # Create DataLoader (handles downloading n_articles internally)
        self.dataloader = create_dataloader(
            data_dir=data_dir,
            tokenizer=self.tokenizer,
            n_articles=n_articles,
            batch_size=self.config.get("batch_size", 32),
            max_length=self.config.get("max_seq_len", 512),
            shuffle=True,
            num_workers=self.config.get("num_workers", 0),
            use_local_articles=use_local,
        )

        print(f"DataLoader created with {len(self.dataloader)} batches")

    def _setup_model(self) -> None:
        """Initializes the transformer model based on configuration."""
        vocab_size = self.tokenizer.vocab_size

        self.model = DecoderOnlyTransformer(
            vocab_size=vocab_size,
            d_model=self.config.get("d_model", 512),
            n_heads=self.config.get("n_heads", 8),
            n_layers=self.config.get("n_layers", 6),
            d_ff=self.config.get("d_ff", 2048),
            max_seq_len=self.config.get("max_seq_len", 512),
            dropout=self.config.get("dropout", 0.1),
        ).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(
            f"Model initialized with {total_params:,} total parameters "
            f"({trainable_params:,} trainable)"
        )

    def _setup_optimizer(self) -> None:
        """Sets up the optimizer, scheduler, and loss function."""
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.config.get("learning_rate", 1e-4),
            weight_decay=self.config.get("weight_decay", 0.01),
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.get("num_epochs", 10),
            eta_min=self.config.get("min_lr", 1e-6),
        )

        # Ignore padding tokens (id=0) in the loss
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def train_epoch(self) -> float:
        """Trains the model for a single epoch.

        Returns:
            Average training loss over the epoch.
        """
        self.model.train()
        total_loss: float = 0.0
        num_batches: int = 0

        pbar = tqdm(self.dataloader, desc=f"Epoch {self.current_epoch + 1}")
        for input_ids, target_ids in pbar:
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(input_ids)

            # Reshape for loss calculation
            logits = logits.view(-1, logits.size(-1))
            target_ids = target_ids.view(-1)

            # Calculate loss
            loss = self.criterion(logits, target_ids)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.get("max_grad_norm", 1.0),
            )

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False) -> None:
        """Saves a model checkpoint to disk.

        Args:
            epoch: Current epoch number (1-based).
            loss: Average loss for the epoch.
            is_best: Whether this checkpoint is the best so far.
        """
        weights_dir = self.config.get("weights_dir", "wikipedia/weights")

        # Make path absolute if relative
        if not os.path.isabs(weights_dir):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            weights_dir = os.path.join(base_dir, weights_dir)

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

        # Save latest checkpoint
        checkpoint_path = os.path.join(weights_dir, f"{model_name}_latest.pt")
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = os.path.join(weights_dir, f"{model_name}_best.pt")
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")

        # Save epoch-specific checkpoint
        epoch_path = os.path.join(weights_dir, f"{model_name}_epoch_{epoch}.pt")
        torch.save(checkpoint, epoch_path)

    def train(self) -> None:
        """Runs the full training loop over all epochs."""
        num_epochs = self.config.get("num_epochs", 10)
        save_every = self.config.get("save_every", 1)

        print(f"Starting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train one epoch
            avg_loss = self.train_epoch()

            # Update learning rate
            self.scheduler.step()

            print(
                f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f} - "
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            )

            is_best = avg_loss < self.best_loss
            if is_best:
                self.best_loss = avg_loss

            if (epoch + 1) % save_every == 0 or is_best:
                self.save_checkpoint(epoch + 1, avg_loss, is_best)

        print("Training completed!")
        print(f"Best loss: {self.best_loss:.4f}")


def main() -> None:
    """CLI entry point for training with a YAML configuration file."""
    if len(sys.argv) < 2:
        print("Usage: python training.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    trainer = Trainer(config_path)
    trainer.train()


if __name__ == "__main__":
    main()

