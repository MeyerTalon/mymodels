"""training report generation for the wikipedia model.

records per-epoch train and validation loss and renders a single readable
line chart to disk, refreshed after every epoch. each run writes to its own
timestamped directory: ``<reports_dir>/<model_name>_<date>_<time>/report.png``.
"""

import os
from datetime import datetime
from typing import List, Optional

import matplotlib

# headless Agg backend: rendering must work during training with no display
# (servers, CI, background runs). must be selected before importing pyplot.
matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402  (import after backend selection)


class TrainingReporter:
    """accumulates per-epoch losses and renders a loss curve to disk.

    the report is a single figure with train and validation loss on the same
    axes so the two curves can be compared directly. it is re-rendered after
    every epoch, so a partially trained run always has an up-to-date chart.
    """

    def __init__(self, model_name: str, reports_dir: str) -> None:
        """creates the timestamped report directory for this run.

        Args:
            model_name: name of the model being trained; used in the directory
                name and chart title.
            reports_dir: base directory under which per-run report folders are
                created.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.model_name = model_name
        self.run_dir = os.path.join(reports_dir, f'{model_name}_{timestamp}')
        os.makedirs(self.run_dir, exist_ok=True)
        self.report_path = os.path.join(self.run_dir, 'report.png')

        self.epochs: List[int] = []
        self.train_losses: List[float] = []
        # validation loss is optional per epoch (``None`` when no val set).
        self.val_losses: List[Optional[float]] = []

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
    ) -> str:
        """records one epoch's losses and re-renders the report.

        Args:
            epoch: 1-based epoch number.
            train_loss: average training loss for the epoch.
            val_loss: average validation loss for the epoch, or ``None`` when
                there is no validation set.

        Returns:
            the path to the written report image.
        """
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        return self._render()

    def _render(self) -> str:
        """draws the current loss curves and writes them to ``report_path``.

        Returns:
            the path to the written report image.
        """
        fig, ax = plt.subplots(figsize=(10, 6), dpi=120)

        ax.plot(
            self.epochs,
            self.train_losses,
            marker='o',
            markersize=5,
            linewidth=2,
            color='#1f77b4',
            label='train loss',
        )

        # validation points may be sparse; plot only the epochs that have one.
        val_epochs = [e for e, v in zip(self.epochs, self.val_losses) if v is not None]
        val_values = [v for v in self.val_losses if v is not None]
        if val_values:
            ax.plot(
                val_epochs,
                val_values,
                marker='s',
                markersize=5,
                linewidth=2,
                color='#d62728',
                label='val loss',
            )
            self._annotate_best_val(ax, val_epochs, val_values)

        self._style_axes(ax)
        fig.tight_layout()
        fig.savefig(self.report_path)
        plt.close(fig)
        return self.report_path

    def _style_axes(self, ax: 'plt.Axes') -> None:
        """applies readable styling and per-epoch gridlines to the axes."""
        ax.set_title(
            f'{self.model_name} — training & validation loss',
            fontsize=14,
            fontweight='bold',
        )
        ax.set_xlabel('epoch', fontsize=12)
        ax.set_ylabel('loss (cross-entropy)', fontsize=12)

        # one integer tick per epoch so each epoch boundary is unambiguous.
        ax.set_xticks(self.epochs)
        ax.margins(x=0.02)

        # vertical gridlines mark where each epoch starts/ends; lighter
        # horizontal lines aid reading loss values off the y-axis.
        ax.grid(axis='x', linestyle='--', linewidth=0.6, alpha=0.5)
        ax.grid(axis='y', linestyle=':', linewidth=0.5, alpha=0.4)
        ax.set_axisbelow(True)

        ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

    @staticmethod
    def _annotate_best_val(
        ax: 'plt.Axes',
        val_epochs: List[int],
        val_values: List[float],
    ) -> None:
        """marks and labels the lowest validation-loss epoch."""
        best_idx = min(range(len(val_values)), key=val_values.__getitem__)
        best_epoch = val_epochs[best_idx]
        best_val = val_values[best_idx]
        ax.annotate(
            f'best val {best_val:.3f} @ epoch {best_epoch}',
            xy=(best_epoch, best_val),
            xytext=(0, 18),
            textcoords='offset points',
            ha='center',
            fontsize=10,
            color='#d62728',
            arrowprops=dict(arrowstyle='->', color='#d62728', linewidth=1),
        )
