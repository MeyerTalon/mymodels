"""training report generation for the mnist model.

records per-epoch train/val loss and accuracy and renders a two-panel chart
to disk, refreshed after every epoch. each run writes to its own timestamped
directory: ``<reports_dir>/<model_name>_<date>_<time>/report.png``.
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
    """accumulates per-epoch loss/accuracy and renders curves to disk.

    the report is a two-panel figure: cross-entropy loss on top, classification
    accuracy on the bottom. it is re-rendered after every epoch, so a
    partially trained run always has an up-to-date chart.
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
        self.val_losses: List[Optional[float]] = []
        self.train_accs: List[float] = []
        self.val_accs: List[Optional[float]] = []

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        train_acc: float = 0.0,
        val_acc: Optional[float] = None,
    ) -> str:
        """records one epoch's metrics and re-renders the report.

        Args:
            epoch: 1-based epoch number.
            train_loss: average training loss for the epoch.
            val_loss: average validation loss, or ``None`` when there is no
                validation set.
            train_acc: training accuracy in ``[0, 1]``.
            val_acc: validation accuracy in ``[0, 1]``, or ``None``.

        Returns:
            the path to the written report image.
        """
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        return self._render()

    def _render(self) -> str:
        """draws the current loss and accuracy curves.

        Returns:
            the path to the written report image.
        """
        fig, (ax_loss, ax_acc) = plt.subplots(
            2, 1, figsize=(10, 8), dpi=120, sharex=True
        )
        self._plot_series(
            ax_loss,
            self.train_losses,
            self.val_losses,
            ylabel='loss (cross-entropy)',
            annotate_best='min',
        )
        self._plot_series(
            ax_acc,
            self.train_accs,
            self.val_accs,
            ylabel='accuracy',
            annotate_best='max',
        )
        ax_acc.set_xlabel('epoch', fontsize=12)
        fig.suptitle(
            f'{self.model_name} — training & validation',
            fontsize=14,
            fontweight='bold',
        )
        fig.tight_layout()
        fig.savefig(self.report_path)
        plt.close(fig)
        return self.report_path

    def _plot_series(
        self,
        ax: 'plt.Axes',
        train_values: List[float],
        val_values: List[Optional[float]],
        ylabel: str,
        annotate_best: str,
    ) -> None:
        """plots train/val series on ``ax`` and marks the best val point."""
        ax.plot(
            self.epochs,
            train_values,
            marker='o',
            markersize=5,
            linewidth=2,
            color='#1f77b4',
            label='train',
        )
        val_epochs = [e for e, v in zip(self.epochs, val_values) if v is not None]
        plotted_vals = [v for v in val_values if v is not None]
        if plotted_vals:
            ax.plot(
                val_epochs,
                plotted_vals,
                marker='s',
                markersize=5,
                linewidth=2,
                color='#d62728',
                label='val',
            )
            self._annotate_best(ax, val_epochs, plotted_vals, annotate_best)

        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xticks(self.epochs)
        ax.margins(x=0.02)
        ax.grid(axis='x', linestyle='--', linewidth=0.6, alpha=0.5)
        ax.grid(axis='y', linestyle=':', linewidth=0.5, alpha=0.4)
        ax.set_axisbelow(True)
        ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

    @staticmethod
    def _annotate_best(
        ax: 'plt.Axes',
        val_epochs: List[int],
        val_values: List[float],
        mode: str,
    ) -> None:
        """marks the best validation epoch (min loss or max accuracy)."""
        if mode == 'max':
            best_idx = max(range(len(val_values)), key=val_values.__getitem__)
        else:
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
