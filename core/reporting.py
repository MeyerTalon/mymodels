from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.figure import Figure

REPORT_FILENAME = 'report.png'
RUN_TIMESTAMP_FORMAT = '%Y%m%d_%H%M%S'
FIGURE_WIDTH_IN = 10
FIGURE_HEIGHT_IN_PER_PANEL = 3
FIGURE_BASE_HEIGHT_IN = 3
FIGURE_DPI = 120
TRAIN_COLOR = '#1f77b4'
VAL_COLOR = '#d62728'
BEST_LABEL_OFFSET_PT = (0, 18)


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    accuracy: float | None = None


@dataclass(frozen=True)
class _Panel:
    label: str
    value: Callable[[EpochMetrics], float | None]
    higher_is_better: bool


LOSS_PANEL = _Panel('loss (cross-entropy)', lambda m: m.loss, higher_is_better=False)
ACCURACY_PANEL = _Panel('accuracy', lambda m: m.accuracy, higher_is_better=True)


class TrainingReporter:
    """Re-renders after every epoch, so an interrupted run still has a current chart."""

    def __init__(self, model_name: str, reports_dir: Path) -> None:
        timestamp = datetime.now(UTC).astimezone().strftime(RUN_TIMESTAMP_FORMAT)
        self.model_name = model_name
        self.run_dir = reports_dir / f'{model_name}_{timestamp}'
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.report_path = self.run_dir / REPORT_FILENAME
        self.epochs: list[int] = []
        self.train: list[EpochMetrics] = []
        self.val: list[EpochMetrics | None] = []

    def log_epoch(
        self, epoch: int, train: EpochMetrics, val: EpochMetrics | None
    ) -> Path:
        self.epochs.append(epoch)
        self.train.append(train)
        self.val.append(val)
        self._render()
        return self.report_path

    def _render(self) -> None:
        panels = [LOSS_PANEL]
        if any(metrics.accuracy is not None for metrics in self.train):
            panels.append(ACCURACY_PANEL)
        figure = Figure(
            figsize=(
                FIGURE_WIDTH_IN,
                FIGURE_BASE_HEIGHT_IN + FIGURE_HEIGHT_IN_PER_PANEL * len(panels),
            ),
            dpi=FIGURE_DPI,
        )
        first_axes: Axes | None = None
        for index, panel in enumerate(panels, start=1):
            axes = figure.add_subplot(len(panels), 1, index, sharex=first_axes)
            first_axes = first_axes or axes
            self._plot_panel(axes, panel)
        if first_axes is not None:
            figure.axes[-1].set_xlabel('epoch', fontsize=12)
        figure.suptitle(
            f'{self.model_name} — training & validation',
            fontsize=14,
            fontweight='bold',
        )
        figure.tight_layout()
        figure.savefig(self.report_path)

    def _plot_panel(self, axes: Axes, panel: _Panel) -> None:
        train_points = _points(self.epochs, [panel.value(m) for m in self.train])
        val_points = _points(
            self.epochs,
            [None if m is None else panel.value(m) for m in self.val],
        )
        _plot_series(axes, train_points, color=TRAIN_COLOR, marker='o', label='train')
        if val_points:
            _plot_series(axes, val_points, color=VAL_COLOR, marker='s', label='val')
            _annotate_best(axes, val_points, higher_is_better=panel.higher_is_better)
        axes.set_ylabel(panel.label, fontsize=12)
        axes.set_xticks(self.epochs)
        axes.margins(x=0.02)
        axes.grid(axis='x', linestyle='--', linewidth=0.6, alpha=0.5)
        axes.grid(axis='y', linestyle=':', linewidth=0.5, alpha=0.4)
        axes.set_axisbelow(True)
        axes.legend(loc='upper right', fontsize=11, framealpha=0.9)


def _points(epochs: list[int], values: list[float | None]) -> list[tuple[int, float]]:
    return [
        (epoch, value)
        for epoch, value in zip(epochs, values, strict=True)
        if value is not None
    ]


def _plot_series(
    axes: Axes,
    points: list[tuple[int, float]],
    *,
    color: str,
    marker: str,
    label: str,
) -> None:
    axes.plot(
        [epoch for epoch, _ in points],
        [value for _, value in points],
        marker=marker,
        markersize=5,
        linewidth=2,
        color=color,
        label=label,
    )


def _annotate_best(
    axes: Axes, points: list[tuple[int, float]], *, higher_is_better: bool
) -> None:
    pick = max if higher_is_better else min
    best_epoch, best_value = pick(points, key=lambda point: point[1])
    axes.annotate(
        f'best val {best_value:.3f} @ epoch {best_epoch}',
        xy=(best_epoch, best_value),
        xytext=BEST_LABEL_OFFSET_PT,
        textcoords='offset points',
        ha='center',
        fontsize=10,
        color=VAL_COLOR,
        arrowprops={'arrowstyle': '->', 'color': VAL_COLOR, 'linewidth': 1},
    )
