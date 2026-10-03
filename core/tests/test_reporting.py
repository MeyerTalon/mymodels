from pathlib import Path

from core.reporting import EpochMetrics, TrainingReporter


def test_log_epoch_writes_report(tmp_path: Path) -> None:
    reporter = TrainingReporter('model', tmp_path)
    reporter.log_epoch(1, EpochMetrics(2.0), None)
    path = reporter.log_epoch(2, EpochMetrics(1.5), EpochMetrics(1.7))
    assert path.is_file()
    assert path.parent.parent == tmp_path
    assert path.parent.name.startswith('model_')


def test_log_epoch_with_accuracy(tmp_path: Path) -> None:
    reporter = TrainingReporter('classifier', tmp_path)
    path = reporter.log_epoch(1, EpochMetrics(0.5, 0.8), EpochMetrics(0.6, 0.75))
    assert path.stat().st_size > 0
