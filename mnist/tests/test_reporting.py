"""tests for the MNIST training reporter."""

import os

from mnist.reporting import TrainingReporter


def test_reporter_creates_timestamped_run_dir(tmp_path):
    reporter = TrainingReporter("tiny", str(tmp_path))
    assert os.path.isdir(reporter.run_dir)
    assert os.path.dirname(reporter.run_dir) == str(tmp_path)
    assert os.path.basename(reporter.run_dir).startswith("tiny_")


def test_log_epoch_writes_nonempty_report(tmp_path):
    reporter = TrainingReporter("tiny", str(tmp_path))
    path = reporter.log_epoch(
        1, train_loss=2.1, val_loss=2.0, train_acc=0.4, val_acc=0.45
    )
    assert path == reporter.report_path
    assert os.path.basename(path) == "report.png"
    assert os.path.getsize(path) > 0


def test_log_epoch_accumulates_and_handles_missing_val(tmp_path):
    reporter = TrainingReporter("tiny", str(tmp_path))
    reporter.log_epoch(1, train_loss=2.1, val_loss=2.0, train_acc=0.4, val_acc=0.45)
    reporter.log_epoch(2, train_loss=1.8, train_acc=0.55)
    reporter.log_epoch(3, train_loss=1.5, val_loss=1.6, train_acc=0.7, val_acc=0.65)

    assert reporter.epochs == [1, 2, 3]
    assert reporter.train_losses == [2.1, 1.8, 1.5]
    assert reporter.val_losses == [2.0, None, 1.6]
    assert reporter.train_accs == [0.4, 0.55, 0.7]
    assert reporter.val_accs == [0.45, None, 0.65]
    assert os.path.getsize(reporter.report_path) > 0
