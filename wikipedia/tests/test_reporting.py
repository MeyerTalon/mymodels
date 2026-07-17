"""tests for the training loss reporter."""

import os

from wikipedia.reporting import TrainingReporter


def test_reporter_creates_timestamped_run_dir(tmp_path):
    reporter = TrainingReporter("tiny", str(tmp_path))
    assert os.path.isdir(reporter.run_dir)
    # the run directory is nested under the reports base and prefixed by name.
    assert os.path.dirname(reporter.run_dir) == str(tmp_path)
    assert os.path.basename(reporter.run_dir).startswith("tiny_")


def test_log_epoch_writes_nonempty_report(tmp_path):
    reporter = TrainingReporter("tiny", str(tmp_path))
    path = reporter.log_epoch(1, train_loss=6.5, val_loss=6.1)
    assert path == reporter.report_path
    assert os.path.basename(path) == "report.png"
    assert os.path.getsize(path) > 0


def test_log_epoch_accumulates_and_handles_missing_val(tmp_path):
    reporter = TrainingReporter("tiny", str(tmp_path))
    reporter.log_epoch(1, train_loss=6.5, val_loss=6.1)
    reporter.log_epoch(2, train_loss=5.9)  # no validation this epoch
    reporter.log_epoch(3, train_loss=5.2, val_loss=5.4)

    assert reporter.epochs == [1, 2, 3]
    assert reporter.train_losses == [6.5, 5.9, 5.2]
    assert reporter.val_losses == [6.1, None, 5.4]
    assert os.path.getsize(reporter.report_path) > 0
