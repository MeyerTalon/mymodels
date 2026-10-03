from pathlib import Path

from core.training import TrainingConfig


def training_config(tmp_path: Path, **overrides: object) -> dict[str, object]:
    return {
        'model_name': 'test_model',
        'num_epochs': 2,
        'batch_size': 2,
        'grad_accum_steps': 1,
        'learning_rate': 1e-3,
        'min_lr': 1e-4,
        'weight_decay': 0.1,
        'warmup_epochs': 0,
        'max_grad_norm': 1.0,
        'precision': 'fp32',
        'val_fraction': 0.0,
        'save_every': 1,
        'log_interval': 1,
        'num_workers': 0,
        'dataset_cache_only': False,
        'data_dir': str(tmp_path / 'data'),
        'weights_dir': str(tmp_path / 'weights'),
        'reports_dir': str(tmp_path / 'reports'),
        **overrides,
    }


def training_settings(tmp_path: Path, **overrides: object) -> TrainingConfig:
    return TrainingConfig.from_config(training_config(tmp_path, **overrides))
