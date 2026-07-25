from __future__ import annotations

from pathlib import Path

import numpy as np

from utils.cnn_training import train_model_run


def test_training_smoke(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    train_x = rng.normal(size=(48, 2, 35)).astype(np.float32)
    val_x = rng.normal(size=(16, 2, 35)).astype(np.float32)
    train_y = (train_x[:, 0].mean(axis=1) - train_x[:, 1].mean(axis=1)).astype(np.float32)
    val_y = (val_x[:, 0].mean(axis=1) - val_x[:, 1].mean(axis=1)).astype(np.float32)
    np.savez(
        tmp_path / "data.npz",
        direct_train_x=train_x,
        direct_train_y=train_y,
        direct_validation_x=val_x,
        direct_validation_y=val_y,
        correction_train_x=train_x,
        correction_train_led_delta_ps=train_y,
        correction_validation_x=val_x,
        correction_validation_led_delta_ps=val_y,
    )
    model_config = {
        "direct_model": {
            "conv_channels": [4, 4],
            "kernel_sizes": [5, 3],
            "pool_after": [1],
            "pool_size": 2,
            "adaptive_pool_bins": 2,
            "fc_units": 8,
            "dropout": 0.0,
            "activation": "relu",
        },
        "correction_model": {
            "conv_channels": [4, 4],
            "kernel_sizes": [5, 3],
            "pool_after": [1],
            "pool_size": 2,
            "adaptive_pool_bins": 2,
            "embedding_units": 8,
            "head_units": 4,
            "dropout": 0.0,
            "activation": "relu",
        },
        "training": {
            "epochs": 2,
            "batch_size": 16,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "loss": "mse",
            "early_stopping_patience": 2,
            "lr_patience": 1,
            "lr_factor": 0.5,
            "min_learning_rate": 1e-6,
            "minimum_improvement": 0.0,
            "gradient_clip_norm": 5.0,
            "mixed_precision": False,
            "num_workers": 0,
            "pin_memory": False,
            "seeds": [1]
        },
        "parallel": {
            "deterministic": True,
            "cpu_threads_per_run": 1,
            "use_data_parallel": False,
            "max_parallel_runs": 1,
            "device_pool": ["cpu"]
        },
    }
    preprocessing = {"normalization": {"type": "train_channel_zscore", "epsilon": 1e-6}}
    result = train_model_run(
        {
            "model_type": "direct",
            "seed": 1,
            "dataset_path": str(tmp_path / "data.npz"),
            "run_dir": str(tmp_path / "run"),
            "model_config": model_config,
            "preprocessing_config": preprocessing,
            "device": "cpu",
        }
    )
    assert Path(result["checkpoint"]).is_file()
    history_path = tmp_path / "run" / "training_history.csv"
    assert history_path.is_file()
    header = history_path.read_text(encoding="utf-8").splitlines()[0]
    for column in (
        "epoch_seconds",
        "elapsed_seconds",
        "eta_max_seconds",
        "estimated_max_finish",
    ):
        assert column in header
