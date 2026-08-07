from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ml_pipeline.study import (
    _metrics,
    _read_results,
    _selection_value,
    _write_results,
)
from ml_pipeline.study_config import load_study_config


def test_rmse_selection_is_independent_of_var_bias_objective() -> None:
    rows = [
        {"rmse_ps": 12.0, "loss": 2.0, "loss_type": "var_bias"},
        {"rmse_ps": 8.0, "loss": 5.0, "loss_type": "var_bias"},
    ]
    assert _selection_value(rows, "rmse") == 10.0
    assert _selection_value(rows, "validation_rmse_ps") == 10.0
    assert _selection_value(rows, "loss") == 3.5


def test_old_mse_rows_can_reconstruct_rmse_but_var_bias_rows_cannot() -> None:
    assert _selection_value([{"loss": 81.0, "loss_type": "mse"}], "rmse") == 9.0
    assert np.isinf(
        _selection_value([{"loss": 81.0, "loss_type": "var_bias"}], "rmse")
    )


def test_common_metrics_persist_rmse_for_bias_aware_loss(monkeypatch) -> None:
    monkeypatch.setattr(
        "ml_pipeline.study.fit_times_ps",
        lambda values, label, config: SimpleNamespace(success=True, ctr_ps=50.0),
    )
    corrected = np.asarray([2.0, 4.0, 10.0])
    true = np.asarray([1.0, 2.0, 3.0])
    baseline = np.asarray([1.0, 2.0, 3.0])
    metrics = _metrics(
        corrected,
        true,
        baseline,
        {"method": "std"},
        {"id": "vb", "type": "var_bias", "bias_weight": 0.01},
        2.0,
    )
    residual = corrected - true
    assert np.isclose(metrics["rmse_ps"], np.sqrt(np.mean(residual**2)))
    assert "loss" in metrics


def test_compact_internal_results_round_trip_rmse(tmp_path: Path) -> None:
    path = tmp_path / "_state" / "all_results.csv"
    row = {
        "row_key": "1234567890abcdef12345678",
        "record_type": "cv_fold",
        "experiment_id": "study",
        "root_id": "run",
        "root_file": "/tmp/run.root",
        "channel_mode": "timing_to_timing",
        "model_id": "linear_svr",
        "model_type": "linear_svr",
        "loss_id": "var_bias_0p01",
        "loss_type": "var_bias",
        "input_transform": "none",
        "window_id": "w1",
        "window_start_ns": -2.0,
        "window_end_ns": 16.0,
        "trial_id": "t1",
        "fold_id": 0,
        "split": "validation",
        "statistic": "raw",
        "status": "completed",
        "n_events": 10,
        "loss": 4.0,
        "rmse_ps": 7.5,
        "bias_ps": 0.2,
        "ctr_ps": 60.0,
    }
    _write_results(path, [row])
    loaded = _read_results(path)
    assert len(loaded) == 1
    assert float(loaded[0]["rmse_ps"]) == 7.5


def test_study_config_canonicalizes_rmse_alias(tmp_path: Path, monkeypatch) -> None:
    root_file = tmp_path / "45V.root"
    root_file.write_bytes(b"")
    model_dir = tmp_path / "config" / "model_spaces"
    model_dir.mkdir(parents=True)
    (model_dir / "linear_svr.json").write_text(
        json.dumps(
            {
                "id": "linear_svr",
                "model_type": "linear_svr",
                "base_train_config": {},
                "supported_losses": ["mse"],
                "search": {"method": "grid", "parameters": {}},
            }
        )
    )
    config = {
        "experiment": {"name": "x", "output_dir": "results/x"},
        "data": {"root_folder": ".", "root_glob": "45V.root"},
        "preprocessing": {},
        "split": {"blind_fraction": 0.15},
        "windows_ns": [{"id": "w", "start_ns": -2, "end_ns": 8}],
        "channel_modes": ["timing_to_timing"],
        "input_transforms": ["none"],
        "losses": [{"id": "mse", "type": "mse"}],
        "models": ["linear_svr"],
        "model_spaces_dir": "config/model_spaces",
        "cross_validation": {"n_splits": 2},
        "selection": {
            "method": "median_mad_z",
            "hyperparameter_metric": "rmse",
            "window_metric": "validation_rmse_ps",
        },
    }
    config_path = tmp_path / "study.json"
    config_path.write_text(json.dumps(config))
    loaded = load_study_config(config_path, tmp_path)
    assert loaded["selection"]["hyperparameter_metric"] == "rmse_ps"
    assert loaded["selection"]["window_metric"] == "rmse_ps"
