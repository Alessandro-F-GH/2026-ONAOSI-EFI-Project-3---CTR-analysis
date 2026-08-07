from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
import torch

from ml_pipeline.dataset import PreparedDataset
from ml_pipeline.evaluation import TrainedModel, _evaluate_model
from ml_pipeline.models.linear_regression import LinearPairRegressor
from ml_pipeline.training import train_model


def _dataset(tmp_path: Path) -> PreparedDataset:
    event_count = 60
    sample_count = 6
    x = np.linspace(-3.0, 3.0, event_count, dtype=np.float32)
    target_ps = 10.0 * x - 3.0
    windows = np.zeros((event_count, 2, sample_count), dtype=np.float32)
    windows[:, 0, 1] = x
    windows[:, 0, 3] = 0.4 * x
    windows[:, 1, 4] = -0.2 * x
    led = np.column_stack(
        [
            np.rint(target_ps * 1000.0).astype(np.int64),
            np.zeros(event_count, dtype=np.int64),
        ]
    )
    directory = tmp_path / "prepared"
    directory.mkdir()
    manifest = {
        "format_version": 4,
        "fingerprint": "linear-regression-test",
        "name": "linear-regression-test",
        "true_tof_ps": 0.0,
        "led_timestamp_source": "energy_channels",
        "cfd_timestamp_source": "energy_channels",
        "ml_window_alignment_source": "energy_channel_led",
        "timing_channel_waveforms_saved": False,
        "waveform_grid": "native_acquisition_samples",
        "input_components": ["energy"],
        "input_component_lengths": [sample_count],
    }
    return PreparedDataset(
        directory=directory,
        manifest=manifest,
        event_id=np.arange(event_count, dtype=np.int64),
        event_index=np.arange(event_count, dtype=np.int64),
        source_file_id=np.zeros(event_count, dtype=np.int32),
        source_run_index=np.zeros(event_count, dtype=np.int32),
        bias_voltage_V=np.full(event_count, 47.0, dtype=np.float32),
        amplitude_mV=np.ones((event_count, 2), dtype=np.float32),
        noise_rms_mV=np.zeros((event_count, 2), dtype=np.float32),
        trigger_index=np.zeros((event_count, 2), dtype=np.int32),
        led_time_fs=led,
        cfd_time_fs=led,
        windows_mV=windows,
        relative_time_ps=np.arange(sample_count, dtype=np.float32) * 100.0,
        train=np.arange(0, 40, dtype=np.int64),
        validation=np.arange(40, 50, dtype=np.int64),
        test=np.empty(0, dtype=np.int64),
        evaluation=np.arange(50, 60, dtype=np.int64),
    )


def _fit_config() -> dict[str, object]:
    return {
        "histogram_range_ps": [-200.0, 200.0],
        "histogram_bin_ps": 10.0,
        "initial_half_width_ps": 100.0,
        "iteration_sigma": 2.5,
        "max_iterations": 1,
        "convergence_tolerance_ps": 0.1,
        "min_events": 1,
        "minimum_fit_bins": 3,
        "minimum_sigma_bins": 0.5,
    }


def test_linear_pair_regressor_is_antisymmetric_before_calibration() -> None:
    model = LinearPairRegressor(input_length=3)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([1.0, -2.0, 0.5]))
        model.pair_output_bias_ps.zero_()
    pair = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 1.0, -1.0]]])
    swapped = pair[:, [1, 0], :]
    assert torch.allclose(model(swapped), -model(pair))


@pytest.mark.parametrize(
    ("regularization", "alpha"),
    [("none", 0.0), ("ridge", 0.01), ("lasso", 1.0e-4)],
)
def test_linear_regression_variants_replay_checkpoint(
    tmp_path: Path,
    regularization: str,
    alpha: float,
) -> None:
    dataset = _dataset(tmp_path)
    run_dir = tmp_path / f"linear_{regularization}"
    config = {
        "datasets": [str(dataset.directory)],
        "model": {
            "type": "linear_regression",
            "name": f"tiny_{regularization}",
            "regularization": regularization,
            "alpha": alpha,
            "loss": {"type": "rmse"},
            "tolerance": 1.0e-8,
            "max_iterations": 50000,
            "ridge_solver": "auto",
            "lasso_selection": "cyclic",
        },
        "training": {
            "device": "cpu",
            "seed": 13,
            "batch_size": 8,
            "normalization_chunk_size": 8,
            "linear_materialization_chunk_size": 7,
            "num_workers": 0,
            "pin_memory": False,
            "baseline_guard_metric": None,
        },
        "fit": _fit_config(),
        "output": {"train_dir": str(run_dir)},
        "artifacts": {
            "save_config": False,
            "save_history": False,
            "save_plots": False,
            "save_last_checkpoint": False,
            "save_summary": True,
        },
        "input_transform": "normalize",
        "prediction": {"input_waveforms": "energy", "target": "prepared_led"},
    }
    summary = train_model(
        config,
        restart=True,
        logger=logging.getLogger("test.linear_regression"),
        prepared_datasets=[dataset],
    )

    assert summary["regularization"] == regularization
    assert Path(summary["coefficient_path"]).is_file()
    coefficient = np.load(summary["coefficient_path"])
    assert coefficient.shape == (dataset.input_length,)
    assert abs(float(summary["final_train_bias_ps"])) < 0.05
    assert summary["best_validation_rmse_ps"] < summary["uncorrected_led_validation_rmse_ps"]

    checkpoint = Path(summary["best_checkpoint"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["context"]["model_type"] == "linear_regression"
    assert payload["context"]["linear_regression"]["regularization"] == regularization
    assert payload["context"]["normalization"]["strategy"] == "feature"

    trained = TrainedModel(
        model_name=summary["model_name"],
        model_type=summary["model_type"],
        checkpoint=checkpoint,
        validation_rmse_ps=summary["best_validation_rmse_ps"],
        train_dir=run_dir,
        input_transform="normalize",
    )
    corrected = _evaluate_model(
        trained,
        dataset,
        {"batch_size": 4, "num_workers": 0, "pin_memory": False},
        torch.device("cpu"),
    )
    assert corrected.shape == (dataset.evaluation.size,)


def test_lasso_produces_sparse_coefficients(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    run_dir = tmp_path / "lasso_sparse"
    config = {
        "datasets": [str(dataset.directory)],
        "model": {
            "type": "linear_regression",
            "name": "sparse_lasso",
            "regularization": "lasso",
            "alpha": 0.05,
            "loss": {"type": "rmse"},
            "tolerance": 1.0e-8,
            "max_iterations": 50000,
            "lasso_selection": "cyclic",
        },
        "training": {
            "device": "cpu",
            "seed": 13,
            "batch_size": 8,
            "normalization_chunk_size": 8,
            "linear_materialization_chunk_size": 7,
            "num_workers": 0,
            "pin_memory": False,
            "baseline_guard_metric": None,
        },
        "fit": _fit_config(),
        "output": {"train_dir": str(run_dir)},
        "artifacts": {"save_summary": True, "save_last_checkpoint": False},
        "input_transform": "normalize",
        "prediction": {"input_waveforms": "energy", "target": "prepared_led"},
    }
    summary = train_model(
        config,
        restart=True,
        logger=logging.getLogger("test.linear_regression.sparse"),
        prepared_datasets=[dataset],
    )
    coefficient = np.load(summary["coefficient_path"])
    assert np.count_nonzero(coefficient) < coefficient.size
