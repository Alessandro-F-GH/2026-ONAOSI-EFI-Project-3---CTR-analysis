from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import pytest
import torch

from ml_pipeline.dataset import PreparedDataset
from ml_pipeline.evaluation import TrainedModel, _evaluate_model
from ml_pipeline.models.linear_svr import LinearPairSVR
from ml_pipeline.training import train_model


def _dataset(tmp_path: Path) -> PreparedDataset:
    event_count = 36
    sample_count = 4
    x = np.linspace(-3.0, 3.0, event_count, dtype=np.float32)
    target_ps = 12.0 * x + 7.0
    windows = np.zeros((event_count, 2, sample_count), dtype=np.float32)
    windows[:, 0, 0] = x
    windows[:, 0, 1] = 0.5 * x
    windows[:, 1, 2] = -0.25 * x
    led = np.column_stack(
        [np.rint(target_ps * 1000.0).astype(np.int64), np.zeros(event_count, dtype=np.int64)]
    )
    directory = tmp_path / "prepared"
    directory.mkdir()
    manifest = {
        "format_version": 2,
        "fingerprint": "linear-svr-test",
        "name": "linear-svr-test",
        "true_tof_ps": 0.0,
        "led_timestamp_source": "energy_channels",
        "cfd_timestamp_source": "energy_channels",
        "ml_window_alignment_source": "energy_channel_led",
        "timing_channel_waveforms_saved": False,
        "waveform_grid": "native_acquisition_samples",
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
        train=np.arange(0, 24, dtype=np.int64),
        validation=np.arange(24, 30, dtype=np.int64),
        test=np.empty(0, dtype=np.int64),
        evaluation=np.arange(30, 36, dtype=np.int64),
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


def test_linear_pair_svr_is_antisymmetric_before_global_calibration() -> None:
    model = LinearPairSVR(input_length=3)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([1.0, -2.0, 0.5]))
        model.pair_output_bias_ps.zero_()
    pair = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 1.0, -1.0]]])
    swapped = pair[:, [1, 0], :]
    assert torch.allclose(model(swapped), -model(pair))


@pytest.mark.parametrize("input_transform", ["none", "normalize"])
def test_linear_svr_scans_all_epsilons_and_replays_checkpoint(
    tmp_path: Path, input_transform: str
) -> None:
    dataset = _dataset(tmp_path)
    run_dir = tmp_path / f"linear_svr_run_{input_transform}"
    config = {
        "datasets": [str(dataset.directory)],
        "model": {
            "type": "linear_svr",
            "name": f"tiny_linear_svr_{input_transform}",
            "C": 100.0,
            "epsilon_values": [0.0, 1.0, 5.0],
            "svm_loss": "epsilon_insensitive",
            "loss": {"type": "variance_bias", "bias_weight": 1.0},
            "tolerance": 1.0e-6,
            "max_iterations": 20000,
            "dual": "auto",
        },
        "training": {
            "device": "cpu",
            "seed": 13,
            "batch_size": 8,
            "normalization_chunk_size": 8,
            "svr_materialization_chunk_size": 7,
            "num_workers": 0,
            "pin_memory": False,
            "baseline_guard_metric": None,
        },
        "fit": _fit_config(),
        "output": {"train_dir": str(run_dir)},
        "plotting": {"dpi": 72},
        "logging": {"level": "INFO"},
        "artifacts": {
            "save_config": False,
            "save_history": True,
            "save_plots": False,
            "save_last_checkpoint": True,
            "save_summary": True,
        },
        "input_transform": input_transform,
        "prediction": {"input_waveforms": "energy", "target": "prepared_led"},
    }
    summary = train_model(
        config,
        restart=True,
        logger=logging.getLogger("test.linear_svr"),
        prepared_datasets=[dataset],
    )

    with (run_dir / "epsilon_scan.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [float(row["epsilon_ps"]) for row in rows] == [0.0, 1.0, 5.0]
    assert sum(row["selected_best"] == "True" for row in rows) == 1
    assert abs(float(summary["final_train_bias_ps"])) < 1.0e-5
    assert summary["best_validation_rmse_ps"] < summary["uncorrected_led_validation_rmse_ps"]

    checkpoint = Path(summary["best_checkpoint"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["context"]["model_type"] == "linear_svr"
    assert payload["context"]["linear_svr"]["epsilon_values_ps"] == [0.0, 1.0, 5.0]
    expected_strategy = "feature" if input_transform == "normalize" else "global"
    assert payload["context"]["normalization"]["strategy"] == expected_strategy

    trained = TrainedModel(
        model_name=summary["model_name"],
        model_type=summary["model_type"],
        checkpoint=checkpoint,
        validation_rmse_ps=summary["best_validation_rmse_ps"],
        train_dir=run_dir,
        input_transform=input_transform,
    )
    corrected = _evaluate_model(
        trained,
        dataset,
        {"batch_size": 4, "num_workers": 0, "pin_memory": False},
        torch.device("cpu"),
    )
    assert corrected.shape == (dataset.evaluation.size,)


def test_linear_svr_selection_losses() -> None:
    from ml_pipeline.models.linear_svr import _selection_value

    metrics = {"variance_ps2": 9.0, "rmse_ps": 5.0, "bias_ps": -4.0}
    assert _selection_value(metrics, loss_type="variance", bias_weight=3.0) == 9.0
    assert _selection_value(metrics, loss_type="rmse", bias_weight=3.0) == 5.0
    assert _selection_value(metrics, loss_type="variance_bias", bias_weight=3.0) == 57.0
