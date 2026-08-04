from __future__ import annotations

import json
from pathlib import Path

import torch

from ml_pipeline.config import load_train_config
from ml_pipeline.models import build_model, count_model_parameters, model_registry
from ml_pipeline.models.cnn_regressor import AntisymmetricCNNRegressor


def _model_config() -> dict:
    return {
        "channels": [4, 8, 12],
        "kernel_sizes": [9, 7, 5],
        "strides": [4, 4, 2],
        "dilations": [1, 1, 1],
        "activation": "silu",
        "normalization": "batch",
        "conv_dropout": 0.0,
        "adaptive_pool_length": None,
        "dense_units": [],
        "dense_dropout": 0.0,
        "loss": {"type": "mse", "bias_weight": 0.0},
        "max_abs_single_channel_output_ps": 1000.0,
    }


def test_cnn_is_auto_discovered_and_lightweight() -> None:
    assert "cnn_regressor" in model_registry()
    count = count_model_parameters("cnn_regressor", _model_config(), 5119)
    assert 0 < count < 20_000


def test_cnn_pair_output_is_antisymmetric_without_pair_bias() -> None:
    model = build_model("cnn_regressor", _model_config(), 1024)
    assert isinstance(model, AntisymmetricCNNRegressor)
    model.eval()
    pair = torch.randn(3, 2, 1024)
    with torch.no_grad():
        forward = model(pair)
        reversed_pair = model(pair[:, [1, 0], :])
    assert forward.shape == (3,)
    assert torch.allclose(reversed_pair, -forward, atol=1e-5, rtol=1e-5)


def test_cnn_checkpoint_rebuild_uses_model_config(tmp_path: Path) -> None:
    model_config = _model_config()
    model = build_model("cnn_regressor", model_config, 513)
    checkpoint = tmp_path / "best.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "epoch": 1,
            "context": {
                "model_type": "cnn_regressor",
                "model_name": "tiny_cnn",
                "model_config": model_config,
                "input_length": 513,
                "input_transform": "differentiate",
                "normalization": {"mean_mV": 0.0, "std_mV": 1.0},
            },
        },
        checkpoint,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    rebuilt = build_model(
        payload["context"]["model_type"],
        payload["context"]["model_config"],
        payload["context"]["input_length"],
    )
    rebuilt.load_state_dict(payload["model_state"])
    assert isinstance(rebuilt, AntisymmetricCNNRegressor)
    assert rebuilt.shared.encoded_length == 17


def test_cnn_trains_and_writes_replayable_checkpoint(tmp_path: Path) -> None:
    import logging
    import numpy as np

    from ml_pipeline.dataset import PreparedDataset
    from ml_pipeline.training import train_model

    event_count = 32
    sample_count = 64
    rng = np.random.default_rng(7)
    shift = np.linspace(-1.0, 1.0, event_count, dtype=np.float32)
    grid = np.linspace(-3.0, 3.0, sample_count, dtype=np.float32)
    base = np.exp(-grid**2)
    windows = np.zeros((event_count, 2, sample_count), dtype=np.float32)
    for index, value in enumerate(shift):
        windows[index, 0] = np.interp(grid - 0.10 * value, grid, base).astype(np.float32)
        windows[index, 1] = np.interp(grid + 0.10 * value, grid, base).astype(np.float32)
        windows[index] += 0.005 * rng.normal(size=(2, sample_count)).astype(np.float32)
    target_ps = 20.0 * shift
    led = np.column_stack(
        [np.rint(target_ps * 1000.0).astype(np.int64), np.zeros(event_count, dtype=np.int64)]
    )
    directory = tmp_path / "prepared"
    directory.mkdir()
    dataset = PreparedDataset(
        directory=directory,
        manifest={
            "format_version": 2,
            "fingerprint": "cnn-test",
            "name": "cnn-test",
            "true_tof_ps": 0.0,
            "led_timestamp_source": "energy_channels",
            "cfd_timestamp_source": "energy_channels",
            "ml_window_alignment_source": "energy_channel_led",
            "timing_channel_waveforms_saved": False,
            "waveform_grid": "native_acquisition_samples",
        },
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
        relative_time_ps=np.arange(sample_count, dtype=np.float32) * 12.5,
        train=np.arange(0, 20, dtype=np.int64),
        validation=np.arange(20, 28, dtype=np.int64),
        test=np.empty(0, dtype=np.int64),
        evaluation=np.arange(28, 32, dtype=np.int64),
    )
    run_dir = tmp_path / "cnn_run"
    config = {
        "datasets": [str(directory)],
        "model": {
            "type": "cnn_regressor",
            "name": "tiny_cnn",
            "channels": [2, 4],
            "kernel_sizes": [7, 5],
            "strides": [4, 2],
            "dilations": [1, 1],
            "activation": "silu",
            "normalization": "none",
            "conv_dropout": 0.0,
            "adaptive_pool_length": None,
            "dense_units": [],
            "dense_dropout": 0.0,
            "loss": {
                "type": "var_bias",
                "bias_weight": 0.5,
                "bias_normalization": "target_std",
            },
            "max_abs_single_channel_output_ps": 1000.0,
        },
        "optimizer": {"learning_rate": 0.005, "weight_decay": 0.0},
        "training": {
            "device": "cpu",
            "seed": 11,
            "initialization_seed": 11,
            "fit_interval_epochs": 1,
            "fit_train_during_training": False,
            "fit_validation_during_training": False,
            "epochs": 2,
            "random_pair_swap": False,
            "batch_size": 8,
            "mixed_precision": False,
            "gradient_clip_norm": 5.0,
            "early_stopping_patience": 2,
            "early_stopping_min_delta_ps": 0.0,
            "normalization_chunk_size": 8,
            "num_workers": 0,
            "pin_memory": False,
            "selection_metric": "validation_rmse",
            "baseline_guard_metric": None,
            "zero_bias_constraint": {"enabled": False, "mode": "residual_mean"},
        },
        "fit": {
            "histogram_range_ps": [-100.0, 100.0],
            "histogram_bin_ps": 5.0,
            "initial_half_width_ps": 50.0,
            "iteration_sigma": 2.5,
            "max_iterations": 1,
            "convergence_tolerance_ps": 0.1,
            "min_events": 1,
            "minimum_fit_bins": 3,
            "minimum_sigma_bins": 0.5,
        },
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
        "input_transform": "none",
        "prediction": {"input_waveforms": "energy", "target": "prepared_led"},
    }
    summary = train_model(
        config,
        restart=True,
        logger=logging.getLogger("test.cnn"),
        prepared_datasets=[dataset],
    )
    checkpoint = Path(summary["best_checkpoint"])
    assert checkpoint.is_file()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["context"]["model_type"] == "cnn_regressor"
    assert abs(float(summary["final_train_bias_ps"])) < 1.0e-4

    from ml_pipeline.evaluation import TrainedModel, _evaluate_model

    trained = TrainedModel(
        model_name=summary["model_name"],
        model_type=summary["model_type"],
        checkpoint=checkpoint,
        validation_rmse_ps=summary["best_validation_rmse_ps"],
        train_dir=run_dir,
        input_transform="none",
        input_waveform_source="energy",
        prediction_target="prepared_led",
    )
    corrected = _evaluate_model(
        trained,
        dataset,
        {
            "batch_size": 4,
            "num_workers": 0,
            "pin_memory": False,
            "output": {"evaluation_dir": str(tmp_path / "evaluation")},
        },
        torch.device("cpu"),
    )
    assert corrected.shape == (dataset.evaluation.size,)
