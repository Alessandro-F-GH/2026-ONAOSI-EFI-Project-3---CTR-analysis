from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from ml_pipeline.dataset import PreparedDataset
from ml_pipeline.evaluation import TrainedModel, _evaluate_model
from ml_pipeline.models.shapelet_regressor import ShapeletPairRegressor
from ml_pipeline.training import train_model


def _fit_config() -> dict[str, object]:
    return {
        "histogram_range_ps": [-300.0, 300.0],
        "histogram_bin_ps": 10.0,
        "initial_half_width_ps": 200.0,
        "iteration_sigma": 2.5,
        "max_iterations": 1,
        "convergence_tolerance_ps": 0.1,
        "min_events": 1,
        "minimum_fit_bins": 3,
        "minimum_sigma_bins": 0.5,
    }


def _dataset(tmp_path: Path) -> PreparedDataset:
    rng = np.random.default_rng(44)
    events = 90
    samples = 41
    latent = np.linspace(-2.5, 2.5, events)
    base = np.exp(-0.5 * ((np.arange(samples) - 22.0) / 6.0) ** 2)
    tail = np.exp(-0.5 * ((np.arange(samples) - 34.0) / 2.0) ** 2)
    windows = np.zeros((events, 2, samples), dtype=np.float32)
    for index, value in enumerate(latent):
        windows[index, 0] = base + 0.25 * value * tail + 0.01 * rng.normal(size=samples)
        windows[index, 1] = base - 0.25 * value * tail + 0.01 * rng.normal(size=samples)
    target_ps = 35.0 * latent - 8.0
    led = np.column_stack(
        [np.rint(target_ps * 1000.0).astype(np.int64), np.zeros(events, dtype=np.int64)]
    )
    directory = tmp_path / "prepared_shapelet"
    directory.mkdir()
    manifest = {
        "format_version": 4,
        "fingerprint": "shapelet-regressor-test",
        "name": "shapelet-regressor-test",
        "true_tof_ps": 0.0,
        "led_timestamp_source": "energy_channels",
        "cfd_timestamp_source": "energy_channels",
        "ml_window_alignment_source": "energy_channel_led",
        "timing_channel_waveforms_saved": False,
        "waveform_grid": "native_acquisition_samples",
        "input_components": ["energy"],
        "input_component_lengths": [samples],
    }
    return PreparedDataset(
        directory=directory,
        manifest=manifest,
        event_id=np.arange(events, dtype=np.int64),
        event_index=np.arange(events, dtype=np.int64),
        source_file_id=np.zeros(events, dtype=np.int32),
        source_run_index=np.zeros(events, dtype=np.int32),
        bias_voltage_V=np.full(events, 47.0, dtype=np.float32),
        amplitude_mV=np.ones((events, 2), dtype=np.float32),
        noise_rms_mV=np.zeros((events, 2), dtype=np.float32),
        trigger_index=np.zeros((events, 2), dtype=np.int32),
        led_time_fs=led,
        cfd_time_fs=led,
        windows_mV=windows,
        relative_time_ps=np.linspace(-2000.0, 6000.0, samples, dtype=np.float32),
        train=np.arange(0, 60, dtype=np.int64),
        validation=np.arange(60, 75, dtype=np.int64),
        test=np.empty(0, dtype=np.int64),
        evaluation=np.arange(75, 90, dtype=np.int64),
    )


def test_radius_zero_dtw_equals_mse() -> None:
    base = {
        "n_shapelets": 1,
        "subsampling_factor": 1,
        "_serialized_shapelet_count": 1,
        "_serialized_shapelet_width": 3,
        "_serialized_subsampled_length": 3,
        "dtw_radius_points": 0,
    }
    mse = ShapeletPairRegressor({**base, "distance_metric": "mse"}, 3)
    dtw = ShapeletPairRegressor({**base, "distance_metric": "dtw"}, 3)
    for model in (mse, dtw):
        with torch.no_grad():
            model.selected_source_indices.copy_(torch.tensor([0, 1, 2]))
            model.shapelet_values[0].copy_(torch.tensor([0.0, 1.0, 0.0]))
            model.shapelet_lengths[0] = 3
            model.shapelet_starts[0] = 0
            model.feature_mean.zero_()
            model.feature_std.fill_(1.0)
            model.coefficient.fill_(1.0)
    pair = torch.tensor([[[0.0, 2.0, 0.0], [0.0, 0.0, 0.0]]])
    assert torch.allclose(mse(pair), dtw(pair), atol=1.0e-6)
    assert torch.allclose(mse(pair), torch.tensor([1.0 / 3.0]), atol=1.0e-6)


def test_shapelet_regressor_trains_and_replays_checkpoint(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    run_dir = tmp_path / "shapelet_run"
    config = {
        "datasets": [str(dataset.directory)],
        "model": {
            "type": "shapelet_regressor",
            "name": "tiny_shapelet",
            "n_shapelets": 3,
            "candidate_pool_size": 5,
            "subsampling_factor": 2,
            "shapelet_lengths_ns": [0.8, 1.2],
            "candidates_per_group": 8,
            "extreme_fraction": 0.2,
            "score_events": 40,
            "redundancy_threshold": 0.99,
            "distance_metric": "dtw",
            "dtw_radius_points": 1,
            "local_z_normalize": False,
            "ridge_alpha": 10.0,
            "loss": {"type": "rmse"},
        },
        "training": {
            "device": "cpu",
            "seed": 17,
            "batch_size": 16,
            "normalization_chunk_size": 16,
            "shapelet_materialization_chunk_size": 16,
            "shapelet_feature_chunk_size": 20,
            "num_workers": 0,
            "pin_memory": False,
            "random_pair_swap": False,
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
        logger=logging.getLogger("test.shapelet_regressor"),
        prepared_datasets=[dataset],
    )
    assert summary["model_type"] == "shapelet_regressor"
    assert summary["n_shapelets"] == 3
    assert summary["subsampling_factor"] == 2
    shapelet_csv = Path(summary["shapelet_csv"])
    assert shapelet_csv.is_file()
    assert len(shapelet_csv.read_text(encoding="utf-8").splitlines()) == 4

    checkpoint = Path(summary["best_checkpoint"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["context"]["shapelet_regressor"]["distance_metric"] == "dtw"
    assert payload["model_state"]["shapelet_values"].shape[0] == 3

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
        {"batch_size": 8, "num_workers": 0, "pin_memory": False},
        torch.device("cpu"),
    )
    assert corrected.shape == (dataset.evaluation.size,)
    assert np.all(np.isfinite(corrected))


def test_model_space_lists_multiple_subsampling_factors() -> None:
    path = Path(__file__).resolve().parents[1] / "config" / "model_spaces" / "shapelet_regressor.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    factors = config["search"]["parameters"]["model.subsampling_factor"]
    assert isinstance(factors, list)
    assert len(factors) >= 2
    assert all(int(value) > 0 for value in factors)


def test_compact_shapelet_plot_reserves_legend_space(tmp_path: Path) -> None:
    from ml_pipeline.study import _plot_compact_shapelet_model

    rows = []
    for fold in range(2):
        for rank in range(1, 4):
            rows.append(
                {
                    "fold_id": fold,
                    "rank": rank,
                    "start_time_ns": 10.0 * rank,
                    "end_time_ns": 10.0 * rank + 2.0,
                    "values": "0 1 0 -1 0",
                    "mean_abs_contribution_ps": 3.0 * rank,
                }
            )
    destination = tmp_path / "compact.png"
    _plot_compact_shapelet_model(
        rows=rows,
        destination=destination,
        title="Compact shapelet test",
        dpi=90,
    )
    assert destination.is_file()
    assert destination.stat().st_size > 0
