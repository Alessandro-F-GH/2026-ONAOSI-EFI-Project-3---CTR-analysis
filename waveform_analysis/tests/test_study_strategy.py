from __future__ import annotations

from pathlib import Path

import numpy as np

from ml_pipeline.dataset import PreparedDataset
from ml_pipeline.input_transform import (
    apply_input_transform,
    transformed_dataset_input_length,
)
from ml_pipeline.prediction import prediction_window_dataset_view
from ml_pipeline.robust_selection import fit_median_mad_z, robust_z_mask
from ml_pipeline.study import _build_preprocess_config


def _dataset(tmp_path: Path) -> PreparedDataset:
    energy = np.arange(2 * 2 * 5, dtype=np.float32).reshape(2, 2, 5)
    timing_aligned_energy = energy + 50.0
    timing = (100 + np.arange(2 * 2 * 4, dtype=np.float32)).reshape(2, 2, 4)
    energy_led = np.asarray([[1000, 900], [1100, 950]], dtype=np.int64)
    timing_led = np.asarray([[2000, 1800], [2200, 1900]], dtype=np.int64)
    return PreparedDataset(
        directory=tmp_path,
        manifest={"fingerprint": "study-test", "true_tof_ps": 0.0},
        event_id=np.arange(2),
        event_index=np.arange(2),
        source_file_id=np.zeros((2, 2), dtype=np.int64),
        source_run_index=np.zeros(2, dtype=np.int64),
        bias_voltage_V=np.zeros(2),
        amplitude_mV=np.zeros((2, 2)),
        noise_rms_mV=np.zeros((2, 2)),
        trigger_index=np.zeros((2, 2), dtype=np.int32),
        led_time_fs=timing_led,
        cfd_time_fs=timing_led,
        windows_mV=energy,
        relative_time_ps=np.asarray([-2000, -1000, 0, 1000, 2000], dtype=np.float32),
        energy_led_time_fs=energy_led,
        timing_led_time_fs=timing_led,
        timing_aligned_energy_windows_mV=timing_aligned_energy,
        timing_windows_mV=timing,
        timing_relative_time_ps=np.asarray([-1500, -500, 500, 1500], dtype=np.float32),
        train=np.asarray([0], dtype=np.int64),
        validation=np.asarray([1], dtype=np.int64),
        evaluation=np.asarray([0, 1], dtype=np.int64),
    )


def test_combined_mode_preserves_modality_and_transform_order(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    view = prediction_window_dataset_view(
        dataset,
        input_waveforms="energy_timing",
        target="timing_led",
        before_ns=2.0,
        after_ns=2.0,
    )
    pair = np.asarray(view.windows_mV[0])
    expected_raw = np.concatenate(
        (dataset.timing_aligned_energy_windows_mV[0], dataset.timing_windows_mV[0]),
        axis=-1,
    )
    np.testing.assert_array_equal(pair, expected_raw)
    assert view.manifest["input_components"] == ["energy", "timing"]
    assert view.manifest["input_component_lengths"] == [5, 4]

    transformed = apply_input_transform(
        pair,
        "concatenate_diff",
        view.manifest["input_component_lengths"],
    )
    expected = np.concatenate(
        (
            dataset.timing_aligned_energy_windows_mV[0],
            np.diff(dataset.timing_aligned_energy_windows_mV[0], axis=-1),
            dataset.timing_windows_mV[0],
            np.diff(dataset.timing_windows_mV[0], axis=-1),
        ),
        axis=-1,
    )
    np.testing.assert_array_equal(transformed, expected)
    assert transformed_dataset_input_length(view, "concatenate_diff") == 16


def test_energy_windows_follow_target_led_alignment(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    energy_target = prediction_window_dataset_view(
        dataset,
        input_waveforms="energy",
        target="energy_led",
        before_ns=2.0,
        after_ns=2.0,
    )
    timing_target = prediction_window_dataset_view(
        dataset,
        input_waveforms="energy",
        target="timing_led",
        before_ns=2.0,
        after_ns=2.0,
    )
    np.testing.assert_array_equal(energy_target.windows_mV[0], dataset.windows_mV[0])
    np.testing.assert_array_equal(
        timing_target.windows_mV[0], dataset.timing_aligned_energy_windows_mV[0]
    )
    assert energy_target.manifest["ml_window_alignment_source"] == "energy_channel_led"
    assert timing_target.manifest["ml_window_alignment_source"] == "timing_channel_led"


def test_robust_z_uses_training_median_and_mad() -> None:
    training = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0, 100.0])
    fitted = fit_median_mad_z(training)
    assert fitted.center_ps == 0.5
    assert fitted.scale_ps > 0.0
    values = np.asarray([0.0, 1.0, 100.0])
    mask = robust_z_mask(values, fitted, 4.0)
    np.testing.assert_array_equal(mask, np.asarray([True, True, False]))


def test_generated_preprocessing_has_modality_specific_denoising(tmp_path: Path) -> None:
    root_file = tmp_path / "run.root"
    root_file.write_bytes(b"placeholder")
    config = {
        "windows_ns": [
            {"before_ns": 4.0, "after_ns": 12.0},
            {"before_ns": 6.0, "after_ns": 20.0},
        ],
        "data": {
            "true_tof_ps": 0.0,
            "channels": {
                "energy": [1, 2],
                "polarities": [1, 1],
                "timing": [3, 4],
                "timing_polarities": [1, 1],
            },
        },
        "preprocessing": {
            "common": {
                "baseline_samples": 500,
                "search_trigger_threshold_mV": 50.0,
                "analysis_crop_ns": {"before": 5.0, "after": 60.0},
                "led_threshold_mV": 7.0,
                "cfd_fraction": 0.045,
            },
            "energy": {
                "denoising": {
                    "enabled": True,
                    "method": "butterworth_lowpass",
                    "cutoff_GHz": 0.5,
                    "order": 4,
                }
            },
            "timing": {"denoising": {"enabled": False}},
            "selection": {},
            "photopeak": {"enabled": False},
        },
        "split": {"blind_fraction": 0.15, "guard_gap_events": 1000},
    }
    generated = _build_preprocess_config(config, root_file, "run", tmp_path / "out")
    assert generated["waveform"]["denoising"]["enabled"] is True
    assert generated["waveform"]["timing_channel_led"]["denoising"]["enabled"] is False
    assert generated["waveform"]["ml_window_ns"] == {"before": 6.0, "after": 20.0}
    assert generated["selection"]["led_outlier_rejection"] == {"enabled": False}
    assert generated["split"]["development_blind"] is True
    assert generated["split"]["train_fraction"] == 0.85
    assert generated["split"]["validation_fraction"] == 0.0
    assert generated["split"]["test_fraction"] == 0.15


def test_combined_mode_checkpoint_replays_window_and_component_transform(tmp_path: Path) -> None:
    import logging

    from ml_pipeline.evaluation import evaluate_trained_model, load_trained_model
    from ml_pipeline.training import train_model
    from ml_pipeline.training_utils import resolve_device

    event_count = 24
    energy_length = 17
    timing_length = 13
    rng = np.random.default_rng(91)
    energy = rng.normal(size=(event_count, 2, energy_length)).astype(np.float32)
    timing_aligned_energy = energy + 0.25
    timing = rng.normal(size=(event_count, 2, timing_length)).astype(np.float32)
    target = np.linspace(-20.0, 20.0, event_count)
    timing_led = np.column_stack(
        [np.rint(target * 1000.0).astype(np.int64), np.zeros(event_count, dtype=np.int64)]
    )
    energy_led = np.column_stack(
        [np.rint((target + 5.0) * 1000.0).astype(np.int64), np.zeros(event_count, dtype=np.int64)]
    )
    dataset = PreparedDataset(
        directory=tmp_path / "canonical",
        manifest={
            "format_version": 2,
            "fingerprint": "combined-replay-test",
            "name": "combined-replay-test",
            "true_tof_ps": 0.0,
            "led_timestamp_source": "timing_channels",
            "cfd_timestamp_source": "timing_channels",
            "ml_window_alignment_source": "timing_channel_led",
            "timing_channel_waveforms_saved": True,
            "waveform_grid": "native_acquisition_samples",
        },
        event_id=np.arange(event_count, dtype=np.int64),
        event_index=np.arange(event_count, dtype=np.int64),
        source_file_id=np.zeros(event_count, dtype=np.int32),
        source_run_index=np.zeros(event_count, dtype=np.int32),
        bias_voltage_V=np.zeros(event_count, dtype=np.float32),
        amplitude_mV=np.ones((event_count, 2), dtype=np.float32),
        noise_rms_mV=np.zeros((event_count, 2), dtype=np.float32),
        trigger_index=np.zeros((event_count, 2), dtype=np.int32),
        led_time_fs=timing_led,
        cfd_time_fs=timing_led,
        windows_mV=energy,
        relative_time_ps=np.linspace(-2000.0, 2000.0, energy_length, dtype=np.float32),
        energy_led_time_fs=energy_led,
        timing_led_time_fs=timing_led,
        timing_aligned_energy_windows_mV=timing_aligned_energy,
        timing_windows_mV=timing,
        timing_relative_time_ps=np.linspace(-1800.0, 1800.0, timing_length, dtype=np.float32),
        train=np.arange(0, 14, dtype=np.int64),
        validation=np.arange(14, 20, dtype=np.int64),
        evaluation=np.arange(20, 24, dtype=np.int64),
    )
    window = prediction_window_dataset_view(
        dataset,
        input_waveforms="energy_timing",
        target="timing_led",
        before_ns=1.5,
        after_ns=1.5,
    )
    run_dir = tmp_path / "combined_run"
    config = {
        "datasets": ["injected"],
        "model": {
            "type": "mlp_regressor",
            "name": "combined_mlp",
            "hidden_units": [4],
            "activation": "silu",
            "loss": {"type": "mse"},
            "dropout": 0.0,
            "batch_norm": False,
            "max_abs_single_channel_output_ps": 1000.0,
        },
        "optimizer": {"learning_rate": 0.001, "weight_decay": 0.0},
        "training": {
            "device": "cpu",
            "seed": 3,
            "fit_interval_epochs": 1,
            "fit_train_during_training": False,
            "fit_validation_during_training": False,
            "epochs": 1,
            "random_pair_swap": False,
            "batch_size": 4,
            "mixed_precision": False,
            "gradient_clip_norm": 5.0,
            "early_stopping_patience": 1,
            "early_stopping_min_delta_ps": 0.0,
            "normalization_chunk_size": 8,
            "num_workers": 0,
            "pin_memory": False,
            "selection_metric": "validation_loss",
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
            "save_history": False,
            "save_plots": False,
            "save_last_checkpoint": False,
            "save_summary": True,
        },
        "input_transform": "concatenate_diff",
        "prediction": {"input_waveforms": "energy_timing", "target": "timing_led"},
    }
    train_model(
        config,
        restart=True,
        logger=logging.getLogger("test.combined"),
        prepared_datasets=[window],
        data_view={"window_before_ns": 1.5, "window_after_ns": 1.5},
    )
    trained = load_trained_model(run_dir)
    prediction = evaluate_trained_model(
        trained,
        dataset,
        {
            "device": "cpu",
            "batch_size": 4,
            "num_workers": 0,
            "pin_memory": False,
            "input_transform_cache_dir": str(tmp_path / "eval_cache"),
            "output": {"evaluation_dir": str(tmp_path / "evaluation")},
        },
        resolve_device("cpu"),
    )
    assert prediction.corrected_ps.shape == (4,)
    assert prediction.input_waveform_source == "energy_timing"
    assert prediction.input_transform == "concatenate_diff"


def test_study_config_discovers_root_folder_instead_of_path_list(tmp_path: Path) -> None:
    import json

    from ml_pipeline.study_config import load_study_config

    roots = tmp_path / "roots"
    roots.mkdir()
    (roots / "b.root").write_bytes(b"")
    (roots / "a.root").write_bytes(b"")
    model_spaces = tmp_path / "model_spaces"
    model_spaces.mkdir()
    (model_spaces / "toy.json").write_text(
        json.dumps(
            {
                "id": "toy",
                "model_type": "mlp_regressor",
                "supported_losses": ["mse"],
                "base_train_config": {},
                "search": {"method": "grid", "parameters": {}},
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "study.json"
    config_path.write_text(
        json.dumps(
            {
                "experiment": {"name": "folder-test", "output_dir": "out"},
                "data": {"root_folder": "roots"},
                "preprocessing": {},
                "windows_ns": [{"start_ns": -1.0, "end_ns": 2.0}],
                "channel_modes": ["energy_to_energy"],
                "input_transforms": ["none"],
                "losses": [{"id": "mse", "type": "mse"}],
                "models": ["toy"],
                "model_spaces_dir": "model_spaces",
                "cross_validation": {"n_splits": 2},
                "selection": {"z_threshold": 4.0},
            }
        ),
        encoding="utf-8",
    )
    resolved = load_study_config(config_path, tmp_path)
    assert [Path(value).name for value in resolved["root_files"]] == ["a.root", "b.root"]
    assert "root_paths" not in resolved["data"]


def test_results_csv_is_numeric_and_metadata_is_not_repeated(tmp_path: Path) -> None:
    import csv
    import json

    from ml_pipeline.study import _read_results, _write_results

    results_path = tmp_path / "all_results.csv"
    root_path = tmp_path / "inputs" / "run_01.root"
    row = {
        "row_key": "0123456789abcdef01234567",
        "record_type": "trial_definition",
        "experiment_id": "compact-study",
        "root_id": "run_01_abcd1234",
        "root_file": str(root_path.resolve()),
        "channel_mode": "energy_to_timing",
        "model_id": "linear_svr",
        "model_type": "linear_svr",
        "loss_id": "mse",
        "loss_type": "mse",
        "input_transform": "concatenate_diff",
        "window_id": "m4_p12",
        "window_start_ns": -4.0,
        "window_end_ns": 12.0,
        "trial_id": "t0001_deadbeef",
        "fold_id": "",
        "split": "",
        "statistic": "",
        "is_selected_hyperparameters": 0,
        "is_selected_window": 0,
        "status": "completed",
        "n_events": "",
        "loss": "",
        "bias_ps": "",
        "ctr_ps": "",
        "baseline_ctr_ps": "",
        "relative_improvement_pct": "",
        "outlier_center_ps": "",
        "outlier_scale_ps": "",
        "outlier_scale_method": "median_mad",
        "outlier_z_threshold": 4.0,
        "runtime_seconds": "",
        "pearson_cv_blind": "",
        "spearman_cv_blind": "",
        "mean_cv_blind_gap_ps": "",
        "blind_rank_of_cv_selected_window": "",
        "blind_regret_ps": "",
        "params_json": json.dumps({"model.C": 1.0, "model.svm_loss": "epsilon_insensitive"}),
        "error": "",
    }
    _write_results(results_path, [row])

    with results_path.open("r", encoding="utf-8", newline="") as stream:
        compact = list(csv.DictReader(stream))
    assert len(compact) == 1
    assert "root_file" not in compact[0]
    assert "params_json" not in compact[0]
    assert "experiment_id" not in compact[0]
    assert "model_type" not in compact[0]
    assert "loss_type" not in compact[0]
    assert "window_id" not in compact[0]
    assert "row_key" not in compact[0]
    for value in compact[0].values():
        if value != "":
            float(value)

    metadata_path = tmp_path / "results_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["roots"][0]["path"] == str(root_path.resolve())
    assert metadata["trials"][0]["parameters"]["model.C"] == 1.0

    decoded = _read_results(results_path)
    assert decoded[0]["row_key"] == row["row_key"]
    assert decoded[0]["root_file"] == row["root_file"]
    assert decoded[0]["model_type"] == row["model_type"]
    assert decoded[0]["loss_type"] == row["loss_type"]
    assert decoded[0]["window_id"] == row["window_id"]
    assert json.loads(decoded[0]["params_json"])["model.svm_loss"] == "epsilon_insensitive"
