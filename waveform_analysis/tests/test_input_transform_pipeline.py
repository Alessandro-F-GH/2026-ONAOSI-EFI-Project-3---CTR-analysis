from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ml_pipeline.dataset import (
    PreparedDataset,
    load_prepared_dataset,
    prepared_dataset_view,
)
from ml_pipeline.evaluation import TrainedModel, _evaluate_model
from ml_pipeline.input_transform import (
    apply_input_transform,
    materialize_training_input_cache,
    resolve_input_transform,
    transform_relative_time_ps,
    transformed_input_length,
)
from ml_pipeline.models import build_model
from ml_pipeline.prediction import prediction_dataset_view, resolve_prediction_config
from ml_pipeline.torch_data import (
    CorrectionDataset,
    Normalization,
    compute_normalization,
)


def _prepared_dataset(tmp_path: Path) -> PreparedDataset:
    event_count = 8
    sample_count = 5
    windows = np.arange(
        event_count * 2 * sample_count, dtype=np.float32
    ).reshape(event_count, 2, sample_count)
    led = np.column_stack(
        [
            np.arange(event_count, dtype=np.int64) * 1_000,
            np.zeros(event_count, dtype=np.int64),
        ]
    )
    cfd = led + np.asarray([100, 0], dtype=np.int64)
    directory = tmp_path / "canonical_prepared"
    directory.mkdir()
    manifest = {
        "format_version": 1,
        "fingerprint": "synthetic-canonical-fingerprint",
        "name": "synthetic",
        "true_tof_ps": 0.0,
        "led_timestamp_source": "energy_channels",
        "cfd_timestamp_source": "energy_channels",
        "ml_window_alignment_source": "energy_channel_led",
        "timing_channel_waveforms_saved": False,
        "waveform_grid": "native_acquisition_samples",
        "is_canonical_prepared_dataset": True,
        "model_input_transform_applied": False,
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
        cfd_time_fs=cfd,
        windows_mV=windows,
        relative_time_ps=np.arange(sample_count, dtype=np.float32) * 100.0,
        energy_led_time_fs=led,
        timing_led_time_fs=led + np.asarray([2_000, 0], dtype=np.int64),
        timing_windows_mV=windows + np.float32(1000.0),
        timing_relative_time_ps=np.arange(sample_count, dtype=np.float32) * 200.0,
        train=np.asarray([0, 1, 2, 3], dtype=np.int64),
        validation=np.asarray([4, 5], dtype=np.int64),
        test=np.empty(0, dtype=np.int64),
        evaluation=np.asarray([6, 7], dtype=np.int64),
    )


def test_input_transform_config_alias_and_conflict() -> None:
    assert resolve_input_transform({}) == "none"
    assert (
        resolve_input_transform({"model": {"input_transform": "first_difference"}})
        == "differentiate"
    )
    assert resolve_input_transform({"input_transform": "concatenate_diff"}) == "concatenate_diff"
    assert resolve_input_transform({"input_transform": "concat_diff"}) == "concatenate_diff"
    with pytest.raises(ValueError, match="Conflicting input transforms"):
        resolve_input_transform(
            {
                "input_transform": "none",
                "model": {"input_transform": "differentiate"},
            }
        )




def test_concatenate_diff_preserves_raw_then_difference_order() -> None:
    waveform = np.asarray([[1.0, 4.0, 9.0, 16.0]], dtype=np.float32)
    time_ps = np.asarray([-100.0, 0.0, 100.0, 200.0], dtype=np.float32)

    transformed = apply_input_transform(waveform, "concatenate_diff")
    expected = np.asarray(
        [[1.0, 4.0, 9.0, 16.0, 3.0, 5.0, 7.0]], dtype=np.float32
    )
    np.testing.assert_array_equal(transformed, expected)
    np.testing.assert_allclose(
        transform_relative_time_ps(time_ps, "concatenate_diff"),
        np.asarray([-100.0, 0.0, 100.0, 200.0, -50.0, 50.0, 150.0]),
    )
    assert transformed_input_length(4, "concatenate_diff") == 7


def test_energy_and_timing_prediction_views(tmp_path: Path) -> None:
    dataset = _prepared_dataset(tmp_path)
    energy = prediction_dataset_view(
        dataset, input_waveforms="energy", target="energy_led"
    )
    timing = prediction_dataset_view(
        dataset, input_waveforms="timing", target="timing_led"
    )
    np.testing.assert_array_equal(energy.windows_mV, dataset.windows_mV)
    np.testing.assert_array_equal(energy.led_time_fs, dataset.energy_led_time_fs)
    np.testing.assert_array_equal(timing.windows_mV, dataset.timing_windows_mV)
    np.testing.assert_array_equal(timing.led_time_fs, dataset.timing_led_time_fs)
    np.testing.assert_array_equal(
        timing.relative_time_ps, dataset.timing_relative_time_ps
    )
    assert resolve_prediction_config({}) == {
        "input_waveforms": "energy",
        "target": "prepared_led",
    }

def test_standard_and_differentiated_training_inputs(tmp_path: Path) -> None:
    canonical = _prepared_dataset(tmp_path)
    cache_root = tmp_path / "run" / "input_cache"

    standard, standard_cache = materialize_training_input_cache(
        canonical, "none", cache_root
    )
    assert standard is canonical
    assert standard_cache is None

    differentiated, cache_dir = materialize_training_input_cache(
        canonical, "differentiate", cache_root, chunk_size=3
    )
    assert cache_dir is not None
    assert cache_dir.is_relative_to(cache_root)
    assert differentiated.directory == canonical.directory
    np.testing.assert_array_equal(
        differentiated.windows_mV,
        np.diff(canonical.windows_mV, axis=-1),
    )
    np.testing.assert_allclose(
        differentiated.relative_time_ps,
        0.5 * (canonical.relative_time_ps[1:] + canonical.relative_time_ps[:-1]),
    )

    concatenated, concatenate_cache_dir = materialize_training_input_cache(
        canonical, "concatenate_diff", cache_root, chunk_size=3
    )
    assert concatenate_cache_dir is not None
    expected_concatenated = np.concatenate(
        (canonical.windows_mV, np.diff(canonical.windows_mV, axis=-1)), axis=-1
    )
    np.testing.assert_array_equal(concatenated.windows_mV, expected_concatenated)
    np.testing.assert_allclose(
        concatenated.relative_time_ps,
        np.concatenate(
            (
                canonical.relative_time_ps,
                0.5 * (
                    canonical.relative_time_ps[1:]
                    + canonical.relative_time_ps[:-1]
                ),
            )
        ),
    )
    manifest = (concatenate_cache_dir / "transform_manifest.json").read_text()
    assert '"component_order": [' in manifest
    assert '"raw_waveform"' in manifest
    assert '"first_difference"' in manifest

    # The transform cache deliberately lacks canonical metadata/splits and cannot
    # be mistaken for a second prepared dataset.
    assert (cache_dir / "transform_manifest.json").is_file()
    assert not (cache_dir / "manifest.json").exists()
    assert not (cache_dir / "splits.npz").exists()
    with pytest.raises(FileNotFoundError):
        load_prepared_dataset(cache_dir)

    normalization = Normalization(mean_mV=0.0, std_mV=1.0)
    cached_training_item = CorrectionDataset(
        differentiated,
        np.asarray([0], dtype=np.int64),
        normalization,
        input_transform="none",
    )[0][0].numpy()
    direct_transformed_item = CorrectionDataset(
        canonical,
        np.asarray([0], dtype=np.int64),
        normalization,
        input_transform="differentiate",
    )[0][0].numpy()
    np.testing.assert_array_equal(cached_training_item, direct_transformed_item)

    # Equal-length experiment windows at different positions must not collide.
    left_view = prepared_dataset_view(canonical, window_start=0, window_stop=4)
    right_view = prepared_dataset_view(canonical, window_start=1, window_stop=5)
    _, left_cache = materialize_training_input_cache(
        left_view, "differentiate", cache_root
    )
    _, right_cache = materialize_training_input_cache(
        right_view, "differentiate", cache_root
    )
    assert left_cache != right_cache


def _write_zero_checkpoint(
    path: Path,
    dataset: PreparedDataset,
    input_transform: str,
) -> TrainedModel:
    model_config = {
        "hidden_units": [],
        "activation": "identity",
        "dropout": 0.0,
        "batch_norm": False,
    }
    model_input_length = int(
        apply_input_transform(dataset.windows_mV[:1], input_transform).shape[-1]
    )
    model = build_model("mlp_regressor", model_config, model_input_length)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    context = {
        "model_type": "mlp_regressor",
        "model_name": f"zero_{input_transform}",
        "model_config": model_config,
        "input_length": model_input_length,
        "source_input_length": dataset.input_length,
        "input_transform": input_transform,
        "normalization": {"mean_mV": 0.0, "std_mV": 1.0},
        "dataset_contract": {
            "led_timestamp_source": dataset.manifest["led_timestamp_source"],
            "cfd_timestamp_source": dataset.manifest["cfd_timestamp_source"],
            "ml_window_alignment_source": dataset.manifest[
                "ml_window_alignment_source"
            ],
            "timing_channel_waveforms_saved": False,
            "waveform_grid": "native_acquisition_samples",
        },
        "relative_time_ps_start": float(
            transform_relative_time_ps(dataset.relative_time_ps, input_transform)[0]
        ),
        "relative_time_ps_stop": float(
            transform_relative_time_ps(dataset.relative_time_ps, input_transform)[-1]
        ),
        "data_view": {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "epoch": 0, "context": context}, path)
    return TrainedModel(
        model_name=context["model_name"],
        model_type="mlp_regressor",
        checkpoint=path,
        validation_rmse_ps=float("nan"),
        train_dir=path.parent,
        input_transform=input_transform,
    )


@pytest.mark.parametrize(
    "input_transform", ["none", "differentiate", "concatenate_diff", "normalize"]
)
def test_evaluation_replays_checkpoint_input_transform(
    tmp_path: Path, input_transform: str
) -> None:
    dataset = _prepared_dataset(tmp_path)
    trained = _write_zero_checkpoint(
        tmp_path / input_transform / "best.pt", dataset, input_transform
    )
    corrected = _evaluate_model(
        trained,
        dataset,
        {"batch_size": 2, "num_workers": 0, "pin_memory": False},
        torch.device("cpu"),
    )
    expected_led_delta = (
        dataset.led_time_fs[dataset.evaluation, 0]
        - dataset.led_time_fs[dataset.evaluation, 1]
    ) / 1000.0
    np.testing.assert_allclose(corrected, expected_led_delta)

@pytest.mark.parametrize(
    "input_transform", ["none", "differentiate", "concatenate_diff", "normalize"]
)
def test_train_then_evaluate_standard_and_differentiated_paths(
    tmp_path: Path, input_transform: str
) -> None:
    import logging

    from ml_pipeline.training import train_model

    dataset = _prepared_dataset(tmp_path)
    run_dir = tmp_path / f"train_{input_transform}"
    config = {
        "datasets": [str(dataset.directory)],
        "input_transform": input_transform,
        "model": {
            "type": "mlp_regressor",
            "name": f"tiny_{input_transform}",
            "hidden_units": [],
            "activation": "identity",
            "dropout": 0.0,
            "batch_norm": False,
            "loss": {"type": "mse"},
        },
        "optimizer": {"learning_rate": 1.0e-3, "weight_decay": 0.0},
        "training": {
            "device": "cpu",
            "seed": 7,
            "epochs": 1,
            "batch_size": 2,
            "mixed_precision": False,
            "early_stopping_patience": 1,
            "early_stopping_min_delta_ps": 0.0,
            "normalization_chunk_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "selection_metric": "validation_rmse",
            "fit_interval_epochs": 0,
            "fit_train_during_training": False,
            "fit_validation_during_training": False,
        },
        "fit": {
            "histogram_range_ps": [-100.0, 100.0],
            "histogram_bin_ps": 10.0,
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
        "artifacts": {
            "save_config": False,
            "save_history": False,
            "save_plots": False,
            "save_last_checkpoint": False,
            "save_summary": True,
        },
    }
    summary = train_model(
        config,
        restart=True,
        logger=logging.getLogger(f"test.{input_transform}"),
        prepared_datasets=[dataset],
    )
    assert summary["input_transform"] == input_transform
    if input_transform == "differentiate":
        assert summary["input_length"] == dataset.input_length - 1
        assert len(summary["input_cache_paths"]) == 1
    elif input_transform == "concatenate_diff":
        assert summary["input_length"] == 2 * dataset.input_length - 1
        assert len(summary["input_cache_paths"]) == 1
    else:
        assert summary["input_length"] == dataset.input_length
        assert summary["input_cache_paths"] == []

    checkpoint = Path(summary["best_checkpoint"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["context"]["input_transform"] == input_transform
    if input_transform == "normalize":
        assert payload["context"]["normalization"]["strategy"] == "feature"
        assert len(payload["context"]["normalization"]["mean_mV"]) == dataset.input_length
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
        {"batch_size": 2, "num_workers": 0, "pin_memory": False},
        torch.device("cpu"),
    )
    assert corrected.shape == (dataset.evaluation.size,)


@pytest.mark.parametrize(
    ("mode", "mean_field"),
    [
        ("prediction_mean", "train_prediction_mean_ps"),
        ("residual_mean", "train_prediction_residual_mean_ps"),
    ],
)
def test_epoch_end_zero_bias_constraint(
    tmp_path: Path, mode: str, mean_field: str
) -> None:
    import csv
    import logging

    from ml_pipeline.training import train_model

    dataset = _prepared_dataset(tmp_path)
    run_dir = tmp_path / f"zero_bias_{mode}"
    config = {
        "datasets": [str(dataset.directory)],
        "prediction": {
            "input_waveforms": "timing",
            "target": "timing_led",
        },
        "input_transform": "none",
        "model": {
            "type": "mlp_regressor",
            "name": f"zero_bias_{mode}",
            "hidden_units": [],
            "activation": "identity",
            "dropout": 0.0,
            "batch_norm": False,
            "loss": {"type": "mse"},
        },
        "optimizer": {"learning_rate": 1.0e-3, "weight_decay": 0.0},
        "training": {
            "device": "cpu",
            "seed": 11,
            "epochs": 2,
            "batch_size": 2,
            "mixed_precision": False,
            "early_stopping_patience": 2,
            "early_stopping_min_delta_ps": 0.0,
            "normalization_chunk_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "selection_metric": "validation_rmse",
            "fit_interval_epochs": 0,
            "fit_train_during_training": False,
            "fit_validation_during_training": False,
            "zero_bias_constraint": {"enabled": True, "mode": mode},
        },
        "fit": {
            "histogram_range_ps": [-100.0, 100.0],
            "histogram_bin_ps": 10.0,
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
        "artifacts": {
            "save_config": False,
            "save_history": True,
            "save_plots": False,
            "save_last_checkpoint": False,
            "save_summary": True,
        },
    }
    summary = train_model(
        config,
        restart=True,
        logger=logging.getLogger(f"test.zero_bias.{mode}"),
        prepared_datasets=[dataset],
    )
    assert summary["input_waveform_source"] == "timing"
    assert summary["prediction_target"] == "timing_led"
    with (run_dir / "training_metrics.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    for row in rows:
        assert abs(float(row[mean_field])) < 1.0e-4
        assert row["zero_bias_constraint_enabled"].lower() == "true"


def test_timing_model_evaluation_replays_waveform_and_target_source(tmp_path: Path) -> None:
    dataset = _prepared_dataset(tmp_path)
    timing_view = prediction_dataset_view(
        dataset, input_waveforms="timing", target="timing_led"
    )
    trained = _write_zero_checkpoint(
        tmp_path / "timing" / "best.pt", timing_view, "none"
    )
    payload = torch.load(trained.checkpoint, map_location="cpu", weights_only=False)
    payload["context"]["input_waveform_source"] = "timing"
    payload["context"]["prediction_target"] = "timing_led"
    torch.save(payload, trained.checkpoint)
    trained = TrainedModel(
        model_name=trained.model_name,
        model_type=trained.model_type,
        checkpoint=trained.checkpoint,
        validation_rmse_ps=trained.validation_rmse_ps,
        train_dir=trained.train_dir,
        input_transform=trained.input_transform,
        input_waveform_source="timing",
        prediction_target="timing_led",
    )
    corrected = _evaluate_model(
        trained,
        dataset,
        {"batch_size": 2, "num_workers": 0, "pin_memory": False},
        torch.device("cpu"),
    )
    expected = (
        dataset.timing_led_time_fs[dataset.evaluation, 0]
        - dataset.timing_led_time_fs[dataset.evaluation, 1]
    ) / 1000.0
    np.testing.assert_allclose(corrected, expected)



def test_normalize_is_featurewise_train_only_zscore(tmp_path: Path) -> None:
    dataset = _prepared_dataset(tmp_path)
    normalized, cache_dir = materialize_training_input_cache(
        dataset, "normalize", tmp_path / "cache"
    )
    assert normalized is dataset
    assert cache_dir is None
    assert transformed_input_length(dataset.input_length, "normalize") == dataset.input_length
    np.testing.assert_array_equal(
        apply_input_transform(dataset.windows_mV[:1], "normalize"),
        dataset.windows_mV[:1],
    )

    statistics = compute_normalization(
        [(dataset, dataset.train)], featurewise=True, chunk_size=2
    )
    assert statistics.mode == "feature"
    training_values = np.asarray(dataset.windows_mV[dataset.train], dtype=np.float64)
    expected_mean = np.mean(training_values, axis=(0, 1))
    expected_std = np.std(training_values, axis=(0, 1))
    np.testing.assert_allclose(statistics.mean_mV, expected_mean)
    np.testing.assert_allclose(statistics.std_mV, expected_std)

    item = CorrectionDataset(
        dataset,
        np.asarray([dataset.train[0]], dtype=np.int64),
        statistics,
    )[0][0].numpy()
    expected_item = (
        dataset.windows_mV[dataset.train[0]] - expected_mean
    ) / expected_std
    np.testing.assert_allclose(item, expected_item, rtol=1e-6, atol=1e-6)

    serialized = statistics.as_dict()
    restored = Normalization.from_dict(serialized)
    assert restored.mode == "feature"
    np.testing.assert_allclose(restored.mean_mV, statistics.mean_mV)
    np.testing.assert_allclose(restored.std_mV, statistics.std_mV)
