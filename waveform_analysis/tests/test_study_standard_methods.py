from pathlib import Path

import numpy as np

from ml_pipeline.dataset import PreparedDataset
from ml_pipeline.robust_selection import RobustLocationScale
from ml_pipeline.study import (
    _evaluate_standard_methods,
    _generate_model_loss_records,
    _read_results,
    _standard_method_delta_ps,
    _write_results,
)
from ml_pipeline.study_config import CHANNEL_MODES


def _pair_times_ps(values: list[float]) -> np.ndarray:
    values_fs = np.rint(np.asarray(values, dtype=np.float64) * 1000.0).astype(np.int64)
    return np.column_stack((values_fs, np.zeros(values_fs.size, dtype=np.int64)))


def _dataset(tmp_path: Path, name: str, offset: float = 0.0) -> PreparedDataset:
    n = 12
    energy_led = _pair_times_ps([offset + value for value in (-28, -20, -14, -8, -3, 2, 7, 13, 19, 26, 32, 39)])
    timing_led = _pair_times_ps([offset + value for value in (-18, -13, -9, -5, -2, 1, 4, 8, 12, 17, 23, 30)])
    energy_cfd = _pair_times_ps([offset + value for value in (-12, -9, -7, -4, -2, 0, 2, 5, 7, 10, 13, 17)])
    timing_cfd = _pair_times_ps([offset + value for value in (-8, -6, -4, -3, -1, 0, 1, 3, 4, 6, 8, 11)])
    generic_cfd = _pair_times_ps([100.0 + value for value in range(n)])
    indices = np.arange(n, dtype=np.int64)
    return PreparedDataset(
        directory=tmp_path / name,
        manifest={"true_tof_ps": 0.0},
        event_id=indices,
        event_index=indices,
        source_file_id=np.zeros((n, 2), dtype=np.int64),
        source_run_index=np.zeros(n, dtype=np.int32),
        bias_voltage_V=np.zeros(n, dtype=np.float64),
        amplitude_mV=np.ones((n, 2), dtype=np.float32),
        noise_rms_mV=np.zeros((n, 2), dtype=np.float32),
        trigger_index=np.zeros((n, 2), dtype=np.int32),
        led_time_fs=timing_led,
        cfd_time_fs=generic_cfd,
        windows_mV=np.zeros((n, 2, 3), dtype=np.float32),
        relative_time_ps=np.asarray([-1000.0, 0.0, 1000.0]),
        energy_led_time_fs=energy_led,
        timing_led_time_fs=timing_led,
        energy_cfd_time_fs=energy_cfd,
        timing_cfd_time_fs=timing_cfd,
        train=indices,
        evaluation=indices,
    )


def test_cfd_uses_same_timestamp_family_as_prediction_target(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, "dataset")
    indices = np.asarray([0, 1, 2], dtype=np.int64)

    energy = _standard_method_delta_ps(dataset, "energy_led", "cfd", indices)
    timing = _standard_method_delta_ps(dataset, "timing_led", "cfd", indices)

    np.testing.assert_allclose(energy, [-12.0, -9.0, -7.0])
    np.testing.assert_allclose(timing, [-8.0, -6.0, -4.0])
    assert not np.allclose(energy, [100.0, 101.0, 102.0])


def test_led_and_cfd_are_written_as_compact_model_rows(tmp_path: Path) -> None:
    development = _dataset(tmp_path, "development")
    blind = _dataset(tmp_path, "blind", offset=1.0)
    robust = RobustLocationScale(0.0, 20.0, "test", 6)
    folds = [
        {
            "fold_id": 0,
            "train": np.asarray([0, 1, 2, 3, 4, 5]),
            "validation": np.asarray([6, 7, 8, 9, 10, 11]),
            "blind": np.asarray([0, 1, 2, 3, 4, 5]),
            "robust": robust,
            "z_threshold": 4.0,
        },
        {
            "fold_id": 1,
            "train": np.asarray([6, 7, 8, 9, 10, 11]),
            "validation": np.asarray([0, 1, 2, 3, 4, 5]),
            "blind": np.asarray([6, 7, 8, 9, 10, 11]),
            "robust": robust,
            "z_threshold": 4.0,
        },
    ]
    config = {
        "experiment": {"name": "study"},
        "standard_methods": ["led", "cfd"],
        "fit": {
            "histogram_range_ps": [-100.0, 100.0],
            "histogram_bin_ps": 5.0,
            "initial_half_width_ps": 100.0,
            "iteration_sigma": 2.5,
            "max_iterations": 3,
            "convergence_tolerance_ps": 0.01,
            "min_events": 3,
            "minimum_fit_bins": 3,
            "minimum_sigma_bins": 0.1,
        },
        "selection": {"window_metric": "rmse_ps"},
        "channel_modes": ["energy_to_energy"],
        "models": [],
        "losses": [],
        "root_files": [str(tmp_path / "47V-run.root")],
        "reporting": {
            "voltage_from_filename": {
                "enabled": True,
                "pattern": r"^(?P<voltage>\d+(?:\.\d+)?)V",
                "group": "voltage",
            }
        },
    }
    rows: list[dict[str, object]] = []

    class Logger:
        def info(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

    _evaluate_standard_methods(
        config=config,
        rows=rows,
        development=development,
        blind=blind,
        root_id="r1",
        root_file=Path("47V-run.root"),
        mode_id="energy_to_energy",
        mode=CHANNEL_MODES["energy_to_energy"],
        folds=folds,
        logger=Logger(),
    )

    mean_rows = [
        row for row in rows
        if row.get("record_type") == "summary" and row.get("statistic") == "mean"
    ]
    assert {(row["model_id"], row["split"]) for row in mean_rows} == {
        ("led", "validation"),
        ("led", "blind"),
        ("cfd", "validation"),
        ("cfd", "blind"),
    }
    assert all(row["model_type"] == "standard_method" for row in mean_rows)
    assert all(row["is_selected_window"] == 1 for row in mean_rows)

    records = _generate_model_loss_records(
        config=config,
        rows=rows,
        root_file=Path("47V-run.root"),
        root_id="r1",
    )
    assert {record["model_id"] for record in records} == {"led", "cfd"}
    assert all(record["loss_id"] == "evaluation_mse" for record in records)
    assert all(np.isnan(record["window_size_ns"]) for record in records)


def test_standard_rows_survive_compact_results_roundtrip(tmp_path: Path) -> None:
    row = {
        "row_key": "1234567890abcdef12345678",
        "record_type": "summary",
        "experiment_id": "study",
        "root_id": "r1",
        "root_file": str(tmp_path / "47V.root"),
        "channel_mode": "energy_to_energy",
        "model_id": "cfd",
        "model_type": "standard_method",
        "loss_id": "evaluation_mse",
        "loss_type": "mse",
        "input_transform": "not_applicable",
        "window_id": "not_applicable",
        "window_start_ns": float("nan"),
        "window_end_ns": float("nan"),
        "trial_id": "not_applicable",
        "fold_id": "",
        "split": "validation",
        "statistic": "mean",
        "is_selected_hyperparameters": 1,
        "is_selected_window": 1,
        "status": "completed",
        "n_events": 100,
        "loss": 4.0,
        "rmse_ps": 2.0,
        "bias_ps": 0.1,
        "ctr_ps": 50.0,
        "baseline_ctr_ps": 60.0,
        "relative_improvement_pct": 16.6667,
    }
    path = tmp_path / "_state" / "all_results.csv"
    _write_results(path, [row])
    restored = _read_results(path)
    assert len(restored) == 1
    assert restored[0]["model_id"] == "cfd"
    assert restored[0]["model_type"] == "standard_method"
    assert restored[0]["window_id"] == "not_applicable"
