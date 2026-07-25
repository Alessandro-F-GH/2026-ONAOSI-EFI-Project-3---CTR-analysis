from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from utils.tof_shift_experiment import (
    assign_artificial_shifts,
    finalize_shift_datasets,
    matched_integer_uniform_half_width,
    rising_crossing_ns,
    shift_signal_on_fixed_grid,
)


def test_matched_integer_uniform_width_for_default_shift() -> None:
    match = matched_integer_uniform_half_width(80)
    assert match["continuous_half_width_ps"] == 113
    assert abs(match["discrete_variance_ps2"] - 2.0 * 80.0**2 / 3.0) < 1e-12
    assert abs(match["continuous_variance_ps2"] - 113.0 * 114.0 / 3.0) < 1e-12


def test_shift_assignment_is_balanced_reproducible_and_variance_matched() -> None:
    selected = np.ones(10001, dtype=bool)
    first = assign_artificial_shifts(
        selected,
        discrete_shifts_ps=[-80, 0, 80],
        continuous_max_abs_ps=None,
        random_seed=7,
    )
    second = assign_artificial_shifts(
        selected,
        discrete_shifts_ps=[-80, 0, 80],
        continuous_max_abs_ps=None,
        random_seed=7,
    )
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)
    discrete, groups, continuous = first
    _, counts = np.unique(discrete, return_counts=True)
    assert int(counts.max() - counts.min()) <= 1
    assert set(np.unique(groups)) == {0, 1, 2}
    assert int(continuous.min()) >= -113
    assert int(continuous.max()) <= 113
    assert np.issubdtype(continuous.dtype, np.integer)


def test_pair_shift_moves_only_selected_detector_channels() -> None:
    import utils.tof_shift_experiment as module

    settings = {
        "channels": {"energy": [1, 2], "timing": [3, 4]},
        "shifted_channels": [1, 3],
    }
    shifts = module._detector_pair_channel_shifts_ps(80, settings)
    assert shifts == {0: 80.0, 1: 0.0, 2: 80.0, 3: 0.0}


def test_finalize_uses_assigned_pair_shift_as_target(tmp_path: Path) -> None:
    discrete_rows = []
    continuous_rows = []
    discrete_shifts = [-80, 0, 80]
    continuous_shifts = [-5, 2, 9]
    for index, (discrete_shift, continuous_shift) in enumerate(
        zip(discrete_shifts, continuous_shifts, strict=True)
    ):
        common = {
            "meta_event_index": index,
            "meta_event_id": index + 100,
            "meta_source_file_id": "1",
            "meta_discrete_group": index,
            "example_feature": float(index),
            "_led_tof_ps": float(index - 1) * 10.0,
        }
        discrete_rows.append(
            {
                **common,
                "meta_shift_mode": "discrete",
                "meta_assigned_shift_ps": discrete_shift,
            }
        )
        continuous_rows.append(
            {
                **common,
                "meta_shift_mode": "continuous",
                "meta_assigned_shift_ps": continuous_shift,
            }
        )
    config = {
        "tof_shift_experiment": {
            "target_column": "target_shift_ps",
            "discrete_filename": "discrete.csv",
            "continuous_filename": "continuous.csv",
            "absolute_window_ns": [-40.0, -20.0],
            "discrete_shifts_ps": [-80, 0, 80],
        }
    }
    summary = finalize_shift_datasets(
        discrete_rows,
        continuous_rows,
        tmp_path,
        config,
    )
    discrete = pd.read_csv(tmp_path / "discrete.csv")
    continuous = pd.read_csv(tmp_path / "continuous.csv")
    np.testing.assert_allclose(discrete["target_shift_ps"], discrete_shifts)
    np.testing.assert_allclose(continuous["target_shift_ps"], continuous_shifts)
    assert summary["rows_per_dataset"] == 3
    assert "_led_tof_ps" not in discrete.columns
    assert "_led_tof_ps" not in continuous.columns
    assert "_led_tof_ps" not in summary["feature_columns"]


def test_scenario_features_are_extracted_after_pair_translation(monkeypatch) -> None:
    from types import SimpleNamespace
    import utils.tof_shift_experiment as module

    # Replace catch22 with a deterministic summary so this test has no
    # external dependency and directly verifies ordering of operations.
    catch22_calls: list[tuple[str, np.ndarray]] = []

    def fake_catch22(signal_mV: np.ndarray, prefix: str) -> dict[str, float]:
        values = np.asarray(signal_mV, dtype=float).copy()
        catch22_calls.append((prefix, values))
        return {
            f"{prefix}_c22_test_mean": float(np.mean(values)),
            f"{prefix}_c22_test_first": float(values[0]),
        }

    monkeypatch.setattr(module, "_catch22_features", fake_catch22)

    time_ns = np.linspace(-45.0, -15.0, 3001)
    # Pulses with a slowly changing envelope ensure that the pair shift changes
    # both catch22 stand-ins and the maximum measured in the fixed window.
    signals = [
        20.0 / (1.0 + np.exp(-(time_ns + 31.0) / 0.15)) + 0.05 * (time_ns + 40.0),
        25.0 / (1.0 + np.exp(-(time_ns + 30.5) / 0.18)) + 0.08 * (time_ns + 40.0),
        18.0 / (1.0 + np.exp(-(time_ns + 30.0) / 0.14)) + 0.04 * (time_ns + 40.0),
        22.0 / (1.0 + np.exp(-(time_ns + 29.5) / 0.16)) + 0.06 * (time_ns + 40.0),
    ]
    basics = [
        SimpleNamespace(corrected_signal_mV=signal, amplitude_mV=float(np.max(signal)))
        for signal in signals
    ]
    settings = {
        "channels": {
            "energy": [1, 2],
            "timing": [3, 4],
            "polarities": [1, 1, 1, 1],
        },
        "window_start_ns": -40.0,
        "window_stop_ns": -20.0,
        "resample_step_ps": 20.0,
        "timing_threshold_mV": 7.0,
        "energy_threshold_mV": 10.0,
        "crossing_mode": "first",
        "shifted_channels": [1, 3],
    }

    features_0, _ = module._scenario_features(
        basics, [time_ns] * 4, 0, settings
    )
    features_80, _ = module._scenario_features(
        basics, [time_ns] * 4, 80, settings
    )

    assert features_0 is not None and features_80 is not None
    assert len(catch22_calls) == 2
    assert [prefix for prefix, _ in catch22_calls] == [
        "timing_difference",
        "timing_difference",
    ]
    grid = module._fixed_grid_ns(settings)
    expected_diff_0 = (
        module.shift_signal_on_fixed_grid(time_ns, signals[2], grid, 0.0)
        - module.shift_signal_on_fixed_grid(time_ns, signals[3], grid, 0.0)
    )
    expected_diff_80 = (
        module.shift_signal_on_fixed_grid(time_ns, signals[2], grid, 80.0)
        - module.shift_signal_on_fixed_grid(time_ns, signals[3], grid, 0.0)
    )
    np.testing.assert_allclose(catch22_calls[0][1], expected_diff_0)
    np.testing.assert_allclose(catch22_calls[1][1], expected_diff_80)

    # Only detector pair 1 (energy ch1 + timing ch3) moves.  The CSV stores
    # one timing arrival-time difference rather than two absolute timestamps.
    assert abs(
        (features_80["timing_delta_t7_ps"] - features_0["timing_delta_t7_ps"])
        - 80.0
    ) < 0.4
    assert "timing_ch3_t7_abs_ps" not in features_80
    assert "timing_ch4_t7_abs_ps" not in features_80
    assert abs((features_80["energy_ch1_t10_abs_ps"] - features_0["energy_ch1_t10_abs_ps"]) - 80.0) < 0.3
    assert abs(features_80["energy_ch2_t10_abs_ps"] - features_0["energy_ch2_t10_abs_ps"]) < 0.3

    # catch22 is evaluated once on the post-shift difference waveform ch3-ch4.
    assert features_80["timing_difference_c22_test_first"] != features_0["timing_difference_c22_test_first"]
    assert not any(key.startswith("timing_ch3_c22_") for key in features_80)
    assert not any(key.startswith("timing_ch4_c22_") for key in features_80)
    assert features_80["energy_ch1_max_amplitude_mV"] != features_0["energy_ch1_max_amplitude_mV"]
    assert features_80["energy_ch2_max_amplitude_mV"] == features_0["energy_ch2_max_amplitude_mV"]

    # The saved energy maximum is the translated-window maximum, not the
    # pre-translation full-record amplitude stored in basics.
    shifted_energy = module.shift_signal_on_fixed_grid(
        time_ns, signals[0], grid, 80.0
    )
    assert shifted_energy is not None
    assert abs(
        features_80["energy_ch1_max_amplitude_mV"] - float(np.max(shifted_energy))
    ) < 1e-12


def test_comparison_feature_sets_use_difference_schema() -> None:
    import importlib.util

    script_path = PROJECT / "scripts" / "compare_tof_shift_models.py"
    spec = importlib.util.spec_from_file_location("compare_tof_shift_models", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    columns = [
        *[f"timing_difference_c22_f{index:02d}" for index in range(22)],
        "timing_delta_t7_ps",
        "energy_ch1_max_amplitude_mV",
        "energy_ch1_t10_abs_ps",
        "energy_ch2_max_amplitude_mV",
        "energy_ch2_t10_abs_ps",
    ]
    assert len(columns) == 27
    assert module._feature_set(columns, "catch22_only") == columns[:22]
    assert module._feature_set(columns, "timing_difference_only") == [
        "timing_delta_t7_ps"
    ]
    assert len(module._feature_set(columns, "scalar_only")) == 5
    assert len(module._feature_set(columns, "all_requested")) == 27


def test_discrete_support_diagnostics_detect_quantized_predictions() -> None:
    import importlib.util

    script_path = PROJECT / "scripts" / "compare_tof_shift_models.py"
    spec = importlib.util.spec_from_file_location(
        "compare_tof_shift_models_support", script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    diagnostics = module._discrete_support_diagnostics(
        prediction_ps=np.array([-79.0, 2.0, 76.0, 41.0]),
        discrete_values_ps=np.array([-80.0, 0.0, 80.0]),
        tolerance_ps=5.0,
    )
    expected_distances = np.array([1.0, 2.0, 4.0, 39.0])
    assert abs(
        diagnostics["mean_distance_to_discrete_support_ps"]
        - float(np.mean(expected_distances))
    ) < 1e-12
    assert abs(
        diagnostics["median_distance_to_discrete_support_ps"]
        - float(np.median(expected_distances))
    ) < 1e-12
    assert diagnostics["fraction_within_discrete_tolerance"] == 0.75
