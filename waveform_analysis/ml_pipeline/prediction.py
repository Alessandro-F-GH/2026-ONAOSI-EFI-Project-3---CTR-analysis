from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from .dataset import PreparedDataset

INPUT_WAVEFORMS_ENERGY = "energy"
INPUT_WAVEFORMS_TIMING = "timing"
INPUT_WAVEFORMS_ENERGY_TIMING = "energy_timing"
TARGET_PREPARED_LED = "prepared_led"
TARGET_ENERGY_LED = "energy_led"
TARGET_TIMING_LED = "timing_led"

SUPPORTED_INPUT_WAVEFORMS = {
    INPUT_WAVEFORMS_ENERGY,
    INPUT_WAVEFORMS_TIMING,
    INPUT_WAVEFORMS_ENERGY_TIMING,
}
SUPPORTED_TARGETS = {TARGET_PREPARED_LED, TARGET_ENERGY_LED, TARGET_TIMING_LED}


def normalize_input_waveforms(value: Any) -> str:
    if value is None:
        return INPUT_WAVEFORMS_ENERGY
    key = str(value).strip().lower()
    aliases = {
        "": INPUT_WAVEFORMS_ENERGY,
        "energy_channels": INPUT_WAVEFORMS_ENERGY,
        "time": INPUT_WAVEFORMS_TIMING,
        "time_channels": INPUT_WAVEFORMS_TIMING,
        "timing_channels": INPUT_WAVEFORMS_TIMING,
        "energy+timing": INPUT_WAVEFORMS_ENERGY_TIMING,
        "energy_timing_channels": INPUT_WAVEFORMS_ENERGY_TIMING,
        "combined": INPUT_WAVEFORMS_ENERGY_TIMING,
    }
    key = aliases.get(key, key)
    if key not in SUPPORTED_INPUT_WAVEFORMS:
        raise ValueError(
            f"Unsupported prediction.input_waveforms {value!r}; expected one of "
            f"{sorted(SUPPORTED_INPUT_WAVEFORMS)}"
        )
    return key


def normalize_prediction_target(value: Any) -> str:
    if value is None:
        return TARGET_PREPARED_LED
    key = str(value).strip().lower()
    aliases = {
        "": TARGET_PREPARED_LED,
        "led": TARGET_PREPARED_LED,
        "prepared": TARGET_PREPARED_LED,
        "energy": TARGET_ENERGY_LED,
        "energy_channels": TARGET_ENERGY_LED,
        "timing": TARGET_TIMING_LED,
        "time": TARGET_TIMING_LED,
        "time_led": TARGET_TIMING_LED,
        "timing_channels": TARGET_TIMING_LED,
    }
    key = aliases.get(key, key)
    if key not in SUPPORTED_TARGETS:
        raise ValueError(
            f"Unsupported prediction.target {value!r}; expected one of "
            f"{sorted(SUPPORTED_TARGETS)}"
        )
    return key


def resolve_prediction_config(config: dict[str, Any]) -> dict[str, str]:
    raw = config.get("prediction")
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("prediction must be an object")
    return {
        "input_waveforms": normalize_input_waveforms(raw.get("input_waveforms")),
        "target": normalize_prediction_target(raw.get("target")),
    }


def _target_array(dataset: PreparedDataset, target: str) -> np.ndarray:
    target = normalize_prediction_target(target)
    if target == TARGET_PREPARED_LED:
        return dataset.led_time_fs
    if target == TARGET_ENERGY_LED:
        if dataset.energy_led_time_fs is None:
            raise ValueError(
                f"Dataset {dataset.directory} does not contain energy LED timestamps; "
                "rebuild preprocessing with the new canonical dataset format"
            )
        return dataset.energy_led_time_fs
    if dataset.timing_led_time_fs is None:
        raise ValueError(
            f"Dataset {dataset.directory} does not contain timing LED timestamps; "
            "configure channels.timing and rebuild preprocessing"
        )
    return dataset.timing_led_time_fs





def _target_uses_timing_led(dataset: PreparedDataset, target: str) -> bool:
    normalized = normalize_prediction_target(target)
    if normalized == TARGET_TIMING_LED:
        return True
    if normalized == TARGET_ENERGY_LED:
        return False
    return (
        str(dataset.manifest.get("led_timestamp_source", "energy_channels"))
        == "timing_channels"
    )


def _energy_anchor_for_target(
    dataset: PreparedDataset, target: str
) -> tuple[np.ndarray | None, str]:
    if _target_uses_timing_led(dataset, target):
        return (
            dataset.timing_aligned_energy_window_anchor_time_fs,
            "timing_aligned_energy_native_anchor",
        )
    return dataset.energy_window_anchor_time_fs, "energy_native_anchor"


class ConcatenatedWaveformArray:
    """Lazy concatenation of waveform families along the sample axis.

    The object intentionally implements only the NumPy-style indexing used by
    the training pipeline.  It avoids duplicating the canonical energy and
    timing arrays for combined-input experiments.
    """

    def __init__(self, left: np.ndarray, right: np.ndarray) -> None:
        if left.shape[:2] != right.shape[:2]:
            raise ValueError("Combined waveform families must share event and detector axes")
        self.left = left
        self.right = right
        self.shape = (*left.shape[:-1], int(left.shape[-1]) + int(right.shape[-1]))
        self.dtype = np.result_type(left.dtype, right.dtype)

    def __getitem__(self, key: Any) -> np.ndarray:
        return np.concatenate((np.asarray(self.left[key]), np.asarray(self.right[key])), axis=-1)


def _slice_by_time(
    windows: np.ndarray,
    relative_time_ps: np.ndarray,
    before_ns: float,
    after_ns: float,
) -> tuple[np.ndarray, np.ndarray]:
    times_ns = np.asarray(relative_time_ps, dtype=np.float64) / 1000.0
    selected = np.flatnonzero(
        (times_ns >= -float(before_ns) - 1e-9)
        & (times_ns <= float(after_ns) + 1e-9)
    )
    if selected.size == 0:
        raise ValueError(
            f"Requested window [-{before_ns}, {after_ns}] ns contains no samples"
        )
    start, stop = int(selected[0]), int(selected[-1]) + 1
    if not np.array_equal(selected, np.arange(start, stop)):
        raise ValueError("Requested physical window is not contiguous")
    return windows[:, :, start:stop], np.asarray(relative_time_ps[start:stop])


def _energy_windows_for_target(
    dataset: PreparedDataset,
    target: str,
) -> tuple[np.ndarray, str]:
    """Choose the energy-waveform alignment associated with the prediction target."""

    if _target_uses_timing_led(dataset, target):
        if dataset.timing_aligned_energy_windows_mV is None:
            raise ValueError(
                f"Dataset {dataset.directory} does not contain timing-LED-aligned energy "
                "waveforms; rebuild it with timing channels enabled"
            )
        return dataset.timing_aligned_energy_windows_mV, "timing_channel_led"
    return dataset.windows_mV, "energy_channel_led"


def prediction_window_dataset_view(
    dataset: PreparedDataset,
    *,
    input_waveforms: str,
    target: str,
    before_ns: float,
    after_ns: float,
) -> PreparedDataset:
    """Resolve one LED-relative window and channel mode without data copies.

    For the combined mode, energy and timing are sliced independently on their
    own native grids, then exposed through a lazy sample-axis concatenation in
    the fixed order ``energy, timing``.
    """

    input_waveforms = normalize_input_waveforms(input_waveforms)
    target = normalize_prediction_target(target)
    target_values = _target_array(dataset, target)

    if input_waveforms == INPUT_WAVEFORMS_ENERGY:
        energy_source, energy_alignment = _energy_windows_for_target(dataset, target)
        anchor_values, anchor_source = _energy_anchor_for_target(dataset, target)
        windows, relative = _slice_by_time(
            energy_source, dataset.relative_time_ps, before_ns, after_ns
        )
        component_lengths = [int(windows.shape[-1])]
        components = ["energy"]
        component_alignments = [energy_alignment]
        factorization_anchor_component = "energy"
    elif input_waveforms == INPUT_WAVEFORMS_TIMING:
        if dataset.timing_windows_mV is None or dataset.timing_relative_time_ps is None:
            raise ValueError(
                f"Dataset {dataset.directory} does not contain timing-channel waveforms"
            )
        anchor_values = dataset.timing_window_anchor_time_fs
        anchor_source = "timing_native_anchor"
        windows, relative = _slice_by_time(
            dataset.timing_windows_mV,
            dataset.timing_relative_time_ps,
            before_ns,
            after_ns,
        )
        component_lengths = [int(windows.shape[-1])]
        components = ["timing"]
        component_alignments = ["timing_channel_led"]
        factorization_anchor_component = "timing"
    else:
        if dataset.timing_windows_mV is None or dataset.timing_relative_time_ps is None:
            raise ValueError(
                f"Dataset {dataset.directory} does not contain timing-channel waveforms"
            )
        energy_source, energy_alignment = _energy_windows_for_target(dataset, target)
        energy_anchor_values, energy_anchor_source = _energy_anchor_for_target(dataset, target)
        energy_windows, energy_time = _slice_by_time(
            energy_source, dataset.relative_time_ps, before_ns, after_ns
        )
        timing_windows, timing_time = _slice_by_time(
            dataset.timing_windows_mV,
            dataset.timing_relative_time_ps,
            before_ns,
            after_ns,
        )
        windows = ConcatenatedWaveformArray(energy_windows, timing_windows)
        relative = np.concatenate((energy_time, timing_time), axis=0)
        component_lengths = [int(energy_windows.shape[-1]), int(timing_windows.shape[-1])]
        components = ["energy", "timing"]
        component_alignments = [energy_alignment, "timing_channel_led"]
        # The timing component is the canonical anchor for the timing-LED target.
        # The energy component remains independently sampled, but its label no
        # longer carries the timing-grid rounding discontinuity.
        anchor_values = dataset.timing_window_anchor_time_fs
        anchor_source = "timing_native_anchor"
        factorization_anchor_component = "timing"
        manifest_component_anchor_sources = [
            energy_anchor_source, "timing_native_anchor"
        ]

    manifest = dict(dataset.manifest)
    manifest["prediction_view"] = {
        "input_waveforms": input_waveforms,
        "target": target,
    }
    manifest["input_components"] = components
    manifest["input_component_lengths"] = component_lengths
    manifest["input_component_alignments"] = component_alignments
    manifest["ml_window_alignment_source"] = (
        component_alignments[0]
        if len(set(component_alignments)) == 1
        else "mixed_target_specific_led"
    )
    manifest["window_before_ns"] = float(before_ns)
    manifest["correction_target_reference"] = "interpolated_led"
    manifest["window_anchor_shift_factored"] = anchor_values is not None
    manifest["factorization_anchor_source"] = anchor_source
    manifest["factorization_anchor_component"] = factorization_anchor_component
    if input_waveforms == INPUT_WAVEFORMS_ENERGY_TIMING:
        manifest["input_component_anchor_sources"] = manifest_component_anchor_sources
    manifest["window_after_ns"] = float(after_ns)
    manifest["input_length"] = int(sum(component_lengths))
    return replace(
        dataset,
        manifest=manifest,
        windows_mV=windows,
        relative_time_ps=relative,
        led_time_fs=target_values,
        window_anchor_time_fs=anchor_values,
    )

def prediction_dataset_view(
    dataset: PreparedDataset,
    *,
    input_waveforms: str,
    target: str,
) -> PreparedDataset:
    """Select the waveform family and timing target without copying event data.

    ``windows_mV`` and ``led_time_fs`` remain the generic fields consumed by the
    existing training/evaluation pipeline. This view only redirects those fields
    to arrays already stored in the single canonical prepared dataset.
    """

    input_waveforms = normalize_input_waveforms(input_waveforms)
    target = normalize_prediction_target(target)
    current_view = dataset.manifest.get("prediction_view")
    if isinstance(current_view, dict) and current_view == {
        "input_waveforms": input_waveforms,
        "target": target,
    }:
        return dataset
    if input_waveforms == INPUT_WAVEFORMS_ENERGY_TIMING:
        before_ns = max(0.0, -float(dataset.relative_time_ps[0]) / 1000.0)
        after_ns = max(0.0, float(dataset.relative_time_ps[-1]) / 1000.0)
        return prediction_window_dataset_view(
            dataset,
            input_waveforms=input_waveforms,
            target=target,
            before_ns=before_ns,
            after_ns=after_ns,
        )
    if input_waveforms == INPUT_WAVEFORMS_ENERGY:
        windows, energy_alignment = _energy_windows_for_target(dataset, target)
        anchor_values, anchor_source = _energy_anchor_for_target(dataset, target)
        relative_time = dataset.relative_time_ps
        component_alignments = [energy_alignment]
        factorization_anchor_component = "energy"
    else:
        if dataset.timing_windows_mV is None or dataset.timing_relative_time_ps is None:
            raise ValueError(
                f"Dataset {dataset.directory} does not contain timing-channel waveform windows; "
                "configure channels.timing and rebuild preprocessing"
            )
        windows = dataset.timing_windows_mV
        anchor_values = dataset.timing_window_anchor_time_fs
        anchor_source = "timing_native_anchor"
        relative_time = dataset.timing_relative_time_ps
        component_alignments = ["timing_channel_led"]
        factorization_anchor_component = "timing"

    target_values = _target_array(dataset, target)
    manifest = dict(dataset.manifest)
    manifest["prediction_view"] = {
        "input_waveforms": input_waveforms,
        "target": target,
    }
    manifest["input_component_alignments"] = component_alignments
    manifest["ml_window_alignment_source"] = component_alignments[0]
    manifest["input_length"] = int(windows.shape[2])
    manifest["correction_target_reference"] = "interpolated_led"
    manifest["window_anchor_shift_factored"] = anchor_values is not None
    manifest["factorization_anchor_source"] = anchor_source
    manifest["factorization_anchor_component"] = factorization_anchor_component
    manifest["relative_time_ps_start"] = float(relative_time[0])
    manifest["relative_time_ps_stop"] = float(relative_time[-1])
    return replace(
        dataset,
        manifest=manifest,
        windows_mV=windows,
        relative_time_ps=relative_time,
        led_time_fs=target_values,
        window_anchor_time_fs=anchor_values,
    )
