from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from .dataset import PreparedDataset

INPUT_WAVEFORMS_ENERGY = "energy"
INPUT_WAVEFORMS_TIMING = "timing"
TARGET_PREPARED_LED = "prepared_led"
TARGET_ENERGY_LED = "energy_led"
TARGET_TIMING_LED = "timing_led"

SUPPORTED_INPUT_WAVEFORMS = {INPUT_WAVEFORMS_ENERGY, INPUT_WAVEFORMS_TIMING}
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
    if input_waveforms == INPUT_WAVEFORMS_ENERGY:
        windows = dataset.windows_mV
        relative_time = dataset.relative_time_ps
    else:
        if dataset.timing_windows_mV is None or dataset.timing_relative_time_ps is None:
            raise ValueError(
                f"Dataset {dataset.directory} does not contain timing-channel waveform windows; "
                "configure channels.timing and rebuild preprocessing"
            )
        windows = dataset.timing_windows_mV
        relative_time = dataset.timing_relative_time_ps

    target_values = _target_array(dataset, target)
    manifest = dict(dataset.manifest)
    manifest["prediction_view"] = {
        "input_waveforms": input_waveforms,
        "target": target,
    }
    manifest["input_length"] = int(windows.shape[2])
    manifest["relative_time_ps_start"] = float(relative_time[0])
    manifest["relative_time_ps_stop"] = float(relative_time[-1])
    return replace(
        dataset,
        manifest=manifest,
        windows_mV=windows,
        relative_time_ps=relative_time,
        led_time_fs=target_values,
    )
