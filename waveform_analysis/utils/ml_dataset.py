from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import awkward as ak
import numpy as np
from scipy.interpolate import CubicSpline

from .io import decode_voltage_mV, iterate_chunks
from .pipeline import SelectionResult, build_selection
from .signal import (
    FEMTOSECONDS_PER_NANOSECOND,
    INVALID_TIME_FS,
    baseline_and_basic_features,
)

LOGGER = logging.getLogger(__name__)
SELECTION_CACHE_FORMAT_VERSION = 1
_WORKER_SETTINGS: dict[str, Any] | None = None


@dataclass(frozen=True)
class DatasetEventPayload:
    event_index: int
    event_id: int
    source_file_id: tuple[int, ...]
    raw_samples: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    vertical_gain_v_per_count: np.ndarray
    vertical_offset_v: np.ndarray
    horizontal_interval_s: np.ndarray
    horizontal_offset_s: np.ndarray


@dataclass(frozen=True)
class EventProcessingResult:
    row: dict[str, Any] | None
    rejection_reason: str | None


def _selection_fingerprint(config: dict[str, Any]) -> str:
    relevant = {
        "channels": config["channels"],
        "baseline_samples": config["waveform"]["baseline_samples"],
        "trigger_threshold_mV": config["waveform"]["trigger_threshold_mV"],
        "selection": config["selection"],
        "photopeak": config["photopeak"],
        "max_events": int(config["io"].get("max_events", 0)),
    }
    payload = json.dumps(relevant, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def extract_selection_features(
    input_path: Path,
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Extract only the quantities required by the existing event selection.

    This deliberately avoids spline resampling and LED/CFD scans. Dataset
    generation is a second pass over only the selected events.
    """
    channels = config["channels"]
    waveform_config = config["waveform"]
    io_config = config["io"]
    polarities = np.asarray(channels["polarities"], dtype=np.int8)
    energy_channels = np.asarray(channels["energy"], dtype=np.int64) - 1
    timing_channels = np.asarray(channels["timing"], dtype=np.int64) - 1

    event_ids: list[int] = []
    amplitudes: list[np.ndarray] = []
    baselines: list[np.ndarray] = []
    noises: list[np.ndarray] = []
    trigger_indices: list[np.ndarray] = []

    max_events = int(io_config.get("max_events", 0))
    entry_stop = max_events if max_events > 0 else None
    progress_every = max(1, int(io_config.get("progress_every", 1000)))
    processed = 0

    LOGGER.info("Selection pass: extracting amplitudes, noise, and trigger indices")
    for chunk in iterate_chunks(
        input_path,
        step_size=io_config.get("step_size", "128 MB"),
        entry_stop=entry_stop,
    ):
        for row_index in range(chunk.event_id.size):
            amplitude = np.full(4, np.nan, dtype=np.float64)
            baseline = np.full(4, np.nan, dtype=np.float64)
            noise = np.full(4, np.nan, dtype=np.float64)
            trigger_index = np.full(4, -1, dtype=np.int32)

            for channel in range(4):
                raw = np.asarray(ak.to_numpy(chunk.samples[channel][row_index]), dtype=np.int16)
                voltage_mV = decode_voltage_mV(
                    raw,
                    float(chunk.vertical_gain_v_per_count[row_index, channel]),
                    float(chunk.vertical_offset_v[row_index, channel]),
                )
                basic = baseline_and_basic_features(
                    voltage_mV,
                    baseline_samples=int(waveform_config["baseline_samples"]),
                    polarity=int(polarities[channel]),
                    trigger_threshold_mV=float(waveform_config["trigger_threshold_mV"]),
                    horizontal_interval_s=float(
                        chunk.horizontal_interval_s[row_index, channel]
                    ),
                    horizontal_offset_s=float(
                        chunk.horizontal_offset_s[row_index, channel]
                    ),
                )
                amplitude[channel] = basic.amplitude_mV
                baseline[channel] = basic.baseline_mV
                noise[channel] = basic.noise_rms_mV
                trigger_index[channel] = basic.trigger_index

            event_ids.append(int(chunk.event_id[row_index]))
            amplitudes.append(amplitude)
            baselines.append(baseline)
            noises.append(noise)
            trigger_indices.append(trigger_index)
            processed += 1
            if processed % progress_every == 0:
                LOGGER.info("Selection pass processed %d events", processed)

    source_stat = input_path.stat()
    LOGGER.info("Selection pass complete: %d events", processed)
    return {
        "selection_cache_format_version": np.asarray(
            SELECTION_CACHE_FORMAT_VERSION, dtype=np.int32
        ),
        "selection_fingerprint": np.asarray(_selection_fingerprint(config)),
        "source_path": np.asarray(str(input_path.resolve())),
        "source_size_bytes": np.asarray(source_stat.st_size, dtype=np.int64),
        "source_mtime_ns": np.asarray(source_stat.st_mtime_ns, dtype=np.int64),
        "event_id": np.asarray(event_ids, dtype=np.int64),
        "amplitude_mV": np.asarray(amplitudes, dtype=np.float64),
        "baseline_mV": np.asarray(baselines, dtype=np.float64),
        "noise_rms_mV": np.asarray(noises, dtype=np.float64),
        "trigger_index": np.asarray(trigger_indices, dtype=np.int32),
        "energy_channels_zero_based": energy_channels,
        "timing_channels_zero_based": timing_channels,
        "polarities": polarities,
    }


def save_selection_features(path: Path, features: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **features)


def load_selection_features(
    path: Path,
    config: dict[str, Any],
    input_path: Path,
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        features = {key: loaded[key] for key in loaded.files}
    required = {
        "selection_cache_format_version",
        "selection_fingerprint",
        "source_path",
        "source_size_bytes",
        "source_mtime_ns",
        "event_id",
        "amplitude_mV",
        "baseline_mV",
        "noise_rms_mV",
        "trigger_index",
        "energy_channels_zero_based",
        "timing_channels_zero_based",
        "polarities",
    }
    missing = sorted(required.difference(features))
    if missing:
        raise ValueError(f"selection cache is incomplete; missing: {', '.join(missing)}")
    if int(features["selection_cache_format_version"]) != SELECTION_CACHE_FORMAT_VERSION:
        raise ValueError("selection cache version differs; regenerate it")
    if str(features["selection_fingerprint"].item()) != _selection_fingerprint(config):
        raise ValueError("selection configuration differs from the cache; regenerate it")

    source = input_path.resolve()
    stat = source.stat()
    if str(features["source_path"].item()) != str(source):
        raise ValueError("selection cache belongs to a different input ROOT file")
    if (
        int(features["source_size_bytes"]) != stat.st_size
        or int(features["source_mtime_ns"]) != stat.st_mtime_ns
    ):
        raise ValueError("input ROOT file changed after the selection cache was created")
    return features


def select_events(
    features: dict[str, np.ndarray],
    config: dict[str, Any],
) -> SelectionResult:
    """Apply the exact same photopeak/trigger/noise selection as CTR analysis."""
    return build_selection(features, config)


def _first_rising_crossing_ns(
    time_ns: np.ndarray,
    signal_mV: np.ndarray,
    threshold_mV: float,
) -> float | None:
    if time_ns.size < 2 or signal_mV.size != time_ns.size:
        return None
    above = np.flatnonzero(signal_mV >= threshold_mV)
    if above.size == 0:
        return None
    index = int(above[0])
    if index <= 0:
        return None
    y0 = float(signal_mV[index - 1])
    y1 = float(signal_mV[index])
    if not (np.isfinite(y0) and np.isfinite(y1)) or y1 == y0:
        return None
    fraction = (float(threshold_mV) - y0) / (y1 - y0)
    if not 0.0 <= fraction <= 1.0:
        return None
    return float(time_ns[index - 1] + fraction * (time_ns[index] - time_ns[index - 1]))


def _last_rising_crossing_before_peak_ns(
    time_ns: np.ndarray,
    signal_mV: np.ndarray,
    threshold_mV: float,
) -> float | None:
    """Return the crossing on the rising edge connected to the pulse peak.

    Selecting the last valid upward crossing before the peak prevents a low
    proportional energy threshold from locking onto an earlier noise excursion.
    """
    if time_ns.size < 2 or signal_mV.size != time_ns.size:
        return None
    peak_index = int(np.argmax(signal_mV))
    if peak_index <= 0:
        return None
    before = signal_mV[:peak_index]
    after = signal_mV[1 : peak_index + 1]
    candidates = np.flatnonzero((before < threshold_mV) & (after >= threshold_mV))
    if candidates.size == 0:
        return None
    index = int(candidates[-1]) + 1
    y0 = float(signal_mV[index - 1])
    y1 = float(signal_mV[index])
    if not (np.isfinite(y0) and np.isfinite(y1)) or y1 == y0:
        return None
    fraction = (float(threshold_mV) - y0) / (y1 - y0)
    if not 0.0 <= fraction <= 1.0:
        return None
    return float(time_ns[index - 1] + fraction * (time_ns[index] - time_ns[index - 1]))


def _local_spline(
    time_ns: np.ndarray,
    signal_mV: np.ndarray,
    start_ns: float,
    stop_ns: float,
) -> CubicSpline | None:
    start_index = max(0, int(np.searchsorted(time_ns, start_ns, side="left")) - 1)
    stop_index = min(
        signal_mV.size,
        int(np.searchsorted(time_ns, stop_ns, side="right")) + 1,
    )
    if stop_index - start_index < 4:
        return None
    local_t = np.asarray(time_ns[start_index:stop_index], dtype=np.float64)
    local_y = np.asarray(signal_mV[start_index:stop_index], dtype=np.float64)
    if np.any(~np.isfinite(local_t)) or np.any(~np.isfinite(local_y)):
        return None
    if np.any(np.diff(local_t) <= 0):
        return None
    return CubicSpline(local_t, local_y, bc_type="not-a-knot", extrapolate=False)


def _find_led_crossing_ns(
    time_ns: np.ndarray,
    signal_mV: np.ndarray,
    trigger_index: int,
    *,
    threshold_mV: float,
    search_before_ns: float,
    search_after_ns: float,
    crossing_step_ps: float,
) -> float | None:
    if trigger_index < 0 or trigger_index >= signal_mV.size:
        return None
    trigger_ns = float(time_ns[trigger_index])
    start_ns = trigger_ns - float(search_before_ns)
    stop_ns = trigger_ns + float(search_after_ns)
    spline = _local_spline(time_ns, signal_mV, start_ns, stop_ns)
    if spline is None:
        return None
    step_ns = float(crossing_step_ps) / 1000.0
    sample_count = int(math.floor((stop_ns - start_ns) / step_ns)) + 1
    if sample_count < 2:
        return None
    grid_ns = start_ns + np.arange(sample_count, dtype=np.float64) * step_ns
    values = np.asarray(spline(grid_ns), dtype=np.float64)
    valid = np.isfinite(values)
    if np.count_nonzero(valid) < 2:
        return None
    return _first_rising_crossing_ns(grid_ns[valid], values[valid], threshold_mV)


def regularized_polynomial_fit(
    relative_time_ns: np.ndarray,
    signal_mV: np.ndarray,
    *,
    half_width_ns: float,
    degree: int,
    l2_regularization: float,
    penalize_intercept: bool,
) -> tuple[np.ndarray, float, float]:
    """Fit y=sum(c_k*x^k) with x normalized to [-1, 1].

    The optimized objective is mean squared residual plus
    ``l2_regularization * ||R c||^2``. Normalizing both the time coordinate and
    the data term makes the regularization hyperparameter stable when the
    resampling step changes.
    """
    x_ns = np.asarray(relative_time_ns, dtype=np.float64)
    y = np.asarray(signal_mV, dtype=np.float64)
    if x_ns.ndim != 1 or y.ndim != 1 or x_ns.size != y.size:
        raise ValueError("polynomial fit inputs must be matching 1D arrays")
    if x_ns.size < degree + 1:
        raise ValueError("not enough samples for requested polynomial degree")
    if half_width_ns <= 0:
        raise ValueError("half_width_ns must be positive")

    x = x_ns / float(half_width_ns)
    vandermonde = np.vander(x, N=degree + 1, increasing=True)
    gram = (vandermonde.T @ vandermonde) / float(x.size)
    rhs = (vandermonde.T @ y) / float(x.size)
    penalty = np.eye(degree + 1, dtype=np.float64)
    if not penalize_intercept:
        penalty[0, 0] = 0.0
    system = gram + float(l2_regularization) * penalty
    try:
        coefficients = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(system, rhs, rcond=None)[0]

    prediction = vandermonde @ coefficients
    residual = y - prediction
    rmse = float(np.sqrt(np.mean(residual * residual)))
    centered = y - float(np.mean(y))
    denominator = float(np.sum(centered * centered))
    r2 = float(1.0 - np.sum(residual * residual) / denominator) if denominator > 0 else np.nan
    return coefficients.astype(np.float64), rmse, r2


def _timing_polynomial_features(
    basic_signal_mV: np.ndarray,
    time_ns: np.ndarray,
    trigger_index: int,
    noise_rms_mV: float,
    channel_number: int,
    settings: dict[str, Any],
) -> tuple[dict[str, float], float] | tuple[None, None]:
    threshold_mV = float(settings["led_threshold_mV"])
    crossing_ns = _find_led_crossing_ns(
        time_ns,
        basic_signal_mV,
        trigger_index,
        threshold_mV=threshold_mV,
        search_before_ns=float(settings["led_search_ns"]["before"]),
        search_after_ns=float(settings["led_search_ns"]["after"]),
        crossing_step_ps=float(settings["crossing_step_ps"]),
    )
    if crossing_ns is None:
        return None, None

    # Match the CTR pipeline storage convention before defining the aligned
    # window and the TOF target.
    crossing_fs = np.int64(np.rint(crossing_ns * FEMTOSECONDS_PER_NANOSECOND))
    if crossing_fs == INVALID_TIME_FS:
        return None, None
    crossing_ns = float(crossing_fs) / FEMTOSECONDS_PER_NANOSECOND

    width_ns = float(settings["timing_window_width_ns"])
    half_width_ns = 0.5 * width_ns
    start_ns = crossing_ns - half_width_ns
    stop_ns = crossing_ns + half_width_ns
    if start_ns < float(time_ns[0]) or stop_ns > float(time_ns[-1]):
        return None, None

    spline = _local_spline(time_ns, basic_signal_mV, start_ns, stop_ns)
    if spline is None:
        return None, None
    step_ns = float(settings["polynomial_resample_step_ps"]) / 1000.0
    sample_count = int(round(width_ns / step_ns)) + 1
    relative_ns = np.linspace(-half_width_ns, half_width_ns, sample_count, dtype=np.float64)
    values = np.asarray(spline(crossing_ns + relative_ns), dtype=np.float64)
    if np.any(~np.isfinite(values)):
        return None, None

    coefficients, rmse, r2 = regularized_polynomial_fit(
        relative_ns,
        values,
        half_width_ns=half_width_ns,
        degree=int(settings["polynomial_degree"]),
        l2_regularization=float(settings["polynomial_l2_regularization"]),
        penalize_intercept=bool(settings["polynomial_penalize_intercept"]),
    )
    prefix = f"timing_ch{channel_number}"
    features: dict[str, float] = {
        f"{prefix}_max_amplitude_mV": float(np.max(basic_signal_mV)),
        f"{prefix}_noise_rms_mV": float(noise_rms_mV),
        f"{prefix}_poly_rmse_mV": rmse,
        f"{prefix}_poly_r2": r2,
    }
    for coefficient_index, coefficient in enumerate(coefficients):
        features[f"{prefix}_poly_c{coefficient_index}"] = float(coefficient)
    return features, crossing_ns


def _fraction_label(fraction: float) -> str:
    percentage = fraction * 100.0
    rounded = round(percentage)
    if abs(percentage - rounded) < 1.0e-9:
        return f"{int(rounded):02d}"
    return f"{percentage:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def _energy_features(
    basic_signal_mV: np.ndarray,
    time_ns: np.ndarray,
    trigger_index: int,
    timing_led_ns: float,
    amplitude_mV: float,
    noise_rms_mV: float,
    energy_channel_number: int,
    timing_channel_number: int,
    settings: dict[str, Any],
) -> dict[str, float] | None:
    if trigger_index < 0 or not np.isfinite(amplitude_mV) or amplitude_mV <= 0:
        return None
    trigger_ns = float(time_ns[trigger_index])
    start_ns = trigger_ns - float(settings["energy_search_ns"]["before"])
    stop_ns = trigger_ns + float(settings["energy_search_ns"]["after"])
    spline = _local_spline(time_ns, basic_signal_mV, start_ns, stop_ns)
    if spline is None:
        return None

    step_ns = float(settings["crossing_step_ps"]) / 1000.0
    sample_count = int(math.floor((stop_ns - start_ns) / step_ns)) + 1
    grid_ns = start_ns + np.arange(sample_count, dtype=np.float64) * step_ns
    values = np.asarray(spline(grid_ns), dtype=np.float64)
    if np.any(~np.isfinite(values)):
        return None

    peak_index = int(np.argmax(values))
    rising_time = grid_ns[: peak_index + 1]
    rising_signal = values[: peak_index + 1]
    if rising_time.size < 2:
        return None

    fractions = [float(item) for item in settings["energy_fractions"]]
    crossings: dict[float, float] = {}
    for fraction in fractions:
        crossing = _last_rising_crossing_before_peak_ns(
            rising_time,
            rising_signal,
            float(amplitude_mV) * fraction,
        )
        if crossing is None:
            return None
        crossings[fraction] = crossing

    prefix = f"energy_ch{energy_channel_number}"
    output: dict[str, float] = {
        f"{prefix}_max_amplitude_mV": float(amplitude_mV),
        f"{prefix}_noise_rms_mV": float(noise_rms_mV),
    }
    for fraction in fractions:
        label = _fraction_label(fraction)
        output[
            f"{prefix}_t{label}_minus_timing_ch{timing_channel_number}_led_ps"
        ] = float((crossings[fraction] - timing_led_ns) * 1000.0)

    for low_index, low_fraction in enumerate(fractions[:-1]):
        for high_fraction in fractions[low_index + 1 :]:
            low_label = _fraction_label(low_fraction)
            high_label = _fraction_label(high_fraction)
            output[f"{prefix}_rise_t{low_label}_to_t{high_label}_ps"] = float(
                (crossings[high_fraction] - crossings[low_fraction]) * 1000.0
            )
    return output


def _event_settings(config: dict[str, Any]) -> dict[str, Any]:
    dataset = config["ml_dataset"]
    waveform = config["waveform"]
    return {
        "channels": config["channels"],
        "baseline_samples": int(waveform["baseline_samples"]),
        "trigger_threshold_mV": float(waveform["trigger_threshold_mV"]),
        "led_threshold_mV": float(dataset["led_threshold_mV"]),
        "led_search_ns": {
            "before": float(dataset["led_search_ns"]["before"]),
            "after": float(dataset["led_search_ns"]["after"]),
        },
        "energy_search_ns": {
            "before": float(dataset["energy_search_ns"]["before"]),
            "after": float(dataset["energy_search_ns"]["after"]),
        },
        "crossing_step_ps": float(dataset["crossing_step_ps"]),
        "timing_window_width_ns": float(dataset["timing_window_width_ns"]),
        "polynomial_resample_step_ps": float(dataset["polynomial"]["resample_step_ps"]),
        "polynomial_degree": int(dataset["polynomial"]["degree"]),
        "polynomial_l2_regularization": float(dataset["polynomial"]["l2_regularization"]),
        "polynomial_penalize_intercept": bool(
            dataset["polynomial"].get("penalize_intercept", False)
        ),
        "energy_fractions": sorted(float(item) for item in dataset["energy_fractions"]),
    }


def process_dataset_event(
    payload: DatasetEventPayload,
    settings: dict[str, Any],
) -> EventProcessingResult:
    channels = settings["channels"]
    energy_channels = [int(item) - 1 for item in channels["energy"]]
    timing_channels = [int(item) - 1 for item in channels["timing"]]
    polarities = [int(item) for item in channels["polarities"]]

    basics = []
    time_axes_ns = []
    for channel in range(4):
        voltage_mV = decode_voltage_mV(
            payload.raw_samples[channel],
            float(payload.vertical_gain_v_per_count[channel]),
            float(payload.vertical_offset_v[channel]),
        )
        basic = baseline_and_basic_features(
            voltage_mV,
            baseline_samples=int(settings["baseline_samples"]),
            polarity=polarities[channel],
            trigger_threshold_mV=float(settings["trigger_threshold_mV"]),
            horizontal_interval_s=float(payload.horizontal_interval_s[channel]),
            horizontal_offset_s=float(payload.horizontal_offset_s[channel]),
        )
        basics.append(basic)
        time_axes_ns.append(
            (
                float(payload.horizontal_offset_s[channel])
                + np.arange(voltage_mV.size, dtype=np.float64)
                * float(payload.horizontal_interval_s[channel])
            )
            * 1.0e9
        )

    row: dict[str, Any] = {
        "meta_event_index": payload.event_index,
        "meta_event_id": payload.event_id,
        "meta_source_file_id": ";".join(str(item) for item in payload.source_file_id),
    }
    timing_led_ns: list[float] = []
    for timing_channel in timing_channels:
        timing_output, led_ns = _timing_polynomial_features(
            basics[timing_channel].corrected_signal_mV,
            time_axes_ns[timing_channel],
            basics[timing_channel].trigger_index,
            basics[timing_channel].noise_rms_mV,
            timing_channel + 1,
            settings,
        )
        if timing_output is None or led_ns is None:
            return EventProcessingResult(None, f"timing_ch{timing_channel + 1}_invalid")
        row.update(timing_output)
        timing_led_ns.append(float(led_ns))

    for detector_index, energy_channel in enumerate(energy_channels):
        timing_channel = timing_channels[detector_index]
        energy_output = _energy_features(
            basics[energy_channel].corrected_signal_mV,
            time_axes_ns[energy_channel],
            basics[energy_channel].trigger_index,
            timing_led_ns[detector_index],
            basics[energy_channel].amplitude_mV,
            basics[energy_channel].noise_rms_mV,
            energy_channel + 1,
            timing_channel + 1,
            settings,
        )
        if energy_output is None:
            return EventProcessingResult(None, f"energy_ch{energy_channel + 1}_invalid")
        row.update(energy_output)

    row["_led_tof_ps"] = float((timing_led_ns[0] - timing_led_ns[1]) * 1000.0)
    return EventProcessingResult(row, None)


def _safe_process_dataset_event(
    payload: DatasetEventPayload,
    settings: dict[str, Any],
) -> EventProcessingResult:
    try:
        return process_dataset_event(payload, settings)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        return EventProcessingResult(
            None,
            f"processing_{type(exc).__name__}",
        )


def _init_dataset_worker(settings: dict[str, Any]) -> None:
    global _WORKER_SETTINGS
    _WORKER_SETTINGS = settings


def _process_dataset_event_worker(payload: DatasetEventPayload) -> EventProcessingResult:
    if _WORKER_SETTINGS is None:
        raise RuntimeError("dataset worker was not initialized")
    return _safe_process_dataset_event(payload, _WORKER_SETTINGS)


def _source_file_id_tuple(value: Any) -> tuple[int, ...]:
    array = np.asarray(value, dtype=np.int64)
    if array.ndim == 0:
        return (int(array),)
    return tuple(int(item) for item in array.reshape(-1))


def _payloads_for_selected_chunk(
    chunk: Any,
    selected_chunk: np.ndarray,
) -> list[DatasetEventPayload]:
    payloads: list[DatasetEventPayload] = []
    for row_index in np.flatnonzero(selected_chunk):
        row = int(row_index)
        payloads.append(
            DatasetEventPayload(
                event_index=int(chunk.event_index[row]),
                event_id=int(chunk.event_id[row]),
                source_file_id=_source_file_id_tuple(chunk.source_file_id[row]),
                raw_samples=tuple(
                    np.asarray(ak.to_numpy(chunk.samples[channel][row]), dtype=np.int16)
                    for channel in range(4)
                ),
                vertical_gain_v_per_count=np.asarray(
                    chunk.vertical_gain_v_per_count[row], dtype=np.float64
                ).copy(),
                vertical_offset_v=np.asarray(
                    chunk.vertical_offset_v[row], dtype=np.float64
                ).copy(),
                horizontal_interval_s=np.asarray(
                    chunk.horizontal_interval_s[row], dtype=np.float64
                ).copy(),
                horizontal_offset_s=np.asarray(
                    chunk.horizontal_offset_s[row], dtype=np.float64
                ).copy(),
            )
        )
    return payloads


def resolve_worker_count(config: dict[str, Any], override: int | None = None) -> int:
    requested = int(
        override if override is not None else config["ml_dataset"]["parallel"].get("workers", 0)
    )
    if requested > 0:
        return requested
    cpu_count = os.cpu_count() or 1
    maximum = max(1, int(config["ml_dataset"]["parallel"].get("max_auto_workers", 8)))
    return min(maximum, max(1, cpu_count - 1))


def generate_dataset_rows(
    input_path: Path,
    selected: np.ndarray,
    config: dict[str, Any],
    *,
    workers_override: int | None = None,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    settings = _event_settings(config)
    io_config = config["io"]
    parallel_config = config["ml_dataset"]["parallel"]
    workers = resolve_worker_count(config, workers_override)
    map_chunksize = max(1, int(parallel_config.get("map_chunksize", 16)))
    progress_every = max(1, int(parallel_config.get("progress_every", 500)))
    max_events = int(io_config.get("max_events", 0))
    entry_stop = max_events if max_events > 0 else None

    LOGGER.info(
        "Dataset pass: %d selected events, %d worker(s), LED=%.3g mV, "
        "window=%.3g ns, polynomial degree=%d, L2=%g",
        int(np.count_nonzero(selected)),
        workers,
        float(settings["led_threshold_mV"]),
        float(settings["timing_window_width_ns"]),
        int(settings["polynomial_degree"]),
        float(settings["polynomial_l2_regularization"]),
    )
    LOGGER.info(
        "Energy proportional thresholds: %s",
        ", ".join(f"{100.0 * item:g}%" for item in settings["energy_fractions"]),
    )

    rows: list[dict[str, Any]] = []
    rejections: Counter[str] = Counter()
    processed_selected = 0
    global_start = 0

    executor: ProcessPoolExecutor | None = None
    if workers > 1:
        executor = ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_dataset_worker,
            initargs=(settings,),
        )

    try:
        for chunk in iterate_chunks(
            input_path,
            step_size=io_config.get("step_size", "128 MB"),
            entry_stop=entry_stop,
        ):
            chunk_size = int(chunk.event_id.size)
            selected_chunk = selected[global_start : global_start + chunk_size]
            if selected_chunk.size != chunk_size:
                raise RuntimeError("selection mask and ROOT event order are inconsistent")
            payloads = _payloads_for_selected_chunk(chunk, selected_chunk)
            global_start += chunk_size
            if not payloads:
                continue

            if executor is None:
                results: Iterable[EventProcessingResult] = (
                    _safe_process_dataset_event(payload, settings) for payload in payloads
                )
            else:
                results = executor.map(
                    _process_dataset_event_worker,
                    payloads,
                    chunksize=map_chunksize,
                )

            for result in results:
                processed_selected += 1
                if result.row is None:
                    rejections[result.rejection_reason or "unknown"] += 1
                else:
                    rows.append(result.row)
                if processed_selected % progress_every == 0:
                    LOGGER.info(
                        "Dataset pass processed %d/%d selected events; accepted=%d",
                        processed_selected,
                        int(np.count_nonzero(selected)),
                        len(rows),
                    )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)

    if global_start != selected.size:
        raise RuntimeError(
            f"processed {global_start} ROOT events but selection mask has {selected.size} entries"
        )
    LOGGER.info(
        "Dataset pass complete: accepted=%d, rejected=%d",
        len(rows),
        sum(rejections.values()),
    )
    return rows, rejections


def _target_center(values: np.ndarray, method: str) -> float:
    if method == "mean":
        return float(np.mean(values))
    if method == "median":
        return float(np.median(values))
    raise ValueError(f"unsupported target center method: {method}")


def finalize_and_write_dataset(
    rows: list[dict[str, Any]],
    output_csv: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("no events passed dataset-specific waveform extraction")
    led_tof_ps = np.asarray([float(row.pop("_led_tof_ps")) for row in rows], dtype=np.float64)
    center_method = str(config["ml_dataset"]["target"].get("center", "mean")).lower()
    center_ps = _target_center(led_tof_ps, center_method)
    target_name = str(
        config["ml_dataset"]["target"].get("column_name", "target_led_residual_ps")
    )
    for row, tof_ps in zip(rows, led_tof_ps, strict=True):
        row[target_name] = float(tof_ps - center_ps)

    metadata_columns = ["meta_event_index", "meta_event_id", "meta_source_file_id"]
    feature_columns = sorted(
        key for key in rows[0] if key not in set(metadata_columns + [target_name])
    )
    fieldnames = metadata_columns + [target_name] + feature_columns
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    target = led_tof_ps - center_ps
    summary = {
        "dataset_csv": str(output_csv),
        "rows": len(rows),
        "columns": len(fieldnames),
        "metadata_columns": metadata_columns,
        "target_column": target_name,
        "feature_columns": feature_columns,
        "target_definition": "LED(ch_timing_a) - LED(ch_timing_b) - configured center",
        "target_center_method": center_method,
        "target_center_led_tof_ps": center_ps,
        "target_mean_ps": float(np.mean(target)),
        "target_std_ps": float(np.std(target, ddof=1)) if target.size > 1 else 0.0,
        "polynomial_time_coordinate": (
            "x=(t-t_LED)/(window_width/2), so x is in [-1,1]; coefficients use increasing powers"
        ),
        "absolute_timestamp_features_included": False,
    }
    LOGGER.info("Dataset written: %s", output_csv)
    LOGGER.info(
        "Target center (%s): %.6f ps; target standard deviation: %.3f ps",
        center_method,
        center_ps,
        summary["target_std_ps"],
    )
    return summary
