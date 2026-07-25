from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline

from utils.io import decode_voltage_mV
from utils.signal import (
    FEMTOSECONDS_PER_NANOSECOND,
    INVALID_TIME_FS,
    baseline_and_basic_features,
)


@dataclass(frozen=True)
class ChannelExtraction:
    amplitude_mV: float
    noise_rms_mV: float
    trigger_index: int
    led_time_fs: np.int64
    cfd_time_fs: np.int64
    window_mV: np.ndarray
    valid: bool


def relative_window_grid_ps(waveform_config: dict[str, Any]) -> np.ndarray:
    base_step = float(waveform_config["upsample_step_ps"])
    factor = int(waveform_config["subsample_factor"])
    before = float(waveform_config["ml_window_ns"]["before"]) * 1000.0
    after = float(waveform_config["ml_window_ns"]["after"]) * 1000.0
    full = np.arange(-before, after + 0.5 * base_step, base_step, dtype=np.float64)
    sampled = full[::factor]
    if sampled.size < 4:
        raise ValueError("ML window contains fewer than four points after subsampling")
    return sampled


def _first_rising_crossing_ns(
    time_ns: np.ndarray, signal_mV: np.ndarray, threshold_mV: float
) -> float:
    threshold = float(threshold_mV)
    if not np.isfinite(threshold) or threshold <= 0:
        return np.nan
    above = np.flatnonzero(signal_mV >= threshold)
    if above.size == 0:
        return np.nan
    index = int(above[0])
    if index <= 0:
        return np.nan
    x0, x1 = float(time_ns[index - 1]), float(time_ns[index])
    y0, y1 = float(signal_mV[index - 1]), float(signal_mV[index])
    if not np.all(np.isfinite([x0, x1, y0, y1])) or y1 == y0:
        return np.nan
    fraction = (threshold - y0) / (y1 - y0)
    if not 0.0 <= fraction <= 1.0:
        return np.nan
    return x0 + fraction * (x1 - x0)


def extract_channel(
    raw_samples: np.ndarray,
    *,
    vertical_gain_v_per_count: float,
    vertical_offset_v: float,
    horizontal_interval_s: float,
    horizontal_offset_s: float,
    polarity: int,
    waveform_config: dict[str, Any],
    relative_grid_ps: np.ndarray,
) -> ChannelExtraction:
    invalid_window = np.full(relative_grid_ps.shape, np.nan, dtype=np.float32)
    voltage_mV = decode_voltage_mV(
        raw_samples, vertical_gain_v_per_count, vertical_offset_v
    )
    basic = baseline_and_basic_features(
        voltage_mV,
        baseline_samples=int(waveform_config["baseline_samples"]),
        polarity=int(polarity),
        trigger_threshold_mV=float(waveform_config["search_trigger_threshold_mV"]),
        horizontal_interval_s=float(horizontal_interval_s),
        horizontal_offset_s=float(horizontal_offset_s),
    )
    invalid = ChannelExtraction(
        amplitude_mV=basic.amplitude_mV,
        noise_rms_mV=basic.noise_rms_mV,
        trigger_index=basic.trigger_index,
        led_time_fs=np.int64(INVALID_TIME_FS),
        cfd_time_fs=np.int64(INVALID_TIME_FS),
        window_mV=invalid_window,
        valid=False,
    )
    if basic.trigger_index < 0:
        return invalid

    signal = np.asarray(basic.corrected_signal_mV, dtype=np.float64)
    time_ns = (
        float(horizontal_offset_s)
        + np.arange(signal.size, dtype=np.float64) * float(horizontal_interval_s)
    ) * 1.0e9
    trigger_ns = float(time_ns[basic.trigger_index])
    crop_start = trigger_ns - float(waveform_config["analysis_crop_ns"]["before"])
    crop_stop = trigger_ns + float(waveform_config["analysis_crop_ns"]["after"])
    start_index = max(0, int(np.searchsorted(time_ns, crop_start, side="left")) - 1)
    stop_index = min(
        signal.size, int(np.searchsorted(time_ns, crop_stop, side="right")) + 1
    )
    if stop_index - start_index < 4:
        return invalid
    crop_time = time_ns[start_index:stop_index]
    crop_signal = signal[start_index:stop_index]
    if np.any(~np.isfinite(crop_time)) or np.any(~np.isfinite(crop_signal)):
        return invalid

    try:
        spline = CubicSpline(
            crop_time, crop_signal, bc_type="not-a-knot", extrapolate=False
        )
    except Exception:
        return invalid
    step_ns = float(waveform_config["upsample_step_ps"]) / 1000.0
    up_time = np.arange(
        crop_time[0], crop_time[-1] + 0.25 * step_ns, step_ns, dtype=np.float64
    )
    if up_time.size < 4:
        return invalid
    up_signal = np.asarray(spline(up_time), dtype=np.float64)
    if np.any(~np.isfinite(up_signal)):
        return invalid
    peak = float(np.max(up_signal))
    led_ns = _first_rising_crossing_ns(
        up_time, up_signal, float(waveform_config["led_threshold_mV"])
    )
    cfd_ns = _first_rising_crossing_ns(
        up_time, up_signal, peak * float(waveform_config["cfd_fraction"])
    )
    if not np.isfinite(led_ns) or not np.isfinite(cfd_ns):
        return invalid

    window_time = led_ns + np.asarray(relative_grid_ps, dtype=np.float64) / 1000.0
    if window_time[0] < crop_time[0] or window_time[-1] > crop_time[-1]:
        return invalid
    window = np.asarray(spline(window_time), dtype=np.float32)
    if np.any(~np.isfinite(window)):
        return invalid
    return ChannelExtraction(
        amplitude_mV=basic.amplitude_mV,
        noise_rms_mV=basic.noise_rms_mV,
        trigger_index=basic.trigger_index,
        led_time_fs=np.int64(np.rint(led_ns * FEMTOSECONDS_PER_NANOSECOND)),
        cfd_time_fs=np.int64(np.rint(cfd_ns * FEMTOSECONDS_PER_NANOSECOND)),
        window_mV=window,
        valid=True,
    )
