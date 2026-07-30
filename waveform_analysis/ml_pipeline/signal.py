from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline

from utils.signal import (
    FEMTOSECONDS_PER_NANOSECOND,
    INVALID_TIME_FS,
    BasicFeatures,
    baseline_and_basic_features,
)

from .denoising import apply_optional_lowpass_denoising


def _decode_voltage_mV(
    raw_samples: np.ndarray,
    vertical_gain_v_per_count: float,
    vertical_offset_v: float,
) -> np.ndarray:
    raw = np.asarray(raw_samples, dtype=np.float64)
    return (raw * float(vertical_gain_v_per_count) - float(vertical_offset_v)) * 1000.0


@dataclass(frozen=True)
class TimingReference:
    trigger_index: int
    led_time_fs: np.int64
    cfd_time_fs: np.int64
    valid: bool


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


def timing_channel_waveform_config(waveform_config: dict[str, Any]) -> dict[str, Any]:
    """Resolve timing-channel LED settings, inheriting energy defaults.

    Only timing extraction options are copied. ML-window settings intentionally
    remain energy-channel-only.
    """

    override = waveform_config.get("timing_channel_led", {})
    if not isinstance(override, dict):
        raise ValueError("waveform.timing_channel_led must be an object")
    keys = (
        "baseline_samples",
        "search_trigger_threshold_mV",
        "analysis_crop_ns",
        "upsample_step_ps",
        "led_threshold_mV",
        "denoising",
    )
    resolved: dict[str, Any] = {}
    for key in keys:
        source = override[key] if key in override else waveform_config[key]
        resolved[key] = deepcopy(source)
    return resolved


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


def _basic_features(
    raw_samples: np.ndarray,
    *,
    vertical_gain_v_per_count: float,
    vertical_offset_v: float,
    horizontal_interval_s: float,
    horizontal_offset_s: float,
    polarity: int,
    extraction_config: dict[str, Any],
) -> BasicFeatures:
    voltage_mV = _decode_voltage_mV(
        raw_samples, vertical_gain_v_per_count, vertical_offset_v
    )
    basic = baseline_and_basic_features(
        voltage_mV,
        baseline_samples=int(extraction_config["baseline_samples"]),
        polarity=int(polarity),
        trigger_threshold_mV=float(extraction_config["search_trigger_threshold_mV"]),
        horizontal_interval_s=float(horizontal_interval_s),
        horizontal_offset_s=float(horizontal_offset_s),
    )

    denoising_config = extraction_config.get("denoising")
    if bool((denoising_config or {}).get("enabled", False)):
        denoised_signal = apply_optional_lowpass_denoising(
            basic.corrected_signal_mV,
            horizontal_interval_s=float(horizontal_interval_s),
            denoising_config=denoising_config,
        )
        # The first pass has already applied the configured polarity and removed
        # the baseline. Recompute features on this positive-oriented signal.
        basic = baseline_and_basic_features(
            denoised_signal,
            baseline_samples=int(extraction_config["baseline_samples"]),
            polarity=1,
            trigger_threshold_mV=float(
                extraction_config["search_trigger_threshold_mV"]
            ),
            horizontal_interval_s=float(horizontal_interval_s),
            horizontal_offset_s=float(horizontal_offset_s),
        )
    return basic


def _timing_from_basic(
    basic: BasicFeatures,
    *,
    horizontal_interval_s: float,
    horizontal_offset_s: float,
    extraction_config: dict[str, Any],
    compute_led: bool = True,
    compute_cfd: bool = True,
) -> TimingReference:
    invalid = TimingReference(
        trigger_index=basic.trigger_index,
        led_time_fs=np.int64(INVALID_TIME_FS),
        cfd_time_fs=np.int64(INVALID_TIME_FS),
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
    crop_start = trigger_ns - float(extraction_config["analysis_crop_ns"]["before"])
    crop_stop = trigger_ns + float(extraction_config["analysis_crop_ns"]["after"])
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
    step_ns = float(extraction_config["upsample_step_ps"]) / 1000.0
    up_time = np.arange(
        crop_time[0], crop_time[-1] + 0.25 * step_ns, step_ns, dtype=np.float64
    )
    if up_time.size < 4:
        return invalid
    up_signal = np.asarray(spline(up_time), dtype=np.float64)
    if np.any(~np.isfinite(up_signal)):
        return invalid
    peak = float(np.max(up_signal))
    if compute_led:
        led_ns = _first_rising_crossing_ns(
            up_time, up_signal, float(extraction_config["led_threshold_mV"])
        )
    else:
        led_ns = np.nan
    if compute_cfd:
        cfd_ns = _first_rising_crossing_ns(
            up_time, up_signal, peak * float(extraction_config["cfd_fraction"])
        )
    else:
        cfd_ns = np.nan
    if (compute_led and not np.isfinite(led_ns)) or (
        compute_cfd and not np.isfinite(cfd_ns)
    ):
        return invalid
    return TimingReference(
        trigger_index=basic.trigger_index,
        led_time_fs=(
            np.int64(np.rint(led_ns * FEMTOSECONDS_PER_NANOSECOND))
            if compute_led
            else np.int64(INVALID_TIME_FS)
        ),
        cfd_time_fs=(
            np.int64(np.rint(cfd_ns * FEMTOSECONDS_PER_NANOSECOND))
            if compute_cfd
            else np.int64(INVALID_TIME_FS)
        ),
        valid=True,
    )


def extract_timing_reference(
    raw_samples: np.ndarray,
    *,
    vertical_gain_v_per_count: float,
    vertical_offset_v: float,
    horizontal_interval_s: float,
    horizontal_offset_s: float,
    polarity: int,
    waveform_config: dict[str, Any],
) -> TimingReference:
    """Extract an LED timestamp from a timing channel only.

    No timing waveform samples are returned or persisted for ML use.
    """

    extraction_config = timing_channel_waveform_config(waveform_config)
    basic = _basic_features(
        np.asarray(raw_samples, dtype=np.int16),
        vertical_gain_v_per_count=vertical_gain_v_per_count,
        vertical_offset_v=vertical_offset_v,
        horizontal_interval_s=horizontal_interval_s,
        horizontal_offset_s=horizontal_offset_s,
        polarity=polarity,
        extraction_config=extraction_config,
    )
    return _timing_from_basic(
        basic,
        horizontal_interval_s=horizontal_interval_s,
        horizontal_offset_s=horizontal_offset_s,
        extraction_config=extraction_config,
        compute_led=True,
        compute_cfd=False,
    )


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
    timing_reference: TimingReference | None = None,
) -> ChannelExtraction:
    """Extract one energy-channel ML window and its timing labels.

    When ``timing_reference`` is supplied, its LED timestamp controls both the
    reported LED value and the absolute alignment of the energy-channel ML
    window. The timing waveform itself is never included in ``window_mV``.
    """

    invalid_window = np.full(relative_grid_ps.shape, np.nan, dtype=np.float32)
    basic = _basic_features(
        np.asarray(raw_samples, dtype=np.int16),
        vertical_gain_v_per_count=vertical_gain_v_per_count,
        vertical_offset_v=vertical_offset_v,
        horizontal_interval_s=horizontal_interval_s,
        horizontal_offset_s=horizontal_offset_s,
        polarity=polarity,
        extraction_config=waveform_config,
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

    energy_timing = _timing_from_basic(
        basic,
        horizontal_interval_s=horizontal_interval_s,
        horizontal_offset_s=horizontal_offset_s,
        extraction_config=waveform_config,
        compute_led=timing_reference is None,
        compute_cfd=True,
    )
    if not energy_timing.valid:
        return invalid
    reference = energy_timing if timing_reference is None else timing_reference
    if not reference.valid:
        return invalid

    signal = np.asarray(basic.corrected_signal_mV, dtype=np.float64)
    time_ns = (
        float(horizontal_offset_s)
        + np.arange(signal.size, dtype=np.float64) * float(horizontal_interval_s)
    ) * 1.0e9
    alignment_ns = float(reference.led_time_fs) / FEMTOSECONDS_PER_NANOSECOND
    relative_ns = np.asarray(relative_grid_ps, dtype=np.float64) / 1000.0
    window_time = alignment_ns + relative_ns

    # Retain the established energy-trigger crop, but expand it when necessary
    # so the externally aligned energy window is fully covered.
    trigger_ns = float(time_ns[basic.trigger_index])
    crop_start = min(
        trigger_ns - float(waveform_config["analysis_crop_ns"]["before"]),
        float(window_time[0]),
    )
    crop_stop = max(
        trigger_ns + float(waveform_config["analysis_crop_ns"]["after"]),
        float(window_time[-1]),
    )
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
    if window_time[0] < crop_time[0] or window_time[-1] > crop_time[-1]:
        return invalid

    try:
        spline = CubicSpline(
            crop_time, crop_signal, bc_type="not-a-knot", extrapolate=False
        )
    except Exception:
        return invalid
    window = np.asarray(spline(window_time), dtype=np.float32)
    if np.any(~np.isfinite(window)):
        return invalid
    return ChannelExtraction(
        amplitude_mV=basic.amplitude_mV,
        noise_rms_mV=basic.noise_rms_mV,
        trigger_index=basic.trigger_index,
        led_time_fs=reference.led_time_fs,
        cfd_time_fs=energy_timing.cfd_time_fs,
        window_mV=window,
        valid=True,
    )
