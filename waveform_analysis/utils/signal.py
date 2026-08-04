from __future__ import annotations

from dataclasses import dataclass

import numpy as np
FEMTOSECONDS_PER_SECOND = 1_000_000_000_000_000
FEMTOSECONDS_PER_NANOSECOND = 1_000_000
INVALID_TIME_FS = np.iinfo(np.int64).min
INVALID_INDEX = -1


@dataclass(frozen=True)
class BasicFeatures:
    baseline_mV: float
    noise_rms_mV: float
    amplitude_mV: float
    peak_index: int
    trigger_index: int
    trigger_time_fs: np.int64
    corrected_signal_mV: np.ndarray


@dataclass(frozen=True)
class TimingFeatures:
    led_times_fs: np.ndarray
    cfd_times_fs: np.ndarray
    cropped_peak_mV: float
    crop_start_fs: np.int64
    crop_stop_fs: np.int64


def baseline_and_basic_features(
    voltage_mV: np.ndarray,
    *,
    baseline_samples: int,
    polarity: int,
    trigger_threshold_mV: float,
    horizontal_interval_s: float,
    horizontal_offset_s: float,
) -> BasicFeatures:
    values = np.asarray(voltage_mV, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("waveform must be one-dimensional with at least two samples")
    if polarity not in (-1, 1):
        raise ValueError("polarity must be +1 or -1")
    if not np.isfinite(horizontal_interval_s) or horizontal_interval_s <= 0:
        raise ValueError("horizontal interval must be finite and positive")
    if not np.isfinite(horizontal_offset_s):
        raise ValueError("horizontal offset must be finite")

    count = min(values.size, max(1, int(baseline_samples)))
    baseline = float(np.mean(values[:count]))
    residual = values[:count] - baseline
    noise = float(np.sqrt(np.mean(residual * residual)))
    corrected = polarity * (values - baseline)
    peak_index = int(np.argmax(corrected))
    amplitude = float(corrected[peak_index])

    crossing = np.flatnonzero(corrected > float(trigger_threshold_mV))
    trigger_index = int(crossing[0]) if crossing.size else INVALID_INDEX
    if trigger_index >= 0:
        trigger_time_s = horizontal_offset_s + trigger_index * horizontal_interval_s
        trigger_time_fs = np.int64(np.rint(trigger_time_s * FEMTOSECONDS_PER_SECOND))
    else:
        trigger_time_fs = np.int64(INVALID_TIME_FS)

    return BasicFeatures(
        baseline_mV=baseline,
        noise_rms_mV=noise,
        amplitude_mV=amplitude,
        peak_index=peak_index,
        trigger_index=trigger_index,
        trigger_time_fs=trigger_time_fs,
        corrected_signal_mV=corrected,
    )


def _sequential_crossings_fs(
    time_ns: np.ndarray,
    signal_mV: np.ndarray,
    thresholds_mV: np.ndarray,
) -> np.ndarray:
    """Match the original C LED search on one rising edge.

    Thresholds are processed in ascending order and the search index is retained,
    as in ``TWaveForm::LED``. Crossing timestamps are rounded to int64 fs only
    after linear interpolation between adjacent native acquisition samples.
    """
    x = np.asarray(time_ns, dtype=np.float64)
    y = np.asarray(signal_mV, dtype=np.float64)
    thresholds = np.asarray(thresholds_mV, dtype=np.float64)
    result = np.full(thresholds.shape, INVALID_TIME_FS, dtype=np.int64)
    if x.size < 2 or x.size != y.size:
        return result

    index = 0
    for out_index, threshold in enumerate(thresholds):
        if not np.isfinite(threshold) or threshold <= 0:
            continue
        while index < y.size and y[index] < threshold:
            index += 1
        if index <= 0 or index >= y.size:
            continue
        y0, y1 = y[index - 1], y[index]
        if not (np.isfinite(y0) and np.isfinite(y1)) or y1 == y0:
            continue
        fraction = (threshold - y0) / (y1 - y0)
        if not 0.0 <= fraction <= 1.0:
            continue
        crossing_ns = x[index - 1] + fraction * (x[index] - x[index - 1])
        result[out_index] = np.int64(
            np.rint(crossing_ns * FEMTOSECONDS_PER_NANOSECOND)
        )
    return result


def _last_rising_crossing_before_peak_fs(
    time_ns: np.ndarray,
    signal_mV: np.ndarray,
    thresholds_mV: np.ndarray,
) -> np.ndarray:
    """Return the final rising crossing before the crop maximum for each threshold."""

    x = np.asarray(time_ns, dtype=np.float64)
    y = np.asarray(signal_mV, dtype=np.float64)
    thresholds = np.asarray(thresholds_mV, dtype=np.float64)
    result = np.full(thresholds.shape, INVALID_TIME_FS, dtype=np.int64)
    if x.size < 2 or x.size != y.size:
        return result
    peak_index = int(np.argmax(y))
    if peak_index <= 0 or not np.isfinite(y[peak_index]):
        return result

    x0 = x[:peak_index]
    x1 = x[1 : peak_index + 1]
    y0 = y[:peak_index]
    y1 = y[1 : peak_index + 1]
    finite = np.isfinite(x0) & np.isfinite(x1) & np.isfinite(y0) & np.isfinite(y1)

    for out_index, threshold in enumerate(thresholds):
        if not np.isfinite(threshold) or threshold <= 0.0:
            continue
        crossing = finite & (y0 < threshold) & (y1 >= threshold)
        indices = np.flatnonzero(crossing)
        if indices.size == 0:
            crossing = finite & (y0 == threshold) & (y1 > threshold)
            indices = np.flatnonzero(crossing)
        if indices.size == 0:
            continue
        lower = int(indices[-1])
        if x1[lower] <= x0[lower] or y1[lower] == y0[lower]:
            continue
        fraction = (threshold - y0[lower]) / (y1[lower] - y0[lower])
        if not 0.0 <= fraction <= 1.0:
            continue
        crossing_ns = x0[lower] + fraction * (x1[lower] - x0[lower])
        result[out_index] = np.int64(np.rint(crossing_ns * FEMTOSECONDS_PER_NANOSECOND))
    return result


def prepare_timing_features(
    corrected_signal_mV: np.ndarray,
    *,
    trigger_index: int,
    horizontal_interval_s: float,
    horizontal_offset_s: float,
    crop_before_ns: float,
    crop_after_ns: float,
    upsample_step_ps: float | None = None,
    led_thresholds_mV: np.ndarray,
    cfd_fractions: np.ndarray,
) -> TimingFeatures:
    """Extract LED/CFD times from the native samples.

    ``upsample_step_ps`` is retained as a deprecated compatibility argument and
    is ignored. Only the final threshold crossing time is interpolated, using the
    two original samples that bracket the crossing.
    """

    del upsample_step_ps
    invalid_led = np.full(np.asarray(led_thresholds_mV).shape, INVALID_TIME_FS, dtype=np.int64)
    invalid_cfd = np.full(np.asarray(cfd_fractions).shape, INVALID_TIME_FS, dtype=np.int64)
    if trigger_index < 0:
        return TimingFeatures(
            invalid_led,
            invalid_cfd,
            np.nan,
            np.int64(INVALID_TIME_FS),
            np.int64(INVALID_TIME_FS),
        )

    signal = np.asarray(corrected_signal_mV, dtype=np.float64)
    sample_index = np.arange(signal.size, dtype=np.float64)
    time_ns = (
        horizontal_offset_s + sample_index * horizontal_interval_s
    ) * 1.0e9
    trigger_ns = float(time_ns[trigger_index])
    requested_start = trigger_ns - float(crop_before_ns)
    requested_stop = trigger_ns + float(crop_after_ns)

    # Keep one original sample outside each boundary so crossings at the crop
    # edge still have a valid bracketing pair.
    start_index = max(0, int(np.searchsorted(time_ns, requested_start, side="left")) - 1)
    stop_index = min(signal.size, int(np.searchsorted(time_ns, requested_stop, side="right")) + 1)
    if stop_index - start_index < 2:
        return TimingFeatures(
            invalid_led,
            invalid_cfd,
            np.nan,
            np.int64(INVALID_TIME_FS),
            np.int64(INVALID_TIME_FS),
        )

    crop_time_ns = time_ns[start_index:stop_index]
    crop_signal = signal[start_index:stop_index]
    if np.any(~np.isfinite(crop_time_ns)) or np.any(~np.isfinite(crop_signal)):
        return TimingFeatures(
            invalid_led,
            invalid_cfd,
            np.nan,
            np.int64(INVALID_TIME_FS),
            np.int64(INVALID_TIME_FS),
        )

    cropped_peak = float(np.max(crop_signal))
    led_times = _sequential_crossings_fs(crop_time_ns, crop_signal, led_thresholds_mV)
    cfd_thresholds = cropped_peak * np.asarray(cfd_fractions, dtype=np.float64)
    cfd_times = _last_rising_crossing_before_peak_fs(
        crop_time_ns, crop_signal, cfd_thresholds
    )

    return TimingFeatures(
        led_times_fs=led_times,
        cfd_times_fs=cfd_times,
        cropped_peak_mV=cropped_peak,
        crop_start_fs=np.int64(
            np.rint(crop_time_ns[0] * FEMTOSECONDS_PER_NANOSECOND)
        ),
        crop_stop_fs=np.int64(
            np.rint(crop_time_ns[-1] * FEMTOSECONDS_PER_NANOSECOND)
        ),
    )
