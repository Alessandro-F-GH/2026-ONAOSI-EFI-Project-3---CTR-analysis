from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.interpolate import CubicSpline


def relative_grid_ns(width_ns: float, step_ps: float) -> np.ndarray:
    count = int(round(float(width_ns) * 1000.0 / float(step_ps)))
    if count < 2:
        raise ValueError("crop grid requires at least two samples")
    step_ns = float(step_ps) / 1000.0
    return (np.arange(count, dtype=np.float64) - (count - 1) / 2.0) * step_ns


def first_rising_crossing_ns(
    time_ns: np.ndarray,
    signal_mV: np.ndarray,
    threshold_mV: float,
) -> float:
    x = np.asarray(time_ns, dtype=np.float64)
    y = np.asarray(signal_mV, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 2:
        return np.nan
    peak_index = int(np.argmax(y))
    if peak_index < 1 or not np.isfinite(y[peak_index]) or y[peak_index] < threshold_mV:
        return np.nan
    candidates = np.flatnonzero(
        (y[:peak_index] < threshold_mV) & (y[1 : peak_index + 1] >= threshold_mV)
    )
    if candidates.size == 0:
        return np.nan
    index = int(candidates[0])
    y0 = float(y[index])
    y1 = float(y[index + 1])
    if not np.isfinite(y0 + y1) or y1 == y0:
        return np.nan
    fraction = (float(threshold_mV) - y0) / (y1 - y0)
    return float(x[index] + fraction * (x[index + 1] - x[index]))


def prepare_signal_sampler(
    time_ns: np.ndarray,
    signal_mV: np.ndarray,
    *,
    interpolation: str,
) -> Callable[[np.ndarray], np.ndarray | None]:
    x = np.asarray(time_ns, dtype=np.float64)
    y = np.asarray(signal_mV, dtype=np.float64)
    if x.size < 4 or x.size != y.size or np.any(np.diff(x) <= 0):
        raise ValueError("invalid waveform for interpolation")
    spline = CubicSpline(x, y, extrapolate=False) if interpolation == "cubic" else None
    if interpolation not in {"linear", "cubic"}:
        raise ValueError(f"unsupported interpolation: {interpolation}")

    def sample(query_ns: np.ndarray) -> np.ndarray | None:
        query = np.asarray(query_ns, dtype=np.float64)
        if query[0] < x[0] or query[-1] > x[-1]:
            return None
        sampled = np.interp(query, x, y) if spline is None else spline(query)
        sampled = np.asarray(sampled, dtype=np.float64)
        if sampled.shape != query.shape or np.any(~np.isfinite(sampled)):
            return None
        return sampled

    return sample


def sample_signal(
    time_ns: np.ndarray,
    signal_mV: np.ndarray,
    query_ns: np.ndarray,
    *,
    interpolation: str,
) -> np.ndarray | None:
    x = np.asarray(time_ns, dtype=np.float64)
    y = np.asarray(signal_mV, dtype=np.float64)
    query = np.asarray(query_ns, dtype=np.float64)
    if x.size < 4 or x.size != y.size or np.any(np.diff(x) <= 0):
        return None
    if query[0] < x[0] or query[-1] > x[-1]:
        return None
    if interpolation == "linear":
        sampled = np.interp(query, x, y)
    elif interpolation == "cubic":
        sampled = CubicSpline(x, y, extrapolate=False)(query)
    else:
        raise ValueError(f"unsupported interpolation: {interpolation}")
    sampled = np.asarray(sampled, dtype=np.float64)
    if sampled.shape != query.shape or np.any(~np.isfinite(sampled)):
        return None
    return sampled


def direct_pair_crop(
    time3_ns: np.ndarray,
    signal3_mV: np.ndarray,
    time4_ns: np.ndarray,
    signal4_mV: np.ndarray,
    *,
    t3_cross_ns: float,
    t4_cross_ns: float,
    shift_ps: float,
    shifted_timing_channel: int,
    relative_grid: np.ndarray,
    interpolation: str,
    sampler3: Callable[[np.ndarray], np.ndarray | None] | None = None,
    sampler4: Callable[[np.ndarray], np.ndarray | None] | None = None,
) -> np.ndarray | None:
    """Build the paper-like pair crop after translating one timing channel."""
    shift_ns = float(shift_ps) / 1000.0
    if shifted_timing_channel == 3:
        shifted_t3 = t3_cross_ns + shift_ns
        shifted_t4 = t4_cross_ns
    elif shifted_timing_channel == 4:
        shifted_t3 = t3_cross_ns
        shifted_t4 = t4_cross_ns + shift_ns
    else:
        raise ValueError("shifted_timing_channel must be 3 or 4")
    center_ns = 0.5 * (shifted_t3 + shifted_t4)
    output_grid = center_ns + relative_grid
    query3 = output_grid - (shift_ns if shifted_timing_channel == 3 else 0.0)
    query4 = output_grid - (shift_ns if shifted_timing_channel == 4 else 0.0)
    crop3 = (
        sampler3(query3)
        if sampler3 is not None
        else sample_signal(time3_ns, signal3_mV, query3, interpolation=interpolation)
    )
    crop4 = (
        sampler4(query4)
        if sampler4 is not None
        else sample_signal(time4_ns, signal4_mV, query4, interpolation=interpolation)
    )
    if crop3 is None or crop4 is None:
        return None
    return np.stack((crop3, crop4), axis=0).astype(np.float32)


def invariant_pair_crop(
    time3_ns: np.ndarray,
    signal3_mV: np.ndarray,
    time4_ns: np.ndarray,
    signal4_mV: np.ndarray,
    *,
    t3_cross_ns: float,
    t4_cross_ns: float,
    relative_grid: np.ndarray,
    interpolation: str,
    sampler3: Callable[[np.ndarray], np.ndarray | None] | None = None,
    sampler4: Callable[[np.ndarray], np.ndarray | None] | None = None,
) -> np.ndarray | None:
    """Crop each waveform around its own threshold crossing."""
    query3 = t3_cross_ns + relative_grid
    query4 = t4_cross_ns + relative_grid
    crop3 = (
        sampler3(query3)
        if sampler3 is not None
        else sample_signal(time3_ns, signal3_mV, query3, interpolation=interpolation)
    )
    crop4 = (
        sampler4(query4)
        if sampler4 is not None
        else sample_signal(time4_ns, signal4_mV, query4, interpolation=interpolation)
    )
    if crop3 is None or crop4 is None:
        return None
    return np.stack((crop3, crop4), axis=0).astype(np.float32)
