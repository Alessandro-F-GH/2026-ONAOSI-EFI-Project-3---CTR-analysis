from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable

import numpy as np

from utils.signal import INVALID_TIME_FS, prepare_timing_features

from ..dataset import PreparedDataset
from ..metrics import residual_metrics


@dataclass(frozen=True)
class FamilySelection:
    family: str
    cfd_enabled: bool
    led_threshold_mV: float
    cfd_fraction: float
    led_validation_sctr_ps: float
    cfd_validation_sctr_ps: float
    led_fold_sctr_ps: tuple[float, ...]
    cfd_fold_sctr_ps: tuple[float, ...]
    led_times_fs: np.ndarray
    cfd_times_fs: np.ndarray
    led_search_low_mV: float
    led_search_high_mV: float
    led_coarse_points: int
    led_refine_points: int
    cfd_coarse_points: int
    cfd_refine_points: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "cfd_enabled": bool(self.cfd_enabled),
            "led_threshold_mV": float(self.led_threshold_mV),
            "cfd_fraction": (
                float(self.cfd_fraction) if self.cfd_enabled else None
            ),
            "led_validation_sctr_ps": float(self.led_validation_sctr_ps),
            "cfd_validation_sctr_ps": (
                float(self.cfd_validation_sctr_ps) if self.cfd_enabled else None
            ),
            "led_fold_sctr_ps": [float(v) for v in self.led_fold_sctr_ps],
            "cfd_fold_sctr_ps": (
                [float(v) for v in self.cfd_fold_sctr_ps]
                if self.cfd_enabled else []
            ),
            "led_search_low_mV": float(self.led_search_low_mV),
            "led_search_high_mV": float(self.led_search_high_mV),
            "led_coarse_points": int(self.led_coarse_points),
            "led_refine_points": int(self.led_refine_points),
            "cfd_coarse_points": int(self.cfd_coarse_points),
            "cfd_refine_points": int(self.cfd_refine_points),
            "selection_metric": "mean_fold_sctr_sample_std",
            "blind_used_for_selection": False,
        }

class ShiftedWaveformArray:
    """Lazy native-sample re-alignment without duplicating prepared waveforms."""

    def __init__(self, base: np.ndarray, shifts_samples: np.ndarray) -> None:
        self.base = base
        self.shifts_samples = np.asarray(shifts_samples, dtype=np.int64)
        if len(base.shape) != 3 or int(base.shape[1]) != 2:
            raise ValueError("ShiftedWaveformArray expects [event, detector, sample]")
        if self.shifts_samples.shape != tuple(base.shape[:2]):
            raise ValueError("Per-event shifts must have shape [event, detector]")
        self.shape = tuple(base.shape)
        self.dtype = np.dtype(getattr(base, "dtype", np.float32))

    def __getitem__(self, key: Any) -> np.ndarray:
        keys = key if isinstance(key, tuple) else (key,)
        keys = (*keys, *([slice(None)] * (3 - len(keys))))
        if len(keys) != 3:
            raise IndexError("Waveform indexing supports at most three axes")
        event_key, detector_key, sample_key = keys

        all_events = np.arange(self.shape[0], dtype=np.int64)
        all_detectors = np.arange(self.shape[1], dtype=np.int64)
        all_samples = np.arange(self.shape[2], dtype=np.int64)
        event_scalar = isinstance(event_key, (int, np.integer))
        detector_scalar = isinstance(detector_key, (int, np.integer))
        sample_scalar = isinstance(sample_key, (int, np.integer))
        events = np.atleast_1d(all_events[event_key]).astype(np.int64, copy=False)
        detectors = np.atleast_1d(all_detectors[detector_key]).astype(np.int64, copy=False)
        samples = np.atleast_1d(all_samples[sample_key]).astype(np.int64, copy=False)

        shifts = self.shifts_samples[np.ix_(events, detectors)]
        source_samples = samples[None, None, :] + shifts[:, :, None]
        valid = (source_samples >= 0) & (source_samples < self.shape[2])
        clipped = np.clip(source_samples, 0, self.shape[2] - 1)
        event_grid = events[:, None, None]
        detector_grid = detectors[None, :, None]
        values = np.asarray(
            self.base[event_grid, detector_grid, clipped],
            dtype=self.dtype,
        )
        if not np.all(valid):
            values = values.copy()
            values[~valid] = np.nan
        if sample_scalar:
            values = np.squeeze(values, axis=2)
        if detector_scalar:
            values = np.squeeze(values, axis=1)
        if event_scalar:
            values = np.squeeze(values, axis=0)
        return values


def _alignment_shifts(
    selected_led_fs: np.ndarray,
    source_anchor_fs: np.ndarray,
    relative_time_ps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(selected_led_fs, dtype=np.int64)
    anchors = np.asarray(source_anchor_fs, dtype=np.int64)
    if selected.shape != anchors.shape or selected.ndim != 2 or selected.shape[1] != 2:
        raise ValueError("Selected LED and source anchors must have shape [event,2]")
    t = np.asarray(relative_time_ps, dtype=np.float64)
    if t.size < 2:
        raise ValueError("Cannot re-align a waveform with fewer than two time samples")
    dt_fs = int(np.rint(float(np.median(np.diff(t))) * 1000.0))
    if dt_fs <= 0:
        raise ValueError("Prepared waveform sample interval must be positive")
    shifts = np.rint(
        (selected.astype(np.float64) - anchors.astype(np.float64)) / float(dt_fs)
    ).astype(np.int64)
    new_anchors = anchors + shifts * np.int64(dt_fs)
    return shifts, new_anchors


def _check_shift_support(
    shifts: np.ndarray,
    relative_time_ps: np.ndarray,
    windows: list[dict[str, Any]],
    *,
    label: str,
) -> None:
    t_ns = np.asarray(relative_time_ps, dtype=np.float64) / 1000.0
    low = min(float(window["start_ns"]) for window in windows)
    high = max(float(window["end_ns"]) for window in windows)
    selected = np.flatnonzero((t_ns >= low - 1e-9) & (t_ns <= high + 1e-9))
    if selected.size == 0:
        raise ValueError(f"No prepared samples cover experiment windows for {label}")
    first = int(selected[0])
    last = int(selected[-1])
    minimum = int(np.min(shifts))
    maximum = int(np.max(shifts))
    if first + minimum < 0 or last + maximum >= t_ns.size:
        raise RuntimeError(
            f"Prepared {label} waveform has insufficient alignment padding for the "
            f"selected LED (sample shifts {minimum}..{maximum}). Increase "
            "standard_methods.alignment_padding_ns and rebuild preprocessing."
        )


def family_for_mode(mode: str) -> str:
    key = str(mode)
    return "timing" if key.endswith("_to_timing") or key == "timing_to_timing" else "energy"


def cfd_enabled_for_family(config: dict[str, Any], family: str) -> bool:
    standard = config.get("standard_methods", {}) or {}
    mapping = standard.get("cfd_enabled_by_family", {}) or {}
    if not isinstance(mapping, dict):
        raise ValueError("standard_methods.cfd_enabled_by_family must be an object")
    return bool(mapping.get(str(family), True))


def cfd_enabled_for_mode(config: dict[str, Any], mode: str) -> bool:
    return cfd_enabled_for_family(config, family_for_mode(mode))

def _family_timing_config(
    config: dict[str, Any], family: str
) -> tuple[float, float, float]:
    preprocessing = config["preprocessing"]
    common = dict(preprocessing.get("common", {}) or {})
    resolved = dict(common)
    resolved.update(dict(preprocessing.get(family, {}) or {}))

    crop = resolved.get(
        "analysis_crop_ns", {"before": 2.0, "after": 40.0}
    )
    before_ns = float(crop["before"])
    after_ns = float(crop["after"])
    reference_threshold_mV = float(
        resolved["search_trigger_threshold_mV"]
    )
    if before_ns <= 0.0 or after_ns <= 0.0:
        raise ValueError(
            f"preprocessing.{family}.analysis_crop_ns must be positive"
        )
    if not np.isfinite(reference_threshold_mV):
        raise ValueError(
            f"preprocessing.{family}.search_trigger_threshold_mV "
            "must be finite"
        )
    return before_ns, after_ns, reference_threshold_mV


def _analysis_bounds(
    relative_time_ps: np.ndarray,
    *,
    reference_time_ps: float,
    before_ns: float,
    after_ns: float,
) -> tuple[int, int]:
    t = np.asarray(relative_time_ps, dtype=np.float64)
    start = max(
        0,
        int(
            np.searchsorted(
                t,
                float(reference_time_ps) - float(before_ns) * 1000.0,
                side="left",
            )
        )
        - 1,
    )
    stop = min(
        t.size,
        int(
            np.searchsorted(
                t,
                float(reference_time_ps) + float(after_ns) * 1000.0,
                side="right",
            )
        )
        + 1,
    )
    if stop - start < 3:
        raise RuntimeError(
            "analysis_crop_ns contains fewer than three native samples"
        )
    return start, stop


def _pulse_peak_after_reference(
    signal_mV: np.ndarray,
    relative_time_ps: np.ndarray,
    *,
    reference_time_ps: float,
    before_ns: float,
    after_ns: float,
) -> tuple[int, int, int, float]:
    y = np.asarray(signal_mV, dtype=np.float64)
    t = np.asarray(relative_time_ps, dtype=np.float64)
    start, stop = _analysis_bounds(
        t,
        reference_time_ps=reference_time_ps,
        before_ns=before_ns,
        after_ns=after_ns,
    )
    ref_sample = int(
        np.clip(
            np.searchsorted(
                t, float(reference_time_ps), side="right"
            )
            - 1,
            start,
            stop - 2,
        )
    )
    peak_search_start = min(stop - 1, ref_sample + 1)
    local = y[peak_search_start:stop]
    if local.size == 0 or not np.any(np.isfinite(local)):
        raise RuntimeError(
            "Cannot identify pulse peak after preprocessing reference edge"
        )
    peak = peak_search_start + int(np.nanargmax(local))
    return start, stop, ref_sample, float(y[peak])


def _same_edge_level_times_fs(
    signal_mV: np.ndarray,
    relative_time_ps: np.ndarray,
    levels_mV: np.ndarray,
    *,
    reference_threshold_mV: float,
    reference_time_ps: float,
    before_ns: float,
    after_ns: float,
) -> np.ndarray:
    """Cross arbitrary levels on the SAME rising edge as the fixed reference.

    No time-distance constraint is applied:
      * level <= Tref: last rising crossing before the reference crossing;
      * level >  Tref: first rising crossing after the reference crossing and
        before the associated pulse peak;
      * level == Tref: exact preprocessing reference timestamp.

    Hence a very low LED threshold may legitimately occur several ns before the
    reference threshold while remaining on the same physical rise.
    """
    y = np.asarray(signal_mV, dtype=np.float64)
    t = np.asarray(relative_time_ps, dtype=np.float64)
    levels = np.asarray(levels_mV, dtype=np.float64).reshape(-1)
    result = np.full(levels.shape, INVALID_TIME_FS, dtype=np.int64)

    start, stop, ref_sample, _peak_value = _pulse_peak_after_reference(
        y,
        t,
        reference_time_ps=reference_time_ps,
        before_ns=before_ns,
        after_ns=after_ns,
    )

    # The associated pulse peak is the largest post-reference sample inside the
    # exact configured analysis crop.
    peak_search_start = min(stop - 1, ref_sample + 1)
    peak = peak_search_start + int(
        np.nanargmax(y[peak_search_start:stop])
    )

    tolerance_ps = (
        1.5 * abs(float(np.median(np.diff(t))))
        if t.size >= 2
        else 0.0
    )

    for output_index, level in enumerate(levels):
        if not np.isfinite(level):
            continue

        if np.isclose(
            float(level),
            float(reference_threshold_mV),
            rtol=0.0,
            atol=1e-12,
        ):
            result[output_index] = np.int64(
                np.rint(float(reference_time_ps) * 1000.0)
            )
            continue

        if float(level) <= float(reference_threshold_mV):
            # Search the complete stored waveform before Tref.
            # The crossing may be several ns before Tref and is still
            # valid as long as it belongs to the same rising edge.
            seg_start = 0
            seg_stop = min(ref_sample + 1, y.size - 1)
            if seg_stop <= seg_start:
                continue
            indices = np.arange(
                seg_start, seg_stop, dtype=np.int64
            )
            prefer_last = True
        else:
            seg_start = max(start, ref_sample)
            seg_stop = min(peak, y.size - 1)
            if seg_stop <= seg_start:
                continue
            indices = np.arange(
                seg_start, seg_stop, dtype=np.int64
            )
            prefer_last = False

        y0 = y[indices]
        y1 = y[indices + 1]
        finite = np.isfinite(y0) & np.isfinite(y1)
        crossing = finite & (
            ((y0 < level) & (y1 >= level))
            | ((y0 == level) & (y1 > level))
        )
        candidates = indices[np.flatnonzero(crossing)]
        if candidates.size == 0:
            continue

        if prefer_last:
            # Same low-level rise: closest crossing from below to the reference.
            candidate_times = t[candidates]
            eligible = candidates[
                candidate_times
                <= float(reference_time_ps) + tolerance_ps
            ]
            if eligible.size == 0:
                continue
            lower = int(eligible[-1])
        else:
            # Same high-level rise: first crossing after the reference.
            candidate_times = t[candidates]
            eligible = candidates[
                candidate_times
                >= float(reference_time_ps) - tolerance_ps
            ]
            if eligible.size == 0:
                continue
            lower = int(eligible[0])

        y0v = float(y[lower])
        y1v = float(y[lower + 1])
        if y1v == y0v:
            continue
        fraction = (float(level) - y0v) / (y1v - y0v)
        if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            continue

        crossing_ps = float(t[lower]) + fraction * (
            float(t[lower + 1]) - float(t[lower])
        )
        result[output_index] = np.int64(
            np.rint(crossing_ps * 1000.0)
        )

    return result

def _first_crossing_levels_fs(relative_time_ps: np.ndarray, signal_mV: np.ndarray, levels_mV: np.ndarray) -> np.ndarray:
    t = np.asarray(relative_time_ps, dtype=np.float64)
    y = np.asarray(signal_mV, dtype=np.float64)
    levels = np.asarray(levels_mV, dtype=np.float64).reshape(-1)
    out = np.full(levels.shape, INVALID_TIME_FS, dtype=np.int64)
    if t.size < 2 or y.size != t.size:
        return out
    y0, y1 = y[:-1], y[1:]
    finite = np.isfinite(y0) & np.isfinite(y1)
    for j, level in enumerate(levels):
        if not np.isfinite(level) or level <= 0.0:
            continue
        candidates = np.flatnonzero(finite & (((y0 < level) & (y1 >= level)) | ((y0 == level) & (y1 > level))))
        for lower in candidates:
            denom = float(y1[lower] - y0[lower])
            if not np.isfinite(denom) or denom == 0.0:
                continue
            f = (float(level) - float(y0[lower])) / denom
            if np.isfinite(f) and 0.0 <= f <= 1.0:
                ps = float(t[lower]) + f * (float(t[lower + 1]) - float(t[lower]))
                out[j] = np.int64(np.rint(ps * 1000.0))
                break
    return out


def _last_crossing_before_peak_levels_fs(relative_time_ps: np.ndarray, signal_mV: np.ndarray, levels_mV: np.ndarray) -> np.ndarray:
    t = np.asarray(relative_time_ps, dtype=np.float64)
    y = np.asarray(signal_mV, dtype=np.float64)
    levels = np.asarray(levels_mV, dtype=np.float64).reshape(-1)
    out = np.full(levels.shape, INVALID_TIME_FS, dtype=np.int64)
    if t.size < 2 or y.size != t.size or not np.any(np.isfinite(y)):
        return out
    peak = int(np.nanargmax(y))
    if peak <= 0:
        return out
    y0, y1 = y[:peak], y[1:peak + 1]
    finite = np.isfinite(y0) & np.isfinite(y1)
    for j, level in enumerate(levels):
        if not np.isfinite(level) or level <= 0.0:
            continue
        candidates = np.flatnonzero(finite & (((y0 < level) & (y1 >= level)) | ((y0 == level) & (y1 > level))))
        if candidates.size == 0:
            continue
        lower = int(candidates[-1])
        denom = float(y1[lower] - y0[lower])
        if not np.isfinite(denom) or denom == 0.0:
            continue
        f = (float(level) - float(y0[lower])) / denom
        if np.isfinite(f) and 0.0 <= f <= 1.0:
            ps = float(t[lower]) + f * (float(t[lower + 1]) - float(t[lower]))
            out[j] = np.int64(np.rint(ps * 1000.0))
    return out

def _family_arrays(
    dataset: PreparedDataset, family: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if family == "energy":
        waves = dataset.windows_mV
        times = dataset.relative_time_ps
        anchors = dataset.energy_window_anchor_time_fs
        references = dataset.energy_led_time_fs
    elif family == "timing":
        if (
            dataset.timing_windows_mV is None
            or dataset.timing_relative_time_ps is None
        ):
            raise ValueError(
                "Adaptive timing standard methods require timing waveforms"
            )
        waves = dataset.timing_windows_mV
        times = dataset.timing_relative_time_ps
        anchors = dataset.timing_window_anchor_time_fs
        references = dataset.timing_led_time_fs
    else:
        raise ValueError(
            f"Unknown standard-method family {family!r}"
        )

    if anchors is None or references is None:
        raise ValueError(
            f"{family} preprocessing reference timestamps/anchors are "
            "unavailable; rebuild preprocessing"
        )

    return (
        np.asarray(waves),
        np.asarray(times, dtype=np.float64),
        np.asarray(anchors, dtype=np.int64),
        np.asarray(references, dtype=np.int64),
    )

def _edge_slice(
    relative_time_ps: np.ndarray, before_ns: float, after_ns: float
) -> tuple[int, int, int]:
    t = np.asarray(relative_time_ps, dtype=np.float64)
    zero = int(np.argmin(np.abs(t)))
    start = max(
        0,
        int(np.searchsorted(t, -float(before_ns) * 1000.0, side="left")),
    )
    stop = min(
        t.size,
        int(np.searchsorted(t, float(after_ns) * 1000.0, side="right")),
    )
    if stop - start < 4 or not start <= zero < stop:
        raise ValueError(
            "Prepared waveform does not contain the configured standard-method edge-search crop"
        )
    return start, stop, zero


def _led_support(
    waves: np.ndarray,
    relative_time_ps: np.ndarray,
    anchors_fs: np.ndarray,
    reference_times_fs: np.ndarray,
    indices: np.ndarray,
    *,
    before_ns: float,
    after_ns: float,
    chunk_size: int,
    configured_range: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Full stored-waveform LED range; independent of preprocessing Tref/crop."""
    del relative_time_ps, anchors_fs, reference_times_fs, before_ns, after_ns
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    common_peak = np.inf
    step = max(1, int(chunk_size))
    for first in range(0, idx.size, step):
        block = np.asarray(waves[idx[first:first + step]], dtype=np.float64)
        peaks = np.nanmax(block, axis=2)
        if np.any(~np.isfinite(peaks)):
            raise RuntimeError("Non-finite pulse maximum in development cohort")
        common_peak = min(common_peak, float(np.min(peaks)))
    low = 1.0 if configured_range is None else max(0.0, float(configured_range[0]))
    high = common_peak if configured_range is None else min(common_peak, float(configured_range[1]))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise RuntimeError(f"No common LED scan range: {low:.6g}..{high:.6g} mV")
    return max(1e-6, low), high - max(1e-6, 1e-8 * max(1.0, abs(high)))

def _extract_grids(
    waves: np.ndarray,
    relative_time_ps: np.ndarray,
    anchors_fs: np.ndarray,
    reference_times_fs: np.ndarray,
    *,
    reference_threshold_mV: float,
    led_thresholds_mV: np.ndarray,
    cfd_fractions: np.ndarray,
    before_ns: float,
    after_ns: float,
    chunk_size: int,
    indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Scan LED/CFD over the complete stored native waveform."""
    del reference_times_fs, reference_threshold_mV, before_ns, after_ns
    thresholds = np.asarray(led_thresholds_mV, dtype=np.float64).reshape(-1)
    fractions = np.asarray(cfd_fractions, dtype=np.float64).reshape(-1)
    n = int(waves.shape[0])
    led = np.full((n, 2, thresholds.size), INVALID_TIME_FS, dtype=np.int64)
    cfd = np.full((n, 2, fractions.size), INVALID_TIME_FS, dtype=np.int64)
    selected = np.arange(n, dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64).reshape(-1)
    t = np.asarray(relative_time_ps, dtype=np.float64)
    step = max(1, int(chunk_size))
    for first in range(0, selected.size, step):
        block_idx = selected[first:first + step]
        block = np.asarray(waves[block_idx], dtype=np.float64)
        for local_row, event_value in enumerate(block_idx):
            event = int(event_value)
            for detector in range(2):
                signal = block[local_row, detector]
                anchor_fs = np.int64(anchors_fs[event, detector])
                if thresholds.size:
                    local_led = _first_crossing_levels_fs(t, signal, thresholds)
                    good = local_led != INVALID_TIME_FS
                    led[event, detector, good] = anchor_fs + local_led[good]
                if fractions.size and np.any(np.isfinite(signal)):
                    levels = float(np.nanmax(signal)) * fractions
                    local_cfd = _last_crossing_before_peak_levels_fs(t, signal, levels)
                    good = local_cfd != INVALID_TIME_FS
                    cfd[event, detector, good] = anchor_fs + local_cfd[good]
    return led, cfd

def _candidate_score(
    grid_fs: np.ndarray,
    candidate_index: int,
    development: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    true_tof_ps: float,
) -> tuple[float, tuple[float, ...]]:
    development = np.asarray(development, dtype=np.int64)
    pair = np.asarray(grid_fs[:, :, candidate_index], dtype=np.int64)
    if np.any(pair[development] == INVALID_TIME_FS):
        return float("inf"), tuple()

    fold_values: list[float] = []
    for _train, score in splits:
        idx = np.asarray(score, dtype=np.int64)
        if idx.size < 2 or np.any(pair[idx] == INVALID_TIME_FS):
            return float("inf"), tuple()
        delta = (
            pair[idx, 0].astype(np.float64)
            - pair[idx, 1].astype(np.float64)
        ) / 1000.0
        metrics = residual_metrics(delta - float(true_tof_ps))
        fold_values.append(float(metrics["ctr_ps"]))

    if not fold_values or not np.all(np.isfinite(fold_values)):
        return float("inf"), tuple(fold_values)
    return float(np.mean(fold_values)), tuple(fold_values)


def _best_candidate(
    grid_fs: np.ndarray,
    parameters: np.ndarray,
    development: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    true_tof_ps: float,
) -> tuple[int, float, tuple[float, ...]]:
    finite: list[tuple[float, int, tuple[float, ...]]] = []
    for index in range(int(parameters.size)):
        score, folds = _candidate_score(
            grid_fs, index, development, splits, true_tof_ps
        )
        if np.isfinite(score):
            finite.append((score, index, folds))
    if not finite:
        raise RuntimeError(
            "No standard-method candidate has complete crossing coverage on the "
            "development selection population"
        )
    score, index, folds = min(finite, key=lambda item: (item[0], item[1]))
    return int(index), float(score), tuple(folds)


def _refined_axis(values: np.ndarray, best_index: int, points: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 1 or int(points) <= 1:
        return arr[[best_index]]
    lo = arr[max(0, best_index - 1)]
    hi = arr[min(arr.size - 1, best_index + 1)]
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return arr[[best_index]]
    return np.linspace(float(lo), float(hi), int(points), dtype=np.float64)


def optimize_family(
    config: dict[str, Any],
    dataset: PreparedDataset,
    development: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    family: str,
    logger: Any,
) -> FamilySelection:
    standard = config.get("standard_methods", {}) or {}
    cfd_enabled = cfd_enabled_for_family(config, family)

    (
        waves,
        times,
        anchors,
        reference_times,
    ) = _family_arrays(dataset, family)
    development = np.asarray(
        development, dtype=np.int64
    )

    before_ns, after_ns, reference_threshold_mV = (
        _family_timing_config(config, family)
    )
    chunk_size = int(
        standard.get("waveform_scan_chunk_size", 1024)
    )

    configured_range = None
    ranges = standard.get("led_range_mV_by_family")
    if isinstance(ranges, dict) and family in ranges:
        values = np.asarray(
            ranges[family], dtype=np.float64
        ).reshape(-1)
        if values.size != 2 or not (
            np.isfinite(values[0])
            and np.isfinite(values[1])
            and values[1] > values[0]
        ):
            raise ValueError(
                f"standard_methods.led_range_mV_by_family.{family} "
                "must be [low, high]"
            )
        configured_range = (
            float(values[0]),
            float(values[1]),
        )

    low, high = _led_support(
        waves,
        times,
        anchors,
        reference_times,
        development,
        before_ns=before_ns,
        after_ns=after_ns,
        chunk_size=chunk_size,
        configured_range=configured_range,
    )

    led_points = max(
        3, int(standard.get("led_grid_points", 121))
    )
    led_axis = np.linspace(
        low, high, led_points, dtype=np.float64
    )

    if cfd_enabled:
        cfd_points = max(
            3, int(standard.get("cfd_grid_points", 81))
        )
        cfd_axis = np.linspace(
            float(standard.get("cfd_min_fraction", 0.02)),
            float(standard.get("cfd_max_fraction", 0.80)),
            cfd_points,
            dtype=np.float64,
        )
        if not (
            0.0 < float(cfd_axis[0])
            < float(cfd_axis[-1])
            <= 1.0
        ):
            raise ValueError(
                "standard_methods CFD range must satisfy "
                "0 < min < max <= 1"
            )
    else:
        cfd_axis = np.empty(0, dtype=np.float64)

    logger.info(
        "Adaptive standards coarse scan | %s | Tref=%.6g mV | "
        "analysis crop=-%.3f..+%.3f ns | LED %.3f..%.3f mV (%d) | CFD %s",
        family,
        reference_threshold_mV,
        before_ns,
        after_ns,
        low,
        high,
        led_axis.size,
        (
            f"{cfd_axis[0]:.4f}..{cfd_axis[-1]:.4f} "
            f"({cfd_axis.size})"
            if cfd_enabled
            else "disabled"
        ),
    )

    led_grid, cfd_grid = _extract_grids(
        waves,
        times,
        anchors,
        reference_times,
        reference_threshold_mV=reference_threshold_mV,
        led_thresholds_mV=led_axis,
        cfd_fractions=cfd_axis,
        before_ns=before_ns,
        after_ns=after_ns,
        chunk_size=chunk_size,
        indices=development,
    )
    led_index, _led_score, _led_folds = (
        _best_candidate(
            led_grid,
            led_axis,
            development,
            splits,
            dataset.true_tof_ps,
        )
    )

    if cfd_enabled:
        cfd_index, _cfd_score, _cfd_folds = (
            _best_candidate(
                cfd_grid,
                cfd_axis,
                development,
                splits,
                dataset.true_tof_ps,
            )
        )
    else:
        cfd_index = -1

    led_fine = _refined_axis(
        led_axis,
        led_index,
        max(3, int(standard.get("led_refine_points", 41))),
    )
    cfd_fine = (
        _refined_axis(
            cfd_axis,
            cfd_index,
            max(
                3,
                int(
                    standard.get(
                        "cfd_refine_points", 41
                    )
                ),
            ),
        )
        if cfd_enabled
        else np.empty(0, dtype=np.float64)
    )

    logger.info(
        "Adaptive standards refine | %s | LED %.3f..%.3f mV (%d) | CFD %s",
        family,
        led_fine[0],
        led_fine[-1],
        led_fine.size,
        (
            f"{cfd_fine[0]:.5f}..{cfd_fine[-1]:.5f} "
            f"({cfd_fine.size})"
            if cfd_enabled
            else "disabled"
        ),
    )

    led_grid_fine, cfd_grid_fine = _extract_grids(
        waves,
        times,
        anchors,
        reference_times,
        reference_threshold_mV=reference_threshold_mV,
        led_thresholds_mV=led_fine,
        cfd_fractions=cfd_fine,
        before_ns=before_ns,
        after_ns=after_ns,
        chunk_size=chunk_size,
        indices=development,
    )

    led_index, led_score, led_folds = _best_candidate(
        led_grid_fine,
        led_fine,
        development,
        splits,
        dataset.true_tof_ps,
    )

    if cfd_enabled:
        cfd_index, cfd_score, cfd_folds = (
            _best_candidate(
                cfd_grid_fine,
                cfd_fine,
                development,
                splits,
                dataset.true_tof_ps,
            )
        )
        selected_cfd_fraction = float(
            cfd_fine[cfd_index]
        )
    else:
        cfd_score = float("nan")
        cfd_folds = tuple()
        selected_cfd_fraction = float("nan")

    # ------------------------------------------------------------
    # Final selected LED timestamps: always extract for all events
    # ------------------------------------------------------------

    selected_led_grid, _ = _extract_grids(
        waves,
        times,
        anchors,
        reference_times,
        reference_threshold_mV=reference_threshold_mV,
        led_thresholds_mV=np.asarray(
            [led_fine[led_index]],
            dtype=np.float64,
        ),
        cfd_fractions=np.empty(0, dtype=np.float64),
        before_ns=before_ns,
        after_ns=after_ns,
        chunk_size=chunk_size,
        indices=None,
    )

    selected_led = np.asarray(
        selected_led_grid[:, :, 0],
        dtype=np.int64,
    )

    if np.any(selected_led == INVALID_TIME_FS):
        missing_led = int(
            np.count_nonzero(
                np.any(
                    selected_led == INVALID_TIME_FS,
                    axis=1,
                )
            )
        )
        raise RuntimeError(
            f"Selected {family} LED threshold is missing a crossing "
            f"for {missing_led} prepared events."
        )


    # ------------------------------------------------------------
    # Final selected CFD timestamps
    # ------------------------------------------------------------

    if cfd_enabled:
        _, selected_cfd_grid = _extract_grids(
            waves,
            times,
            anchors,
            reference_times,
            reference_threshold_mV=reference_threshold_mV,
            led_thresholds_mV=np.empty(
                0,
                dtype=np.float64,
            ),
            cfd_fractions=np.asarray(
                [selected_cfd_fraction],
                dtype=np.float64,
            ),
            before_ns=before_ns,
            after_ns=after_ns,
            chunk_size=chunk_size,
            indices=None,
        )

        selected_cfd = np.asarray(
            selected_cfd_grid[:, :, 0],
            dtype=np.int64,
        )

        # If CFD fails for either detector, use LED for BOTH
        # detectors for that event.
        cfd_fallback_events = np.any(
            selected_cfd == INVALID_TIME_FS,
            axis=1,
        )

        n_fallback = int(
            np.count_nonzero(cfd_fallback_events)
        )

        if n_fallback:
            selected_cfd[cfd_fallback_events, :] = (
                selected_led[cfd_fallback_events, :]
            )

            logger.warning(
                "Adaptive standards | %s | CFD fallback to LED "
                "for %d/%d events (%.3f%%)",
                family,
                n_fallback,
                selected_cfd.shape[0],
                100.0
                * n_fallback
                / max(1, selected_cfd.shape[0]),
            )

    else:
        # CFD disabled for this family.
        # Keep a valid placeholder internally, but no CFD result
        # should be reported/evaluated for this family.
        selected_cfd = selected_led.copy()

    selection = FamilySelection(
        family=family,
        cfd_enabled=bool(cfd_enabled),
        led_threshold_mV=float(led_fine[led_index]),
        cfd_fraction=selected_cfd_fraction,
        led_validation_sctr_ps=float(led_score),
        cfd_validation_sctr_ps=float(cfd_score),
        led_fold_sctr_ps=tuple(
            float(v) for v in led_folds
        ),
        cfd_fold_sctr_ps=tuple(
            float(v) for v in cfd_folds
        ),
        led_times_fs=selected_led,
        cfd_times_fs=selected_cfd,
        led_search_low_mV=float(low),
        led_search_high_mV=float(high),
        led_coarse_points=int(led_axis.size),
        led_refine_points=int(led_fine.size),
        cfd_coarse_points=int(cfd_axis.size),
        cfd_refine_points=int(cfd_fine.size),
    )

    if cfd_enabled:
        logger.info(
            "Adaptive standards selected | %s | Tref %.6g mV | "
            "LED %.6g mV -> %.3f ps s-CTR | "
            "CFD %.6g -> %.3f ps s-CTR",
            family,
            reference_threshold_mV,
            selection.led_threshold_mV,
            selection.led_validation_sctr_ps,
            selection.cfd_fraction,
            selection.cfd_validation_sctr_ps,
        )
    else:
        logger.info(
            "Adaptive standards selected | %s | Tref %.6g mV | "
            "LED %.6g mV -> %.3f ps s-CTR | CFD disabled",
            family,
            reference_threshold_mV,
            selection.led_threshold_mV,
            selection.led_validation_sctr_ps,
        )

    return selection

def apply_selections(
    config: dict[str, Any],
    dataset: PreparedDataset,
    selections: dict[str, FamilySelection],
) -> PreparedDataset:
    manifest = dict(dataset.manifest)
    manifest["adaptive_standard_methods"] = {
        family: selection.as_dict()
        for family, selection in selections.items()
    }
    manifest["adaptive_reference_edge_protocol"] = {
        family: {
            "analysis_crop_before_ns": _family_timing_config(config, family)[0],
            "analysis_crop_after_ns": _family_timing_config(config, family)[1],
            "reference_threshold_mV": _family_timing_config(config, family)[2],
            "led_edge_rule": "same rising edge; lower levels before reference, higher levels after reference",
        }
        for family in selections
    }
    manifest["ml_window_alignment_source"] = "adaptive_selected_led_native_anchor"
    manifest["window_anchor_shift_factored"] = True

    kwargs: dict[str, Any] = {"manifest": manifest}
    energy = selections.get("energy")
    timing = selections.get("timing")

    if energy is not None:
        if dataset.energy_window_anchor_time_fs is None:
            raise ValueError("Energy preprocessing anchors are required for LED re-alignment")
        energy_shifts, energy_anchors = _alignment_shifts(
            energy.led_times_fs,
            dataset.energy_window_anchor_time_fs,
            dataset.relative_time_ps,
        )
        _check_shift_support(
            energy_shifts,
            dataset.relative_time_ps,
            config["windows_ns"],
            label="energy",
        )
        kwargs["windows_mV"] = ShiftedWaveformArray(
            dataset.windows_mV, energy_shifts
        )
        if dataset.denoised_windows_mV is not None:
            kwargs["denoised_windows_mV"] = ShiftedWaveformArray(
                dataset.denoised_windows_mV, energy_shifts
            )
        kwargs["energy_window_anchor_time_fs"] = energy_anchors
        kwargs["energy_led_time_fs"] = energy.led_times_fs
        kwargs["energy_cfd_time_fs"] = energy.cfd_times_fs
        kwargs["led_time_fs"] = energy.led_times_fs
        kwargs["cfd_time_fs"] = energy.cfd_times_fs
        kwargs["window_anchor_time_fs"] = energy_anchors

    if timing is not None:
        if dataset.timing_window_anchor_time_fs is None:
            raise ValueError("Timing preprocessing anchors are required for LED re-alignment")
        timing_shifts, timing_anchors = _alignment_shifts(
            timing.led_times_fs,
            dataset.timing_window_anchor_time_fs,
            dataset.timing_relative_time_ps,
        )
        _check_shift_support(
            timing_shifts,
            dataset.timing_relative_time_ps,
            config["windows_ns"],
            label="timing",
        )
        if dataset.timing_windows_mV is not None:
            kwargs["timing_windows_mV"] = ShiftedWaveformArray(
                dataset.timing_windows_mV, timing_shifts
            )
        if dataset.denoised_timing_windows_mV is not None:
            kwargs["denoised_timing_windows_mV"] = ShiftedWaveformArray(
                dataset.denoised_timing_windows_mV, timing_shifts
            )
        kwargs["timing_window_anchor_time_fs"] = timing_anchors
        kwargs["timing_led_time_fs"] = timing.led_times_fs
        kwargs["timing_cfd_time_fs"] = timing.cfd_times_fs

        if dataset.timing_aligned_energy_windows_mV is not None:
            if dataset.timing_aligned_energy_window_anchor_time_fs is None:
                raise ValueError(
                    "Timing-aligned energy preprocessing anchors are required"
                )
            aligned_shifts, aligned_anchors = _alignment_shifts(
                timing.led_times_fs,
                dataset.timing_aligned_energy_window_anchor_time_fs,
                dataset.relative_time_ps,
            )
            _check_shift_support(
                aligned_shifts,
                dataset.relative_time_ps,
                config["windows_ns"],
                label="timing-aligned energy",
            )
            kwargs["timing_aligned_energy_windows_mV"] = ShiftedWaveformArray(
                dataset.timing_aligned_energy_windows_mV,
                aligned_shifts,
            )
            if dataset.denoised_timing_aligned_energy_windows_mV is not None:
                kwargs["denoised_timing_aligned_energy_windows_mV"] = ShiftedWaveformArray(
                    dataset.denoised_timing_aligned_energy_windows_mV,
                    aligned_shifts,
                )
            kwargs["timing_aligned_energy_window_anchor_time_fs"] = aligned_anchors

    return replace(dataset, **kwargs)


def optimize_standard_methods(
    config: dict[str, Any],
    dataset: PreparedDataset,
    development: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    families: Iterable[str],
    logger: Any,
) -> tuple[PreparedDataset, dict[str, FamilySelection]]:
    if not bool((config.get("standard_methods", {}) or {}).get("enabled", True)):
        return dataset, {}
    selections: dict[str, FamilySelection] = {}
    for family in sorted(set(str(v) for v in families)):
        selections[family] = optimize_family(
            config,
            dataset,
            development,
            splits,
            family=family,
            logger=logger,
        )
    return apply_selections(config, dataset, selections), selections


def parameter_payload(
    selections: dict[str, FamilySelection],
    mode: str,
    model: str,
) -> dict[str, Any]:
    family = family_for_mode(mode)
    selection = selections.get(family)
    if selection is None:
        return {}
    if str(model) == "led":
        return {
            "family": family,
            "led_threshold_mV": float(selection.led_threshold_mV),
            "selection_metric": "sctr",
            "validation_sctr_ps": float(selection.led_validation_sctr_ps),
        }
    if str(model) == "cfd":
        if not selection.cfd_enabled:
            return {}
        return {
            "family": family,
            "cfd_fraction": float(selection.cfd_fraction),
            "selection_metric": "sctr",
            "validation_sctr_ps": float(selection.cfd_validation_sctr_ps),
        }
    return {}

