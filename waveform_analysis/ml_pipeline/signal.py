from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

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
    window_anchor_time_fs: np.int64
    reference_aligned_window_mV: np.ndarray | None
    reference_aligned_window_anchor_time_fs: np.int64
    valid: bool


def relative_window_grid_ps(
    waveform_config: dict[str, Any],
    native_interval_s: float,
) -> np.ndarray:
    """Return sample offsets for a canonical window on the native time grid.

    The old pipeline generated a dense cubic-spline grid using
    ``upsample_step_ps`` and then optionally subsampled it. Those options are now
    deprecated: every saved point is an acquired sample and adjacent points are
    separated by the hardware sampling interval.
    """

    interval_s = float(native_interval_s)
    if not np.isfinite(interval_s) or interval_s <= 0.0:
        raise ValueError("native_interval_s must be finite and positive")
    interval_ps = interval_s * 1.0e12
    before_ps = float(waveform_config["ml_window_ns"]["before"]) * 1000.0
    after_ps = float(waveform_config["ml_window_ns"]["after"]) * 1000.0
    before_samples = int(np.floor(before_ps / interval_ps + 1e-9))
    after_samples = int(np.floor(after_ps / interval_ps + 1e-9))
    offsets = np.arange(-before_samples, after_samples + 1, dtype=np.int64)
    if offsets.size < 4:
        raise ValueError("ML window contains fewer than four native samples")
    return offsets.astype(np.float64) * interval_ps


def timing_channel_waveform_config(waveform_config: dict[str, Any]) -> dict[str, Any]:
    """Resolve timing-channel LED settings, inheriting energy defaults.

    Only timing extraction options are copied. ML-window settings intentionally
    remain energy-channel-only. ``upsample_step_ps`` is deliberately excluded:
    LED extraction uses local interpolation on the native samples.
    """

    override = waveform_config.get("timing_channel_led", {})
    if not isinstance(override, dict):
        raise ValueError("waveform.timing_channel_led must be an object")
    keys = (
        "baseline_samples",
        "search_trigger_threshold_mV",
        "analysis_crop_ns",
        "led_threshold_mV",
        "cfd_fraction",
        "ml_window_ns",
        "denoising",
    )
    resolved: dict[str, Any] = {}
    for key in keys:
        source = override[key] if key in override else waveform_config[key]
        resolved[key] = deepcopy(source)
    return resolved


def _first_rising_crossing_ns(
    time_ns: np.ndarray,
    signal_mV: np.ndarray,
    threshold_mV: float,
) -> float:
    """Interpolate the first rising threshold crossing from two native samples."""

    x = np.asarray(time_ns, dtype=np.float64)
    y = np.asarray(signal_mV, dtype=np.float64)
    threshold = float(threshold_mV)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 2:
        return np.nan
    if not np.isfinite(threshold) or threshold <= 0.0:
        return np.nan
    finite = np.isfinite(x[:-1]) & np.isfinite(x[1:]) & np.isfinite(y[:-1]) & np.isfinite(y[1:])
    crossing = finite & (y[:-1] < threshold) & (y[1:] >= threshold)
    indices = np.flatnonzero(crossing)
    if indices.size == 0:
        # Preserve exact hits on the lower sample of a rising segment.
        crossing = finite & (y[:-1] == threshold) & (y[1:] > threshold)
        indices = np.flatnonzero(crossing)
    if indices.size == 0:
        return np.nan
    lower = int(indices[0])
    x0, x1 = float(x[lower]), float(x[lower + 1])
    y0, y1 = float(y[lower]), float(y[lower + 1])
    if x1 <= x0 or y1 == y0:
        return np.nan
    fraction = (threshold - y0) / (y1 - y0)
    if not 0.0 <= fraction <= 1.0:
        return np.nan
    return x0 + fraction * (x1 - x0)


def _last_rising_crossing_before_peak_ns(
    time_ns: np.ndarray,
    signal_mV: np.ndarray,
    threshold_mV: float,
) -> float:
    """Interpolate the physical rising-edge crossing immediately before the peak.

    A CFD threshold can be low enough that baseline noise or a pre-pulse crosses it
    inside the analysis crop. Selecting the first crossing then gives a timestamp
    unrelated to the main pulse. Restricting the search to samples up to the crop
    maximum and taking the last rising crossing identifies the crossing connected
    to the pulse that defines the CFD amplitude.
    """

    x = np.asarray(time_ns, dtype=np.float64)
    y = np.asarray(signal_mV, dtype=np.float64)
    threshold = float(threshold_mV)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 2:
        return np.nan
    if not np.isfinite(threshold) or threshold <= 0.0:
        return np.nan

    peak_index = int(np.argmax(y))
    if peak_index <= 0 or not np.isfinite(y[peak_index]):
        return np.nan

    finite = (
        np.isfinite(x[:peak_index])
        & np.isfinite(x[1 : peak_index + 1])
        & np.isfinite(y[:peak_index])
        & np.isfinite(y[1 : peak_index + 1])
    )
    crossing = finite & (y[:peak_index] < threshold) & (y[1 : peak_index + 1] >= threshold)
    indices = np.flatnonzero(crossing)
    if indices.size == 0:
        crossing = finite & (y[:peak_index] == threshold) & (y[1 : peak_index + 1] > threshold)
        indices = np.flatnonzero(crossing)
    if indices.size == 0:
        return np.nan

    lower = int(indices[-1])
    x0, x1 = float(x[lower]), float(x[lower + 1])
    y0, y1 = float(y[lower]), float(y[lower + 1])
    if x1 <= x0 or y1 == y0:
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
    if stop_index - start_index < 2:
        return invalid
    crop_time = time_ns[start_index:stop_index]
    crop_signal = signal[start_index:stop_index]
    if np.any(~np.isfinite(crop_time)) or np.any(~np.isfinite(crop_signal)):
        return invalid

    amplitude_mV = float(np.max(crop_signal))
    if not np.isfinite(amplitude_mV) or amplitude_mV <= 0.0:
        return invalid
    led_ns = (
        _first_rising_crossing_ns(
            crop_time,
            crop_signal,
            float(extraction_config["led_threshold_mV"]),
        )
        if compute_led
        else np.nan
    )
    cfd_ns = (
        _last_rising_crossing_before_peak_ns(
            crop_time,
            crop_signal,
            amplitude_mV * float(extraction_config["cfd_fraction"]),
        )
        if compute_cfd
        else np.nan
    )
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
    """Extract an interpolated LED timestamp from a timing channel.

    This compatibility helper returns timing metadata only. Canonical
    preprocessing uses :func:`extract_timing_channel` when timing waveforms are
    configured as an available ML input.
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


def _native_window(
    signal_mV: np.ndarray,
    *,
    horizontal_interval_s: float,
    horizontal_offset_s: float,
    alignment_ns: float,
    relative_grid_ps: np.ndarray,
    return_anchor: bool = False,
) -> np.ndarray | None | tuple[np.ndarray | None, np.int64]:
    interval_s = float(horizontal_interval_s)
    interval_ps = interval_s * 1.0e12
    relative = np.asarray(relative_grid_ps, dtype=np.float64)
    sample_offsets = np.rint(relative / interval_ps).astype(np.int64)
    if not np.allclose(
        relative,
        sample_offsets.astype(np.float64) * interval_ps,
        rtol=0.0,
        atol=max(1e-6, abs(interval_ps) * 1e-9),
    ):
        raise ValueError("Requested ML window is not on the waveform's native sample grid")

    first_time_ns = float(horizontal_offset_s) * 1.0e9
    interval_ns = interval_s * 1.0e9
    anchor = int(np.rint((float(alignment_ns) - first_time_ns) / interval_ns))
    indices = anchor + sample_offsets
    anchor_time_s = float(horizontal_offset_s) + anchor * interval_s
    anchor_time_fs = np.int64(np.rint(anchor_time_s * 1.0e15))
    if indices.size == 0 or int(indices[0]) < 0 or int(indices[-1]) >= signal_mV.size:
        return (None, anchor_time_fs) if return_anchor else None
    window = np.asarray(signal_mV[indices], dtype=np.float32).copy()
    return (window, anchor_time_fs) if return_anchor else window


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
    compute_cfd: bool = True,
) -> ChannelExtraction:
    """Extract one native-grid waveform window and its timing labels.

    LED and CFD timestamps use linear interpolation only between the two native
    samples bracketing the crossing. The saved ML window itself is never
    interpolated: it is a direct slice of acquired samples, aligned to the native
    sample nearest the channel's own LED timestamp.  When ``timing_reference``
    is supplied, a second window aligned to that external LED is returned in
    ``reference_aligned_window_mV``.  Keeping both alignments is essential for
    experiments whose target can be either energy LED or timing LED.
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
        window_anchor_time_fs=np.int64(INVALID_TIME_FS),
        reference_aligned_window_mV=None,
        reference_aligned_window_anchor_time_fs=np.int64(INVALID_TIME_FS),
        valid=False,
    )
    if basic.trigger_index < 0:
        return invalid

    channel_timing = _timing_from_basic(
        basic,
        horizontal_interval_s=horizontal_interval_s,
        horizontal_offset_s=horizontal_offset_s,
        extraction_config=waveform_config,
        compute_led=True,
        compute_cfd=bool(compute_cfd),
    )
    if not channel_timing.valid:
        return invalid
    alignment_ns = float(channel_timing.led_time_fs) / FEMTOSECONDS_PER_NANOSECOND
    window, window_anchor_time_fs = _native_window(
        np.asarray(basic.corrected_signal_mV, dtype=np.float64),
        horizontal_interval_s=horizontal_interval_s,
        horizontal_offset_s=horizontal_offset_s,
        alignment_ns=alignment_ns,
        relative_grid_ps=relative_grid_ps,
        return_anchor=True,
    )
    if window is None or np.any(~np.isfinite(window)):
        return invalid

    reference_window: np.ndarray | None = None
    reference_anchor_time_fs = np.int64(INVALID_TIME_FS)
    if timing_reference is not None:
        if timing_reference.valid:
            reference_alignment_ns = (
                float(timing_reference.led_time_fs) / FEMTOSECONDS_PER_NANOSECOND
            )
            reference_window, reference_anchor_time_fs = _native_window(
                np.asarray(basic.corrected_signal_mV, dtype=np.float64),
                horizontal_interval_s=horizontal_interval_s,
                horizontal_offset_s=horizontal_offset_s,
                alignment_ns=reference_alignment_ns,
                relative_grid_ps=relative_grid_ps,
                return_anchor=True,
            )
            if reference_window is not None and np.any(~np.isfinite(reference_window)):
                reference_window = None
                reference_anchor_time_fs = np.int64(INVALID_TIME_FS)
    return ChannelExtraction(
        amplitude_mV=basic.amplitude_mV,
        noise_rms_mV=basic.noise_rms_mV,
        trigger_index=basic.trigger_index,
        led_time_fs=channel_timing.led_time_fs,
        cfd_time_fs=channel_timing.cfd_time_fs,
        window_mV=window,
        window_anchor_time_fs=window_anchor_time_fs,
        reference_aligned_window_mV=reference_window,
        reference_aligned_window_anchor_time_fs=reference_anchor_time_fs,
        valid=True,
    )


def extract_timing_channel(
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
    """Extract a timing-channel waveform plus precomputed LED and CFD timestamps.

    Both standard-method timestamps are materialized during preprocessing.  The
    evaluator later reads them from the prepared dataset and does not recompute
    either crossing from the saved ML window.
    """

    resolved = timing_channel_waveform_config(waveform_config)
    return extract_channel(
        raw_samples,
        vertical_gain_v_per_count=vertical_gain_v_per_count,
        vertical_offset_v=vertical_offset_v,
        horizontal_interval_s=horizontal_interval_s,
        horizontal_offset_s=horizontal_offset_s,
        polarity=polarity,
        waveform_config=resolved,
        relative_grid_ps=relative_grid_ps,
        timing_reference=None,
        compute_cfd=True,
    )
