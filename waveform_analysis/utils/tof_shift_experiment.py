from __future__ import annotations

import csv
import logging
import math
import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.interpolate import CubicSpline

LOGGER = logging.getLogger(__name__)
_WORKER_SETTINGS: dict[str, Any] | None = None


@dataclass(frozen=True)
class ShiftExperimentPayload:
    event_index: int
    event_id: int
    source_file_id: tuple[int, ...]
    raw_samples: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    vertical_gain_v_per_count: np.ndarray
    vertical_offset_v: np.ndarray
    horizontal_interval_s: np.ndarray
    horizontal_offset_s: np.ndarray
    discrete_shift_ps: int
    discrete_group: int
    continuous_shift_ps: int


@dataclass(frozen=True)
class ShiftExperimentResult:
    discrete_row: dict[str, Any] | None
    continuous_row: dict[str, Any] | None
    rejection_reason: str | None


def matched_integer_uniform_half_width(discrete_half_width_ps: int) -> dict[str, float | int]:
    """Return the integer-uniform half-width with variance closest to {-a, 0, +a}.

    For equally likely discrete shifts {-a, 0, +a},
        Var = 2 a^2 / 3.

    For an integer-uniform shift on {-b, ..., +b},
        Var = b(b+1) / 3.

    The exact positive solution is generally non-integer, so the closest of
    floor(b*) and ceil(b*) is selected.
    """
    a = int(discrete_half_width_ps)
    if a <= 0:
        raise ValueError("discrete_half_width_ps must be positive")

    target_variance = 2.0 * float(a * a) / 3.0
    exact_half_width = 0.5 * (math.sqrt(1.0 + 8.0 * float(a * a)) - 1.0)
    candidates = sorted({max(1, math.floor(exact_half_width)), math.ceil(exact_half_width)})
    b = min(
        candidates,
        key=lambda value: abs(float(value * (value + 1)) / 3.0 - target_variance),
    )
    uniform_variance = float(b * (b + 1)) / 3.0
    return {
        "discrete_half_width_ps": a,
        "discrete_variance_ps2": target_variance,
        "exact_uniform_half_width_ps": exact_half_width,
        "continuous_half_width_ps": int(b),
        "continuous_variance_ps2": uniform_variance,
        "relative_variance_error": (uniform_variance - target_variance) / target_variance,
    }


def assign_artificial_shifts(
    selected: np.ndarray,
    *,
    discrete_shifts_ps: list[int],
    continuous_max_abs_ps: int | None,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create balanced three-point shifts and integer-uniform continuous shifts.

    If ``continuous_max_abs_ps`` is ``None``, its half-width is chosen so that
    the theoretical variance of U{-b, ..., +b} is as close as possible to the
    variance of an equally likely {-a, 0, +a} distribution.

    Arrays have the same length as the ROOT event stream. Values outside the
    selected mask are zero and are never processed.
    """
    selected_mask = np.asarray(selected, dtype=bool).reshape(-1)
    selected_indices = np.flatnonzero(selected_mask)
    if selected_indices.size == 0:
        raise ValueError("cannot assign shifts because no events are selected")

    shifts_array = np.asarray(discrete_shifts_ps, dtype=np.int32)
    if shifts_array.shape != (3,):
        raise ValueError("discrete_shifts_ps must contain exactly three values")
    if len(set(int(item) for item in shifts_array)) != 3:
        raise ValueError("discrete shifts must be unique")
    shifts_array = np.sort(shifts_array)
    if not (
        int(shifts_array[1]) == 0
        and int(shifts_array[0]) == -int(shifts_array[2])
        and int(shifts_array[2]) > 0
    ):
        raise ValueError("discrete shifts must have the form [-a, 0, +a]")

    if continuous_max_abs_ps is None:
        variance_match = matched_integer_uniform_half_width(int(shifts_array[2]))
        continuous_half_width = int(variance_match["continuous_half_width_ps"])
    else:
        continuous_half_width = int(continuous_max_abs_ps)
        if continuous_half_width < 1:
            raise ValueError("continuous_max_abs_ps must be positive")

    rng = np.random.default_rng(int(random_seed))

    # Balanced assignment: group counts differ by at most one.
    order = rng.permutation(selected_indices.size)
    balanced_groups = np.arange(selected_indices.size, dtype=np.int32) % 3
    group_for_selected = np.empty(selected_indices.size, dtype=np.int32)
    group_for_selected[order] = balanced_groups

    discrete = np.zeros(selected_mask.size, dtype=np.int32)
    groups = np.full(selected_mask.size, -1, dtype=np.int16)
    discrete[selected_indices] = shifts_array[group_for_selected]
    groups[selected_indices] = group_for_selected.astype(np.int16)

    continuous = np.zeros(selected_mask.size, dtype=np.int32)
    continuous[selected_indices] = rng.integers(
        -continuous_half_width,
        continuous_half_width + 1,
        size=selected_indices.size,
        dtype=np.int32,
    )
    return discrete, groups, continuous


def _source_file_id_tuple(value: Any) -> tuple[int, ...]:
    array = np.asarray(value, dtype=np.int64)
    if array.ndim == 0:
        return (int(array),)
    return tuple(int(item) for item in array.reshape(-1))


def _sanitize_feature_name(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", str(value)).strip("_")
    return cleaned or "unnamed"


def _catch22_features(signal_mV: np.ndarray, prefix: str) -> dict[str, float] | None:
    try:
        import pycatch22  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on user environment
        raise RuntimeError(
            "pycatch22 is required for the TOF-shift experiment. "
            "Install the repository requirements before running the generator."
        ) from exc

    values = np.asarray(signal_mV, dtype=np.float64).reshape(-1)
    if values.size < 20 or np.any(~np.isfinite(values)):
        return None
    try:
        result = pycatch22.catch22_all(
            values.tolist(),
            catch24=False,
            short_names=True,
        )
    except Exception:
        return None
    names = result.get("names", [])
    feature_values = result.get("values", [])
    if len(names) != 22 or len(feature_values) != 22:
        return None
    output: dict[str, float] = {}
    for name, value in zip(names, feature_values, strict=True):
        numeric = float(value)
        if not np.isfinite(numeric):
            return None
        output[f"{prefix}_c22_{_sanitize_feature_name(name)}"] = numeric
    return output


def _fixed_grid_ns(settings: dict[str, Any]) -> np.ndarray:
    start_ns = float(settings["window_start_ns"])
    stop_ns = float(settings["window_stop_ns"])
    step_ns = float(settings["resample_step_ps"]) / 1000.0
    count = int(round((stop_ns - start_ns) / step_ns)) + 1
    if count < 20:
        raise ValueError("fixed window must contain at least 20 samples for catch22")
    return np.linspace(start_ns, stop_ns, count, dtype=np.float64)


def shift_signal_on_fixed_grid(
    time_ns: np.ndarray,
    signal_mV: np.ndarray,
    fixed_grid_ns: np.ndarray,
    shift_ps: float,
) -> np.ndarray | None:
    """Delay a waveform by ``shift_ps`` and sample it on one absolute grid.

    A positive delay d is represented by y_shifted(t)=y_original(t-d).
    """
    time_values = np.asarray(time_ns, dtype=np.float64).reshape(-1)
    signal_values = np.asarray(signal_mV, dtype=np.float64).reshape(-1)
    grid = np.asarray(fixed_grid_ns, dtype=np.float64).reshape(-1)
    if time_values.size < 4 or time_values.size != signal_values.size:
        return None
    if np.any(~np.isfinite(time_values)) or np.any(~np.isfinite(signal_values)):
        return None
    if np.any(np.diff(time_values) <= 0):
        return None

    query_ns = grid - float(shift_ps) / 1000.0
    if query_ns[0] < time_values[0] or query_ns[-1] > time_values[-1]:
        return None
    start = max(0, int(np.searchsorted(time_values, query_ns[0], side="left")) - 2)
    stop = min(
        time_values.size,
        int(np.searchsorted(time_values, query_ns[-1], side="right")) + 2,
    )
    if stop - start < 4:
        return None
    try:
        spline = CubicSpline(
            time_values[start:stop],
            signal_values[start:stop],
            bc_type="not-a-knot",
            extrapolate=False,
        )
        shifted = np.asarray(spline(query_ns), dtype=np.float64)
    except (ValueError, FloatingPointError):
        return None
    if shifted.size != grid.size or np.any(~np.isfinite(shifted)):
        return None
    return shifted


def rising_crossing_ns(
    time_ns: np.ndarray,
    signal_mV: np.ndarray,
    threshold_mV: float,
    *,
    mode: str,
) -> float | None:
    time_values = np.asarray(time_ns, dtype=np.float64).reshape(-1)
    signal_values = np.asarray(signal_mV, dtype=np.float64).reshape(-1)
    if time_values.size < 2 or time_values.size != signal_values.size:
        return None
    threshold = float(threshold_mV)
    if mode == "first":
        candidates = np.flatnonzero(
            (signal_values[:-1] < threshold) & (signal_values[1:] >= threshold)
        )
        if candidates.size == 0:
            return None
        left_index = int(candidates[0])
    elif mode == "last_before_peak":
        peak_index = int(np.argmax(signal_values))
        if peak_index <= 0:
            return None
        candidates = np.flatnonzero(
            (signal_values[:peak_index] < threshold)
            & (signal_values[1 : peak_index + 1] >= threshold)
        )
        if candidates.size == 0:
            return None
        left_index = int(candidates[-1])
    else:
        raise ValueError("crossing mode must be 'first' or 'last_before_peak'")

    right_index = left_index + 1
    y0 = float(signal_values[left_index])
    y1 = float(signal_values[right_index])
    if not (np.isfinite(y0) and np.isfinite(y1)) or y1 == y0:
        return None
    fraction = (threshold - y0) / (y1 - y0)
    if not 0.0 <= fraction <= 1.0:
        return None
    return float(
        time_values[left_index]
        + fraction * (time_values[right_index] - time_values[left_index])
    )


def _detector_pair_channel_shifts_ps(
    pair_shift_ps: int,
    settings: dict[str, Any],
) -> dict[int, float]:
    """Shift only the configured detector pair and leave the other pair fixed.

    With the default channel mapping, detector pair 1 is energy channel 1 and
    timing channel 3.  A positive shift therefore changes the measured LED
    difference by +shift while preserving the local energy/timing alignment of
    that detector.
    """
    channels = settings["channels"]
    all_channels = {
        int(item) - 1
        for item in list(channels["energy"]) + list(channels["timing"])
    }
    shifted_channels = {int(item) - 1 for item in settings["shifted_channels"]}
    if not shifted_channels:
        raise ValueError("shifted_channels cannot be empty")
    if not shifted_channels.issubset(all_channels):
        raise ValueError("shifted_channels must belong to configured energy/timing channels")
    shift = float(pair_shift_ps)
    return {channel: (shift if channel in shifted_channels else 0.0) for channel in all_channels}


def _base_timing_tof_ps(
    basics: list[Any],
    time_axes_ns: list[np.ndarray],
    settings: dict[str, Any],
) -> float | None:
    fixed_grid = _fixed_grid_ns(settings)
    timing_channels = [
        int(item) - 1 for item in settings["channels"]["timing"]
    ]
    crossings: list[float] = []
    for channel in timing_channels:
        window = shift_signal_on_fixed_grid(
            time_axes_ns[channel],
            basics[channel].corrected_signal_mV,
            fixed_grid,
            0.0,
        )
        if window is None:
            return None
        crossing = rising_crossing_ns(
            fixed_grid,
            window,
            float(settings["timing_threshold_mV"]),
            mode=str(settings["crossing_mode"]),
        )
        if crossing is None:
            return None
        crossings.append(crossing)
    return float((crossings[0] - crossings[1]) * 1000.0)


def _scenario_features(
    basics: list[Any],
    time_axes_ns: list[np.ndarray],
    artificial_shift_ps: int,
    settings: dict[str, Any],
) -> tuple[dict[str, float], tuple[float, float]] | tuple[None, None]:
    fixed_grid = _fixed_grid_ns(settings)
    channel_shift = _detector_pair_channel_shifts_ps(artificial_shift_ps, settings)
    channels = settings["channels"]
    timing_channels = [int(item) - 1 for item in channels["timing"]]
    energy_channels = [int(item) - 1 for item in channels["energy"]]
    timing_threshold = float(settings["timing_threshold_mV"])
    energy_threshold = float(settings["energy_threshold_mV"])
    crossing_mode = str(settings["crossing_mode"])

    shifted_windows: dict[int, np.ndarray] = {}
    for channel in set(timing_channels + energy_channels):
        shifted = shift_signal_on_fixed_grid(
            time_axes_ns[channel],
            basics[channel].corrected_signal_mV,
            fixed_grid,
            channel_shift[channel],
        )
        if shifted is None:
            return None, None
        shifted_windows[channel] = shifted

    output: dict[str, float] = {}
    timing_crossings: list[float] = []
    for channel in timing_channels:
        crossing = rising_crossing_ns(
            fixed_grid,
            shifted_windows[channel],
            timing_threshold,
            mode=crossing_mode,
        )
        if crossing is None:
            return None, None
        timing_crossings.append(crossing)

    # Build one timing series after all requested waveform translations.
    # The orientation matches the LED TOF convention used by the pipeline:
    #     timing difference = channel 3 - channel 4.
    timing_difference_signal = (
        shifted_windows[timing_channels[0]]
        - shifted_windows[timing_channels[1]]
    )
    difference_catch22 = _catch22_features(
        timing_difference_signal,
        prefix="timing_difference",
    )
    if difference_catch22 is None:
        return None, None
    output.update(difference_catch22)

    # Store only the inter-detector arrival-time difference, not the two
    # absolute timing coordinates.  A shift applied to detector pair 1 changes
    # this quantity by exactly the assigned artificial TOF shift.
    output[f"timing_delta_t{timing_threshold:g}_ps"] = float(
        (timing_crossings[0] - timing_crossings[1]) * 1000.0
    )

    for channel in energy_channels:
        channel_number = channel + 1
        crossing = rising_crossing_ns(
            fixed_grid,
            shifted_windows[channel],
            energy_threshold,
            mode=crossing_mode,
        )
        if crossing is None:
            return None, None
        # The energy amplitude must be extracted from the translated waveform
        # inside the same fixed absolute window used for every other feature.
        # Using basics[channel].amplitude_mV here would leak a pre-translation,
        # full-record feature and violate the experiment definition.
        output[f"energy_ch{channel_number}_max_amplitude_mV"] = float(
            np.max(shifted_windows[channel])
        )
        output[f"energy_ch{channel_number}_t{energy_threshold:g}_abs_ps"] = (
            crossing * 1000.0
        )
    return output, (timing_crossings[0], timing_crossings[1])


def _event_settings(config: dict[str, Any]) -> dict[str, Any]:
    experiment = config["tof_shift_experiment"]
    window = experiment["absolute_window_ns"]
    return {
        "channels": config["channels"],
        "baseline_samples": int(config["waveform"]["baseline_samples"]),
        "trigger_threshold_mV": float(config["waveform"]["trigger_threshold_mV"]),
        "window_start_ns": float(window[0]),
        "window_stop_ns": float(window[1]),
        "resample_step_ps": float(experiment["resample_step_ps"]),
        "timing_threshold_mV": float(experiment["timing_threshold_mV"]),
        "energy_threshold_mV": float(experiment["energy_threshold_mV"]),
        "crossing_mode": str(experiment.get("crossing_mode", "last_before_peak")),
        "shifted_channels": [
            int(item) for item in experiment.get("shifted_channels", [1, 3])
        ],
    }


def process_shift_experiment_event(
    payload: ShiftExperimentPayload,
    settings: dict[str, Any],
) -> ShiftExperimentResult:
    from .io import decode_voltage_mV
    from .signal import baseline_and_basic_features

    polarities = [int(item) for item in settings["channels"]["polarities"]]
    basics: list[Any] = []
    time_axes_ns: list[np.ndarray] = []
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

    base_led_tof_ps = _base_timing_tof_ps(basics, time_axes_ns, settings)
    if base_led_tof_ps is None:
        return ShiftExperimentResult(None, None, "base_led_invalid")

    discrete_features, _ = _scenario_features(
        basics,
        time_axes_ns,
        payload.discrete_shift_ps,
        settings,
    )
    if discrete_features is None:
        return ShiftExperimentResult(None, None, "discrete_waveform_invalid")
    continuous_features, _ = _scenario_features(
        basics,
        time_axes_ns,
        payload.continuous_shift_ps,
        settings,
    )
    if continuous_features is None:
        return ShiftExperimentResult(None, None, "continuous_waveform_invalid")

    common_metadata = {
        "meta_event_index": payload.event_index,
        "meta_event_id": payload.event_id,
        "meta_source_file_id": ";".join(str(item) for item in payload.source_file_id),
    }
    discrete_row: dict[str, Any] = {
        **common_metadata,
        "meta_shift_mode": "discrete",
        "meta_assigned_shift_ps": int(payload.discrete_shift_ps),
        "meta_discrete_group": int(payload.discrete_group),
        "_led_tof_ps": float(base_led_tof_ps),
        **discrete_features,
    }
    continuous_row: dict[str, Any] = {
        **common_metadata,
        "meta_shift_mode": "continuous",
        "meta_assigned_shift_ps": int(payload.continuous_shift_ps),
        "meta_discrete_group": -1,
        "_led_tof_ps": float(base_led_tof_ps),
        **continuous_features,
    }
    return ShiftExperimentResult(discrete_row, continuous_row, None)


def _safe_process_shift_event(
    payload: ShiftExperimentPayload,
    settings: dict[str, Any],
) -> ShiftExperimentResult:
    try:
        return process_shift_experiment_event(payload, settings)
    except RuntimeError:
        raise
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        return ShiftExperimentResult(
            None,
            None,
            f"processing_{type(exc).__name__}",
        )


def _init_worker(settings: dict[str, Any]) -> None:
    global _WORKER_SETTINGS
    _WORKER_SETTINGS = settings


def _worker(payload: ShiftExperimentPayload) -> ShiftExperimentResult:
    if _WORKER_SETTINGS is None:
        raise RuntimeError("shift experiment worker was not initialized")
    return _safe_process_shift_event(payload, _WORKER_SETTINGS)


def _payloads_for_chunk(
    chunk: Any,
    selected_chunk: np.ndarray,
    discrete_chunk: np.ndarray,
    group_chunk: np.ndarray,
    continuous_chunk: np.ndarray,
) -> list[ShiftExperimentPayload]:
    import awkward as ak  # type: ignore

    payloads: list[ShiftExperimentPayload] = []
    for row_index in np.flatnonzero(selected_chunk):
        row = int(row_index)
        payloads.append(
            ShiftExperimentPayload(
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
                discrete_shift_ps=int(discrete_chunk[row]),
                discrete_group=int(group_chunk[row]),
                continuous_shift_ps=int(continuous_chunk[row]),
            )
        )
    return payloads


def resolve_worker_count(config: dict[str, Any], override: int | None = None) -> int:
    parallel = config["tof_shift_experiment"]["parallel"]
    requested = int(override if override is not None else parallel.get("workers", 0))
    if requested > 0:
        return requested
    cpu_count = os.cpu_count() or 1
    maximum = max(1, int(parallel.get("max_auto_workers", 8)))
    return min(maximum, max(1, cpu_count - 1))


def generate_shift_dataset_rows(
    input_path: Path,
    selected: np.ndarray,
    discrete_shifts: np.ndarray,
    discrete_groups: np.ndarray,
    continuous_shifts: np.ndarray,
    config: dict[str, Any],
    *,
    workers_override: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    from .io import iterate_chunks

    selected_mask = np.asarray(selected, dtype=bool).reshape(-1)
    for values, name in (
        (discrete_shifts, "discrete shifts"),
        (discrete_groups, "discrete groups"),
        (continuous_shifts, "continuous shifts"),
    ):
        if np.asarray(values).reshape(-1).size != selected_mask.size:
            raise ValueError(f"{name} length differs from selection mask")

    settings = _event_settings(config)
    io_config = config["io"]
    parallel = config["tof_shift_experiment"]["parallel"]
    workers = resolve_worker_count(config, workers_override)
    chunksize = max(1, int(parallel.get("map_chunksize", 8)))
    progress_every = max(1, int(parallel.get("progress_every", 250)))
    max_events = int(io_config.get("max_events", 0))
    entry_stop = max_events if max_events > 0 else None

    LOGGER.info(
        "Shift dataset pass: selected=%d, workers=%d, window=[%.3f, %.3f] ns, step=%.3f ps",
        int(np.count_nonzero(selected_mask)),
        workers,
        settings["window_start_ns"],
        settings["window_stop_ns"],
        settings["resample_step_ps"],
    )
    LOGGER.info(
        "Features: catch22(timing ch3-ch4), timing delta at %.3f mV, energy threshold %.3f mV",
        settings["timing_threshold_mV"],
        settings["energy_threshold_mV"],
    )

    discrete_rows: list[dict[str, Any]] = []
    continuous_rows: list[dict[str, Any]] = []
    rejections: Counter[str] = Counter()
    global_start = 0
    processed = 0
    executor: ProcessPoolExecutor | None = None
    if workers > 1:
        executor = ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(settings,),
        )

    try:
        for chunk in iterate_chunks(
            input_path,
            step_size=io_config.get("step_size", "128 MB"),
            entry_stop=entry_stop,
        ):
            chunk_size = int(chunk.event_id.size)
            stop = global_start + chunk_size
            selected_chunk = selected_mask[global_start:stop]
            discrete_chunk = np.asarray(discrete_shifts)[global_start:stop]
            group_chunk = np.asarray(discrete_groups)[global_start:stop]
            continuous_chunk = np.asarray(continuous_shifts)[global_start:stop]
            if selected_chunk.size != chunk_size:
                raise RuntimeError("selection mask and ROOT event order are inconsistent")
            payloads = _payloads_for_chunk(
                chunk,
                selected_chunk,
                discrete_chunk,
                group_chunk,
                continuous_chunk,
            )
            global_start = stop
            if not payloads:
                continue
            if executor is None:
                results: Iterable[ShiftExperimentResult] = (
                    _safe_process_shift_event(payload, settings) for payload in payloads
                )
            else:
                results = executor.map(_worker, payloads, chunksize=chunksize)

            for result in results:
                processed += 1
                if result.discrete_row is None or result.continuous_row is None:
                    rejections[result.rejection_reason or "unknown"] += 1
                else:
                    discrete_rows.append(result.discrete_row)
                    continuous_rows.append(result.continuous_row)
                if processed % progress_every == 0:
                    LOGGER.info(
                        "Shift dataset pass processed %d/%d; paired accepted=%d",
                        processed,
                        int(np.count_nonzero(selected_mask)),
                        len(discrete_rows),
                    )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)

    if global_start != selected_mask.size:
        raise RuntimeError(
            f"processed {global_start} ROOT events but selection has {selected_mask.size}"
        )
    if len(discrete_rows) != len(continuous_rows):
        raise RuntimeError("discrete and continuous datasets lost row alignment")
    LOGGER.info(
        "Shift dataset pass complete: paired accepted=%d, rejected=%d",
        len(discrete_rows),
        sum(rejections.values()),
    )
    return discrete_rows, continuous_rows, rejections


def _write_rows(rows: list[dict[str, Any]], path: Path, target_column: str) -> list[str]:
    """Write public dataset fields while keeping internal diagnostics private.

    Keys beginning with ``_`` (for example ``_led_tof_ps`` used by the MAD
    filter) are intentionally retained in memory until dataset finalization,
    but must never be serialized as model inputs.  Build sanitized row copies
    before passing them to ``csv.DictWriter`` so private fields cannot trigger
    a schema error or leak into the CSV.
    """
    if not rows:
        raise ValueError("cannot write an empty dataset")

    metadata = [
        "meta_event_index",
        "meta_event_id",
        "meta_source_file_id",
        "meta_shift_mode",
        "meta_assigned_shift_ps",
        "meta_discrete_group",
    ]
    public_rows = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in rows
    ]

    expected_keys = set(public_rows[0])
    for row_index, row in enumerate(public_rows[1:], start=1):
        row_keys = set(row)
        if row_keys != expected_keys:
            missing = sorted(expected_keys - row_keys)
            extra = sorted(row_keys - expected_keys)
            raise ValueError(
                "inconsistent dataset row schema at row "
                f"{row_index}: missing={missing}, extra={extra}"
            )

    required = set(metadata + [target_column])
    missing_required = sorted(required - expected_keys)
    if missing_required:
        raise ValueError(
            f"dataset rows are missing required fields: {missing_required}"
        )

    feature_columns = sorted(expected_keys - required)
    fieldnames = metadata + [target_column] + feature_columns
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(public_rows)
    return feature_columns


def finalize_shift_datasets(
    discrete_rows: list[dict[str, Any]],
    continuous_rows: list[dict[str, Any]],
    output_directory: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Write paired datasets whose target is the shift of one detector pair."""
    if not discrete_rows or len(discrete_rows) != len(continuous_rows):
        raise RuntimeError("paired discrete/continuous rows are empty or misaligned")

    experiment = config["tof_shift_experiment"]
    target_column = str(experiment.get("target_column", "target_shift_ps"))

    for discrete_row, continuous_row in zip(
        discrete_rows, continuous_rows, strict=True
    ):
        if discrete_row["meta_event_index"] != continuous_row["meta_event_index"]:
            raise RuntimeError("paired dataset event order changed")
        discrete_row[target_column] = float(discrete_row["meta_assigned_shift_ps"])
        continuous_row[target_column] = float(continuous_row["meta_assigned_shift_ps"])

    discrete_path = output_directory / str(
        experiment.get("discrete_filename", "tof_discrete.csv")
    )
    continuous_path = output_directory / str(
        experiment.get("continuous_filename", "tof_continuous.csv")
    )
    discrete_features = _write_rows(discrete_rows, discrete_path, target_column)
    continuous_features = _write_rows(continuous_rows, continuous_path, target_column)
    if discrete_features != continuous_features:
        raise RuntimeError("paired datasets have different feature schemas")

    discrete_target = np.asarray(
        [float(row[target_column]) for row in discrete_rows], dtype=np.float64
    )
    continuous_target = np.asarray(
        [float(row[target_column]) for row in continuous_rows], dtype=np.float64
    )

    shifts = np.asarray(
        [int(item) for item in experiment.get("discrete_shifts_ps", [-80, 0, 80])],
        dtype=np.int64,
    )
    if shifts.shape != (3,) or not (
        int(np.sort(shifts)[1]) == 0
        and int(np.sort(shifts)[0]) == -int(np.sort(shifts)[2])
    ):
        raise ValueError("discrete_shifts_ps must have the form [-a, 0, +a]")
    variance_match = matched_integer_uniform_half_width(int(np.max(shifts)))

    summary = {
        "rows_per_dataset": len(discrete_rows),
        "target_column": target_column,
        "feature_columns": discrete_features,
        "number_of_features": len(discrete_features),
        "discrete_csv": str(discrete_path),
        "continuous_csv": str(continuous_path),
        "discrete_target_mean_ps": float(np.mean(discrete_target)),
        "continuous_target_mean_ps": float(np.mean(continuous_target)),
        "discrete_target_std_ps": float(np.std(discrete_target, ddof=1)),
        "continuous_target_std_ps": float(np.std(continuous_target, ddof=1)),
        "discrete_target_variance_ps2": float(np.var(discrete_target, ddof=1)),
        "continuous_target_variance_ps2": float(np.var(continuous_target, ddof=1)),
        "theoretical_variance_match": variance_match,
        "shift_application": "assigned shift applied only to configured detector-pair channels",
        "shifted_channels": [
            int(item) for item in experiment.get("shifted_channels", [1, 3])
        ],
        "absolute_window_ns": [
            float(experiment["absolute_window_ns"][0]),
            float(experiment["absolute_window_ns"][1]),
        ],
        "target_definition": "assigned shift of one detector pair in integer picoseconds",
    }
    LOGGER.info("Discrete dataset written: %s", discrete_path)
    LOGGER.info("Continuous dataset written: %s", continuous_path)
    LOGGER.info(
        "Empirical target variances: discrete=%.3f ps^2, continuous=%.3f ps^2",
        summary["discrete_target_variance_ps2"],
        summary["continuous_target_variance_ps2"],
    )
    return summary

