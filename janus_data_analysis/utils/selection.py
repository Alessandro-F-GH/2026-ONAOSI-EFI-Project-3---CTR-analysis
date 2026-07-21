from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

from .binary_io import iter_events, read_header
from .tabular import read_table, write_table
from .models import (
    EnergyMeasurements,
    EnergySelectionResult,
    Measurements,
    PeakSelection,
    SelectionResult,
)

ENERGY_SELECTION_FIELDS = [
    "event_index",
    "duration_a_lsb",
    "duration_b_lsb",
    "duration_selected",
]
ENERGY_SELECTION_DEBUG_FIELDS = [
    "event_index",
    "duration_a_lsb",
    "duration_b_lsb",
    "energy_a_lsb",
    "energy_b_lsb",
    "duration_selected",
]

SELECTION_FIELDS = [
    "event_index",
    "duration_a_lsb",
    "duration_b_lsb",
    "energy_a_lsb",
    "time_a_lsb",
    "energy_b_lsb",
    "time_b_lsb",
    "alignment_a_lsb",
    "alignment_b_lsb",
    "timing_lsb",
    "duration_selected",
    "alignment_selected",
]


def _single_interval(hits, channel: int) -> tuple[int, int] | None:
    leading = [hit.toa_lsb for hit in hits if hit.channel == channel and hit.edge == 1]
    trailing = [hit.toa_lsb for hit in hits if hit.channel == channel and hit.edge == 0]
    if len(leading) != 1 or len(trailing) != 1 or trailing[0] <= leading[0]:
        return None
    return leading[0], trailing[0]


def _single_leading(hits, channel: int) -> int | None:
    values = [hit.toa_lsb for hit in hits if hit.channel == channel and hit.edge == 1]
    return values[0] if len(values) == 1 else None


def collect_energy_measurements(
    path: str | Path,
    cfg: dict,
) -> tuple[EnergyMeasurements, float]:
    """Read the candidate-preserving binary using only the energy channels."""
    channels = cfg["channels"]
    event_index: list[int] = []
    duration_a: list[int] = []
    duration_b: list[int] = []
    energy_a: list[int] = []
    energy_b: list[int] = []
    with Path(path).open("rb") as handle:
        meta = read_header(handle)
        for event in iter_events(handle, meta):
            interval_a = _single_interval(event.hits, int(channels["signal_a"]))
            interval_b = _single_interval(event.hits, int(channels["signal_b"]))
            if interval_a is None or interval_b is None:
                continue
            event_index.append(event.event_index)
            energy_a.append(interval_a[0])
            duration_a.append(interval_a[1] - interval_a[0])
            energy_b.append(interval_b[0])
            duration_b.append(interval_b[1] - interval_b[0])
    measurements = EnergyMeasurements(
        event_index=np.asarray(event_index, dtype=np.int64),
        duration_a_lsb=np.asarray(duration_a, dtype=np.int64),
        duration_b_lsb=np.asarray(duration_b, dtype=np.int64),
        energy_a_lsb=np.asarray(energy_a, dtype=np.int64),
        energy_b_lsb=np.asarray(energy_b, dtype=np.int64),
    )
    return measurements, meta.toa_lsb_ps


def collect_measurements(path: str | Path, cfg: dict) -> tuple[Measurements, float]:
    """Read the final matched binary, which must contain one timing lead per pair."""
    channels = cfg["channels"]
    event_index: list[int] = []
    duration_a: list[int] = []
    duration_b: list[int] = []
    energy_a: list[int] = []
    time_a: list[int] = []
    energy_b: list[int] = []
    time_b: list[int] = []
    with Path(path).open("rb") as handle:
        meta = read_header(handle)
        for event in iter_events(handle, meta):
            interval_a = _single_interval(event.hits, int(channels["signal_a"]))
            interval_b = _single_interval(event.hits, int(channels["signal_b"]))
            timing_a = _single_leading(event.hits, int(channels["time_a"]))
            timing_b = _single_leading(event.hits, int(channels["time_b"]))
            if interval_a is None or interval_b is None or timing_a is None or timing_b is None:
                continue
            event_index.append(event.event_index)
            energy_a.append(interval_a[0])
            duration_a.append(interval_a[1] - interval_a[0])
            time_a.append(timing_a)
            energy_b.append(interval_b[0])
            duration_b.append(interval_b[1] - interval_b[0])
            time_b.append(timing_b)
    measurements = Measurements(
        event_index=np.asarray(event_index, dtype=np.int64),
        duration_a_lsb=np.asarray(duration_a, dtype=np.int64),
        duration_b_lsb=np.asarray(duration_b, dtype=np.int64),
        energy_a_lsb=np.asarray(energy_a, dtype=np.int64),
        time_a_lsb=np.asarray(time_a, dtype=np.int64),
        energy_b_lsb=np.asarray(energy_b, dtype=np.int64),
        time_b_lsb=np.asarray(time_b, dtype=np.int64),
    )
    return measurements, meta.toa_lsb_ps


def _robust_center_scale(
    values: np.ndarray,
    minimum_scale: float = 0.0,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.int64)
    if values.size == 0:
        return math.nan, math.nan
    center = float(np.median(values))
    scale = 1.4826 * float(np.median(np.abs(values.astype(float) - center)))
    if not math.isfinite(scale) or scale <= 0:
        scale = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    return center, max(scale, float(minimum_scale))


def select_peak(values_lsb: np.ndarray, cfg: dict) -> PeakSelection:
    values = np.asarray(values_lsb, dtype=np.int64)
    if values.size < int(cfg["min_events"]):
        raise RuntimeError(f"Only {values.size} events available for peak selection")
    minimum = int(values.min())
    maximum = int(values.max())
    if minimum == maximum:
        return PeakSelection(minimum, maximum, float(minimum), float(minimum), 1.0)

    bandwidth_factor = float(cfg["kde_bandwidth_factor"])

    def bandwidth(kde):
        return kde.scotts_factor() * bandwidth_factor

    kde = gaussian_kde(values.astype(float), bw_method=bandwidth)
    padding = max(1.0, 0.03 * (maximum - minimum))
    grid = np.linspace(
        minimum - padding,
        maximum + padding,
        int(cfg["kde_grid_points"]),
    )
    density = kde(grid)
    peaks, _ = find_peaks(
        density,
        height=float(np.max(density)) * float(cfg["min_peak_height_fraction"]),
    )
    if peaks.size == 0:
        peak_index = int(np.argmax(density))
    elif str(cfg.get("peak_choice", "rightmost_significant")).lower() == "highest_significant":
        peak_index = int(peaks[np.argmax(density[peaks])])
    else:
        peak_index = int(peaks[np.argmax(grid[peaks])])

    half_height = 0.5 * float(density[peak_index])
    left_index = peak_index
    while left_index > 0 and density[left_index] > half_height:
        left_index -= 1
    right_index = peak_index
    while right_index < density.size - 1 and density[right_index] > half_height:
        right_index += 1
    fwhm_low = float(grid[left_index])
    fwhm_high = float(grid[right_index])
    core = values[(values >= math.floor(fwhm_low)) & (values <= math.ceil(fwhm_high))]
    center, scale = _robust_center_scale(core if core.size >= 2 else values)
    if not math.isfinite(scale) or scale <= 0:
        scale = max((fwhm_high - fwhm_low) / 2.355, 1.0)
    low = math.floor(center - float(cfg["left_sigma_multiplier"]) * scale)
    high = math.ceil(center + float(cfg["right_sigma_multiplier"]) * scale)
    if low >= high:
        low, high = minimum, maximum
    selected_count = np.count_nonzero((values >= low) & (values <= high))
    if selected_count < int(cfg.get("minimum_events_in_interval", 5)):
        raise RuntimeError("Selected duration interval contains too few events")
    return PeakSelection(int(low), int(high), float(grid[peak_index]), center, scale)


def select_energy_events(
    measurements: EnergyMeasurements,
    cfg: dict,
) -> EnergySelectionResult:
    peak_a = select_peak(measurements.duration_a_lsb, cfg["peak_selection"])
    peak_b = select_peak(measurements.duration_b_lsb, cfg["peak_selection"])
    duration_mask = (
        (measurements.duration_a_lsb >= peak_a.low_lsb)
        & (measurements.duration_a_lsb <= peak_a.high_lsb)
        & (measurements.duration_b_lsb >= peak_b.low_lsb)
        & (measurements.duration_b_lsb <= peak_b.high_lsb)
    )
    return EnergySelectionResult(peak_a, peak_b, duration_mask)


def _alignment_keep(
    values: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, float, float, bool]:
    values = np.asarray(values, dtype=np.int64)
    method = str(cfg["method"]).lower()
    if method == "standard":
        center = float(np.mean(values)) if values.size else math.nan
        scale = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        scale = max(scale, float(cfg["minimum_scale_lsb"]))
    else:
        center, scale = _robust_center_scale(values, float(cfg["minimum_scale_lsb"]))
    applied = (
        bool(cfg["enabled"])
        and values.size >= int(cfg["minimum_events"])
        and math.isfinite(center)
        and math.isfinite(scale)
        and scale > 0
    )
    if not applied:
        return np.ones(values.size, dtype=bool), center, scale, False
    keep = np.abs(values.astype(float) - center) <= float(cfg["z_threshold"]) * scale
    return keep, center, scale, True


def select_matched_events(
    measurements: Measurements,
    peak_a: PeakSelection,
    peak_b: PeakSelection,
    cfg: dict,
) -> SelectionResult:
    """Apply the original peak interval and then the alignment filter after matching."""
    duration_mask = (
        (measurements.duration_a_lsb >= peak_a.low_lsb)
        & (measurements.duration_a_lsb <= peak_a.high_lsb)
        & (measurements.duration_b_lsb >= peak_b.low_lsb)
        & (measurements.duration_b_lsb <= peak_b.high_lsb)
    )
    selected_indices = np.flatnonzero(duration_mask)
    keep_a, center_a, scale_a, _ = _alignment_keep(
        measurements.alignment_a_lsb[selected_indices],
        cfg["alignment_filter"],
    )
    keep_b, center_b, scale_b, _ = _alignment_keep(
        measurements.alignment_b_lsb[selected_indices],
        cfg["alignment_filter"],
    )
    local_keep = keep_a & keep_b
    alignment_mask = np.zeros(measurements.size, dtype=bool)
    alignment_mask[selected_indices] = local_keep
    final_mask = duration_mask & alignment_mask
    return SelectionResult(
        peak_a=peak_a,
        peak_b=peak_b,
        duration_mask=duration_mask,
        alignment_mask=alignment_mask,
        final_mask=final_mask,
        alignment_a_center_lsb=center_a,
        alignment_a_scale_lsb=scale_a,
        alignment_b_center_lsb=center_b,
        alignment_b_scale_lsb=scale_b,
    )


def select_events(measurements: Measurements, cfg: dict) -> SelectionResult:
    peak_a = select_peak(measurements.duration_a_lsb, cfg["peak_selection"])
    peak_b = select_peak(measurements.duration_b_lsb, cfg["peak_selection"])
    return select_matched_events(measurements, peak_a, peak_b, cfg)


def write_energy_selection_csv(
    path: str | Path,
    measurements: EnergyMeasurements,
    selection: EnergySelectionResult,
    diagnostic_mode: str = "compact",
) -> None:
    debug = str(diagnostic_mode).lower() == "debug"
    rows = []
    for index in range(measurements.size):
        row = {
            "event_index": int(measurements.event_index[index]),
            "duration_a_lsb": int(measurements.duration_a_lsb[index]),
            "duration_b_lsb": int(measurements.duration_b_lsb[index]),
            "duration_selected": int(selection.duration_mask[index]),
        }
        if debug:
            row["energy_a_lsb"] = int(measurements.energy_a_lsb[index])
            row["energy_b_lsb"] = int(measurements.energy_b_lsb[index])
        rows.append(row)
    write_table(
        path,
        ENERGY_SELECTION_DEBUG_FIELDS if debug else ENERGY_SELECTION_FIELDS,
        rows,
    )


def load_energy_selection_csv(
    path: str | Path,
) -> tuple[EnergyMeasurements, np.ndarray]:
    rows = read_table(path)

    def array(name: str, default: int = 0) -> np.ndarray:
        return np.asarray(
            [int(row.get(name, default) or default) for row in rows],
            dtype=np.int64,
        )

    measurements = EnergyMeasurements(
        event_index=array("event_index"),
        duration_a_lsb=array("duration_a_lsb"),
        duration_b_lsb=array("duration_b_lsb"),
        energy_a_lsb=array("energy_a_lsb"),
        energy_b_lsb=array("energy_b_lsb"),
    )
    mask = np.asarray(
        [bool(int(row["duration_selected"])) for row in rows],
        dtype=bool,
    )
    return measurements, mask


def write_selection_csv(
    path: str | Path,
    measurements: Measurements,
    selection: SelectionResult,
    diagnostic_mode: str = "compact",
) -> None:
    debug = str(diagnostic_mode).lower() == "debug"
    rows = []
    for index in range(measurements.size):
        row = {
            "event_index": int(measurements.event_index[index]),
            "duration_a_lsb": int(measurements.duration_a_lsb[index]),
            "duration_b_lsb": int(measurements.duration_b_lsb[index]),
            "energy_a_lsb": int(measurements.energy_a_lsb[index]),
            "time_a_lsb": int(measurements.time_a_lsb[index]),
            "energy_b_lsb": int(measurements.energy_b_lsb[index]),
            "time_b_lsb": int(measurements.time_b_lsb[index]),
            "duration_selected": int(selection.duration_mask[index]),
            "alignment_selected": int(selection.alignment_mask[index]),
        }
        if debug:
            row["alignment_a_lsb"] = int(measurements.alignment_a_lsb[index])
            row["alignment_b_lsb"] = int(measurements.alignment_b_lsb[index])
            row["timing_lsb"] = int(measurements.timing_lsb[index])
        rows.append(row)
    fields = SELECTION_FIELDS if debug else [
        field
        for field in SELECTION_FIELDS
        if field not in {"alignment_a_lsb", "alignment_b_lsb", "timing_lsb"}
    ]
    write_table(path, fields, rows)


def load_selection_csv(
    path: str | Path,
) -> tuple[Measurements, np.ndarray, np.ndarray]:
    rows = read_table(path)

    def array(name: str) -> np.ndarray:
        return np.asarray([int(row[name]) for row in rows], dtype=np.int64)

    measurements = Measurements(
        event_index=array("event_index"),
        duration_a_lsb=array("duration_a_lsb"),
        duration_b_lsb=array("duration_b_lsb"),
        energy_a_lsb=array("energy_a_lsb"),
        time_a_lsb=array("time_a_lsb"),
        energy_b_lsb=array("energy_b_lsb"),
        time_b_lsb=array("time_b_lsb"),
    )
    duration_mask = np.asarray(
        [bool(int(row["duration_selected"])) for row in rows],
        dtype=bool,
    )
    alignment_mask = np.asarray(
        [bool(int(row["alignment_selected"])) for row in rows],
        dtype=bool,
    )
    return measurements, duration_mask, alignment_mask
