from __future__ import annotations

"""Generate presentation-only figures from the current CTR-analysis repository.

Intended location in the repository:
    waveform_analysis/scripts/make_presentation_plots.py

The script does NOT rerun ML training and does NOT recompute CTR from predictions.
It uses:
  * one permanent prepared dataset for real waveform examples;
  * <run>/results.csv + <run>/manifest.json for the multithreshold performance scan.

Default output:
    waveform_analysis/results/presentation/plots/

Generated LaTeX-facing filenames:
    data_energy_waveform_example.pdf
    data_timing_waveform_example.pdf
    led_cfd_waveform_schematic.pdf
    windowing_native_grid.pdf
    multithreshold_waveform_crossings.pdf
    results_multithreshold_ctr_vs_threshold_count.pdf
    results_multithreshold_ctr_vs_threshold_count_energy_to_energy.pdf
    results_multithreshold_ctr_vs_threshold_count_energy_to_timing.pdf
    results_multithreshold_best_thresholds.pdf
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------------------------------
# Repository imports
# -----------------------------------------------------------------------------
def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    candidates = [here.parent, *here.parents, Path.cwd().resolve()]
    for candidate in candidates:
        if (candidate / "ml_pipeline").is_dir():
            return candidate
        if (candidate / "waveform_analysis" / "ml_pipeline").is_dir():
            return candidate / "waveform_analysis"
    # Standard intended placement: waveform_analysis/scripts/<this file>.
    return here.parents[1]


PROJECT = _find_project_root()
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from ml_pipeline.dataset import PreparedDataset, load_prepared_dataset


FEMTOSECONDS_PER_PICOSECOND = 1000.0


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _as_int(value: Any, default: int = -1) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _save(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def _mode_label(mode: str) -> str:
    return {
        "energy_to_energy": "Energy waveform → energy LED",
        "energy_to_timing": "Energy waveform → timing LED",
        "timing_to_timing": "Timing waveform → timing LED",
    }.get(mode, mode.replace("_", " "))


def _median_voltage(dataset: PreparedDataset) -> float:
    values = np.asarray(dataset.bias_voltage_V, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def _candidate_parameters(run_manifest: dict[str, Any], candidate_id: int) -> dict[str, Any]:
    mapping = run_manifest.get("candidate_parameters", {}) or {}
    value = mapping.get(str(int(candidate_id)), {})
    return value if isinstance(value, dict) else {}


def _codebook_id(run_manifest: dict[str, Any], family: str, name: str) -> int | None:
    codebooks = run_manifest.get("codebooks", {}) or {}
    mapping = codebooks.get(family, {}) or {}
    if name not in mapping:
        return None
    return int(mapping[name])


def _source_file_name(dataset: PreparedDataset) -> str:
    source = str(dataset.manifest.get("source_root", ""))
    return Path(source).name if source else ""


def _dataset_candidates(prepared: Path) -> list[Path]:
    prepared = prepared.resolve()
    if (prepared / "manifest.json").is_file():
        return [prepared]
    output = [p for p in prepared.iterdir() if p.is_dir() and (p / "manifest.json").is_file()]
    return sorted(output, key=lambda p: p.name)


def _choose_dataset(prepared: Path, requested_voltage: float | None) -> PreparedDataset:
    candidates = _dataset_candidates(prepared)
    if not candidates:
        raise FileNotFoundError(
            f"No prepared dataset containing manifest.json found in {prepared}"
        )

    loaded: list[tuple[float, PreparedDataset]] = []
    for directory in candidates:
        try:
            ds = load_prepared_dataset(directory)
        except Exception as exc:
            print(f"warning: skipping {directory}: {exc}", file=sys.stderr)
            continue
        loaded.append((_median_voltage(ds), ds))

    if not loaded:
        raise RuntimeError(f"Could not load any prepared dataset from {prepared}")

    if requested_voltage is not None:
        return min(
            loaded,
            key=lambda item: abs(item[0] - float(requested_voltage))
            if np.isfinite(item[0]) else float("inf"),
        )[1]

    finite = sorted((item for item in loaded if np.isfinite(item[0])), key=lambda x: x[0])
    if finite:
        # Middle voltage makes a visually representative default for a presentation.
        return finite[len(finite) // 2][1]
    return loaded[0][1]


def _waveform_config(dataset: PreparedDataset) -> dict[str, Any]:
    raw_manifest = dataset.manifest.get("raw_cache_manifest", {}) or {}
    preprocessing = raw_manifest.get("preprocessing", {}) or {}
    waveform = preprocessing.get("waveform", {}) or {}
    return waveform if isinstance(waveform, dict) else {}


def _energy_led_threshold(dataset: PreparedDataset) -> float:
    return float(_waveform_config(dataset).get("led_threshold_mV", 10.0))


def _energy_cfd_fraction(dataset: PreparedDataset) -> float:
    return float(_waveform_config(dataset).get("cfd_fraction", 0.2))


def _all_mt_thresholds(run_manifest: dict[str, Any]) -> list[float]:
    values: set[float] = set()
    for descriptor in (run_manifest.get("candidate_parameters", {}) or {}).values():
        if not isinstance(descriptor, dict):
            continue
        if str(descriptor.get("family", "")) != "multithreshold_svr":
            continue
        for value in descriptor.get("thresholds_mV", []) or []:
            try:
                values.add(float(value))
            except (TypeError, ValueError):
                pass
    return sorted(values)


def _representative_event(dataset: PreparedDataset, explicit: int | None = None) -> int:
    n = int(dataset.event_id.size)
    if n <= 0:
        raise RuntimeError("Prepared dataset is empty")
    if explicit is not None:
        if not 0 <= int(explicit) < n:
            raise IndexError(f"event index {explicit} is outside [0, {n - 1}]")
        return int(explicit)

    energy = np.asarray(dataset.windows_mV)
    finite = np.all(np.isfinite(energy), axis=(1, 2))
    if dataset.timing_windows_mV is not None:
        finite &= np.all(np.isfinite(np.asarray(dataset.timing_windows_mV)), axis=(1, 2))

    good = np.flatnonzero(finite)
    if good.size == 0:
        raise RuntimeError("No event has finite waveforms for the requested presentation plots")

    # Prefer a typical photopeak event rather than an extreme high/low pulse.
    amp = np.asarray(dataset.amplitude_mV, dtype=np.float64)
    pair_mean = np.nanmean(amp, axis=1)
    median_amp = float(np.nanmedian(pair_mean[good]))
    return int(good[np.nanargmin(np.abs(pair_mean[good] - median_amp))])


# -----------------------------------------------------------------------------
# Crossing helpers used only to DRAW the same interpolation used by the study.
# They do not enter ML metrics or model selection.
# -----------------------------------------------------------------------------
def _last_rising_crossing_before_peak(
    time_ps: np.ndarray,
    signal_mV: np.ndarray,
    threshold_mV: float,
) -> float:
    t = np.asarray(time_ps, dtype=np.float64)
    y = np.asarray(signal_mV, dtype=np.float64)
    if t.ndim != 1 or y.ndim != 1 or t.size != y.size or t.size < 2:
        return float("nan")
    peak = int(np.nanargmax(y))
    if peak <= 0:
        return float("nan")
    y0 = y[:peak]
    y1 = y[1 : peak + 1]
    finite = np.isfinite(y0) & np.isfinite(y1)
    crossings = finite & (y0 < float(threshold_mV)) & (y1 >= float(threshold_mV))
    loc = np.flatnonzero(crossings)
    if loc.size == 0:
        return float("nan")
    i = int(loc[-1])
    if y1[i] == y0[i]:
        return float("nan")
    fraction = (float(threshold_mV) - y0[i]) / (y1[i] - y0[i])
    return float(t[i] + fraction * (t[i + 1] - t[i]))


def _first_rising_crossing(
    time_ps: np.ndarray,
    signal_mV: np.ndarray,
    threshold_mV: float,
) -> float:
    t = np.asarray(time_ps, dtype=np.float64)
    y = np.asarray(signal_mV, dtype=np.float64)
    if t.ndim != 1 or y.ndim != 1 or t.size != y.size or t.size < 2:
        return float("nan")
    finite = np.isfinite(y[:-1]) & np.isfinite(y[1:])
    crossings = finite & (y[:-1] < float(threshold_mV)) & (y[1:] >= float(threshold_mV))
    loc = np.flatnonzero(crossings)
    if loc.size == 0:
        return float("nan")
    i = int(loc[0])
    y0, y1 = float(y[i]), float(y[i + 1])
    if y1 == y0:
        return float("nan")
    fraction = (float(threshold_mV) - y0) / (y1 - y0)
    return float(t[i] + fraction * (t[i + 1] - t[i]))


# -----------------------------------------------------------------------------
# Real waveform examples
# -----------------------------------------------------------------------------
def plot_energy_waveform_example(
    dataset: PreparedDataset, event: int, output: Path, dpi: int
) -> None:
    t_ns = np.asarray(dataset.relative_time_ps, dtype=np.float64) / 1000.0
    waves = np.asarray(dataset.windows_mV[event], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    ax.plot(t_ns, waves[0], linewidth=1.5, label="Detector 1")
    ax.plot(t_ns, waves[1], linewidth=1.5, label="Detector 2")
    ax.axvline(0.0, linestyle="--", linewidth=1.0, label="native LED anchor")
    ax.set_xlabel("Time relative to native LED anchor [ns]")
    ax.set_ylabel("Voltage [mV]")
    ax.set_title(f"Energy-channel waveform pair · event {event}")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=3)
    _save(fig, output / "data_energy_waveform_example.pdf", dpi)


def plot_timing_waveform_example(
    dataset: PreparedDataset, event: int, output: Path, dpi: int
) -> None:
    if dataset.timing_windows_mV is None or dataset.timing_relative_time_ps is None:
        print("warning: timing waveform arrays are unavailable; timing example not generated", file=sys.stderr)
        return

    t_ns = np.asarray(dataset.timing_relative_time_ps, dtype=np.float64) / 1000.0
    waves = np.asarray(dataset.timing_windows_mV[event], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    ax.plot(t_ns, waves[0], linewidth=1.5, label="Detector 1")
    ax.plot(t_ns, waves[1], linewidth=1.5, label="Detector 2")
    ax.axvline(0.0, linestyle="--", linewidth=1.0, label="native LED anchor")
    ax.set_xlabel("Time relative to native LED anchor [ns]")
    ax.set_ylabel("Voltage [mV]")
    ax.set_title(f"Timing-channel waveform pair · event {event}")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=3)
    _save(fig, output / "data_timing_waveform_example.pdf", dpi)


def plot_led_cfd_example(
    dataset: PreparedDataset, event: int, output: Path, dpi: int, detector: int = 0
) -> None:
    if dataset.energy_led_time_fs is None or dataset.energy_cfd_time_fs is None:
        raise RuntimeError("Prepared dataset has no energy LED/CFD timestamps")
    anchors = dataset.energy_window_anchor_time_fs
    if anchors is None:
        anchors = dataset.window_anchor_time_fs
    if anchors is None:
        raise RuntimeError("Prepared dataset has no energy window-anchor timestamps")

    t_ps = np.asarray(dataset.relative_time_ps, dtype=np.float64)
    t_ns = t_ps / 1000.0
    y = np.asarray(dataset.windows_mV[event, detector], dtype=np.float64)

    led_threshold = _energy_led_threshold(dataset)
    cfd_fraction = _energy_cfd_fraction(dataset)
    amplitude = float(np.asarray(dataset.amplitude_mV)[event, detector])
    cfd_threshold = amplitude * cfd_fraction

    anchor_fs = float(np.asarray(anchors)[event, detector])
    led_ns = (
        float(np.asarray(dataset.energy_led_time_fs)[event, detector]) - anchor_fs
    ) / 1.0e6
    cfd_ns = (
        float(np.asarray(dataset.energy_cfd_time_fs)[event, detector]) - anchor_fs
    ) / 1.0e6

    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    ax.plot(t_ns, y, linewidth=1.6, label="raw native samples")
    ax.scatter(t_ns, y, s=10, zorder=3)
    ax.axhline(led_threshold, linestyle="--", linewidth=1.1,
               label=f"LED = {led_threshold:g} mV")
    ax.axhline(cfd_threshold, linestyle=":", linewidth=1.2,
               label=f"CFD = {100*cfd_fraction:g}% of amplitude")
    ax.axvline(led_ns, linestyle="--", linewidth=1.1, label="LED crossing")
    ax.axvline(cfd_ns, linestyle=":", linewidth=1.2, label="CFD crossing")
    ax.set_xlabel("Time relative to native LED anchor [ns]")
    ax.set_ylabel("Voltage [mV]")
    ax.set_title("LED and CFD on a real energy-channel waveform")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=2)
    _save(fig, output / "led_cfd_waveform_schematic.pdf", dpi)


def _selected_window_from_run(
    run_manifest: dict[str, Any], rows: list[dict[str, str]], dataset: PreparedDataset
) -> tuple[float, float] | None:
    file_name = _source_file_name(dataset)
    voltage = _median_voltage(dataset)
    codebooks = run_manifest.get("codebooks", {}) or {}
    file_map = codebooks.get("file", {}) or {}
    file_id = file_map.get(file_name)

    # Prefer a selected energy-to-energy waveform candidate, then energy-to-timing.
    for mode_name in ("energy_to_energy", "energy_to_timing"):
        mode_id = _codebook_id(run_manifest, "mode", mode_name)
        if mode_id is None:
            continue
        candidates = []
        for row in rows:
            if _as_int(row.get("stage")) != 0 or _as_int(row.get("selected")) != 1:
                continue
            if _as_int(row.get("mode_id")) != mode_id:
                continue
            descriptor = _candidate_parameters(run_manifest, _as_int(row.get("candidate_id")))
            if descriptor.get("family") == "multithreshold_svr":
                continue
            if file_id is not None and _as_int(row.get("file_id")) != int(file_id):
                continue
            candidates.append((abs(_as_float(row.get("voltage_V")) - voltage), descriptor))
        if candidates:
            descriptor = min(candidates, key=lambda x: x[0])[1]
            window_id = descriptor.get("window")
            # Candidate descriptor stores the window id, while run manifest stores only
            # materialized window. The actual start/end can often be inferred from id;
            # if not, fall back to materialized limits below.
            if isinstance(window_id, str):
                import re
                match = re.search(r"m(?P<before>[0-9.]+)_p(?P<after>[0-9.]+)", window_id)
                if match:
                    return float(match.group("before")), float(match.group("after"))

    materialized = run_manifest.get("materialized_window_ns", {}) or {}
    if "before" in materialized and "after" in materialized:
        return float(materialized["before"]), float(materialized["after"])
    return None


def plot_windowing_example(
    dataset: PreparedDataset,
    event: int,
    output: Path,
    dpi: int,
    run_manifest: dict[str, Any],
    rows: list[dict[str, str]],
    detector: int = 0,
) -> None:
    if dataset.energy_led_time_fs is None:
        raise RuntimeError("Prepared dataset has no energy LED timestamps")
    anchors = dataset.energy_window_anchor_time_fs
    if anchors is None:
        anchors = dataset.window_anchor_time_fs
    if anchors is None:
        raise RuntimeError("Prepared dataset has no energy window anchors")

    y = np.asarray(dataset.windows_mV[event, detector], dtype=np.float64)
    rel_anchor_ps = np.asarray(dataset.relative_time_ps, dtype=np.float64)
    anchor_fs = float(np.asarray(anchors)[event, detector])
    led_fs = float(np.asarray(dataset.energy_led_time_fs)[event, detector])

    # Shift native sample coordinates so the exact interpolated LED crossing is t=0.
    anchor_minus_led_ps = (anchor_fs - led_fs) / FEMTOSECONDS_PER_PICOSECOND
    t_led_ns = (rel_anchor_ps + anchor_minus_led_ps) / 1000.0
    anchor_ns = anchor_minus_led_ps / 1000.0

    chosen_window = _selected_window_from_run(run_manifest, rows, dataset)

    fig, ax = plt.subplots(figsize=(9.4, 4.2))
    ax.plot(t_led_ns, y, linewidth=1.35, label="native acquired waveform")
    ax.scatter(t_led_ns, y, s=13, zorder=3, label="acquired samples")
    ax.axvline(0.0, linestyle="--", linewidth=1.2, label="interpolated LED")
    ax.axvline(anchor_ns, linestyle=":", linewidth=1.2, label="nearest native anchor")

    if chosen_window is not None:
        before_ns, after_ns = chosen_window
        ax.axvspan(-before_ns, after_ns, alpha=0.10,
                   label=f"ML window [−{before_ns:g}, +{after_ns:g}] ns")

    # Zoom around the physically relevant area while retaining the full selected window
    # when it is compact enough for a slide.
    if chosen_window is not None and sum(chosen_window) <= 30:
        ax.set_xlim(-chosen_window[0] - 0.5, chosen_window[1] + 0.5)
    else:
        left = max(float(np.min(t_led_ns)), -5.0)
        right = min(float(np.max(t_led_ns)), 12.0)
        if right > left:
            ax.set_xlim(left, right)

    ax.set_xlabel("Time relative to interpolated LED [ns]")
    ax.set_ylabel("Voltage [mV]")
    ax.set_title("Native-grid windowing: interpolation only defines the anchor")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=2)
    _save(fig, output / "windowing_native_grid.pdf", dpi)


# -----------------------------------------------------------------------------
# Multithreshold examples and threshold-count scan
# -----------------------------------------------------------------------------
def _selected_mt_descriptor(
    run_manifest: dict[str, Any],
    rows: list[dict[str, str]],
    dataset: PreparedDataset,
    preferred_mode: str = "energy_to_energy",
) -> dict[str, Any] | None:
    mt_id = _codebook_id(run_manifest, "model", "multithreshold_svr")
    if mt_id is None:
        return None
    mode_id = _codebook_id(run_manifest, "mode", preferred_mode)
    if mode_id is None:
        return None

    file_name = _source_file_name(dataset)
    file_id = (run_manifest.get("codebooks", {}) or {}).get("file", {}).get(file_name)
    voltage = _median_voltage(dataset)

    matches: list[tuple[float, float, dict[str, Any]]] = []
    for row in rows:
        if _as_int(row.get("stage")) != 0:
            continue
        if _as_int(row.get("model_id")) != mt_id or _as_int(row.get("mode_id")) != mode_id:
            continue
        if _as_int(row.get("selected")) != 1:
            continue
        if file_id is not None and _as_int(row.get("file_id")) != int(file_id):
            continue
        descriptor = _candidate_parameters(run_manifest, _as_int(row.get("candidate_id")))
        if not descriptor:
            continue
        dv = abs(_as_float(row.get("voltage_V")) - voltage)
        ctr = _as_float(row.get("ctr_ps"), float("inf"))
        matches.append((dv, ctr, descriptor))

    if not matches:
        return None
    return min(matches, key=lambda x: (x[0], x[1]))[2]


def _valid_thresholds_for_waveform(
    t_ps: np.ndarray, y: np.ndarray, thresholds: Iterable[float]
) -> list[tuple[float, float]]:
    output: list[tuple[float, float]] = []
    for threshold in thresholds:
        crossing = _last_rising_crossing_before_peak(t_ps, y, float(threshold))
        if np.isfinite(crossing):
            output.append((float(threshold), float(crossing)))
    return output


def plot_multithreshold_example(
    dataset: PreparedDataset,
    event: int,
    output: Path,
    dpi: int,
    run_manifest: dict[str, Any],
    detector: int = 0,
) -> None:
    thresholds = _all_mt_thresholds(run_manifest)
    if not thresholds:
        print("warning: no multithreshold candidate thresholds found in run manifest", file=sys.stderr)
        return

    t_ps = np.asarray(dataset.relative_time_ps, dtype=np.float64)
    t_ns = t_ps / 1000.0
    y = np.asarray(dataset.windows_mV[event, detector], dtype=np.float64)
    crossings = _valid_thresholds_for_waveform(t_ps, y, thresholds)

    fig, ax = plt.subplots(figsize=(9.4, 4.5))
    ax.plot(t_ns, y, linewidth=1.55, label="energy waveform")
    ax.scatter(t_ns, y, s=9, zorder=3)
    for threshold, crossing_ps in crossings:
        ax.axhline(threshold, linewidth=0.85, alpha=0.55)
        ax.scatter([crossing_ps / 1000.0], [threshold], s=30, zorder=4)
        ax.annotate(
            f"{threshold:g} mV",
            xy=(crossing_ps / 1000.0, threshold),
            xytext=(5, 3), textcoords="offset points", fontsize=8,
        )
    ax.set_xlabel("Time relative to native LED anchor [ns]")
    ax.set_ylabel("Voltage [mV]")
    ax.set_title("Fixed-threshold timing on a real energy-channel waveform")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    _save(fig, output / "multithreshold_waveform_crossings.pdf", dpi)


def plot_best_multithresholds(
    dataset: PreparedDataset,
    event: int,
    output: Path,
    dpi: int,
    run_manifest: dict[str, Any],
    rows: list[dict[str, str]],
    preferred_mode: str,
) -> None:
    descriptor = _selected_mt_descriptor(run_manifest, rows, dataset, preferred_mode)
    if descriptor is None:
        print(
            f"warning: no selected multithreshold candidate found for {preferred_mode}; "
            "best-threshold figure not generated",
            file=sys.stderr,
        )
        return

    thresholds = [float(v) for v in descriptor.get("thresholds_mV", [])]
    if not thresholds:
        return

    t_ps = np.asarray(dataset.relative_time_ps, dtype=np.float64)
    t_ns = t_ps / 1000.0
    waves = np.asarray(dataset.windows_mV[event], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(9.4, 4.5))
    for detector in range(2):
        ax.plot(t_ns, waves[detector], linewidth=1.45, label=f"Detector {detector + 1}")
        for threshold, crossing_ps in _valid_thresholds_for_waveform(t_ps, waves[detector], thresholds):
            ax.scatter([crossing_ps / 1000.0], [threshold], s=28, zorder=4)
    for threshold in thresholds:
        ax.axhline(threshold, linewidth=0.9, alpha=0.55)

    voltage = _median_voltage(dataset)
    kernel = descriptor.get("kernel", "")
    title = f"Selected fixed thresholds · {_mode_label(preferred_mode)}"
    if np.isfinite(voltage):
        title += f" · {voltage:g} V"
    if kernel:
        title += f" · {kernel} SVR"
    ax.set_title(title)
    ax.set_xlabel("Time relative to native energy-LED anchor [ns]")
    ax.set_ylabel("Voltage [mV]")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=2)
    _save(fig, output / "results_multithreshold_best_thresholds.pdf", dpi)


def _mt_best_by_count(
    rows: list[dict[str, str]], run_manifest: dict[str, Any]
) -> dict[str, dict[float, dict[int, float]]]:
    """Return mode -> voltage -> threshold_count -> best development CTR.

    The compact results.csv stores every development candidate as stage=0.
    We intentionally use these rows rather than blind stage=1: threshold count is
    a hyperparameter and therefore must be visualized from the selection data.
    """
    mt_id = _codebook_id(run_manifest, "model", "multithreshold_svr")
    if mt_id is None:
        raise RuntimeError("Run manifest has no multithreshold_svr model code")

    mode_codebook = (run_manifest.get("codebooks", {}) or {}).get("mode", {}) or {}
    id_to_mode = {int(value): str(name) for name, value in mode_codebook.items()}
    energy_modes = {"energy_to_energy", "energy_to_timing"}

    output: dict[str, dict[float, dict[int, float]]] = {}
    for row in rows:
        if _as_int(row.get("stage")) != 0:
            continue
        if _as_int(row.get("model_id")) != mt_id:
            continue
        mode = id_to_mode.get(_as_int(row.get("mode_id")), "")
        if mode not in energy_modes:
            continue
        descriptor = _candidate_parameters(run_manifest, _as_int(row.get("candidate_id")))
        thresholds = descriptor.get("thresholds_mV", []) if descriptor else []
        if not isinstance(thresholds, list) or not thresholds:
            continue
        count = len(thresholds)
        voltage = _as_float(row.get("voltage_V"))
        ctr = _as_float(row.get("ctr_ps"))
        if not np.isfinite(voltage) or not np.isfinite(ctr):
            continue
        current = output.setdefault(mode, {}).setdefault(voltage, {}).get(count)
        if current is None or ctr < current:
            output[mode][voltage][count] = ctr
    return output


def _plot_threshold_count_axis(
    ax: plt.Axes, mode: str, voltage_data: dict[float, dict[int, float]]
) -> None:
    for voltage in sorted(voltage_data):
        points = sorted(voltage_data[voltage].items())
        if not points:
            continue
        x = [int(k) for k, _ in points]
        y = [float(v) for _, v in points]
        ax.plot(x, y, marker="o", linewidth=1.35, label=f"{voltage:g} V")
    all_counts = sorted({count for values in voltage_data.values() for count in values})
    if all_counts:
        ax.set_xticks(all_counts)
    ax.set_xlabel("Number of fixed thresholds")
    ax.set_ylabel("Best validation s-CTR [ps]")
    ax.set_title(_mode_label(mode))
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=min(5, max(1, len(voltage_data))))


def plot_threshold_count_performance(
    rows: list[dict[str, str]], run_manifest: dict[str, Any], output: Path, dpi: int
) -> None:
    data = _mt_best_by_count(rows, run_manifest)
    modes = [mode for mode in ("energy_to_energy", "energy_to_timing") if mode in data]
    if not modes:
        raise RuntimeError(
            "No stage=0 multithreshold candidates for energy_to_energy/energy_to_timing "
            "were found in results.csv"
        )

    # Combined wide figure expected by the current LaTeX presentation.
    fig, axes = plt.subplots(len(modes), 1, figsize=(9.5, 3.8 * len(modes)), squeeze=False)
    for ax, mode in zip(axes[:, 0], modes):
        _plot_threshold_count_axis(ax, mode, data[mode])
    fig.suptitle(
        "Fixed-threshold SVR: performance versus readout complexity\n"
        "best development candidate at each threshold count",
        fontsize=13,
    )
    _save(fig, output / "results_multithreshold_ctr_vs_threshold_count.pdf", dpi)

    # Also save one wide figure per energy mode; handy if the Beamer slide is split later.
    for mode in modes:
        fig, ax = plt.subplots(figsize=(9.5, 4.2))
        _plot_threshold_count_axis(ax, mode, data[mode])
        _save(
            fig,
            output / f"results_multithreshold_ctr_vs_threshold_count_{mode}.pdf",
            dpi,
        )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the real-signal and multithreshold figures used by the "
            "CTR waveform-ML Beamer presentation."
        )
    )
    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help="Experiment run directory containing results.csv and manifest.json",
    )
    parser.add_argument(
        "--prepared",
        type=Path,
        default=None,
        help=(
            "Prepared dataset directory, or parent containing one subdirectory per file. "
            "Default: use prepared_dir recorded in the run manifest."
        ),
    )
    parser.add_argument(
        "--voltage",
        type=float,
        default=None,
        help="Voltage used to choose an example dataset when --prepared is a parent directory",
    )
    parser.add_argument(
        "--event-index",
        type=int,
        default=None,
        help="Prepared-dataset row to plot; default chooses a typical-amplitude finite event",
    )
    parser.add_argument(
        "--mt-mode",
        choices=("energy_to_energy", "energy_to_timing"),
        default="energy_to_energy",
        help="Mode used to select the best-threshold waveform example",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT / "results" / "presentation" / "plots",
        help="Presentation plot directory",
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run.resolve()
    if run_dir.is_file():
        run_dir = run_dir.parent
    results_path = run_dir / "results.csv"
    manifest_path = run_dir / "manifest.json"
    if not results_path.is_file():
        raise FileNotFoundError(f"results.csv not found: {results_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"run manifest.json not found: {manifest_path}")

    run_manifest = _read_json(manifest_path)
    rows = _read_csv(results_path)

    prepared = args.prepared
    if prepared is None:
        recorded = run_manifest.get("prepared_dir")
        if not recorded:
            raise ValueError("Run manifest has no prepared_dir; pass --prepared explicitly")
        prepared = Path(str(recorded))
    if not prepared.is_absolute():
        # First interpret relative paths from waveform_analysis, matching experiment configs.
        prepared = (PROJECT / prepared).resolve()

    dataset = _choose_dataset(prepared, args.voltage)
    event = _representative_event(dataset, args.event_index)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    print(f"run:      {run_dir}")
    print(f"prepared: {dataset.directory}")
    print(f"voltage:  {_median_voltage(dataset):g} V")
    print(f"event:    {event}")
    print(f"output:   {output}")

    plot_energy_waveform_example(dataset, event, output, args.dpi)
    plot_timing_waveform_example(dataset, event, output, args.dpi)
    plot_led_cfd_example(dataset, event, output, args.dpi)
    plot_windowing_example(dataset, event, output, args.dpi, run_manifest, rows)
    plot_multithreshold_example(dataset, event, output, args.dpi, run_manifest)
    plot_threshold_count_performance(rows, run_manifest, output, args.dpi)
    plot_best_multithresholds(
        dataset, event, output, args.dpi, run_manifest, rows, args.mt_mode
    )


if __name__ == "__main__":
    main()
