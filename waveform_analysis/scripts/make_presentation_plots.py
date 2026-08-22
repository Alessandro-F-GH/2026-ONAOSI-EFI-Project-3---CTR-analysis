from __future__ import annotations

"""Generate presentation-only figures from the current CTR-analysis repository.

Intended location in the repository:
    waveform_analysis/scripts/make_presentation_plots.py

The script does NOT rerun ML training. Final presentation CTR values are recomputed from saved blind residual artifacts with the repository Gaussian fitter.
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
    xai_energy_to_energy_linear_svr_waveform_importance.pdf
    xai_energy_to_energy_cnn_waveform_importance.pdf
    xai_timing_to_timing_linear_svr_waveform_importance.pdf
    xai_timing_to_timing_cnn_waveform_importance.pdf
The XAI waveform background uses contrast-enhanced discrete importance levels with fixed-width time averaging.
"""

import argparse
import csv
import json
import math
import re
import hashlib
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import numpy as np

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "figure.titlesize": 17,
})

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
from ml_pipeline.metrics import fit_times_ps
from ml_pipeline.reporting import short_model_label, short_mode_label, format_ctr
from ml_pipeline.prepared_data import input_variant_dataset_view
from ml_pipeline.prediction import prediction_window_dataset_view
from ml_pipeline.study_config import CHANNEL_MODES


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
    # tight_layout() is incompatible with some figures that use auxiliary axes
    # (e.g. colorbars) or constrained_layout.  Respect constrained_layout when
    # it is enabled and avoid the warning/noisy layout fallback.
    if not fig.get_constrained_layout():
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



def _default_colors(count: int) -> list[str]:
    """Return presentation-friendly colors from Matplotlib's active color cycle."""
    palette = list(plt.rcParams["axes.prop_cycle"].by_key().get("color", []))
    if not palette:
        palette = [f"C{i}" for i in range(max(1, count))]
    return [palette[i % len(palette)] for i in range(count)]


def _rising_edge_xlim(
    time_ps: np.ndarray,
    waveforms_mV: np.ndarray,
    *,
    low_fraction: float = 0.03,
    high_fraction: float = 0.50,
    pad_before_ns: float = 0.6,
    pad_after_ns: float = 1.5,
    fallback_before_ns: float = 2.0,
    fallback_after_ns: float = 10.0,
) -> tuple[float, float]:
    """Compact x-range around the first informative pulse rise.

    This is presentation-only.  It deliberately avoids plotting the long pulse
    tail because the timing information discussed in the slides is concentrated
    around the early rising edge.
    """
    t_ps = np.asarray(time_ps, dtype=np.float64)
    waves = np.asarray(waveforms_mV, dtype=np.float64)
    if waves.ndim == 1:
        waves = waves[None, :]

    low_crossings: list[float] = []
    high_crossings: list[float] = []
    for y in waves:
        if y.size != t_ps.size or not np.any(np.isfinite(y)):
            continue
        peak = float(np.nanmax(y))
        baseline = float(np.nanmedian(y[: max(3, min(100, y.size // 10 or 3))]))
        amplitude = peak - baseline
        if not np.isfinite(amplitude) or amplitude <= 0:
            continue
        low = baseline + low_fraction * amplitude
        high = baseline + high_fraction * amplitude
        t_low = _first_rising_crossing(t_ps, y, low)
        t_high = _first_rising_crossing(t_ps, y, high)
        if np.isfinite(t_low):
            low_crossings.append(t_low)
        if np.isfinite(t_high):
            high_crossings.append(t_high)

    t_min_ns = float(np.nanmin(t_ps)) / 1000.0
    t_max_ns = float(np.nanmax(t_ps)) / 1000.0
    if low_crossings and high_crossings:
        left = min(low_crossings) / 1000.0 - pad_before_ns
        right = max(high_crossings) / 1000.0 + pad_after_ns
    else:
        left = max(t_min_ns, -fallback_before_ns)
        right = min(t_max_ns, fallback_after_ns)

    left = max(left, t_min_ns)
    right = min(right, t_max_ns)
    if not np.isfinite(left) or not np.isfinite(right) or right <= left:
        return t_min_ns, t_max_ns
    if right - left < 1.0:
        center = 0.5 * (left + right)
        left = max(t_min_ns, center - 0.5)
        right = min(t_max_ns, center + 0.5)
    return left, right


def _crossing_xlim(
    crossings_ps: Iterable[float],
    time_ps: np.ndarray,
    *,
    pad_before_ns: float = 0.45,
    pad_after_ns: float = 0.85,
) -> tuple[float, float]:
    finite = np.asarray(
        [float(v) for v in crossings_ps if np.isfinite(float(v))],
        dtype=np.float64,
    )
    t = np.asarray(time_ps, dtype=np.float64)
    t_min_ns = float(np.nanmin(t)) / 1000.0
    t_max_ns = float(np.nanmax(t)) / 1000.0
    if finite.size == 0:
        return max(t_min_ns, -1.5), min(t_max_ns, 4.0)
    left = max(t_min_ns, float(np.min(finite)) / 1000.0 - pad_before_ns)
    right = min(t_max_ns, float(np.max(finite)) / 1000.0 + pad_after_ns)
    if right - left < 1.0:
        center = 0.5 * (left + right)
        left = max(t_min_ns, center - 0.5)
        right = min(t_max_ns, center + 0.5)
    return left, right


# -----------------------------------------------------------------------------
# Real waveform examples
# -----------------------------------------------------------------------------

def plot_energy_waveform_example(
    dataset: PreparedDataset, event: int, output: Path, dpi: int
) -> None:
    t_ps = np.asarray(dataset.relative_time_ps, dtype=np.float64)
    t_ns = t_ps / 1000.0
    waves = np.asarray(dataset.windows_mV[event], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    ax.plot(t_ns, waves[0], linewidth=1.7, label="Detector 1")
    ax.set_xlabel("Time [ns]")
    ax.set_ylabel("Voltage [mV]")
    ax.set_title("Energy-channel signal example")
    ax.grid(alpha=0.20)
    _save(fig, output / "data_energy_waveform_example.pdf", dpi)


def plot_timing_waveform_example(
    dataset: PreparedDataset, event: int, output: Path, dpi: int
) -> None:
    if dataset.timing_windows_mV is None or dataset.timing_relative_time_ps is None:
        print("warning: timing waveform arrays are unavailable; timing example not generated", file=sys.stderr)
        return

    t_ps = np.asarray(dataset.timing_relative_time_ps, dtype=np.float64)
    t_ns = t_ps / 1000.0
    waves = np.asarray(dataset.timing_windows_mV[event], dtype=np.float64)
    left, right = _rising_edge_xlim(
        t_ps,
        waves,
        low_fraction=0.03,
        high_fraction=0.60,
        pad_before_ns=0.35,
        pad_after_ns=0.8,
        fallback_before_ns=1.0,
        fallback_after_ns=4.0,
    )

    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    ax.plot(t_ns, waves[0], linewidth=1.7, label="Detector 1")
    ax.set_xlabel("Time [ns]")
    ax.set_ylabel("Voltage [mV]")
    ax.set_title("Timing-channel signal example")
    ax.grid(alpha=0.20)
    _save(fig, output / "data_timing_waveform_example.pdf", dpi)

def plot_led_cfd_example(
    dataset: PreparedDataset,
    event: int,
    output: Path,
    dpi: int,
) -> None:
    if dataset.energy_led_time_fs is None or dataset.energy_cfd_time_fs is None:
        raise RuntimeError("Prepared dataset has no energy LED/CFD timestamps")

    anchors = dataset.energy_window_anchor_time_fs
    if anchors is None:
        anchors = dataset.window_anchor_time_fs
    if anchors is None:
        raise RuntimeError("Prepared dataset has no energy window-anchor timestamps")

    # ------------------------------------------------------------------
    # Waveforms
    # ------------------------------------------------------------------
    t_ps = np.asarray(dataset.relative_time_ps, dtype=np.float64)
    t_ns = t_ps / 1000.0

    waves = np.asarray(dataset.windows_mV[event], dtype=np.float64)
    y1 = waves[0]
    y2 = waves[1]

    # ------------------------------------------------------------------
    # LED / CFD settings
    # ------------------------------------------------------------------
    led_threshold = _energy_led_threshold(dataset)
    cfd_fraction = _energy_cfd_fraction(dataset)

    amplitude = np.asarray(dataset.amplitude_mV, dtype=np.float64)
    cfd_threshold_1 = float(amplitude[event, 0]) * cfd_fraction
    cfd_threshold_2 = float(amplitude[event, 1]) * cfd_fraction

    anchor_fs = np.asarray(anchors, dtype=np.float64)[event]
    led_fs = np.asarray(dataset.energy_led_time_fs, dtype=np.float64)[event]
    cfd_fs = np.asarray(dataset.energy_cfd_time_fs, dtype=np.float64)[event]

    # ------------------------------------------------------------------
    # Common event-time reference
    #
    # Prepared waveforms are expressed relative to their own native anchors.
    # Shift both onto the same time axis so their actual timing separation
    # is visible.
    # ------------------------------------------------------------------
    reference_fs = 0.5 * (float(led_fs[0]) + float(led_fs[1]))

    x1_ns = t_ns + (float(anchor_fs[0]) - reference_fs) / 1.0e6
    x2_ns = t_ns + (float(anchor_fs[1]) - reference_fs) / 1.0e6

    led_1_ns = (float(led_fs[0]) - reference_fs) / 1.0e6
    led_2_ns = (float(led_fs[1]) - reference_fs) / 1.0e6

    cfd_1_ns = (float(cfd_fs[0]) - reference_fs) / 1.0e6
    cfd_2_ns = (float(cfd_fs[1]) - reference_fs) / 1.0e6

    # ------------------------------------------------------------------
    # Compact presentation window around all four crossings
    # ------------------------------------------------------------------
    crossings_ns = np.asarray(
        [led_1_ns, led_2_ns, cfd_1_ns, cfd_2_ns],
        dtype=np.float64,
    )

    left = float(np.min(crossings_ns)) - 0.20
    right = float(np.max(crossings_ns)) + 0.10

    # Do not exceed available waveform support.
    left = max(left, float(min(np.nanmin(x1_ns), np.nanmin(x2_ns))))
    right = min(right, float(max(np.nanmax(x1_ns), np.nanmax(x2_ns))))

    local1 = (x1_ns >= left) & (x1_ns <= right)
    local2 = (x2_ns >= left) & (x2_ns <= right)

    local_min = min(
        0.0,
        float(np.nanmin(y1[local1])) if np.any(local1) else float(np.nanmin(y1)),
        float(np.nanmin(y2[local2])) if np.any(local2) else float(np.nanmin(y2)),
    )

    local_max = max(
        float(np.nanmax(y1[local1])) if np.any(local1) else float(np.nanmax(y1)),
        float(np.nanmax(y2[local2])) if np.any(local2) else float(np.nanmax(y2)),
        led_threshold,
        cfd_threshold_1,
        cfd_threshold_2,
    )

    amplitude_span = max(1.0, local_max - local_min)

    # Extra vertical space for Δt annotations.
    ymin = local_min - 0.06 * amplitude_span
    ymax = local_max - 0.1 * amplitude_span

    # ------------------------------------------------------------------
    # Colors
    # ------------------------------------------------------------------
    waveform_1_color = "tab:blue"
    waveform_2_color = "tab:orange"

    # Crossing/estimator colors deliberately differ from waveform colors.
    led_color = "crimson"
    cfd_color = "forestgreen"

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.2, 4.7))

    ax.plot(
        x1_ns,
        y1,
        linewidth=1.9,
        color=waveform_1_color,
        label="Detector 1 waveform",
        zorder=2,
    )
    ax.plot(
        x2_ns,
        y2,
        linewidth=1.9,
        color=waveform_2_color,
        label="Detector 2 waveform",
        zorder=2,
    )

    # ------------------------------------------------------------------
    # LED threshold + crossings
    # ------------------------------------------------------------------
    ax.axhline(
        led_threshold,
        linestyle="--",
        linewidth=1.25,
        color=led_color,
        alpha=0.85,
        label=f"LED threshold ({led_threshold:g} mV)",
        zorder=1,
    )

    ax.scatter(
        [led_1_ns, led_2_ns],
        [led_threshold, led_threshold],
        s=60,
        marker="o",
        color=led_color,
        edgecolors="white",
        linewidths=0.8,
        zorder=6,
        label="LED crossings",
    )

    # ------------------------------------------------------------------
    # CFD crossings
    # ------------------------------------------------------------------
    ax.scatter(
        [cfd_1_ns, cfd_2_ns],
        [cfd_threshold_1, cfd_threshold_2],
        s=60,
        marker="s",
        color=cfd_color,
        edgecolors="white",
        linewidths=0.8,
        zorder=6,
        label="CFD crossings",
    )

    # ------------------------------------------------------------------
    # Vertical guides from the crossing points
    # ------------------------------------------------------------------
    for x in (led_1_ns, led_2_ns):
        ax.vlines(
            x,
            ymin,
            led_threshold,
            color=led_color,
            linestyle="--",
            linewidth=2,
            alpha=0.45,
            zorder=0,
        )

    for x, level in (
        (cfd_1_ns, cfd_threshold_1),
        (cfd_2_ns, cfd_threshold_2),
    ):
        ax.vlines(
            x,
            ymin,
            level,
            color=cfd_color,
            linestyle=":",
            linewidth=2,
            alpha=0.45,
            zorder=0,
        )

    # ------------------------------------------------------------------
    # Explicit Δt_LED and Δt_CFD annotations
    # ------------------------------------------------------------------
    y_led_arrow = 1
    y_cfd_arrow = 2

    ax.annotate(
        "",
        xy=(led_1_ns, y_led_arrow),
        xytext=(led_2_ns, y_led_arrow),
        arrowprops=dict(
            arrowstyle="<->",
            color=led_color,
            linewidth=1.7,
        ),
        zorder=5,
    )

    ax.text(
        0.5 * (led_1_ns + led_2_ns),
        y_led_arrow + 0.025 * amplitude_span,
        r"$\Delta t_{\mathrm{LED}}$",
        color=led_color,
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight="bold",
    )

    ax.annotate(
        "",
        xy=(cfd_1_ns, y_cfd_arrow),
        xytext=(cfd_2_ns, y_cfd_arrow),
        arrowprops=dict(
            arrowstyle="<->",
            color=cfd_color,
            linewidth=1.7,
        ),
        zorder=5,
    )

    ax.text(
        0.5 * (cfd_1_ns + cfd_2_ns),
        y_cfd_arrow + 0.025 * amplitude_span,
        r"$\Delta t_{\mathrm{CFD}}$",
        color=cfd_color,
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight="bold",
    )

    # ------------------------------------------------------------------
    # Final formatting
    # ------------------------------------------------------------------
    ax.set_xlim(left, right)
    ax.set_ylim(ymin, ymax)

    ax.set_xlabel("Time [ns]")
    ax.set_ylabel("Voltage [mV]")
    ax.set_title("LED/CFD timing example")

    ax.grid(alpha=0.18)

    ax.legend(
        frameon=False,
        loc="upper left",
        ncol=2,
        fontsize=9,
        columnspacing=1.0,
        handlelength=2.0,
    )

    _save(
        fig,
        output / "led_cfd_waveform_schematic.pdf",
        dpi,
    )

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

    # Exact interpolated LED becomes t=0; waveform samples remain on the native grid.
    anchor_minus_led_ps = (anchor_fs - led_fs) / FEMTOSECONDS_PER_PICOSECOND
    t_led_ns = (rel_anchor_ps + anchor_minus_led_ps) / 1000.0
    anchor_ns = anchor_minus_led_ps / 1000.0
    chosen_window = _selected_window_from_run(run_manifest, rows, dataset)

    # Presentation figure: one close-up for the discrete shift and one compact
    # rising-edge view for the ML window.  The long waveform tail is omitted.
    fig, (ax_local, ax_window) = plt.subplots(
        1, 2, figsize=(10.2, 4.0), gridspec_kw={"width_ratios": [0.9, 1.35]}
    )

    # Left: show the native grid around the interpolation anchor.
    local_half_width_ns = 0.075
    local = np.abs(t_led_ns) <= local_half_width_ns
    if np.count_nonzero(local) < 8:
        nearest = int(np.argmin(np.abs(t_led_ns)))
        lo = max(0, nearest - 6)
        hi = min(t_led_ns.size, nearest + 7)
        local = np.zeros(t_led_ns.size, dtype=bool)
        local[lo:hi] = True
    ax_local.plot(t_led_ns[local], y[local], linewidth=1.3)
    ax_local.scatter(t_led_ns[local], y[local], s=28, zorder=4, label="native samples")
    ax_local.axvline(0.0, linestyle="--", linewidth=1.2, label="interpolated LED")
    ax_local.axvline(anchor_ns, linestyle=":", linewidth=1.2, label="nearest sample")
    ax_local.set_title("Anchor shift")
    ax_local.set_xlabel("LED-relative time [ns]")
    ax_local.set_ylabel("Voltage [mV]")
    ax_local.grid(alpha=0.20)
    ax_local.legend(frameon=False, fontsize=8)

    # Right: show only the informative early waveform region, not the full pulse.
    left, right = _rising_edge_xlim(
        rel_anchor_ps + anchor_minus_led_ps,
        y,
        low_fraction=0.02,
        high_fraction=0.55,
        pad_before_ns=0.8,
        pad_after_ns=1.4,
        fallback_before_ns=2.0,
        fallback_after_ns=8.0,
    )
    if chosen_window is not None:
        before_ns, after_ns = chosen_window
        left = min(left, max(-before_ns, -2.5))
        right = min(right, 12.0)
        visible_left = max(left, -before_ns)
        visible_right = min(right, after_ns)
        if visible_right > visible_left:
            label = f"ML window [−{before_ns:g}, +{after_ns:g}] ns"
            if after_ns > right + 1e-9:
                label += " (cropped view)"
            ax_window.axvspan(visible_left, visible_right, alpha=0.12, label=label)

    ax_window.plot(t_led_ns, y, linewidth=1.6, label="native waveform")
    ax_window.axvline(0.0, linestyle="--", linewidth=1.1, label="LED anchor")
    ax_window.set_xlim(left, right)
    ax_window.set_title("Window passed to ML")
    ax_window.set_xlabel("LED-relative time [ns]")
    ax_window.grid(alpha=0.20)
    ax_window.legend(frameon=False, fontsize=8)

    fig.suptitle("Discrete windowing: shift the grid reference, not the samples", fontsize=13)
    _save(fig, output / "windowing_native_grid.pdf", dpi)



# -----------------------------------------------------------------------------
# XAI waveform examples: waveform background + normalized importance profile
# -----------------------------------------------------------------------------
def _artifact_model_key(name: str) -> str:
    """Match the model-key sanitization used by ml_pipeline.study."""
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(name)).strip("_") or "model"


def _dataset_file_id(
    dataset: PreparedDataset,
    run_manifest: dict[str, Any],
) -> int | None:
    """Resolve the run file id corresponding to the chosen prepared dataset."""
    file_map = (run_manifest.get("codebooks", {}) or {}).get("file", {}) or {}
    source_name = _source_file_name(dataset)
    if source_name in file_map:
        return int(file_map[source_name])

    # Robust fallback: choose the file whose filename voltage is closest to the
    # selected prepared dataset voltage.
    voltage = _median_voltage(dataset)
    if not np.isfinite(voltage):
        return None

    candidates: list[tuple[float, int]] = []
    for name, file_id in file_map.items():
        match = re.search(r"(?P<v>\d+(?:\.\d+)?)V", str(name))
        if match is None:
            continue
        candidates.append(
            (abs(float(match.group("v")) - float(voltage)), int(file_id))
        )
    return min(candidates, default=(float("inf"), -1))[1] if candidates else None


def _final_model_meta(
    dataset: PreparedDataset,
    run_manifest: dict[str, Any],
    *,
    mode: str,
    model: str,
) -> dict[str, Any] | None:
    """Find selected final-model metadata for dataset/mode/model."""
    file_id = _dataset_file_id(dataset, run_manifest)
    if file_id is None:
        return None

    matches: list[dict[str, Any]] = []
    for value in (run_manifest.get("final_models", {}) or {}).values():
        if not isinstance(value, dict):
            continue
        if int(value.get("file_id", -1)) != int(file_id):
            continue
        if str(value.get("mode", "")) != mode:
            continue
        if str(value.get("model", "")) != model:
            continue
        matches.append(value)

    if not matches:
        return None
    if len(matches) > 1:
        # There should normally be exactly one selected final model.
        matches.sort(key=lambda item: int(item.get("candidate_id", -1)), reverse=True)
    return dict(matches[0])


def _xai_profile_from_final_cache(
    run_dir: Path,
    dataset: PreparedDataset,
    run_manifest: dict[str, Any],
    *,
    mode: str,
    model: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    """Load the selected model's numerical XAI profile from final cache.

    study.py stores xai_time_ps/xai_importance in the final-cache NPZ.  A run can
    contain stale cache files from previous candidate selections, so candidates
    are ranked by agreement with the selected final model's physical window.
    """
    meta = _final_model_meta(
        dataset,
        run_manifest,
        mode=mode,
        model=model,
    )
    if meta is None:
        return None

    file_id = int(meta["file_id"])
    mode_id = int(meta.get(
        "mode_id",
        _codebook_id(run_manifest, "mode", mode) or -1,
    ))
    if mode_id < 0:
        return None

    model_key = _artifact_model_key(model)
    artifacts_root = run_dir / "artifacts"

    cache_roots = sorted(
        [
            path
            for path in artifacts_root.glob("final_cache_v*")
            if path.is_dir()
        ],
        key=lambda path: path.name,
        reverse=True,
    )

    window = meta.get("window", {}) or {}
    before_ns = float(window.get("before_ns", 0.0))
    after_ns = float(window.get("after_ns", 0.0))
    expected_left_ns = -before_ns
    expected_right_ns = after_ns

    candidates: list[
        tuple[float, float, Path, np.ndarray, np.ndarray]
    ] = []

    for cache_root in cache_roots:
        model_dir = (
            cache_root
            / f"f{file_id}_m{mode_id}"
            / model_key
        )
        if not model_dir.is_dir():
            continue

        for npz_path in model_dir.glob("*.npz"):
            try:
                with np.load(npz_path) as arrays:
                    if (
                        "xai_time_ps" not in arrays
                        or "xai_importance" not in arrays
                    ):
                        continue
                    time_ps = np.asarray(
                        arrays["xai_time_ps"],
                        dtype=np.float64,
                    ).reshape(-1)
                    importance = np.asarray(
                        arrays["xai_importance"],
                        dtype=np.float64,
                    ).reshape(-1)
            except Exception:
                continue

            if (
                time_ps.size < 2
                or importance.size != time_ps.size
                or np.any(~np.isfinite(time_ps))
                or np.any(~np.isfinite(importance))
            ):
                continue

            # Prefer the cache whose XAI time support matches the selected
            # candidate window. mtime breaks ties in favor of the newest cache.
            left_ns = float(np.min(time_ps)) / 1000.0
            right_ns = float(np.max(time_ps)) / 1000.0
            window_error = (
                abs(left_ns - expected_left_ns)
                + abs(right_ns - expected_right_ns)
            )
            try:
                mtime = float(npz_path.stat().st_mtime)
            except OSError:
                mtime = 0.0

            candidates.append(
                (
                    window_error,
                    -mtime,
                    npz_path,
                    time_ps,
                    importance,
                )
            )

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]))
    _score, _neg_mtime, path, time_ps, importance = candidates[0]

    print(
        f"xai:      {mode} / {model} -> "
        f"{path.relative_to(run_dir)}"
    )
    return time_ps, importance, meta


def _normalize_abs_importance(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = np.abs(values)
    peak = float(np.max(values)) if values.size else 0.0
    if not np.isfinite(peak) or peak <= 0.0:
        return np.zeros_like(values)
    return values / peak


def _contrast_enhanced_unit_values(
    values: np.ndarray | list[float],
    *,
    gamma: float,
    n_levels: int,
) -> np.ndarray:
    """Boost visual separation and quantize to a small number of discrete levels.

    gamma < 1 expands low/mid values so they are not all visually washed out.
    Quantization then creates clearly distinct background bands for slides.
    """
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return x
    x = np.clip(x, 0.0, 1.0)

    gamma = float(gamma)
    if not np.isfinite(gamma) or gamma <= 0.0:
        gamma = 1.0
    x = np.power(x, gamma)

    levels = max(2, int(n_levels))
    # Convert to discrete band centers in [0, 1].
    idx = np.minimum((x * levels).astype(int), levels - 1)
    return (idx + 0.5) / levels


def _regional_importance(
    time_ns: np.ndarray,
    importance: np.ndarray,
    *,
    window_ns: float,
) -> list[tuple[float, float, float]]:
    """Fixed-width time windows with mean absolute normalized importance."""
    time_ns = np.asarray(time_ns, dtype=np.float64).reshape(-1)
    importance = _normalize_abs_importance(importance)

    if time_ns.size != importance.size or time_ns.size < 2:
        raise ValueError("XAI time and importance arrays must have matching length >= 2")
    if not np.isfinite(window_ns) or float(window_ns) <= 0.0:
        raise ValueError("window_ns must be positive")

    dt = np.diff(time_ns)
    finite_dt = dt[np.isfinite(dt) & (dt > 0)]
    half_step = (
        0.5 * float(np.median(finite_dt))
        if finite_dt.size
        else 0.0
    )
    left = float(time_ns[0]) - half_step
    right = float(time_ns[-1]) + half_step

    window_ns = float(window_ns)
    n_bins = max(1, int(np.ceil((right - left) / window_ns)))
    edges = left + window_ns * np.arange(n_bins + 1, dtype=np.float64)
    if edges[-1] < right:
        edges = np.append(edges, right)
    else:
        edges[-1] = right

    regions: list[tuple[float, float, float]] = []

    for index, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if index == len(edges) - 2:
            mask = (time_ns >= lo) & (time_ns <= hi)
        else:
            mask = (time_ns >= lo) & (time_ns < hi)

        if np.any(mask):
            mean_importance = float(np.mean(np.abs(importance[mask])))
        else:
            center = 0.5 * (lo + hi)
            nearest = int(np.argmin(np.abs(time_ns - center)))
            mean_importance = float(abs(importance[nearest]))

        regions.append((float(lo), float(hi), mean_importance))

    return regions


def _selected_waveform_view_for_xai(
    dataset: PreparedDataset,
    *,
    mode: str,
    meta: dict[str, Any],
) -> PreparedDataset:
    """Rebuild exactly the selected model's channel/variant/window data view."""
    if mode not in CHANNEL_MODES:
        raise ValueError(f"Unsupported XAI mode: {mode}")

    variant = str(meta.get("variant", "raw"))
    source = input_variant_dataset_view(dataset, variant)

    window = meta.get("window", {}) or {}
    if "before_ns" not in window or "after_ns" not in window:
        raise ValueError(
            f"Final model metadata for {mode}/{meta.get('model')} "
            "does not contain before_ns/after_ns"
        )

    input_waveforms, target = CHANNEL_MODES[mode]
    return prediction_window_dataset_view(
        source,
        input_waveforms=input_waveforms,
        target=target,
        before_ns=float(window["before_ns"]),
        after_ns=float(window["after_ns"]),
    )


def plot_xai_waveform_importance(
    run_dir: Path,
    dataset: PreparedDataset,
    event: int,
    run_manifest: dict[str, Any],
    output: Path,
    dpi: int,
    *,
    mode: str,
    model: str,
    region_window_ns: float,
    n_levels: int,
    contrast_gamma: float,
) -> None:
    """Waveform over regional XAI background + normalized importance subplot."""
    loaded = _xai_profile_from_final_cache(
        run_dir,
        dataset,
        run_manifest,
        mode=mode,
        model=model,
    )
    if loaded is None:
        print(
            f"warning: numerical XAI profile unavailable for "
            f"{mode}/{model}; skipping waveform-importance plot",
            file=sys.stderr,
        )
        return

    xai_time_ps, raw_importance, meta = loaded
    importance = _normalize_abs_importance(raw_importance)
    xai_time_ns = np.asarray(xai_time_ps, dtype=np.float64) / 1000.0

    view = _selected_waveform_view_for_xai(
        dataset,
        mode=mode,
        meta=meta,
    )
    if not 0 <= int(event) < int(view.event_id.size):
        raise IndexError(
            f"event index {event} is outside the selected XAI view"
        )

    waveform_time_ns = (
        np.asarray(view.relative_time_ps, dtype=np.float64) / 1000.0
    )
    waves = np.asarray(view.windows_mV[event], dtype=np.float64)
    if waves.ndim != 2 or waves.shape[0] != 2:
        raise ValueError(
            f"Expected two detector waveforms, got shape {waves.shape}"
        )

    regions = _regional_importance(
        xai_time_ns,
        importance,
        window_ns=region_window_ns,
    )

    fig, (ax_wave, ax_importance) = plt.subplots(
        2,
        1,
        figsize=(9.2, 5.4),
        sharex=True,
        gridspec_kw={
            "height_ratios": [2.8, 1.0],
            "hspace": 0.08,
        },
        constrained_layout=True,
    )

    cmap = mpl.colormaps["YlOrRd"]

    # Stronger visual separation for presentation:
    # 1) contrast expansion (gamma < 1),
    # 2) discretization into a small number of distinct levels.
    region_strength = _contrast_enhanced_unit_values(
        [region_mean for _lo, _hi, region_mean in regions],
        gamma=contrast_gamma,
        n_levels=n_levels,
    )

    boundaries = np.linspace(0.0, 1.0, int(n_levels) + 1)
    norm = mpl.colors.BoundaryNorm(boundaries, cmap.N, clip=True)

    for (lo, hi, _region_mean), display_strength in zip(regions, region_strength):
        ax_wave.axvspan(
            lo,
            hi,
            color=cmap(norm(float(display_strength))),
            alpha=0.12 + 0.58 * float(display_strength),
            linewidth=0.0,
            zorder=0,
        )

    ax_wave.plot(
        waveform_time_ns,
        waves[0],
        linewidth=1.55,
        label="Detector 1",
        zorder=3,
    )
    ax_wave.plot(
        waveform_time_ns,
        waves[1],
        linewidth=1.55,
        label="Detector 2",
        zorder=3,
    )
    ax_wave.axvline(
        0.0,
        linestyle="--",
        linewidth=0.9,
        color="0.25",
        alpha=0.8,
        label="LED anchor",
        zorder=2,
    )

    # Both panels use exactly the XAI support. This avoids showing waveform
    # regions that were not inputs to the selected model.
    x_left = min(region[0] for region in regions)
    x_right = max(region[1] for region in regions)
    ax_wave.set_xlim(x_left, x_right)

    ax_wave.set_ylabel("Voltage [mV]")
    ax_wave.set_title(
        f"{short_model_label(model)} · {short_mode_label(mode)}"
    )
    ax_wave.grid(alpha=0.18)
  

    ax_importance.plot(
        xai_time_ns,
        importance,
        linewidth=1.55,
    )
    ax_importance.fill_between(
        xai_time_ns,
        0.0,
        importance,
        alpha=0.15,
    )
    ax_importance.set_ylim(0.0, 1.05)
    ax_importance.set_ylabel("Normalized\n|importance|")
    ax_importance.set_xlabel("LED-relative time [ns]")
    ax_importance.grid(alpha=0.18)

    # Side scale for the background levels.
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=[ax_wave, ax_importance],
        location="right",
        fraction=0.055,
        pad=0.02,
        boundaries=boundaries,
        ticks=0.5 * (boundaries[:-1] + boundaries[1:]),
        spacing="proportional",
    )
    cbar.ax.set_yticklabels([f"{v:.2f}" for v in 0.5 * (boundaries[:-1] + boundaries[1:])])
    cbar.set_label("Regional mean normalized |importance|")

    filename = (
        f"xai_{mode}_{_artifact_model_key(model)}_"
        "waveform_importance.pdf"
    )
    _save(fig, output / filename, dpi)


def plot_xai_waveform_importance_examples(
    run_dir: Path,
    dataset: PreparedDataset,
    event: int,
    run_manifest: dict[str, Any],
    output: Path,
    dpi: int,
    *,
    models: Iterable[str],
    region_window_ns: float,
    n_levels: int,
    contrast_gamma: float,
) -> None:
    """Generate requested XAI waveform plots for energy→energy and timing→timing."""
    available_modes = (run_manifest.get("codebooks", {}) or {}).get("mode", {}) or {}

    for mode in ("energy_to_energy", "timing_to_timing"):
        if mode not in available_modes:
            continue
        for model in models:
            plot_xai_waveform_importance(
                run_dir,
                dataset,
                event,
                run_manifest,
                output,
                dpi,
                mode=mode,
                model=str(model),
                region_window_ns=float(region_window_ns),
                n_levels=int(n_levels),
                contrast_gamma=float(contrast_gamma),
            )



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
    if not crossings:
        print("warning: selected event has no valid multithreshold crossings", file=sys.stderr)
        return

    left, right = _crossing_xlim(
        [crossing for _threshold, crossing in crossings],
        t_ps,
        pad_before_ns=0.55,
        pad_after_ns=1.0,
    )
    local = (t_ns >= left) & (t_ns <= right)
    local_peak = float(np.nanmax(y[local])) if np.any(local) else float(np.nanmax(y))
    colors = _default_colors(len(crossings))

    fig, ax = plt.subplots(figsize=(8.9, 4.3))
    ax.plot(t_ns, y, linewidth=1.8, color="0.25", label="energy waveform")

    # Color encodes threshold.  No numeric labels are placed on the waveform:
    # all threshold values live in the legend, which is much cleaner on a slide.
    for color, (threshold, crossing_ps) in zip(colors, crossings):
        ax.axhline(
            threshold,
            linewidth=1.15,
            alpha=0.85,
            color=color,
            label=f"{threshold:g} mV",
        )
        ax.scatter(
            [crossing_ps / 1000.0],
            [threshold],
            s=46,
            zorder=5,
            color=color,
            edgecolors="white",
            linewidths=0.7,
        )

    ax.set_xlim(left, right)
    ymax = max(local_peak, max(threshold for threshold, _ in crossings))
    ax.set_ylim(min(-5.0, float(np.nanmin(y[local])) if np.any(local) else -5.0), ymax * 1.12)
    ax.set_xlabel("Time relative to native LED anchor [ns]")
    ax.set_ylabel("Voltage [mV]")
    ax.set_title("Multithreshold readout: interpolated rising-edge crossings")
    ax.grid(alpha=0.20)
    ax.legend(frameon=False, ncol=3, title="Threshold")
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

    crossing_by_detector: list[list[tuple[float, float]]] = [
        _valid_thresholds_for_waveform(t_ps, waves[detector], thresholds)
        for detector in range(2)
    ]
    all_crossings = [
        crossing
        for detector_crossings in crossing_by_detector
        for _threshold, crossing in detector_crossings
    ]
    left, right = _crossing_xlim(
        all_crossings,
        t_ps,
        pad_before_ns=0.55,
        pad_after_ns=1.0,
    )

    colors = _default_colors(len(thresholds))
    threshold_colors = {float(threshold): colors[i] for i, threshold in enumerate(thresholds)}
    detector_markers = ["o", "s"]

    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    ax.plot(t_ns, waves[0], linewidth=1.5, color="0.25", label="Detector 1 waveform")
    ax.plot(t_ns, waves[1], linewidth=1.5, linestyle="--", color="0.45", label="Detector 2 waveform")

    for threshold in thresholds:
        color = threshold_colors[float(threshold)]
        ax.axhline(threshold, linewidth=1.05, alpha=0.75, color=color)

    for detector in range(2):
        for threshold, crossing_ps in crossing_by_detector[detector]:
            ax.scatter(
                [crossing_ps / 1000.0],
                [threshold],
                s=46,
                marker=detector_markers[detector],
                color=threshold_colors[float(threshold)],
                edgecolors="white",
                linewidths=0.7,
                zorder=5,
            )

    voltage = _median_voltage(dataset)
    kernel = descriptor.get("kernel", "")
    title = f"Selected multithreshold input · {_mode_label(preferred_mode)}"
    if np.isfinite(voltage):
        title += f" · {voltage:g} V"
    if kernel:
        title += f" · {kernel} SVR"
    ax.set_title(title)
    ax.set_xlim(left, right)

    local = (t_ns >= left) & (t_ns <= right)
    local_max = float(np.nanmax(waves[:, local])) if np.any(local) else float(np.nanmax(waves))
    ax.set_ylim(
        min(-5.0, float(np.nanmin(waves[:, local])) if np.any(local) else -5.0),
        max(local_max, max(thresholds)) * 1.12,
    )
    ax.set_xlabel("Time relative to native energy-LED anchor [ns]")
    ax.set_ylabel("Voltage [mV]")
    ax.grid(alpha=0.20)

    threshold_handles = [
        Line2D([0], [0], color=threshold_colors[float(threshold)], lw=2.0, label=f"{threshold:g} mV")
        for threshold in thresholds
    ]
    detector_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="0.35",
               markeredgecolor="white", markersize=7, label="Detector 1 crossing"),
        Line2D([0], [0], marker="s", linestyle="none", markerfacecolor="0.35",
               markeredgecolor="white", markersize=7, label="Detector 2 crossing"),
    ]
    waveform_handles = [
        Line2D([0], [0], color="0.25", lw=1.5, label="Detector 1 waveform"),
        Line2D([0], [0], color="0.45", lw=1.5, linestyle="--", label="Detector 2 waveform"),
    ]
    ax.legend(
        handles=waveform_handles + detector_handles + threshold_handles,
        frameon=False,
        ncol=3,
        fontsize=8,
    )
    _save(fig, output / "results_multithreshold_best_thresholds.pdf", dpi)

def _mt_best_by_count(
    run_dir: Path,
    rows: list[dict[str, str]],
    run_manifest: dict[str, Any],
) -> dict[str, dict[float, dict[int, float]]]:
    """Return mode -> voltage -> threshold_count -> improvement over LED [%].

    Multithreshold candidates come from stage=0 rows in results.csv.
    The matching development/validation LED baseline comes from report_results.csv.

    Improvement:
        100 * (CTR_LED - CTR_MT) / CTR_LED
    """
    mt_id = _codebook_id(run_manifest, "model", "multithreshold_svr")
    if mt_id is None:
        raise RuntimeError("Run manifest has no multithreshold_svr model code")

    mode_codebook = (run_manifest.get("codebooks", {}) or {}).get("mode", {}) or {}
    id_to_mode = {int(value): str(name) for name, value in mode_codebook.items()}

    energy_modes = {"energy_to_energy", "energy_to_timing"}

    # ------------------------------------------------------------------
    # LED validation baselines from report_results.csv
    # ------------------------------------------------------------------
    report = _report_rows(run_dir)

    led_ctr: dict[str, dict[float, float]] = {}

    for row in report:
        if str(row.get("stage_name", "")).lower() != "validation":
            continue

        if str(row.get("model", "")).lower() != "led":
            continue

        mode = str(row.get("mode", ""))
        if mode not in energy_modes:
            continue

        voltage = _as_float(row.get("voltage_V"))
        ctr = _as_float(row.get("ctr_ps"))

        if not np.isfinite(voltage) or not np.isfinite(ctr):
            continue

        led_ctr.setdefault(mode, {})[voltage] = ctr

    if not led_ctr:
        raise RuntimeError(
            "No LED validation baselines found in report_results.csv"
        )

    # ------------------------------------------------------------------
    # Best MT candidate at each threshold count from results.csv
    # ------------------------------------------------------------------
    mt_ctr: dict[str, dict[float, dict[int, float]]] = {}

    for row in rows:
        if _as_int(row.get("stage")) != 0:
            continue

        if _as_int(row.get("model_id")) != mt_id:
            continue

        mode = id_to_mode.get(_as_int(row.get("mode_id")), "")
        if mode not in energy_modes:
            continue

        descriptor = _candidate_parameters(
            run_manifest,
            _as_int(row.get("candidate_id")),
        )

        thresholds = descriptor.get("thresholds_mV", []) if descriptor else []

        if not isinstance(thresholds, list) or not thresholds:
            continue

        count = len(thresholds)
        voltage = _as_float(row.get("voltage_V"))
        ctr = _as_float(row.get("ctr_ps"))

        if not np.isfinite(voltage) or not np.isfinite(ctr):
            continue

        previous = (
            mt_ctr
            .setdefault(mode, {})
            .setdefault(voltage, {})
            .get(count)
        )

        if previous is None or ctr < previous:
            mt_ctr[mode][voltage][count] = ctr

    # ------------------------------------------------------------------
    # Convert absolute MT CTR -> relative improvement over LED
    # ------------------------------------------------------------------
    output: dict[str, dict[float, dict[int, float]]] = {}

    for mode, voltage_data in mt_ctr.items():
        for voltage, count_data in voltage_data.items():

            baseline = led_ctr.get(mode, {}).get(voltage)

            if baseline is None:
                print(
                    f"warning: no LED validation baseline for "
                    f"{mode} at {voltage:g} V",
                    file=sys.stderr,
                )
                continue

            if not np.isfinite(baseline) or baseline <= 0:
                continue

            for count, ctr in count_data.items():

                improvement_pct = (
                    100.0 * (baseline - ctr) / baseline
                )

                output.setdefault(mode, {}).setdefault(
                    voltage, {}
                )[count] = improvement_pct

    return output


def _plot_threshold_count_axis(
    ax: plt.Axes,
    mode: str,
    voltage_data: dict[float, dict[int, float]],
) -> None:
    for voltage in sorted(voltage_data):
        points = sorted(voltage_data[voltage].items())
        if not points:
            continue

        x = [int(k) for k, _ in points]
        y = [float(v) for _, v in points]

        ax.plot(
            x,
            y,
            marker="o",
            linewidth=1.5,
            markersize=5.5,
            label=f"{voltage:g} V",
        )

    all_counts = sorted(
        {
            count
            for values in voltage_data.values()
            for count in values
        }
    )

    if all_counts:
        ax.set_xticks(all_counts)


    ax.set_xlabel("Number of fixed thresholds")
    ax.set_ylabel("CTR improvement over LED [%]")

    # No subplot title
    ax.grid(alpha=0.22)
    ax.legend(
        frameon=False,
        ncol=min(5, max(1, len(voltage_data))),
    )


def plot_threshold_count_performance(
    run_dir: Path,
    rows: list[dict[str, str]],
    run_manifest: dict[str, Any],
    output: Path,
    dpi: int,
) -> None:
    data = _mt_best_by_count(
        run_dir,
        rows,
        run_manifest,
    )

    modes = [
        mode
        for mode in ("energy_to_energy", "energy_to_timing")
        if mode in data
    ]

    if not modes:
        raise RuntimeError(
            "No stage=0 multithreshold candidates with matching LED validation "
            "baseline were found for energy_to_energy/energy_to_timing"
        )

    # Combined figure
    fig, axes = plt.subplots(
        len(modes),
        1,
        figsize=(9.5, 3.8 * len(modes)),
        squeeze=False,
    )

    for ax, mode in zip(axes[:, 0], modes):
        _plot_threshold_count_axis(
            ax,
            mode,
            data[mode],
        )

    # No global title
    _save(
        fig,
        output / "results_multithreshold_ctr_vs_threshold_count.pdf",
        dpi,
    )

    # Individual plot for each mode
    for mode in modes:
        fig, ax = plt.subplots(figsize=(9.5, 4.2))

        _plot_threshold_count_axis(
            ax,
            mode,
            data[mode],
        )

        _save(
            fig,
            output
            / f"results_multithreshold_ctr_vs_threshold_count_{mode}.pdf",
            dpi,
        )


# -----------------------------------------------------------------------------
# Gaussian CTR evaluation for presentation results
# -----------------------------------------------------------------------------
def _presentation_fit_config(run_manifest: dict[str, Any]) -> dict[str, Any]:
    """Return the exact experiment-level Gaussian fit configuration."""
    fit_config = run_manifest.get("fit")
    if not isinstance(fit_config, dict):
        raise ValueError(
            "Run manifest does not contain a valid top-level 'fit' configuration. "
            "The presentation plots require the same global fitter used by the run."
        )
    return fit_config


def _stable_seed(base: int, *parts: Any) -> int:
    payload = "|".join(str(v) for v in (base, *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little") & 0x7FFFFFFF


def _fit_ctr_gaussian(
    values_ps: np.ndarray,
    *,
    method: str,
    fit_config: dict[str, Any],
):
    """Fit CTR with the repository's global all-event Gaussian fitter.

    No event is silently discarded. Evaluation artifacts are expected to contain
    one finite residual per blind event, matching the scientific evaluation rule.
    """
    values = np.asarray(values_ps, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise RuntimeError(f"{method}: empty timing distribution")
    if np.any(~np.isfinite(values)):
        bad = int(np.count_nonzero(~np.isfinite(values)))
        raise RuntimeError(
            f"{method}: found {bad} non-finite residuals; refusing to drop events "
            "during presentation-only CTR evaluation"
        )

    fit = fit_times_ps(values, method, fit_config)
    if not fit.success or not np.isfinite(fit.ctr_ps):
        raise RuntimeError(f"{method}: Gaussian CTR fit failed: {fit.message}")
    return fit


def _bootstrap_gaussian_ctr_uncertainty(
    values_ps: np.ndarray,
    *,
    method: str,
    fit_config: dict[str, Any],
    n_bootstrap: int,
    seed: int,
) -> tuple[float, int]:
    """Event-resampling uncertainty with a full Gaussian refit per bootstrap draw."""
    values = np.asarray(values_ps, dtype=np.float64).reshape(-1)
    if values.size < 3 or int(n_bootstrap) <= 1:
        return float("nan"), 0
    if np.any(~np.isfinite(values)):
        raise RuntimeError(f"{method}: bootstrap input contains non-finite values")

    rng = np.random.default_rng(int(seed))
    draws: list[float] = []
    for draw_index in range(int(n_bootstrap)):
        sample = values[rng.integers(0, values.size, size=values.size)]
        try:
            fit = fit_times_ps(
                sample,
                f"{method} bootstrap {draw_index}",
                fit_config,
            )
        except Exception:
            continue
        if fit.success and np.isfinite(fit.ctr_ps):
            draws.append(float(fit.ctr_ps))

    minimum_success = max(10, int(math.ceil(0.80 * int(n_bootstrap))))
    if len(draws) < minimum_success:
        raise RuntimeError(
            f"{method}: only {len(draws)}/{n_bootstrap} Gaussian bootstrap fits "
            f"succeeded; need at least {minimum_success}"
        )

    return float(np.std(np.asarray(draws, dtype=np.float64), ddof=1)), len(draws)


def _bootstrap_samples_from_run(
    run_manifest: dict[str, Any], override: int | None
) -> int:
    if override is not None:
        return max(0, int(override))
    reporting = run_manifest.get("reporting", {}) or {}
    # Current repository name first; legacy presentation key as fallback.
    value = reporting.get(
        "ctr_uncertainty_bootstrap_samples",
        reporting.get("bootstrap_samples", 1000),
    )
    return max(0, int(value))


def _evaluation_methods(npz_path: Path, meta_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path) as arrays:
        methods: dict[str, np.ndarray] = {}
        if "blind_led" in arrays:
            methods["led"] = np.asarray(arrays["blind_led"], dtype=np.float64)
        if "blind_cfd" in arrays:
            methods["cfd"] = np.asarray(arrays["blind_cfd"], dtype=np.float64)

        model_map = (_read_json(meta_path).get("models", {}) or {})
        for model_name, key in model_map.items():
            arr_key = f"blind__{key}"
            if arr_key in arrays:
                methods[str(model_name)] = np.asarray(arrays[arr_key], dtype=np.float64)

    return methods


def _all_evaluation_artifacts(
    run_dir: Path,
    run_manifest: dict[str, Any],
    *,
    mode: str,
) -> list[tuple[float, int, Path, Path]]:
    mode_id = _codebook_id(run_manifest, "mode", mode)
    if mode_id is None:
        return []

    files = (run_manifest.get("codebooks", {}) or {}).get("file", {}) or {}
    root = run_dir / "artifacts" / "evaluations"
    output: list[tuple[float, int, Path, Path]] = []

    for name, file_id_raw in files.items():
        match = re.search(r"(?P<v>\d+(?:\.\d+)?)V", str(name))
        if match is None:
            continue
        voltage = float(match.group("v"))
        file_id = int(file_id_raw)
        stem = f"f{file_id}_m{mode_id}"
        npz = root / f"{stem}.npz"
        meta = root / f"{stem}.json"
        if npz.is_file() and meta.is_file():
            output.append((voltage, file_id, npz, meta))

    return sorted(output, key=lambda item: item[0])


def _gaussian_blind_ctr_rows(
    run_dir: Path,
    run_manifest: dict[str, Any],
    *,
    mode: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    """Re-evaluate every blind curve directly from saved residual artifacts."""
    fit_config = _presentation_fit_config(run_manifest)
    output: list[dict[str, Any]] = []

    for voltage, file_id, npz_path, meta_path in _all_evaluation_artifacts(
        run_dir, run_manifest, mode=mode
    ):
        methods = _evaluation_methods(npz_path, meta_path)
        if mode == "timing_to_timing":
            methods.pop("cfd", None)
        for model in _PRESENTATION_MODEL_ORDER:
            if model not in methods:
                continue
            values = np.asarray(methods[model], dtype=np.float64).reshape(-1)
            fit = _fit_ctr_gaussian(
                values,
                method=f"{mode} {voltage:g} V {model}",
                fit_config=fit_config,
            )
            uncertainty, n_success = _bootstrap_gaussian_ctr_uncertainty(
                values,
                method=f"{mode} {voltage:g} V {model}",
                fit_config=fit_config,
                n_bootstrap=bootstrap_samples,
                seed=_stable_seed(
                    bootstrap_seed, file_id, mode, model, "presentation-gaussian-bootstrap"
                ),
            )
            output.append(
                {
                    "voltage": float(voltage),
                    "ctr": float(fit.ctr_ps),
                    "ctr_err": float(uncertainty),
                    "model": model,
                    "fit": fit,
                    "n": int(values.size),
                    "bootstrap_success": int(n_success),
                }
            )

    return output



# -----------------------------------------------------------------------------
# Presentation result plots
# -----------------------------------------------------------------------------
_PRESENTATION_MODEL_ORDER = [
    "led", "cfd", "linear_svr", "constructive_mlp", "cnn",
    "multithreshold_svr",
]
_PRESENTATION_MODEL_LABELS = {
    "led": "LED",
    "cfd": "CFD",
    "linear_svr": "Linear SVR",
    "constructive_mlp": "Constructive MLP",
    "cnn": "CNN",
    "multithreshold_svr": "Fixed-threshold SVR",
}


def _report_rows(run_dir: Path) -> list[dict[str, str]]:
    path = run_dir / "report_results.csv"
    return _read_csv(path) if path.is_file() else []


def _blind_ctr_rows(
    run_dir: Path,
    rows: list[dict[str, str]],
    run_manifest: dict[str, Any],
    mode: str,
) -> list[dict[str, Any]]:
    report = _report_rows(run_dir)
    if report:
        out = []
        for row in report:
            if str(row.get("stage_name", "")).lower() != "blind":
                continue
            if str(row.get("mode", "")) != mode:
                continue
            voltage = _as_float(row.get("voltage_V"))
            ctr = _as_float(row.get("ctr_ps"))
            ctr_err = _as_float(row.get("ctr_err_ps"))
            model = str(row.get("model", ""))
            if np.isfinite(voltage) and np.isfinite(ctr) and model:
                out.append({
                    "voltage": voltage,
                    "ctr": ctr,
                    "ctr_err": ctr_err,
                    "model": model,
                })
        if out:
            return out

    mode_id = _codebook_id(run_manifest, "mode", mode)
    if mode_id is None:
        return []
    model_map = (run_manifest.get("codebooks", {}) or {}).get("model", {}) or {}
    id_to_model = {int(v): str(k) for k, v in model_map.items()}
    out = []
    for row in rows:
        if _as_int(row.get("stage")) != 1:
            continue
        if _as_int(row.get("mode_id")) != mode_id:
            continue
        voltage = _as_float(row.get("voltage_V"))
        ctr = _as_float(row.get("ctr_ps"))
        ctr_err = _as_float(row.get("ctr_err_ps"))
        model = id_to_model.get(_as_int(row.get("model_id")), "")
        if np.isfinite(voltage) and np.isfinite(ctr) and model:
            out.append({
                "voltage": voltage,
                "ctr": ctr,
                "ctr_err": ctr_err,
                "model": model,
            })
    return out


def plot_ctr_vs_voltage_presentation(
    run_dir: Path,
    rows: list[dict[str, str]],
    run_manifest: dict[str, Any],
    output: Path,
    dpi: int,
    *,
    mode: str,
    filename: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> None:
    # rows is retained in the signature for backward compatibility with callers;
    # final blind CTR values are intentionally recomputed from residual artifacts.
    _ = rows
    data = _gaussian_blind_ctr_rows(
        run_dir,
        run_manifest,
        mode=mode,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    if not data:
        print(
            f"warning: no blind evaluation artifacts for {mode}; "
            "Gaussian CTR-vs-voltage plot not generated",
            file=sys.stderr,
        )
        return

    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    for model in _PRESENTATION_MODEL_ORDER:
        points = sorted(
            [row for row in data if row["model"] == model],
            key=lambda row: float(row["voltage"]),
        )
        if not points:
            continue

        x = np.asarray([row["voltage"] for row in points], dtype=np.float64)
        y = np.asarray([row["ctr"] for row in points], dtype=np.float64)
        yerr = np.asarray([row["ctr_err"] for row in points], dtype=np.float64)

        label = short_model_label(model)
        if np.all(np.isfinite(yerr)):
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker="o",
                linewidth=1.2,
                markersize=5.5,
                capsize=2.5,
                elinewidth=1.0,
                label=label,
            )
        else:
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=1.2,
                markersize=5.5,
                label=label,
            )

    ax.set_title(short_mode_label(mode))
    ax.set_xlabel("Bias voltage [V]")
    ax.set_ylabel("Blind Gaussian-fit CTR [ps]")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=2, fontsize=8, loc="best")
    _save(fig, output / filename, dpi)


def _evaluation_artifact(
    run_dir: Path,
    run_manifest: dict[str, Any],
    *,
    mode: str,
    requested_voltage: float | None,
) -> tuple[Path, Path, float] | None:
    mode_id = _codebook_id(run_manifest, "mode", mode)
    if mode_id is None:
        return None

    files = (run_manifest.get("codebooks", {}) or {}).get("file", {}) or {}
    candidates = []
    for name, file_id in files.items():
        match = re.search(r"(?P<v>\d+(?:\.\d+)?)V", str(name))
        if match:
            candidates.append((float(match.group("v")), int(file_id)))
    if not candidates:
        return None

    if requested_voltage is None:
        candidates.sort()
        voltage, file_id = candidates[len(candidates) // 2]
    else:
        voltage, file_id = min(
            candidates,
            key=lambda p: abs(p[0] - float(requested_voltage)),
        )

    root = run_dir / "artifacts" / "evaluations"
    stem = f"f{file_id}_m{mode_id}"
    npz = root / f"{stem}.npz"
    meta = root / f"{stem}.json"
    if not npz.is_file() or not meta.is_file():
        return None
    return npz, meta, voltage


def plot_blind_distribution_presentation(
    run_dir: Path,
    run_manifest: dict[str, Any],
    output: Path,
    dpi: int,
    *,
    mode: str,
    filename: str,
    requested_voltage: float | None,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> None:
    selected = _evaluation_artifact(
        run_dir, run_manifest, mode=mode, requested_voltage=requested_voltage
    )
    if selected is None:
        print(f"warning: no evaluation artifact for {mode}", file=sys.stderr)
        return

    npz_path, meta_path, voltage = selected
    methods = _evaluation_methods(npz_path, meta_path)
    if mode == "timing_to_timing":
        methods.pop("cfd", None)
    methods = {m: methods[m] for m in _PRESENTATION_MODEL_ORDER if m in methods}
    if not methods:
        return

    fit_config = _presentation_fit_config(run_manifest)

    # Use one robust visible range for all methods, but absolute event counts.
    finite_groups = [
        np.asarray(v, dtype=np.float64).reshape(-1)
        for v in methods.values()
        if np.asarray(v).size
    ]
    all_values = np.concatenate(finite_groups)
    if np.any(~np.isfinite(all_values)):
        raise RuntimeError(
            f"{mode}: non-finite blind residual found; presentation plot will not "
            "silently remove evaluation events"
        )

    med = float(np.median(all_values))
    mad = float(np.median(np.abs(all_values - med)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.std(all_values, ddof=1))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    # Compact presentation range:
    # retain essentially all of each distribution while ignoring extreme tails.
    range_limits = []

    for values in methods.values():
        values = np.asarray(values, dtype=np.float64).reshape(-1)

        if values.size < 2:
            continue

        q_lo, q_hi = np.percentile(values, [0.2, 99.8])
        range_limits.append((q_lo, q_hi))

    if not range_limits:
        return

    visible_lo = min(lo for lo, _ in range_limits)
    visible_hi = max(hi for _, hi in range_limits)

    # Small visual padding.
    span = visible_hi - visible_lo
    visible_lo -= 0.04 * span
    visible_hi += 0.04 * span

    fig, ax = plt.subplots(figsize=(9.0, 4.4))

    for model_index, model in enumerate(_PRESENTATION_MODEL_ORDER):
        if model not in methods:
            continue

        values = np.asarray(methods[model], dtype=np.float64).reshape(-1)
        fit = _fit_ctr_gaussian(
            values,
            method=f"{mode} {voltage:g} V {model}",
            fit_config=fit_config,
        )
        uncertainty, _n_success = _bootstrap_gaussian_ctr_uncertainty(
            values,
            method=f"{mode} {voltage:g} V {model}",
            fit_config=fit_config,
            n_bootstrap=bootstrap_samples,
            seed=_stable_seed(
                bootstrap_seed,
                mode,
                voltage,
                model,
                "presentation-distribution-bootstrap",
            ),
        )

        # Plot the exact histogram selected by the Gaussian fitter. Counts are
        # absolute; fit.expected is therefore directly comparable bin-by-bin.
        label = f"{short_model_label(model)} — CTR {format_ctr(fit.ctr_ps, uncertainty)}"
        ax.stairs(
            fit.counts,
            fit.edges_ps,
            linewidth=1.45,
            label=label,
        )

    ax.set_xlim(visible_lo, visible_hi)

    ax.set_xlabel("Residual timing error [ps]")
    ax.set_ylabel("Counts")

    # No internal title: the Beamer frame already gives the context.
    # ax.set_title(...)

    ax.grid(alpha=0.22)

    ax.legend(
        frameon=False,
        ncol=2,
        fontsize=7.5,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        borderaxespad=0.0,
        columnspacing=1.2,
        handlelength=2.0,
    )

    _save(fig, output / filename, dpi)


def plot_main_result_slides(
    run_dir: Path,
    rows: list[dict[str, str]],
    run_manifest: dict[str, Any],
    output: Path,
    dpi: int,
    *,
    requested_voltage: float | None,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> None:
    if _codebook_id(run_manifest, "mode", "energy_to_energy") is not None:
        plot_ctr_vs_voltage_presentation(
            run_dir, rows, run_manifest, output, dpi,
            mode="energy_to_energy",
            filename="results_energy_ctr_vs_voltage.pdf",
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        plot_blind_distribution_presentation(
            run_dir, run_manifest, output, dpi,
            mode="energy_to_energy",
            filename="results_energy_blind_distribution_example.pdf",
            requested_voltage=requested_voltage,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )

    if _codebook_id(run_manifest, "mode", "timing_to_timing") is not None:
        plot_ctr_vs_voltage_presentation(
            run_dir, rows, run_manifest, output, dpi,
            mode="timing_to_timing",
            filename="results_timing_ctr_vs_voltage.pdf",
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        plot_blind_distribution_presentation(
            run_dir, run_manifest, output, dpi,
            mode="timing_to_timing",
            filename="results_timing_blind_distribution_example.pdf",
            requested_voltage=requested_voltage,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
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
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=None,
        help=(
            "Event-resampling Gaussian-fit bootstrap draws for final result plots. "
            "Default: reporting.bootstrap_samples from the run manifest, else 1000."
        ),
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=12345,
        help="Base seed for deterministic presentation bootstrap resampling",
    )
    parser.add_argument(
        "--xai-models",
        nargs="+",
        default=["linear_svr", "cnn"],
        help=(
            "Selected waveform models for waveform+importance figures. "
            "Default: linear_svr cnn."
        ),
    )
    parser.add_argument(
        "--xai-window-ns",
        type=float,
        default=1.0,
        help=(
            "Fixed time width [ns] used to average the XAI importance for the "
            "waveform-background heatmap."
        ),
    )
    parser.add_argument(
        "--xai-levels",
        type=int,
        default=6,
        help=(
            "Number of discrete background-importance levels used in the waveform "
            "panel. Higher values give smoother shading; lower values give more contrast."
        ),
    )
    parser.add_argument(
        "--xai-contrast-gamma",
        type=float,
        default=0.55,
        help=(
            "Contrast-enhancement gamma for background importance visualization. "
            "Values below 1 increase visual separation of low/mid importance regions."
        ),
    )
    parser.add_argument(
        "--skip-xai-waveform-importance",
        action="store_true",
        help="Do not generate waveform/background XAI presentation figures.",
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

    bootstrap_samples = _bootstrap_samples_from_run(
        run_manifest, args.bootstrap_samples
    )

    print(f"run:      {run_dir}")
    print(f"prepared: {dataset.directory}")
    print(f"voltage:  {_median_voltage(dataset):g} V")
    print(f"event:    {event}")
    print(f"output:   {output}")
    print(f"bootstrap:{bootstrap_samples} Gaussian-refit draws")

    plot_energy_waveform_example(dataset, event, output, args.dpi)
    plot_timing_waveform_example(dataset, event, output, args.dpi)
    plot_led_cfd_example(dataset, event, output, args.dpi)
    plot_windowing_example(dataset, event, output, args.dpi, run_manifest, rows)
    plot_multithreshold_example(dataset, event, output, args.dpi, run_manifest)
    plot_threshold_count_performance(
    run_dir,
    rows,
    run_manifest,
    output,
    args.dpi,
)
    plot_best_multithresholds(
        dataset, event, output, args.dpi, run_manifest, rows, args.mt_mode
    )

    if not args.skip_xai_waveform_importance:
        plot_xai_waveform_importance_examples(
            run_dir,
            dataset,
            event,
            run_manifest,
            output,
            args.dpi,
            models=args.xai_models,
            region_window_ns=args.xai_window_ns,
            n_levels=args.xai_levels,
            contrast_gamma=args.xai_contrast_gamma,
        )

    plot_main_result_slides(
        run_dir,
        rows,
        run_manifest,
        output,
        args.dpi,
        requested_voltage=args.voltage,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )


if __name__ == "__main__":
    main()
