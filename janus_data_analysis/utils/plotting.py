from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .models import (
    EnergyMeasurements,
    EnergySelectionResult,
    Measurements,
    SelectionResult,
)


def _common_edges(values_a: np.ndarray, values_b: np.ndarray, max_bins: int) -> np.ndarray:
    minimum = int(min(values_a.min(), values_b.min()))
    maximum = int(max(values_a.max(), values_b.max()))
    width = max(1, int(math.ceil((maximum - minimum + 1) / max_bins)))
    start = math.floor(minimum / width) * width
    bins = max(1, int(math.ceil((maximum - start + 1) / width)))
    return start - 0.5 + np.arange(bins + 1, dtype=float) * width


def _ps_to_ns(values_ps: np.ndarray | float) -> np.ndarray | float:
    return np.asarray(values_ps, dtype=float) / 1000.0 if isinstance(values_ps, np.ndarray) else float(values_ps) / 1000.0


def plot_peak_selection(
    path: str | Path,
    run_id: str,
    measurements: Measurements | EnergyMeasurements,
    selection: SelectionResult | EnergySelectionResult,
    toa_lsb_ps: float,
    cfg: dict,
) -> None:
    import matplotlib.pyplot as plt

    # Imposta font globale più grande
    plt.rcParams.update({'font.size': 16})

    edges_lsb = _common_edges(
        measurements.duration_a_lsb,
        measurements.duration_b_lsb,
        int(cfg["plots"]["max_histogram_bins"]),
    )
    edges_ns = _ps_to_ns(edges_lsb * toa_lsb_ps)
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.0), sharex=True)
    definitions = (
        (measurements.duration_a_lsb, selection.peak_a, "ch1"),
        (measurements.duration_b_lsb, selection.peak_b, "ch5"),
    )
    for axis, (values, peak, label) in zip(axes, definitions):
        axis.hist(_ps_to_ns(values.astype(float) * toa_lsb_ps), bins=edges_ns, alpha=0.75)
        span = axis.axvspan(
            _ps_to_ns(peak.low_lsb * toa_lsb_ps),
            _ps_to_ns(peak.high_lsb * toa_lsb_ps),
            alpha=0.18, color='orange', label='selected region'
        )
        axis.text(0.02, 0.92, label, transform=axis.transAxes, ha="left", va="top", fontsize=18)
        axis.set_ylabel("Events", fontsize=18)
        #axis.set_xlim(20, 110)  # Imposta intervallo X
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Energy-pulse duration [ns]", fontsize=18)
    fig.suptitle(f"{run_id} — Energy-duration peak selection", fontsize=20)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=int(cfg["plots"]["dpi"]))
    plt.close(fig)

def _matching_annotation(model) -> str:
    return (
        "average-delay matcher\n"
        f"mean={model.average_delay_lsb:.2f} LSB\n"
        f"std={model.delay_std_lsb:.2f} LSB\n"
        f"robust scale={model.robust_scale_lsb:.2f} LSB\n"
        f"N={model.input_samples} → {model.training_samples} "
        f"(rejected={model.outliers_rejected})\n"
        f"clip={model.outlier_z_threshold:g} MAD-σ, "
        f"iterations={model.outlier_iterations}"
    )

def _plot_matching_panels(
    path: str | Path,
    run_id: str,
    datasets: dict,
    models: dict,
    toa_lsb_ps: float,
    cfg: dict,
    title_suffix: str,
    delay_window_ns: float | None,
    x_ranges_lsb: dict[str, tuple[float, float]],
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 8.0), sharex=False)
    for axis, pair in zip(axes, ("a", "b")):
        model = models[pair]
        x_lsb, y_lsb = datasets[pair]
        x_lsb = np.asarray(x_lsb, dtype=float)
        y_lsb = np.asarray(y_lsb, dtype=float)
        if x_lsb.size:
            axis.scatter(
                x_lsb * toa_lsb_ps / 1000.0,
                y_lsb * toa_lsb_ps / 1000.0,
                s=9,
                alpha=0.35,
                label="Events",
            )
        grid_low, grid_high = x_ranges_lsb[pair]
        grid_low = float(grid_low)
        grid_high = float(grid_high)
        if grid_high <= grid_low:
            grid_high = grid_low + 1.0
        x_grid_lsb = np.linspace(
            grid_low,
            grid_high,
            int(cfg["plots"]["matching_curve_points"]),
        )
        y_grid_lsb = model.predict(x_grid_lsb)
        axis.plot(
            x_grid_lsb * toa_lsb_ps / 1000.0,
            y_grid_lsb * toa_lsb_ps / 1000.0,
            linewidth=2.1,
            color="tab:orange",
            label="Average delay",
        )
        axis.text(
            0.98,
            0.96,
            _matching_annotation(model),
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "0.75"},
        )
        if delay_window_ns is not None:
            axis.set_ylim(0.0, float(delay_window_ns))
        axis.set_xlim(
            grid_low * toa_lsb_ps / 1000.0,
            grid_high * toa_lsb_ps / 1000.0,
        )
        axis.set_ylabel("Delay [ns]")
        axis.set_title(
            f"ch{model.energy_channel} ToT → ch{model.timing_channel} delay"
        )
        axis.grid(alpha=0.2)
        axis.legend(loc="upper left")
    axes[-1].set_xlabel("Energy-channel ToT [ns]")
    axes[0].set_xlabel("Energy-channel ToT [ns]")
    fig.suptitle(f"{run_id} — Average-delay matching ({title_suffix})")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=int(cfg["plots"]["dpi"]))
    plt.close(fig)


def plot_matching_training(
    path: str | Path,
    run_id: str,
    samples: dict,
    models: dict,
    toa_lsb_ps: float,
    cfg: dict,
    peak_selection: EnergySelectionResult,
) -> None:
    datasets = {
        pair: (
            samples[pair].energy_duration_lsb,
            samples[pair].delay_lsb,
        )
        for pair in ("a", "b")
    }
    _plot_matching_panels(
        path,
        run_id,
        datasets,
        models,
        toa_lsb_ps,
        cfg,
        "filtered training after energy selection",
        float(cfg["matching_model"]["training"]["window_ns"]),
        {
            "a": (peak_selection.peak_a.low_lsb, peak_selection.peak_a.high_lsb),
            "b": (peak_selection.peak_b.low_lsb, peak_selection.peak_b.high_lsb),
        },
    )


def plot_matching_total(
    path: str | Path,
    run_id: str,
    rows: list[dict],
    models: dict,
    toa_lsb_ps: float,
    cfg: dict,
    peak_selection: EnergySelectionResult,
) -> None:
    datasets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for pair in ("a", "b"):
        selected = [
            row
            for row in rows
            if str(row.get("pair", "")) == pair
            and str(row.get("accepted", "")).lower() in {"1", "true"}
            and str(row.get("event_accepted", "")).lower() in {"1", "true"}
            and str(row.get("energy_duration_lsb", "")) != ""
            and str(row.get("selected_delay_lsb", "")) != ""
        ]
        datasets[pair] = (
            np.asarray(
                [float(row["energy_duration_lsb"]) for row in selected],
                dtype=float,
            ),
            np.asarray(
                [float(row["selected_delay_lsb"]) for row in selected],
                dtype=float,
            ),
        )
    _plot_matching_panels(
        path,
        run_id,
        datasets,
        models,
        toa_lsb_ps,
        cfg,
        "post-matching events",
        None,
        {
            "a": (peak_selection.peak_a.low_lsb, peak_selection.peak_a.high_lsb),
            "b": (peak_selection.peak_b.low_lsb, peak_selection.peak_b.high_lsb),
        },
    )
