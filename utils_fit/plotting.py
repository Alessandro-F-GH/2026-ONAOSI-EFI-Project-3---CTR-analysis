from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .gaussian import FitResult


def plot_gaussian_fit(
    result: FitResult | None,
    path: str | Path,
    *,
    dpi: int = 180,
    title: str | None = None,
    xlabel: str = "Time difference [ps]",
) -> None:
    if result is None or not result.success or result.edges_ps.size < 2:
        return
    edges = np.asarray(result.edges_ps, dtype=np.float64)
    counts = np.asarray(result.counts, dtype=np.float64)
    expected = np.asarray(result.expected, dtype=np.float64)
    widths = np.diff(edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if counts.size != widths.size or expected.size != centers.size:
        raise ValueError("Malformed FitResult histogram arrays")

    fig, ax = plt.subplots(figsize=(9.2, 6.1))
    bars = ax.bar(edges[:-1], counts, width=widths, align="edge", alpha=0.65, label="Data")
    error_text = f" ± {result.ctr_error_ps:.1f}" if np.isfinite(result.ctr_error_ps) else ""
    line, = ax.plot(
        centers, expected, linewidth=2.2,
        label=(
            "Gaussian fit\n"
            f"μ={result.mean_ps:.1f} ps\n"
            f"σ={result.sigma_ps:.1f} ps\n"
            f"CTR={result.ctr_ps:.1f}{error_text} ps\n"
            f"D/ndof={result.chi2_ndof:.3g}"
        ),
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Events / bin")
    ax.set_title(title or f"{result.method} — parameter {result.parameter:g} — timing fit")
    ax.grid(alpha=0.2)
    first = ax.legend(handles=[bars], labels=["Data"], loc="upper left")
    ax.add_artist(first)
    ax.legend(handles=[line], loc="upper right")

    selected_fraction = 100.0 * result.n_selected / result.n_total if result.n_total else 0.0
    ax.text(
        0.02, 0.04,
        f"selected={result.n_selected} ({selected_fraction:.1f}%)\n"
        f"valid timing pairs={result.n_valid}\n"
        f"bin={result.bin_width_ps:g} ps, phase={result.bin_phase_ps:g} ps",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9},
    )
    fig.tight_layout()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def plot_ctr_comparison(
    pico_rows: list[dict[str, Any]],
    scope_rows: list[dict[str, Any]],
    path: str | Path,
    *,
    dpi: int = 180,
    title: str = "Pico-TDC vs oscilloscope LED",
) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    if scope_rows:
        ordered = sorted(scope_rows, key=lambda row: float(row["voltage_V"]))
        x = np.asarray([float(r["voltage_V"]) for r in ordered])
        y = np.asarray([float(r["ctr_ps"]) for r in ordered])
        e = np.asarray([float(r.get("ctr_error_ps", np.nan)) for r in ordered])
        if np.any(np.isfinite(e)):
            ax.errorbar(
                x, y, yerr=np.where(np.isfinite(e), e, 0.0),
                marker="o", capsize=3, linewidth=1.8,
                label=str(ordered[0]["series_label"]),
            )
        else:
            ax.plot(x, y, marker="o", linewidth=1.8, label=str(ordered[0]["series_label"]))

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in pico_rows:
        threshold = row.get("timing_threshold_mV", "")
        key = "unknown" if threshold in ("", None) or not np.isfinite(float(threshold)) else f"{float(threshold):g}"
        groups.setdefault(key, []).append(row)

    for threshold, rows in sorted(groups.items()):
        ordered = sorted(rows, key=lambda row: float(row["voltage_V"]))
        x = np.asarray([float(r["voltage_V"]) for r in ordered])
        y = np.asarray([float(r["ctr_ps"]) for r in ordered])
        e = np.asarray([float(r.get("ctr_error_ps", np.nan)) for r in ordered])
        label = (
            "Pico-TDC timing channels"
            if threshold == "unknown"
            else f"Pico-TDC timing channels · T_th={threshold} mV"
        )
        if np.any(np.isfinite(e)):
            ax.errorbar(
                x, y, yerr=np.where(np.isfinite(e), e, 0.0),
                marker="s", capsize=3, linewidth=1.6, label=label,
            )
        else:
            ax.plot(x, y, marker="s", linewidth=1.6, label=label)

    ax.set_xlabel("Bias voltage [V]")
    ax.set_ylabel("CTR FWHM [ps]")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
