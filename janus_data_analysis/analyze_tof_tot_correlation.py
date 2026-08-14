#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from utils.config import load_config
from utils.selection import load_selection_csv
from utils.tabular import table_path


SUMMARY_FIELDS = [
    "run_id",
    "events_total",
    "events_selected",
    "selection_fraction",
    "tof_mean_lsb",
    "tof_sigma_lsb",
    "pearson_tot1",
    "pearson_tot5",
    "pearson_delta_tot",
    "spearman_tot1",
    "spearman_tot5",
    "spearman_delta_tot",
    "slope_tof_vs_tot1_lsb_per_lsb",
    "slope_tof_vs_tot5_lsb_per_lsb",
    "slope_tof_vs_delta_tot_lsb_per_lsb",
    "corr2d_intercept_lsb",
    "corr2d_beta_tot1_lsb_per_lsb",
    "corr2d_beta_tot5_lsb_per_lsb",
    "tof_sigma_corrected_lsb",
    "sigma_improvement_percent",
]


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(x * y) / denom)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        avg_rank = 0.5 * (start + stop - 1) + 1.0
        ranks[order[start:stop]] = avg_rank
        start = stop
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    return _pearson(_average_ranks(x), _average_ranks(y))


def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or np.all(x == x[0]):
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def _fit_plane(
    tot1: np.ndarray,
    tot5: np.ndarray,
    tof: np.ndarray,
) -> tuple[float, float, float]:
    x = np.column_stack(
        [
            np.ones(tot1.size, dtype=float),
            np.asarray(tot1, dtype=float),
            np.asarray(tot5, dtype=float),
        ]
    )
    y = np.asarray(tof, dtype=float)
    beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    return float(beta[0]), float(beta[1]), float(beta[2])


def _binned_profile(
    x: np.ndarray,
    y: np.ndarray,
    max_bins: int = 40,
    min_count: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0:
        return np.array([]), np.array([]), np.array([])

    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if not math.isfinite(xmin) or not math.isfinite(xmax) or xmin == xmax:
        return np.array([]), np.array([]), np.array([])

    nbins = min(max_bins, max(10, int(np.sqrt(x.size))))
    edges = np.linspace(xmin, xmax, nbins + 1)

    centers = []
    medians = []
    sigmas = []
    for i in range(nbins):
        if i == nbins - 1:
            mask = (x >= edges[i]) & (x <= edges[i + 1])
        else:
            mask = (x >= edges[i]) & (x < edges[i + 1])
        count = int(np.sum(mask))
        if count < min_count:
            continue
        xb = x[mask]
        yb = y[mask]
        centers.append(float(np.median(xb)))
        medians.append(float(np.median(yb)))
        sigmas.append(float(np.std(yb, ddof=1)) if count > 1 else float("nan"))

    return np.asarray(centers), np.asarray(medians), np.asarray(sigmas)


def _save_summary(path: Path, rows: Iterable[dict[str, object]]()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_scatter_with_profile(
    path: Path,
    run_id: str,
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    slope: float | None = None,
    intercept: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.0))

    ax.scatter(x, y, s=8, alpha=0.18, linewidths=0, label="events")

    centers, medians, _ = _binned_profile(x, y)
    if centers.size:
        ax.plot(centers, medians, "o-", color="black", lw=2, ms=4, label="binned median")

    if slope is not None and intercept is not None and math.isfinite(slope) and math.isfinite(intercept):
        xx = np.linspace(float(np.min(x)), float(np.max(x)), 200)
        ax.plot(xx, intercept + slope * xx, color="tab:red", lw=2, label=f"linear fit: y = {intercept:.2f} + {slope:.4f} x")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{run_id} — {title}")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_before_after_hist(
    path: Path,
    run_id: str,
    tof: np.ndarray,
    tof_corr: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.0))

    lo = min(float(np.min(tof)), float(np.min(tof_corr)))
    hi = max(float(np.max(tof)), float(np.max(tof_corr)))
    if lo == hi:
        hi = lo + 1.0
    bins = np.linspace(lo, hi, 100)

    ax.hist(tof, bins=bins, histtype="step", lw=2, label=f"raw, sigma={np.std(tof, ddof=1):.2f} LSB")
    ax.hist(tof_corr, bins=bins, histtype="step", lw=2, label=f"corrected, sigma={np.std(tof_corr, ddof=1):.2f} LSB")

    ax.set_xlabel("TOF [LSB]")
    ax.set_ylabel("Events")
    ax.set_title(f"{run_id} — TOF before/after ToT correction")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def analyze_run(run_dir: Path, cfg: dict) -> dict[str, object] | None:
    run_id = run_dir.name
    selection_path = table_path(run_dir / "csv", "selection", cfg)
    if not selection_path.exists():
        print(f"[skip] {run_id}: missing {selection_path.name}")
        return None

    try:
        measurements, duration_mask, alignment_mask = load_selection_csv(selection_path)
    except Exception as exc:
        print(f"[skip] {run_id}: failed to read selection table: {exc}")
    
        return None

    final_mask = duration_mask & alignment_mask
    n_total = int(measurements.size)
    n_sel = int(np.sum(final_mask))
    if n_sel < 10:
        print(f"[skip] {run_id}: too few selected events ({n_sel})")
        return None

    tot1 = measurements.duration_a_lsb[final_mask].astype(float)
    tot5 = measurements.duration_b_lsb[final_mask].astype(float)
    tof = (measurements.time_b_lsb[final_mask] - measurements.time_a_lsb[final_mask]).astype(float)
    delta_tot = tot5 - tot1

    pearson_tot1 = _pearson(tot1, tof)
    pearson_tot5 = _pearson(tot5, tof)
    pearson_delta = _pearson(delta_tot, tof)

    spearman_tot1 = _spearman(tot1, tof)
    spearman_tot5 = _spearman(tot5, tof)
    spearman_delta = _spearman(delta_tot, tof)

    slope_tot1, intercept_tot1 = _fit_line(tot1, tof)
    slope_tot5, intercept_tot5 = _fit_line(tot5, tof)
    slope_delta, intercept_delta = _fit_line(delta_tot, tof)

    intercept_2d, beta_tot1, beta_tot5 = _fit_plane(tot1, tot5, tof)
    tof_corr = tof - (intercept_2d + beta_tot1 * tot1 + beta_tot5 * tot5) + np.mean(tof)

    sigma_raw = float(np.std(tof, ddof=1))
    sigma_corr = float(np.std(tof_corr, ddof=1))
    improvement = float(100.0 * (sigma_raw - sigma_corr) / sigma_raw) if sigma_raw > 0.0 else float("nan")

    print(f"{run_id}: total in selection table = {measurements.size}")
    print(f"{run_id}: duration-selected     = {np.sum(duration_mask)}")
    print(f"{run_id}: alignment-selected    = {np.sum(alignment_mask)}")
    print(f"{run_id}: final selected        = {np.sum(duration_mask & alignment_mask)}")

    plot_dir = run_dir / "plots"
    csv_dir = run_dir / "csv"

    _plot_scatter_with_profile(
        plot_dir / "tof_vs_tot_ch1.png",
        run_id,
        tot1,
        tof,
        "ToT ch1 [LSB]",
        "TOF = t7 - t3 [LSB]",
        "TOF vs ToT(ch1)",
        slope_tot1,
        intercept_tot1,
    )
    _plot_scatter_with_profile(
        plot_dir / "tof_vs_tot_ch5.png",
        run_id,
        tot5,
        tof,
        "ToT ch5 [LSB]",
        "TOF = t7 - t3 [LSB]",
        "TOF vs ToT(ch5)",
        slope_tot5,
        intercept_tot5,
    )
    _plot_scatter_with_profile(
        plot_dir / "tof_vs_delta_tot.png",
        run_id,
        delta_tot,
        tof,
        "Delta ToT = ToT(ch5) - ToT(ch1) [LSB]",
        "TOF = t7 - t3 [LSB]",
        "TOF vs Delta ToT",
        slope_delta,
        intercept_delta,
    )
    _plot_before_after_hist(
        plot_dir / "tof_before_after_tot_correction.png",
        run_id,
        tof,
        tof_corr,
    )

    detail_rows = [
        {"metric": "events_total", "value": n_total},
        {"metric": "events_selected", "value": n_sel},
        {"metric": "selection_fraction", "value": n_sel / n_total if n_total else float("nan")},
        {"metric": "tof_mean_lsb", "value": float(np.mean(tof))},
        {"metric": "tof_sigma_lsb", "value": sigma_raw},
        {"metric": "pearson_tot1", "value": pearson_tot1},
        {"metric": "pearson_tot5", "value": pearson_tot5},
        {"metric": "pearson_delta_tot", "value": pearson_delta},
        {"metric": "spearman_tot1", "value": spearman_tot1},
        {"metric": "spearman_tot5", "value": spearman_tot5},
        {"metric": "spearman_delta_tot", "value": spearman_delta},
        {"metric": "slope_tof_vs_tot1_lsb_per_lsb", "value": slope_tot1},
        {"metric": "slope_tof_vs_tot5_lsb_per_lsb", "value": slope_tot5},
        {"metric": "slope_tof_vs_delta_tot_lsb_per_lsb", "value": slope_delta},
        {"metric": "corr2d_intercept_lsb", "value": intercept_2d},
        {"metric": "corr2d_beta_tot1_lsb_per_lsb", "value": beta_tot1},
        {"metric": "corr2d_beta_tot5_lsb_per_lsb", "value": beta_tot5},
        {"metric": "tof_sigma_corrected_lsb", "value": sigma_corr},
        {"metric": "sigma_improvement_percent", "value": improvement},
    ]

    detail_path = csv_dir / "tof_tot_correlation.csv"
    with detail_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for row in detail_rows:
            writer.writerow(row)

    print(
        f"[ok] {run_id}: selected={n_sel}, "
        f"r(delta_tot, tof)={pearson_delta:.4f}, "
        f"sigma_raw={sigma_raw:.3f}, sigma_corr={sigma_corr:.3f}"
    )

    return {
        "run_id": run_id,
        "events_total": n_total,
        "events_selected": n_sel,
        "selection_fraction": n_sel / n_total if n_total else float("nan"),
        "tof_mean_lsb": float(np.mean(tof)),
        "tof_sigma_lsb": sigma_raw,
        "pearson_tot1": pearson_tot1,
        "pearson_tot5": pearson_tot5,
        "pearson_delta_tot": pearson_delta,
        "spearman_tot1": spearman_tot1,
        "spearman_tot5": spearman_tot5,
        "spearman_delta_tot": spearman_delta,
        "slope_tof_vs_tot1_lsb_per_lsb": slope_tot1,
        "slope_tof_vs_tot5_lsb_per_lsb": slope_tot5,
        "slope_tof_vs_delta_tot_lsb_per_lsb": slope_delta,
        "corr2d_intercept_lsb": intercept_2d,
        "corr2d_beta_tot1_lsb_per_lsb": beta_tot1,
        "corr2d_beta_tot5_lsb_per_lsb": beta_tot5,
        "tof_sigma_corrected_lsb": sigma_corr,
        "sigma_improvement_percent": improvement,
    }


def _discover_run_dirs(root: Path) -> list[Path]:
    run_dirs = []
    for path in root.glob("outputs/*/analysis/Run*"):
        if path.is_dir():
            run_dirs.append(path)
    return sorted(run_dirs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze correlation between TOF = ch7 - ch3 and ToT of associated "
            "energy channels ch1 and ch5 using existing Janus selection outputs."
        )
    )
    parser.add_argument(
        "run_output",
        nargs="?",
        help="Path to one run output directory, for example outputs/07-13/analysis/Run4700",
    )
    parser.add_argument(
        "-c",
        "--config",
        default="config/janus_pipeline.json",
        help="Repository config path",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Analyze every outputs/*/analysis/Run* directory containing a selection table",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    cfg = load_config(root / args.config, root)

    if args.all_runs:
        run_dirs = _discover_run_dirs(root)
    elif args.run_output:
        run_dirs = [Path(args.run_output)]
        if not run_dirs[0].is_absolute():
            run_dirs[0] = (root / run_dirs[0]).resolve()
    else:
        parser.error("provide run_output or use --all-runs")
        return

    results = []
    for run_dir in run_dirs:
        if not run_dir.exists():
            print(f"[skip] missing directory: {run_dir}")
            continue
        row = analyze_run(run_dir, cfg)
        if row is not None:
            results.append(row)

    if args.all_runs and results:
        summary_path = root / "outputs" / "tof_tot_correlation_summary.csv"
        _save_summary(summary_path, results)
        print(f"[done] wrote summary: {summary_path}")


if __name__ == "__main__":
    main()