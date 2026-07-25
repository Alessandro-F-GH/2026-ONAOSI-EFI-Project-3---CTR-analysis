from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from utils.fit import FitResult

_STYLE = {
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "legend.fontsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "axes.axisbelow": True,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
}


def _save(figure: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_training_history(history: list[dict[str, Any]], directory: Path, dpi: int) -> None:
    if not history:
        return
    epochs = np.asarray([row["epoch"] for row in history], dtype=int)
    with plt.rc_context(_STYLE):
        figure, axis = plt.subplots(figsize=(8.5, 5.5))
        axis.plot(epochs, [row["train_loss"] for row in history], label="Train")
        axis.plot(epochs, [row["validation_loss"] for row in history], label="Validation")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Optimization metric [ps]")
        axis.set_title("Training objective")
        axis.legend()
        figure.tight_layout()
        _save(figure, directory / "loss_curves.png", dpi)

    with plt.rc_context(_STYLE):
        figure, axis = plt.subplots(figsize=(8.5, 5.5))
        axis.plot(
            epochs,
            [row["train_corrected_std_ps"] for row in history],
            label="Train corrected σ",
        )
        axis.plot(
            epochs,
            [row["validation_corrected_std_ps"] for row in history],
            label="Validation corrected σ",
        )
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Corrected standard deviation [ps]")
        axis.set_title("Variance-reduction metric")
        axis.legend()
        figure.tight_layout()
        _save(figure, directory / "corrected_std_curves.png", dpi)

    with plt.rc_context(_STYLE):
        figure, axis = plt.subplots(figsize=(8.5, 5.5))
        train_ctr = np.asarray([row.get("train_ctr_ps", np.nan) for row in history], dtype=float)
        validation_ctr = np.asarray(
            [row.get("validation_ctr_ps", np.nan) for row in history], dtype=float
        )
        axis.plot(epochs, train_ctr, label="Train")
        axis.plot(epochs, validation_ctr, label="Validation")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Gaussian-fit CTR FWHM [ps]")
        axis.set_title("CTR during training")
        axis.legend()
        figure.tight_layout()
        _save(figure, directory / "ctr_curves.png", dpi)

    with plt.rc_context(_STYLE):
        figure, axis = plt.subplots(figsize=(8.5, 5.5))
        axis.plot(epochs, [row["learning_rate"] for row in history])
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Learning rate")
        axis.set_title("Learning-rate schedule")
        figure.tight_layout()
        _save(figure, directory / "learning_rate_curve.png", dpi)


def plot_method_fit(
    result: FitResult,
    path: Path,
    *,
    true_tof_ps: float,
    bias_ps: float,
    dpi: int,
) -> None:
    if not result.success or result.edges_ps.size < 2:
        return
    edges = np.asarray(result.edges_ps, dtype=float)
    counts = np.asarray(result.counts, dtype=float)
    expected = np.asarray(result.expected, dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    fit_width = max(result.fit_high_ps - result.fit_low_ps, float(np.median(widths)))
    margin = max(0.08 * fit_width, 2.0 * float(np.median(widths)))

    with plt.rc_context(_STYLE):
        figure, axis = plt.subplots(figsize=(9.2, 6.1))
        axis.bar(
            edges[:-1],
            counts,
            width=widths,
            align="edge",
            alpha=0.65,
            label="Test data",
        )
        axis.axvspan(
            result.fit_low_ps,
            result.fit_high_ps,
            alpha=0.12,
            label="Iterative fit interval",
        )
        axis.plot(centers, expected, linewidth=2.2, label="Gaussian fit")
        axis.axvline(true_tof_ps, linestyle="--", linewidth=1.5, label="True TOF")
        annotation = (
            f"μ = {result.mean_ps:.2f} ± {result.mean_error_ps:.2f} ps\n"
            f"bias = {bias_ps:.2f} ps\n"
            f"CTR = {result.ctr_ps:.2f} ± {result.ctr_error_ps:.2f} ps\n"
            f"χ²/ndof = {result.chi2_ndof:.3g}\n"
            f"N = {result.n_valid}"
        )
        axis.text(
            0.98,
            0.96,
            annotation,
            transform=axis.transAxes,
            ha="right",
            va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
        )
        axis.set_xlim(result.fit_low_ps - margin, result.fit_high_ps + margin)
        axis.set_xlabel("Calibrated TOF estimate [ps]")
        axis.set_ylabel("Events / bin")
        axis.set_title(f"{result.method}: blind-test Gaussian CTR fit")
        axis.legend(loc="upper left")
        figure.tight_layout()
        _save(figure, path, dpi)


def plot_metric_comparison(metrics: list[dict[str, Any]], path: Path, dpi: int) -> None:
    methods = [str(item["method"]) for item in metrics]
    ctr = np.asarray([float(item["ctr_ps"]) for item in metrics])
    ctr_error = np.asarray([float(item["ctr_error_ps"]) for item in metrics])
    bias = np.asarray([float(item["bias_ps"]) for item in metrics])
    x = np.arange(len(methods))
    with plt.rc_context(_STYLE):
        figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.2))
        axes[0].bar(x, ctr, yerr=ctr_error, capsize=4)
        axes[0].set_xticks(x, methods, rotation=15)
        axes[0].set_ylabel("CTR FWHM [ps]")
        axes[0].set_title("Blind-test CTR")
        axes[1].bar(x, bias)
        axes[1].axhline(0.0, linewidth=1.0)
        axes[1].set_xticks(x, methods, rotation=15)
        axes[1].set_ylabel("Bias [ps]")
        axes[1].set_title("Blind-test bias")
        figure.tight_layout()
        _save(figure, path, dpi)
