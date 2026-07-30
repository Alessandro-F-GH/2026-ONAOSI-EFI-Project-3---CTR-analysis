from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

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
    figure.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)


def plot_training_history(history: list[dict[str, Any]], directory: Path, dpi: int) -> None:
    if not history:
        return
    epochs = np.asarray([row["epoch"] for row in history], dtype=int)
    plots = (
        ("rmse", "RMSE [ps]", "training_rmse.png", "train_rmse_ps", "validation_rmse_ps"),
        ("CTR", "CTR [ps]", "training_ctr.png", "train_ctr_ps", "validation_ctr_ps"),
        ("bias", "Gaussian bias [ps]", "training_bias.png", "train_bias_ps", "validation_bias_ps"),
    )
    for label, ylabel, filename, train_key, validation_key in plots:
        with plt.rc_context(_STYLE):
            figure, axis = plt.subplots(figsize=(8.5, 5.5))
            axis.plot(epochs, [row[train_key] for row in history], label=f"Train {label}")
            axis.plot(epochs, [row[validation_key] for row in history], label=f"Validation {label}")
            axis.set_xlabel("Epoch")
            axis.set_ylabel(ylabel)
            axis.legend()
            _save(figure, directory / filename, dpi)


def plot_metric_bars(rows: list[dict[str, Any]], output_dir: Path, dpi: int) -> None:
    if not rows:
        return
    labels = [str(row["method"]) for row in rows]
    x = np.arange(len(labels))
    for key, ylabel, filename in (
        ("ctr_ps", "CTR [ps]", "ctr_comparison.png"),
        ("gaussian_bias_ps", "Gaussian bias [ps]", "bias_comparison.png"),
    ):
        with plt.rc_context(_STYLE):
            figure, axis = plt.subplots(figsize=(max(9.0, 1.4 * len(labels)), 5.8))
            axis.bar(x, [float(row.get(key, np.nan)) for row in rows])
            axis.axhline(0.0, linewidth=0.8)
            axis.set_ylabel(ylabel)
            axis.set_xticks(x)
            axis.set_xticklabels(labels, rotation=25, ha="right")
            figure.tight_layout()
            _save(figure, output_dir / filename, dpi)
