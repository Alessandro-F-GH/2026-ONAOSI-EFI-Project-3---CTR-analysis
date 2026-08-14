from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def plot_training_history(history: list[dict[str, Any]], directory: Path, dpi: int) -> None:
    """Optional trainer diagnostic; the compact experiment disables this output."""
    if not history:
        return
    directory.mkdir(parents=True, exist_ok=True)
    epochs = [int(row["epoch"]) for row in history]
    for train_key, validation_key, ylabel, filename in (
        ("train_rmse_ps", "validation_rmse_ps", "RMSE [ps]", "training_rmse.png"),
        ("train_ctr_ps", "validation_ctr_ps", "CTR [ps]", "training_ctr.png"),
        ("train_bias_ps", "validation_bias_ps", "Bias [ps]", "training_bias.png"),
    ):
        if not all(train_key in row and validation_key in row for row in history):
            continue
        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        ax.plot(epochs, [row[train_key] for row in history], label="fit")
        ax.plot(epochs, [row[validation_key] for row in history], label="early stop")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.minorticks_on()
        ax.grid(True, which="major", alpha=0.3)
        ax.grid(True, which="minor", alpha=0.12)
        ax.legend()
        fig.tight_layout()
        fig.savefig(directory / filename, dpi=int(dpi), bbox_inches="tight")
        plt.close(fig)
