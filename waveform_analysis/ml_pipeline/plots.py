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


def plot_top_correction_event(
    dataset: Any,
    record: dict[str, Any],
    path: Path,
    *,
    input_transform: str,
    input_waveform_source: str,
    prediction_target: str,
    model_name: str,
    dpi: int,
) -> None:
    """Plot the waveform pair and timing movement for one useful correction."""

    from .input_transform import apply_input_transform, transform_relative_time_ps

    row = int(record["dataset_index"])
    raw_pair = np.asarray(dataset.windows_mV[row], dtype=np.float64)
    raw_time = np.asarray(dataset.relative_time_ps, dtype=np.float64)
    model_pair = np.asarray(
        apply_input_transform(raw_pair, input_transform), dtype=np.float64
    )
    model_time = transform_relative_time_ps(raw_time, input_transform)

    with plt.rc_context(_STYLE):
        figure, axes = plt.subplots(
            3,
            1,
            figsize=(10.5, 9.0),
            gridspec_kw={"height_ratios": [2.2, 1.6, 1.0]},
        )
        waveform_axis, difference_axis, timing_axis = axes

        waveform_axis.plot(raw_time, raw_pair[0], label="Detector 1")
        waveform_axis.plot(raw_time, raw_pair[1], label="Detector 2")
        waveform_axis.set_ylabel("Signal [mV]")
        waveform_axis.set_title(
            f"Rank {int(record['rank'])}: +{float(record['improvement_ps']):.3f} ps closer to true TOF"
        )
        waveform_axis.legend()

        difference_axis.plot(
            model_time,
            model_pair[0] - model_pair[1],
            label="Detector 1 - detector 2",
        )
        difference_axis.axhline(0.0, linewidth=0.8)
        difference_axis.set_xlabel("Relative time [ps]")
        difference_axis.set_ylabel(
            "Difference of first differences [mV]"
            if input_transform == "differentiate"
            else (
                "Raw then first-difference pair feature [mV]"
                if input_transform == "concatenate_diff"
                else "Pair sample difference [mV]"
            )
        )
        difference_axis.set_title("Model-input asymmetry (diagnostic)")
        difference_axis.legend()

        true_value = float(record["true_tof_ps"])
        raw_value = float(record["raw_ps"])
        corrected_value = float(record["corrected_ps"])
        timing_axis.scatter(
            [true_value, raw_value, corrected_value],
            [0.0, 0.0, 0.0],
            s=[90, 75, 75],
            zorder=3,
        )
        timing_axis.annotate(
            "true TOF",
            (true_value, 0.0),
            xytext=(0, 18),
            textcoords="offset points",
            ha="center",
        )
        timing_axis.annotate(
            "raw",
            (raw_value, 0.0),
            xytext=(0, 18),
            textcoords="offset points",
            ha="center",
        )
        timing_axis.annotate(
            "corrected",
            (corrected_value, 0.0),
            xytext=(0, -26),
            textcoords="offset points",
            ha="center",
        )
        timing_axis.annotate(
            "",
            xy=(corrected_value, 0.0),
            xytext=(raw_value, 0.0),
            arrowprops={"arrowstyle": "->", "linewidth": 1.8},
        )
        span = max(
            abs(raw_value - true_value),
            abs(corrected_value - true_value),
            10.0,
        )
        timing_axis.set_xlim(true_value - 1.35 * span, true_value + 1.35 * span)
        timing_axis.set_ylim(-1.0, 1.0)
        timing_axis.set_yticks([])
        timing_axis.set_xlabel("Time difference [ps]")
        timing_axis.set_title(
            f"raw {raw_value:.3f} ps - prediction {float(record['predicted_correction_ps']):.3f} ps "
            f"= corrected {corrected_value:.3f} ps"
        )

        figure.suptitle(
            f"{model_name} | event_id={record['event_id']} | row={row} | "
            f"input={input_waveform_source}/{input_transform} | target={prediction_target}",
            fontsize=12,
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        _save(figure, path, dpi)


def plot_model_output_correlation(
    matrix: np.ndarray,
    labels: list[str],
    path: Path,
    *,
    dpi: int,
    annotate: bool = True,
) -> None:
    """Plot a Pearson matrix for per-event model correction outputs."""

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("Correlation matrix must be square")
    if values.shape[0] != len(labels):
        raise ValueError("Correlation labels do not match matrix dimensions")
    if not labels:
        return

    masked = np.ma.masked_invalid(values)
    size = max(7.0, 0.82 * len(labels) + 3.0)
    with plt.rc_context(_STYLE):
        figure, axis = plt.subplots(figsize=(size, size))
        image = axis.imshow(masked, vmin=-1.0, vmax=1.0, cmap="coolwarm")
        axis.grid(False)
        positions = np.arange(len(labels))
        axis.set_xticks(positions)
        axis.set_yticks(positions)
        axis.set_xticklabels(labels, rotation=40, ha="right")
        axis.set_yticklabels(labels)
        axis.set_title("Model correction-output correlation")
        colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        colorbar.set_label("Pearson correlation")

        if annotate:
            for row in range(values.shape[0]):
                for column in range(values.shape[1]):
                    value = values[row, column]
                    text = "n/a" if not np.isfinite(value) else f"{value:.3f}"
                    axis.text(
                        column,
                        row,
                        text,
                        ha="center",
                        va="center",
                        color="white" if np.isfinite(value) and abs(value) >= 0.55 else "black",
                        fontsize=max(7.0, 11.0 - 0.25 * len(labels)),
                    )

        figure.tight_layout()
        _save(figure, path, dpi)
