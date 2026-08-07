#!/usr/bin/env python3
"""Plot time-resolved linear-model coefficient norms for different windows.

The study stores only the CV-selected hyperparameter trial for each physical
window in ``linear_model_weights.csv``.  Checkpoints are not required.

For every feature/time position, the default plotted quantity is the RMS
coefficient over CV folds:

    weight_norm(t) = sqrt(mean_k(weight_k(t)^2)).

For a single linear feature this is the Euclidean norm of the fold-coefficient
vector divided by sqrt(K).  It is positive, robust to fold-dependent sign
changes, and equals ``abs(weight)`` when there is only one fold.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "root_file",
    "channel_mode",
    "model_id",
    "model_type",
    "loss_id",
    "input_transform",
    "window_id",
    "window_start_ns",
    "window_end_ns",
    "fold_id",
    "regularization",
    "alpha",
    "component",
    "feature_kind",
    "relative_time_ns",
    "weight_normalized",
    "weight_physical_ps_per_mV",
    "is_selected_window",
}


def _optional_exact_filter(
    frame: pd.DataFrame,
    column: str,
    value: str | None,
) -> pd.DataFrame:
    if value is None:
        return frame
    return frame[frame[column].astype(str) == str(value)].copy()


def _safe_name(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")
    return text or "plot"


def _aggregate(values: pd.Series, method: str) -> float:
    array = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if array.size == 0:
        return float("nan")
    if method == "rms":
        return float(np.sqrt(np.mean(array * array)))
    if method == "mean_abs":
        return float(np.mean(np.abs(array)))
    if method == "mean_signed":
        return float(np.mean(array))
    raise ValueError(f"Unsupported aggregation: {method}")


def _series_label(row: pd.Series) -> str:
    alpha = float(row["alpha"])
    alpha_text = f", alpha={alpha:.3g}" if row["regularization"] != "none" else ""
    selected = " [selected]" if int(row["is_selected_window"]) == 1 else ""
    return (
        f"{row['window_id']} "
        f"[{float(row['window_start_ns']):g}, {float(row['window_end_ns']):g}] ns"
        f"{alpha_text}{selected}"
    )


def _plot_group(
    group: pd.DataFrame,
    *,
    output_dir: Path,
    weight_column: str,
    weight_space: str,
    aggregation: str,
    dpi: int,
    show: bool,
) -> pd.DataFrame:
    identity_columns = [
        "root_id",
        "root_file",
        "file_name",
        "channel_mode",
        "model_id",
        "loss_id",
        "input_transform",
        "component",
        "feature_kind",
    ]
    identity = group.iloc[0]

    grouping = [
        "window_id",
        "window_start_ns",
        "window_end_ns",
        "window_length_ns",
        "regularization",
        "alpha",
        "is_selected_window",
        "relative_time_ns",
    ]
    summary = (
        group.groupby(grouping, dropna=False)
        .agg(
            weight_norm=(weight_column, lambda values: _aggregate(values, aggregation)),
            weight_mean=(weight_column, "mean"),
            weight_std=(weight_column, "std"),
            fold_count=("fold_id", "nunique"),
            cv_mean_ctr_ps=("cv_mean_ctr_ps", "first"),
            cv_mean_rmse_ps=("cv_mean_rmse_ps", "first"),
        )
        .reset_index()
    )
    for column in identity_columns:
        summary[column] = identity[column]

    figure, axis = plt.subplots(figsize=(9.0, 5.5))
    window_order = (
        summary[["window_id", "window_start_ns", "window_end_ns", "window_length_ns"]]
        .drop_duplicates()
        .sort_values(["window_length_ns", "window_start_ns", "window_end_ns"])
    )
    for window_id in window_order["window_id"]:
        curve = summary[summary["window_id"] == window_id].sort_values(
            "relative_time_ns"
        )
        axis.plot(
            curve["relative_time_ns"],
            curve["weight_norm"],
            marker="o",
            markersize=3.5,
            linewidth=1.4,
            label=_series_label(curve.iloc[0]),
        )

    axis.axvline(0.0, linestyle="--", linewidth=1.0)
    axis.set_xlabel("Time relative to native window anchor [ns]")
    if weight_space == "physical":
        axis.set_ylabel(
            f"{aggregation.replace('_', ' ').upper()} coefficient [ps/mV]"
        )
    else:
        axis.set_ylabel(
            f"{aggregation.replace('_', ' ').upper()} normalized coefficient [ps/z-score]"
        )
    axis.set_title(
        "Linear coefficient norm vs time\n"
        f"{identity['file_name']} | {identity['channel_mode']} | "
        f"{identity['model_id']} | {identity['loss_id']} | "
        f"{identity['input_transform']} | {identity['component']} "
        f"{identity['feature_kind']}"
    )
    axis.grid(True, alpha=0.3)
    axis.legend(title="Window", fontsize=8)
    figure.tight_layout()

    stem = "__".join(
        _safe_name(identity[column])
        for column in (
            "file_name",
            "channel_mode",
            "model_id",
            "loss_id",
            "input_transform",
            "component",
            "feature_kind",
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / f"{stem}.png", dpi=dpi)
    summary.to_csv(output_dir / f"{stem}.csv", index=False)
    if show:
        plt.show()
    plt.close(figure)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot fold-RMS linear-regression coefficients versus relative time "
            "for every tested physical window."
        )
    )
    parser.add_argument(
        "--study-dir",
        type=Path,
        required=True,
        help="Study directory containing linear_model_weights.csv.",
    )
    parser.add_argument("--file", dest="file_name", help="ROOT basename or root_id.")
    parser.add_argument("--channel-mode")
    parser.add_argument("--model-id")
    parser.add_argument("--loss-id")
    parser.add_argument("--transform", dest="input_transform")
    parser.add_argument(
        "--regularization",
        choices=("none", "ridge", "lasso"),
    )
    parser.add_argument("--component", help="For example energy or timing.")
    parser.add_argument(
        "--feature-kind",
        choices=("raw", "first_difference"),
    )
    parser.add_argument(
        "--selected-window-only",
        action="store_true",
        help="Plot only the CV-selected window instead of comparing all windows.",
    )
    parser.add_argument(
        "--weight-space",
        choices=("physical", "normalized"),
        default="physical",
    )
    parser.add_argument(
        "--aggregation",
        choices=("rms", "mean_abs", "mean_signed"),
        default="rms",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    study_dir = args.study_dir.resolve()
    source = study_dir / "linear_model_weights.csv"
    if not source.is_file():
        raise FileNotFoundError(
            f"Linear weight table not found: {source}. Run/resume the study with "
            "a linear_regression/ridge_regression/lasso_regression model enabled."
        )

    frame = pd.read_csv(source)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(
            "linear_model_weights.csv is missing columns: "
            + ", ".join(sorted(missing))
        )
    frame = frame[frame["model_type"] == "linear_regression"].copy()
    frame["file_name"] = frame["root_file"].map(lambda value: Path(str(value)).name)

    if args.file_name is not None:
        wanted = str(args.file_name)
        frame = frame[
            (frame["file_name"].astype(str) == wanted)
            | (frame["root_id"].astype(str) == wanted)
        ].copy()
    for column, value in (
        ("channel_mode", args.channel_mode),
        ("model_id", args.model_id),
        ("loss_id", args.loss_id),
        ("input_transform", args.input_transform),
        ("regularization", args.regularization),
        ("component", args.component),
        ("feature_kind", args.feature_kind),
    ):
        frame = _optional_exact_filter(frame, column, value)
    if args.selected_window_only:
        frame = frame[pd.to_numeric(frame["is_selected_window"], errors="coerce") == 1]
    if frame.empty:
        raise ValueError("No linear-weight rows match the requested filters")

    numeric = [
        "window_start_ns",
        "window_end_ns",
        "window_length_ns",
        "fold_id",
        "alpha",
        "relative_time_ns",
        "weight_normalized",
        "weight_physical_ps_per_mV",
        "cv_mean_ctr_ps",
        "cv_mean_rmse_ps",
        "is_selected_window",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["relative_time_ns"])

    weight_column = (
        "weight_physical_ps_per_mV"
        if args.weight_space == "physical"
        else "weight_normalized"
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else study_dir / "summary_plots" / "linear_weights"
    )

    group_columns = [
        "root_id",
        "root_file",
        "file_name",
        "channel_mode",
        "model_id",
        "loss_id",
        "input_transform",
        "component",
        "feature_kind",
    ]
    summaries: list[pd.DataFrame] = []
    for _identity, group in frame.groupby(group_columns, dropna=False):
        summaries.append(
            _plot_group(
                group,
                output_dir=output_dir,
                weight_column=weight_column,
                weight_space=args.weight_space,
                aggregation=args.aggregation,
                dpi=args.dpi,
                show=args.show,
            )
        )

    combined = pd.concat(summaries, ignore_index=True)
    combined.to_csv(output_dir / "linear_weight_norms_all.csv", index=False)
    print(f"Created {len(summaries)} plot(s) under {output_dir}")


if __name__ == "__main__":
    main()
