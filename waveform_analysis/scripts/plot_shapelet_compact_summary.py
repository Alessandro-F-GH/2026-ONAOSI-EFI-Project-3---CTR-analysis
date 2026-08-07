#!/usr/bin/env python3
"""Create one compact summary figure for a fitted shapelet-correction model.

The script is a standalone companion to ``run_shapelet_correction_study.py``
and ``plot_shapelet_correction_events.py``. It reuses their exact preprocessing,
fold reconstruction, shapelet-distance implementation, and fixed-alpha Ridge
fit without registering anything in the main ``ml_pipeline`` study.

For one file/channel/window/fold/K model, the output figure contains:

1. selected shapelet waveforms drawn at their physical time positions;
2. mean absolute linear contribution of each shapelet across the evaluation split;
3. mean signed contribution for the top positive, top negative, and
   smallest-absolute linear-correction event groups.

The Ridge intercept and anchor shift are deliberately excluded from all
importance quantities. Every contribution shown is

    contribution_ik = beta_k * standardized_distance_ik

in ps.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

_EVENT_SCRIPT = Path(__file__).with_name("plot_shapelet_correction_events.py")
_SPEC = importlib.util.spec_from_file_location(
    "_shapelet_event_diagnostics_for_compact_summary",
    _EVENT_SCRIPT,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - defensive
    raise RuntimeError(f"Unable to load event diagnostics module: {_EVENT_SCRIPT}")
_EVENT = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _EVENT
_SPEC.loader.exec_module(_EVENT)

from ml_pipeline.common import setup_logging
from ml_pipeline.input_transform import (
    INPUT_TRANSFORM_NORMALIZE,
    materialize_training_input_cache,
)
from ml_pipeline.prediction import prediction_window_dataset_view
from ml_pipeline.study import _ensure_preprocessed, _fold_masks, _root_id
from ml_pipeline.study_config import CHANNEL_MODES, load_study_config
from ml_pipeline.torch_data import compute_normalization, factored_correction_target_ps


ShapeletCandidate = _EVENT.ShapeletCandidate
build_undersampling_plan = _EVENT.build_undersampling_plan
materialize_shapelet_features = _EVENT.materialize_shapelet_features
_materialize_difference_matrix = _EVENT._materialize_difference_matrix


CATEGORY_ORDER = ("negative", "near_zero", "positive")
CATEGORY_LABELS = {
    "negative": "Most negative",
    "near_zero": "Smallest |linear|",
    "positive": "Most positive",
}


def _safe_display_shapelet(values: np.ndarray) -> np.ndarray:
    """Return a zero-centred, unit-scale shapelet for display only."""

    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(vector)
    if not np.any(finite):
        return np.zeros_like(vector)
    centre = float(np.nanmean(vector))
    scale = float(np.nanstd(vector, ddof=0))
    if not math.isfinite(scale) or scale <= 1.0e-12:
        return np.zeros_like(vector)
    return (vector - centre) / scale


def category_positions(
    selections: Sequence[Any],
) -> dict[str, np.ndarray]:
    """Convert EventSelection objects to local-position arrays by category."""

    grouped: dict[str, list[int]] = {name: [] for name in CATEGORY_ORDER}
    for selection in selections:
        if selection.category in grouped:
            grouped[selection.category].append(int(selection.local_position))
    return {
        name: np.asarray(grouped[name], dtype=np.int64)
        for name in CATEGORY_ORDER
    }


def summarize_shapelet_contributions(
    contributions: np.ndarray,
    category_indices: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return global mean absolute and category mean signed contributions."""

    matrix = np.asarray(contributions, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("contributions must have shape (events, shapelets)")
    mean_absolute = np.nanmean(np.abs(matrix), axis=0)
    category_means = np.full(
        (matrix.shape[1], len(CATEGORY_ORDER)),
        np.nan,
        dtype=np.float64,
    )
    for column, category in enumerate(CATEGORY_ORDER):
        indices = np.asarray(category_indices.get(category, []), dtype=np.int64)
        if indices.size:
            category_means[:, column] = np.nanmean(matrix[indices], axis=0)
    return mean_absolute, category_means


def _importance_linewidths(mean_absolute: np.ndarray) -> np.ndarray:
    values = np.asarray(mean_absolute, dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.full(values.shape, 1.5, dtype=np.float64)
    maximum = float(np.nanmax(values[finite]))
    if maximum <= 1.0e-12:
        return np.full(values.shape, 1.5, dtype=np.float64)
    return 1.2 + 4.0 * np.clip(values / maximum, 0.0, 1.0)


def create_compact_summary_figure(
    *,
    output_path: Path,
    shapelets: Sequence[ShapeletCandidate],
    catalog: pd.DataFrame,
    coefficients: np.ndarray,
    contributions: np.ndarray,
    category_indices: Mapping[str, np.ndarray],
    title: str,
    subtitle: str,
    analysis_start_ns: float,
    analysis_end_ns: float,
    dpi: int = 180,
) -> pd.DataFrame:
    """Render and save the compact summary; return its numeric summary table."""

    if not shapelets:
        raise ValueError("At least one shapelet is required")
    if len(shapelets) != len(catalog):
        raise ValueError("shapelets and catalog must have the same length")

    coefficient_vector = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    if coefficient_vector.size != len(shapelets):
        raise ValueError("Coefficient count does not match shapelet count")

    mean_absolute, category_means = summarize_shapelet_contributions(
        contributions,
        category_indices,
    )
    line_widths = _importance_linewidths(mean_absolute)

    ranks = pd.to_numeric(catalog["rank"], errors="coerce").to_numpy(int)
    order = np.argsort(ranks)
    ranks = ranks[order]
    ordered_shapelets = [shapelets[index] for index in order]
    coefficient_vector = coefficient_vector[order]
    mean_absolute = mean_absolute[order]
    category_means = category_means[order]
    line_widths = line_widths[order]
    ordered_catalog = catalog.iloc[order].reset_index(drop=True)

    figure = plt.figure(figsize=(14.2, max(6.6, 0.48 * len(shapelets) + 3.4)))
    grid = figure.add_gridspec(
        1,
        3,
        width_ratios=(4.7, 1.65, 2.25),
        left=0.075,
        right=0.955,
        bottom=0.12,
        top=0.79,
        wspace=0.42,
    )
    shape_axis = figure.add_subplot(grid[0, 0])
    importance_axis = figure.add_subplot(grid[0, 1])
    heatmap_axis = figure.add_subplot(grid[0, 2])

    row_positions = np.arange(len(ordered_shapelets), dtype=float)
    amplitude_scale = 0.28

    for row_position, candidate, width in zip(
        row_positions,
        ordered_shapelets,
        line_widths,
    ):
        values = _safe_display_shapelet(candidate.values)
        times_ns = np.linspace(
            float(candidate.start_time_ps) / 1000.0,
            float(candidate.end_time_ps) / 1000.0,
            values.size,
        )
        shape_axis.hlines(
            row_position,
            times_ns[0],
            times_ns[-1],
            linewidth=0.7,
            alpha=0.30,
        )
        shape_axis.plot(
            times_ns,
            row_position + amplitude_scale * values,
            linewidth=float(width),
            solid_capstyle="round",
        )

    shape_axis.axvline(0.0, linestyle="--", linewidth=1.1)
    shape_axis.set_xlim(float(analysis_start_ns), float(analysis_end_ns))
    shape_axis.set_ylim(len(ordered_shapelets) - 0.35, -0.65)
    shape_axis.set_yticks(row_positions)
    shape_axis.set_yticklabels([f"S{rank}" for rank in ranks])
    shape_axis.set_xlabel("Time relative to LED anchor [ns]")
    shape_axis.set_ylabel("Selected shapelet")
    shape_axis.set_title(
        "Shapelet pattern at physical position\n"
        "line width ∝ mean absolute contribution"
    )
    shape_axis.grid(axis="x", alpha=0.22)

    importance_axis.barh(row_positions, mean_absolute)
    importance_axis.set_ylim(len(ordered_shapelets) - 0.35, -0.65)
    importance_axis.set_yticks(row_positions)
    importance_axis.set_yticklabels([])
    importance_axis.set_xlabel("Mean |contribution| [ps]")
    importance_axis.set_title("Global importance")
    importance_axis.grid(axis="x", alpha=0.22)
    maximum_importance = float(np.nanmax(mean_absolute)) if mean_absolute.size else 0.0
    annotation_offset = 0.025 * maximum_importance if maximum_importance > 0 else 0.02
    for row_position, value in zip(row_positions, mean_absolute):
        if math.isfinite(float(value)):
            importance_axis.text(
                float(value) + annotation_offset,
                row_position,
                f"{float(value):.2f}",
                va="center",
                ha="left",
                fontsize=8.5,
            )
    if maximum_importance > 0:
        importance_axis.set_xlim(0.0, maximum_importance * 1.24)

    finite_heatmap = category_means[np.isfinite(category_means)]
    heat_limit = (
        float(np.max(np.abs(finite_heatmap)))
        if finite_heatmap.size
        else 1.0
    )
    if heat_limit <= 1.0e-12:
        heat_limit = 1.0
    image = heatmap_axis.imshow(
        category_means,
        aspect="auto",
        interpolation="nearest",
        vmin=-heat_limit,
        vmax=heat_limit,
        cmap="coolwarm",
    )
    heatmap_axis.set_xticks(np.arange(len(CATEGORY_ORDER)))
    heatmap_axis.set_xticklabels(
        [CATEGORY_LABELS[name] for name in CATEGORY_ORDER],
        rotation=35,
        ha="right",
    )
    heatmap_axis.set_yticks(row_positions)
    heatmap_axis.set_yticklabels([])
    heatmap_axis.set_title("Mean signed contribution [ps]")

    for row in range(category_means.shape[0]):
        for column in range(category_means.shape[1]):
            value = category_means[row, column]
            if math.isfinite(float(value)):
                heatmap_axis.text(
                    column,
                    row,
                    f"{float(value):+.1f}",
                    ha="center",
                    va="center",
                    fontsize=8.0,
                )

    colorbar = figure.colorbar(
        image,
        ax=heatmap_axis,
        fraction=0.052,
        pad=0.045,
    )
    colorbar.set_label("Contribution [ps]")

    figure.suptitle(title, y=0.975, fontsize=15, fontweight="semibold")
    figure.text(
        0.5,
        0.925,
        subtitle,
        ha="center",
        va="center",
        fontsize=10,
    )

    legend_handles = [
        Line2D([0], [0], linewidth=1.2, label="Lower mean |contribution|"),
        Line2D([0], [0], linewidth=5.2, label="Higher mean |contribution|"),
        Line2D([0], [0], linestyle="--", linewidth=1.1, label="LED anchor"),
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.875),
        ncol=3,
        frameon=False,
        handlelength=3.0,
        columnspacing=2.2,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)

    summary = pd.DataFrame(
        {
            "rank": ranks,
            "candidate_id": [candidate.candidate_id for candidate in ordered_shapelets],
            "component": [candidate.component_name for candidate in ordered_shapelets],
            "start_time_ns": [
                float(candidate.start_time_ps) / 1000.0
                for candidate in ordered_shapelets
            ],
            "end_time_ns": [
                float(candidate.end_time_ps) / 1000.0
                for candidate in ordered_shapelets
            ],
            "duration_ns": [candidate.duration_ns for candidate in ordered_shapelets],
            "ridge_coefficient": coefficient_vector,
            "mean_absolute_contribution_ps": mean_absolute,
            "mean_negative_group_contribution_ps": category_means[:, 0],
            "mean_near_zero_group_contribution_ps": category_means[:, 1],
            "mean_positive_group_contribution_ps": category_means[:, 2],
        }
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one compact shapelet summary showing physical location, waveform "
            "shape, global mean absolute contribution, and signed event-group contribution."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--study-dir",
        type=Path,
        default=Path("results/adhoc/shapelet_correction_study"),
    )
    parser.add_argument("--file", required=True)
    parser.add_argument(
        "--channel-mode",
        required=True,
        choices=sorted(CHANNEL_MODES),
    )
    parser.add_argument("--window-id", required=True)
    parser.add_argument("--fold-id", type=int, default=0)
    parser.add_argument("--n-shapelets", type=int, default=10)
    parser.add_argument("--split", choices=["validation", "blind"], default="validation")
    parser.add_argument(
        "--top-k-events",
        type=int,
        default=10,
        help=(
            "Events per positive, negative, and smallest-absolute linear-contribution "
            "group used in the contribution heatmap"
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rebuild-preprocessing", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--feature-chunk-size", type=int, default=512)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.n_shapelets <= 0:
        raise ValueError("--n-shapelets must be positive")
    if args.top_k_events <= 0:
        raise ValueError("--top-k-events must be positive")

    study_dir = _EVENT._resolve_path(args.study_dir)
    resolved, fold_rows, shapelet_rows, value_rows = _EVENT._load_study_tables(study_dir)
    config = load_study_config(args.config, PROJECT)
    root_file = _EVENT._resolve_root_file(args.file, config, fold_rows)
    root_id = _root_id(root_file)

    if args.channel_mode not in resolved.get("channel_modes", []):
        raise ValueError(
            f"Channel mode {args.channel_mode!r} is absent from the saved shapelet study"
        )
    resolved_windows = {
        str(window["id"]): window for window in resolved.get("windows", [])
    }
    if args.window_id not in resolved_windows:
        raise ValueError(
            f"Window {args.window_id!r} is absent from the saved shapelet study; "
            f"available: {sorted(resolved_windows)}"
        )
    window = resolved_windows[args.window_id]
    transform = str(resolved["input_transform"])
    undersampling_factor = int(resolved["undersampling_factor"])
    position_mode = str(resolved["position_mode"])
    distance_metric = str(resolved.get("distance_metric", "mse"))
    dtw_radius_points = int(resolved.get("dtw_radius_points", 0))
    local_z_normalize = bool(resolved.get("local_z_normalize", False))

    default_output_dir = (
        study_dir
        / "event_diagnostics"
        / _EVENT._safe_name(root_file.stem)
        / _EVENT._safe_name(args.channel_mode)
        / _EVENT._safe_name(args.window_id)
        / f"fold_{args.fold_id}"
        / f"k_{args.n_shapelets}"
        / args.split
    )
    output_path = (
        _EVENT._resolve_path(args.output)
        if args.output is not None
        else default_output_dir / "compact_shapelet_summary.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(
        output_path.parent / "compact_shapelet_summary.log",
        "INFO",
    )

    fold_table = fold_rows.copy()
    for column in ("fold_id", "selected_shapelet_count", "ridge_alpha"):
        fold_table[column] = pd.to_numeric(fold_table[column], errors="coerce")
    fold_row = _EVENT._single_matching_row(
        fold_table,
        (fold_table["root_id"].astype(str) == root_id)
        & (fold_table["file_name"].astype(str) == root_file.name)
        & (fold_table["channel_mode"].astype(str) == args.channel_mode)
        & (fold_table["window_id"].astype(str) == args.window_id)
        & (fold_table["fold_id"] == int(args.fold_id))
        & (fold_table["selected_shapelet_count"] == int(args.n_shapelets)),
        description="fold-result",
    )
    ridge_alpha = float(fold_row["ridge_alpha"])

    shapelets, catalog = _EVENT._reconstruct_shapelets(
        shapelet_rows,
        value_rows,
        root_id=root_id,
        file_name=root_file.name,
        channel_mode=args.channel_mode,
        window_id=args.window_id,
        fold_id=args.fold_id,
        count=args.n_shapelets,
    )

    development, blind = _ensure_preprocessed(
        config,
        root_file,
        root_id,
        study_dir,
        rebuild=bool(args.rebuild_preprocessing),
        logger=logger,
    )
    mode = CHANNEL_MODES[args.channel_mode]
    folds = _fold_masks(
        development,
        blind,
        mode["target"],
        config["cross_validation"],
        config["selection"],
    )
    fold_matches = [
        fold for fold in folds if int(fold["fold_id"]) == int(args.fold_id)
    ]
    if len(fold_matches) != 1:
        raise ValueError(
            f"Fold {args.fold_id} not found; available: "
            f"{[int(fold['fold_id']) for fold in folds]}"
        )
    fold = fold_matches[0]
    train_indices = np.asarray(fold["train"], dtype=np.int64)
    evaluation_indices = np.asarray(
        fold["validation"] if args.split == "validation" else fold["blind"],
        dtype=np.int64,
    )
    if evaluation_indices.size == 0:
        raise ValueError(f"The requested {args.split} split contains no events")

    development_view = prediction_window_dataset_view(
        development,
        input_waveforms=mode["input_waveforms"],
        target=mode["target"],
        before_ns=float(window["before_ns"]),
        after_ns=float(window["after_ns"]),
    )
    blind_view = prediction_window_dataset_view(
        blind,
        input_waveforms=mode["input_waveforms"],
        target=mode["target"],
        before_ns=float(window["before_ns"]),
        after_ns=float(window["after_ns"]),
    )
    transform_cache = study_dir / "transform_cache" / root_id / args.channel_mode / args.window_id
    transformed_development, _ = materialize_training_input_cache(
        development_view,
        transform,
        transform_cache / "development",
        chunk_size=int(args.chunk_size),
        rebuild=False,
        logger=logger,
    )
    transformed_blind, _ = materialize_training_input_cache(
        blind_view,
        transform,
        transform_cache / "blind",
        chunk_size=int(args.chunk_size),
        rebuild=False,
        logger=logger,
    )

    component_lengths = transformed_development.manifest.get(
        "input_component_lengths",
        [int(transformed_development.windows_mV.shape[-1])],
    )
    component_names = transformed_development.manifest.get(
        "input_components",
        ["waveform"],
    )
    selected_features, components = build_undersampling_plan(
        transformed_development.relative_time_ps,
        component_lengths,
        component_names,
        undersampling_factor,
    )
    normalization = compute_normalization(
        [(transformed_development, train_indices)],
        chunk_size=int(args.chunk_size),
        featurewise=transform == INPUT_TRANSFORM_NORMALIZE,
    )

    x_train = _materialize_difference_matrix(
        transformed_development,
        train_indices,
        normalization.std_mV,
        selected_features,
        chunk_size=int(args.chunk_size),
    )
    evaluation_dataset = (
        transformed_development if args.split == "validation" else transformed_blind
    )
    x_evaluation = _materialize_difference_matrix(
        evaluation_dataset,
        evaluation_indices,
        normalization.std_mV,
        selected_features,
        chunk_size=int(args.chunk_size),
    )
    y_train = factored_correction_target_ps(transformed_development, train_indices)

    train_shapelet_features = materialize_shapelet_features(
        x_train,
        shapelets,
        components,
        position_mode=position_mode,
        local_z_normalize=local_z_normalize,
        distance_metric=distance_metric,
        dtw_radius_points=dtw_radius_points,
        feature_chunk_size=int(args.feature_chunk_size),
    )
    evaluation_shapelet_features = materialize_shapelet_features(
        x_evaluation,
        shapelets,
        components,
        position_mode=position_mode,
        local_z_normalize=local_z_normalize,
        distance_metric=distance_metric,
        dtw_radius_points=dtw_radius_points,
        feature_chunk_size=int(args.feature_chunk_size),
    )
    model, feature_mean, feature_std = _EVENT.fit_fixed_alpha_ridge(
        train_shapelet_features,
        y_train,
        ridge_alpha,
    )
    standardized_evaluation = (
        evaluation_shapelet_features - feature_mean
    ) / feature_std
    coefficients = np.asarray(model.coef_, dtype=np.float64).reshape(-1)
    contributions = standardized_evaluation * coefficients[None, :]
    linear_contribution = np.sum(contributions, axis=1)

    selections = _EVENT.select_event_groups(
        linear_contribution,
        args.top_k_events,
    )
    grouped_positions = category_positions(selections)

    title = (
        f"Compact shapelet summary — {root_file.name}, {args.channel_mode}, "
        f"{args.window_id}"
    )
    subtitle = (
        f"fold {args.fold_id} | K={args.n_shapelets} | {args.split} | "
        f"Ridge α={ridge_alpha:g} | distance={distance_metric}"
        + (f"(r={dtw_radius_points})" if distance_metric == "dtw" else "")
        + f" | top {args.top_k_events} events per group | "
        "intercept and anchor shift excluded"
    )
    summary = create_compact_summary_figure(
        output_path=output_path,
        shapelets=shapelets,
        catalog=catalog,
        coefficients=coefficients,
        contributions=contributions,
        category_indices=grouped_positions,
        title=title,
        subtitle=subtitle,
        analysis_start_ns=float(window["start_ns"]),
        analysis_end_ns=float(window["end_ns"]),
        dpi=int(args.dpi),
    )
    summary_path = output_path.with_name(output_path.stem + "_data.csv")
    summary.to_csv(summary_path, index=False)
    metadata = {
        "source_shapelet_study": str(study_dir),
        "source_config": str(Path(args.config).resolve()),
        "root_file": str(root_file),
        "root_id": root_id,
        "channel_mode": args.channel_mode,
        "window_id": args.window_id,
        "fold_id": int(args.fold_id),
        "n_shapelets": int(args.n_shapelets),
        "ridge_alpha": ridge_alpha,
        "split": args.split,
        "top_k_events": int(args.top_k_events),
        "distance_metric": distance_metric,
        "dtw_radius_points": dtw_radius_points,
        "contribution_definition": "beta_k * standardized_shapelet_distance_ik",
        "global_importance_definition": "mean over evaluation events of abs(contribution_ik)",
        "group_heatmap_definition": "mean signed contribution_ik over selected event group",
        "intercept_and_anchor_shift_included": False,
        "figure": str(output_path),
        "summary_csv": str(summary_path),
    }
    output_path.with_name(output_path.stem + "_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info("Compact shapelet summary written | figure=%s", output_path)
    logger.info("Compact shapelet summary data written | csv=%s", summary_path)


if __name__ == "__main__":
    main()
