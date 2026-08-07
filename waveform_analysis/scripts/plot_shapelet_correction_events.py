#!/usr/bin/env python3
"""Plot original waveforms for events selected by a fitted shapelet correction.

This is a standalone diagnostic companion to ``run_shapelet_correction_study.py``.
It does not alter or register anything in the main ``ml_pipeline`` study.

For one file/channel/window/fold/K model, the script reconstructs the exact
training representation from the saved shapelet catalog, refits Ridge with the
alpha recorded in ``fold_results.csv``, and evaluates either the validation or
blind split. It then plots:

* the top-k largest positive linear shapelet contributions;
* the top-k most negative linear shapelet contributions;
* the k linear shapelet contributions closest to zero.

Every event figure contains the original detector waveforms, their raw
difference in the component used by the event's largest-contributing shapelet,
and the exact transformed/undersampled segment used to compute that shapelet
distance. The annotation reports, in ps, the original LED pair measurement,
the shapelet-model correction, the known anchor shift, the total correction
applied to LED, and the final corrected pair measurement.

Sign convention
---------------
``final_result_ps = original_led_ps - applied_correction_ps``

where
``applied_correction_ps = model_correction_ps + anchor_shift_ps``.
The event ranking defaults to the intercept-free linear shapelet contribution
``sum_k beta_k z_ik``. The fitted intercept is reported separately. Optional
``model`` and ``applied`` ranking modes remain available for diagnostics.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

# Reuse the exact representation and shapelet-distance implementation from the
# standalone study without turning ``scripts`` into an importable package.
_SHAPELET_SCRIPT = Path(__file__).with_name("run_shapelet_correction_study.py")
_SPEC = importlib.util.spec_from_file_location(
    "_shapelet_correction_study_for_diagnostics",
    _SHAPELET_SCRIPT,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - defensive
    raise RuntimeError(f"Unable to load shapelet study module: {_SHAPELET_SCRIPT}")
_SHAPELET = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _SHAPELET
_SPEC.loader.exec_module(_SHAPELET)

from ml_pipeline.common import setup_logging
from ml_pipeline.input_transform import (
    INPUT_TRANSFORM_NORMALIZE,
    materialize_training_input_cache,
)
from ml_pipeline.prediction import prediction_window_dataset_view
from ml_pipeline.study import (
    _delta_ps,
    _ensure_preprocessed,
    _fold_masks,
    _root_id,
)
from ml_pipeline.study_config import CHANNEL_MODES, load_study_config
from ml_pipeline.torch_data import (
    compute_normalization,
    factored_correction_target_ps,
    window_anchor_shift_pair_ps,
)


ShapeletCandidate = _SHAPELET.ShapeletCandidate
UndersampledComponent = _SHAPELET.UndersampledComponent
build_undersampling_plan = _SHAPELET.build_undersampling_plan
materialize_shapelet_features = _SHAPELET.materialize_shapelet_features
_materialize_difference_matrix = _SHAPELET._materialize_difference_matrix


@dataclass(frozen=True)
class EventSelection:
    category: str
    category_rank: int
    local_position: int


EVENT_OUTPUT_COLUMNS = [
    "category",
    "category_rank",
    "split",
    "fold_id",
    "dataset_row_index",
    "event_id",
    "event_index",
    "source_file_id",
    "source_run_index",
    "model_intercept_ps",
    "linear_contribution_ps",
    "model_correction_ps",
    "anchor_shift_ps",
    "applied_correction_ps",
    "ranking_correction_ps",
    "original_led_ps",
    "final_result_ps",
    "true_tof_ps",
    "original_led_residual_ps",
    "final_residual_ps",
    "true_factored_correction_ps",
    "prediction_error_ps",
    "dominant_shapelet_rank",
    "dominant_candidate_id",
    "distance_metric",
    "dtw_radius_points",
    "dominant_component",
    "dominant_start_time_ns",
    "dominant_end_time_ns",
    "dominant_distance",
    "dominant_standardized_distance",
    "dominant_coefficient",
    "dominant_contribution_ps",
    "plot_file",
]


def _resolve_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT / path).resolve()


def _python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _safe_name(value: object) -> str:
    text = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in str(value)
    ).strip("_")
    return text or "value"


def _single_matching_row(
    frame: pd.DataFrame,
    mask: np.ndarray | pd.Series,
    *,
    description: str,
) -> pd.Series:
    selected = frame.loc[mask]
    if selected.empty:
        raise ValueError(f"No {description} row matches the requested model")
    if len(selected) > 1:
        preview = selected.head(20).to_string(index=False)
        raise ValueError(
            f"Expected one {description} row, found {len(selected)}. Matching rows:\n{preview}"
        )
    return selected.iloc[0]


def _load_study_tables(study_dir: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    resolved_path = study_dir / "resolved_shapelet_correction_study.json"
    fold_path = study_dir / "fold_results.csv"
    shapelet_path = study_dir / "selected_shapelets.csv"
    value_path = study_dir / "shapelet_values.csv"
    for path in (resolved_path, fold_path, shapelet_path, value_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required shapelet-study artifact not found: {path}")
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    return (
        resolved,
        pd.read_csv(fold_path),
        pd.read_csv(shapelet_path),
        pd.read_csv(value_path),
    )


def _resolve_root_file(
    requested: str,
    config: dict[str, Any],
    fold_rows: pd.DataFrame,
) -> Path:
    candidates = [Path(value) for value in config["root_files"]]
    matched = [
        path
        for path in candidates
        if str(path) == requested or path.name == requested or path.stem == requested
    ]
    if not matched and "root_file" in fold_rows.columns:
        recorded = [Path(value) for value in fold_rows["root_file"].dropna().astype(str).unique()]
        matched = [
            path
            for path in recorded
            if str(path) == requested or path.name == requested or path.stem == requested
        ]
    if len(matched) != 1:
        available = sorted({path.name for path in candidates})
        raise ValueError(
            f"--file {requested!r} matched {len(matched)} ROOT files; available: {available}"
        )
    return matched[0]


def _reconstruct_shapelets(
    metadata: pd.DataFrame,
    values: pd.DataFrame,
    *,
    root_id: str,
    file_name: str,
    channel_mode: str,
    window_id: str,
    fold_id: int,
    count: int,
) -> tuple[list[ShapeletCandidate], pd.DataFrame]:
    numeric_columns = [
        "fold_id",
        "rank",
        "component_index",
        "start_index",
        "length_points",
        "start_time_ns",
        "end_time_ns",
        "duration_ns",
        "source_event_index",
        "source_target_ps",
    ]
    table = metadata.copy()
    for column in numeric_columns:
        if column in table.columns:
            table[column] = pd.to_numeric(table[column], errors="coerce")
    mask = (
        (table["root_id"].astype(str) == str(root_id))
        & (table["file_name"].astype(str) == str(file_name))
        & (table["channel_mode"].astype(str) == str(channel_mode))
        & (table["window_id"].astype(str) == str(window_id))
        & (table["fold_id"] == int(fold_id))
        & (table["rank"] <= int(count))
    )
    selected = table.loc[mask].sort_values("rank").copy()
    if len(selected) != int(count):
        raise ValueError(
            f"Requested {count} shapelets, but found {len(selected)} catalog rows for "
            f"{file_name}/{channel_mode}/{window_id}/fold {fold_id}"
        )

    candidates: list[ShapeletCandidate] = []
    for _, row in selected.iterrows():
        value_mask = (
            (values["row_key"].astype(str) == str(row["row_key"]))
            & (values["candidate_id"].astype(str) == str(row["candidate_id"]))
        )
        candidate_values = values.loc[value_mask].copy()
        candidate_values["sample_index"] = pd.to_numeric(
            candidate_values["sample_index"], errors="coerce"
        )
        candidate_values = candidate_values.sort_values("sample_index")
        vector = pd.to_numeric(candidate_values["value"], errors="coerce").to_numpy(float)
        expected = int(row["length_points"])
        if vector.size != expected or np.any(~np.isfinite(vector)):
            raise ValueError(
                f"Shapelet {row['candidate_id']} has {vector.size} valid values; expected {expected}"
            )
        candidates.append(
            ShapeletCandidate(
                candidate_id=str(row["candidate_id"]),
                component_index=int(row["component_index"]),
                component_name=str(row["component_name"]),
                start_index=int(row["start_index"]),
                length_points=expected,
                start_time_ps=float(row["start_time_ns"]) * 1000.0,
                end_time_ps=float(row["end_time_ns"]) * 1000.0,
                duration_ns=float(row["duration_ns"]),
                source_event_index=int(row["source_event_index"]),
                source_group=str(row["source_group"]),
                source_target_ps=float(row["source_target_ps"]),
                values=vector,
            )
        )
    return candidates, selected.reset_index(drop=True)


def fit_fixed_alpha_ridge(
    features: np.ndarray,
    target: np.ndarray,
    alpha: float,
) -> tuple[Any, np.ndarray, np.ndarray]:
    """Fit the same standardized Ridge form used by the shapelet study."""

    try:
        from sklearn.linear_model import Ridge
    except ImportError as exc:  # pragma: no cover - dependency error
        raise RuntimeError(
            "Shapelet event diagnostics require scikit-learn. Install it with "
            "'python -m pip install scikit-learn'."
        ) from exc

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    feature_mean = np.mean(x, axis=0)
    feature_std = np.std(x, axis=0, ddof=0)
    feature_std = np.where(feature_std > 1.0e-12, feature_std, 1.0)
    standardized = (x - feature_mean) / feature_std
    model = Ridge(alpha=float(alpha), fit_intercept=True)
    model.fit(standardized, y)
    return model, feature_mean, feature_std


def select_event_groups(corrections: np.ndarray, top_k: int) -> list[EventSelection]:
    """Return positive, negative, and near-zero event selections.

    Positive and negative groups use strict signs. The near-zero group is an
    independent control sample and may overlap an extreme group only in a
    degenerate distribution with too few signed events.
    """

    values = np.asarray(corrections, dtype=np.float64).reshape(-1)
    if int(top_k) <= 0:
        raise ValueError("top_k must be positive")
    finite = np.isfinite(values)
    positive = np.flatnonzero(finite & (values > 0.0))
    negative = np.flatnonzero(finite & (values < 0.0))
    all_finite = np.flatnonzero(finite)

    positive_order = positive[np.argsort(-values[positive])][: int(top_k)]
    negative_order = negative[np.argsort(values[negative])][: int(top_k)]
    zero_order = all_finite[np.argsort(np.abs(values[all_finite]))][: int(top_k)]

    output: list[EventSelection] = []
    for category, indices in (
        ("positive", positive_order),
        ("negative", negative_order),
        ("near_zero", zero_order),
    ):
        output.extend(
            EventSelection(category, rank, int(position))
            for rank, position in enumerate(indices, start=1)
        )
    return output


def _raw_components(dataset_view: Any) -> list[tuple[str, int, int, np.ndarray]]:
    names = list(dataset_view.manifest.get("input_components", ["waveform"]))
    lengths = [
        int(value)
        for value in dataset_view.manifest.get(
            "input_component_lengths", [int(dataset_view.windows_mV.shape[-1])]
        )
    ]
    times = np.asarray(dataset_view.relative_time_ps, dtype=np.float64)
    if len(names) != len(lengths) or sum(lengths) != times.size:
        raise ValueError("Raw input component metadata is inconsistent")
    output: list[tuple[str, int, int, np.ndarray]] = []
    cursor = 0
    for name, length in zip(names, lengths):
        output.append((str(name), cursor, cursor + length, times[cursor : cursor + length]))
        cursor += length
    return output


def _base_component_name(component_name: str) -> str:
    name = str(component_name)
    suffixes = ("_first_difference", "_raw")
    for suffix in suffixes:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    if name in {"raw_waveform", "first_difference", "waveform"}:
        return "waveform"
    return name


def _find_raw_component(
    raw_components: Sequence[tuple[str, int, int, np.ndarray]],
    model_component_name: str,
) -> tuple[str, int, int, np.ndarray]:
    base = _base_component_name(model_component_name)
    for item in raw_components:
        if item[0] == base:
            return item
    if len(raw_components) == 1:
        return raw_components[0]
    raise ValueError(
        f"Cannot map transformed component {model_component_name!r} to raw components "
        f"{[item[0] for item in raw_components]}"
    )


def _format_ps(value: float) -> str:
    return f"{float(value):+.3f} ps"


def _plot_event(
    *,
    output_path: Path,
    selection: EventSelection,
    raw_view: Any,
    raw_dataset_index: int,
    x_model_event: np.ndarray,
    components: Sequence[UndersampledComponent],
    shapelets: Sequence[ShapeletCandidate],
    shapelet_metadata: pd.DataFrame,
    feature_event: np.ndarray,
    standardized_feature_event: np.ndarray,
    coefficients: np.ndarray,
    position_mode: str,
    local_z_normalize: bool,
    distance_metric: str,
    dtw_radius_points: int,
    model_intercept_ps: float,
    linear_contribution_ps: float,
    model_correction_ps: float,
    anchor_shift_ps: float,
    applied_correction_ps: float,
    original_led_ps: float,
    final_result_ps: float,
    true_tof_ps: float,
    true_factored_correction_ps: float,
    event_id: Any,
    event_index: Any,
    dpi: int,
) -> dict[str, Any]:
    contributions = coefficients * standardized_feature_event
    dominant_position = int(np.argmax(np.abs(contributions)))
    candidate = shapelets[dominant_position]
    metadata = shapelet_metadata.iloc[dominant_position]
    component = components[candidate.component_index]

    pair = np.asarray(raw_view.windows_mV[int(raw_dataset_index)], dtype=np.float64)
    if pair.ndim != 2 or pair.shape[0] != 2:
        raise ValueError("Expected original waveform pair with shape (2, samples)")
    raw_components = _raw_components(raw_view)
    dominant_raw = _find_raw_component(raw_components, candidate.component_name)

    figure_rows = len(raw_components) + 2
    figure, axes = plt.subplots(
        figure_rows,
        1,
        figsize=(11.5, 3.0 * figure_rows),
        sharex=False,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    component_values = np.asarray(
        x_model_event[component.output_start : component.output_stop], dtype=np.float64
    )
    length = int(candidate.length_points)
    shapelet_values_for_distance = np.asarray(candidate.values, dtype=np.float64)
    if local_z_normalize:
        shapelet_values_for_distance = _SHAPELET._local_z(
            shapelet_values_for_distance[None, :]
        )[0]
    if position_mode == "fixed":
        match_start_index = int(candidate.start_index)
    elif position_mode == "sliding":
        if str(distance_metric).lower() == "dtw":
            raise ValueError("DTW diagnostics require fixed-position shapelets")
        windows = np.lib.stride_tricks.sliding_window_view(
            component_values, window_shape=length
        )
        windows_for_distance = (
            _SHAPELET._local_z(windows) if local_z_normalize else windows
        )
        distances = np.mean(
            (windows_for_distance - shapelet_values_for_distance[None, :]) ** 2,
            axis=1,
        )
        match_start_index = int(np.argmin(distances))
    else:
        raise ValueError(f"Unsupported position mode: {position_mode!r}")

    match_times_ps = component.relative_time_ps[
        match_start_index : match_start_index + length
    ]
    highlight_start = float(match_times_ps[0] / 1000.0)
    highlight_end = float(match_times_ps[-1] / 1000.0)
    base_component = dominant_raw[0]

    for axis, (name, start, stop, times_ps) in zip(axes, raw_components):
        times_ns = times_ps / 1000.0
        axis.plot(times_ns, pair[0, start:stop], label="Detector 1")
        axis.plot(times_ns, pair[1, start:stop], label="Detector 2")
        if name == base_component:
            axis.axvspan(
                highlight_start,
                highlight_end,
                alpha=0.18,
                label=f"Dominant shapelet region (rank {int(metadata['rank'])})",
            )
        axis.axvline(0.0, linestyle="--", linewidth=1.0, label="LED-aligned reference")
        axis.set_ylabel("Amplitude [mV]")
        axis.set_title(f"Original {name} waveform pair")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")

    difference_axis = axes[len(raw_components)]
    raw_name, raw_start, raw_stop, raw_times_ps = dominant_raw
    raw_times_ns = raw_times_ps / 1000.0
    raw_difference = pair[0, raw_start:raw_stop] - pair[1, raw_start:raw_stop]
    difference_axis.plot(raw_times_ns, raw_difference, label="Detector 1 − Detector 2")
    difference_axis.axvspan(
        highlight_start,
        highlight_end,
        alpha=0.22,
        label="Region used for dominant shapelet distance",
    )
    difference_axis.axvline(0.0, linestyle="--", linewidth=1.0, label="LED-aligned reference")
    difference_axis.set_xlabel("Time relative to LED anchor [ns]")
    difference_axis.set_ylabel("Difference [mV]")
    difference_axis.set_title(f"Original {raw_name} difference waveform")
    difference_axis.grid(True, alpha=0.25)
    difference_axis.legend(loc="best")

    match_axis = axes[-1]
    segment_start = component.output_start + match_start_index
    segment_stop = segment_start + length
    model_segment = np.asarray(
        x_model_event[segment_start:segment_stop], dtype=np.float64
    )
    shapelet_times_ns = match_times_ps / 1000.0
    display_segment = (
        _SHAPELET._local_z(model_segment[None, :])[0]
        if local_z_normalize
        else model_segment
    )
    display_shapelet = (
        shapelet_values_for_distance if local_z_normalize else candidate.values
    )
    match_axis.plot(
        shapelet_times_ns,
        display_segment,
        marker="o",
        markersize=3,
        label="Event segment used by distance",
    )
    match_axis.plot(
        shapelet_times_ns,
        display_shapelet,
        marker="o",
        markersize=3,
        label=f"Selected shapelet rank {int(metadata['rank'])}",
    )
    if str(distance_metric).lower() == "dtw":
        _, warping_path = _SHAPELET.constrained_dtw_distance(
            display_segment,
            display_shapelet,
            int(dtw_radius_points),
            return_path=True,
        )
        for event_point, shapelet_point in warping_path:
            match_axis.plot(
                [shapelet_times_ns[event_point], shapelet_times_ns[shapelet_point]],
                [display_segment[event_point], display_shapelet[shapelet_point]],
                linewidth=0.45,
                alpha=0.22,
            )
    match_axis.set_xlabel("Time relative to LED anchor [ns]")
    match_axis.set_ylabel(
        "Local-z model-space difference"
        if local_z_normalize
        else "Model-space difference"
    )
    metric_label = (
        f"constrained DTW (radius ±{int(dtw_radius_points)} retained points)"
        if str(distance_metric).lower() == "dtw"
        else "mean squared Euclidean"
    )
    match_axis.set_title(
        "Exact transformed and undersampled shapelet comparison\n"
        f"{metric_label}; distance={feature_event[dominant_position]:.6g}, "
        f"standardized={standardized_feature_event[dominant_position]:+.3f}, "
        f"coefficient={coefficients[dominant_position]:+.3f}, "
        f"contribution={contributions[dominant_position]:+.3f} ps"
    )
    match_axis.grid(True, alpha=0.25)
    match_axis.legend(loc="best")

    original_residual = original_led_ps - true_tof_ps
    final_residual = final_result_ps - true_tof_ps
    summary_text = (
        f"Original LED Δt: {_format_ps(original_led_ps)}\n"
        f"Ridge intercept: {_format_ps(model_intercept_ps)}\n"
        f"Linear shapelet contribution: {_format_ps(linear_contribution_ps)}\n"
        f"Shapelet-model correction: {_format_ps(model_correction_ps)}\n"
        f"Known anchor shift: {_format_ps(anchor_shift_ps)}\n"
        f"Applied correction: {_format_ps(applied_correction_ps)}\n"
        f"Final result = LED − correction: {_format_ps(final_result_ps)}\n"
        f"True TOF: {_format_ps(true_tof_ps)}\n"
        f"Residual before / after: {_format_ps(original_residual)} / {_format_ps(final_residual)}"
    )
    figure.suptitle(
        f"{selection.category.replace('_', ' ').title()} linear contribution #{selection.category_rank} | "
        f"event_id={event_id} | event_index={event_index}\n"
        f"Dominant shapelet rank {int(metadata['rank'])} in {candidate.component_name}, "
        f"{highlight_start:.3f}–{highlight_end:.3f} ns",
        fontsize=13,
    )
    figure.text(
        0.995,
        0.995,
        summary_text,
        ha="right",
        va="top",
        fontsize=9.5,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)

    return {
        "dominant_shapelet_rank": int(metadata["rank"]),
        "dominant_candidate_id": candidate.candidate_id,
        "distance_metric": str(distance_metric),
        "dtw_radius_points": int(dtw_radius_points),
        "dominant_component": candidate.component_name,
        "dominant_start_time_ns": highlight_start,
        "dominant_end_time_ns": highlight_end,
        "dominant_distance": float(feature_event[dominant_position]),
        "dominant_standardized_distance": float(
            standardized_feature_event[dominant_position]
        ),
        "dominant_coefficient": float(coefficients[dominant_position]),
        "dominant_contribution_ps": float(contributions[dominant_position]),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot original waveform pairs for the largest positive, largest negative, "
            "and smallest-absolute linear shapelet contributions."
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
        "--top-k",
        type=int,
        default=10,
        help="Number of positive, negative, and near-zero events to plot",
    )
    parser.add_argument(
        "--ranking-correction",
        choices=["linear", "model", "applied"],
        default="linear",
        help=(
            "Rank by the intercept-free linear shapelet contribution (default), "
            "the full Ridge model correction, or the applied correction including "
            "the known anchor shift"
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--rebuild-preprocessing", action="store_true")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--feature-chunk-size", type=int, default=512)
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.n_shapelets <= 0:
        raise ValueError("--n-shapelets must be positive")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")

    study_dir = _resolve_path(args.study_dir)
    resolved, fold_rows, shapelet_rows, value_rows = _load_study_tables(study_dir)
    config = load_study_config(args.config, PROJECT)
    root_file = _resolve_root_file(args.file, config, fold_rows)
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

    output = (
        _resolve_path(args.output_dir)
        if args.output_dir is not None
        else study_dir
        / "event_diagnostics"
        / _safe_name(root_file.stem)
        / _safe_name(args.channel_mode)
        / _safe_name(args.window_id)
        / f"fold_{args.fold_id}"
        / f"k_{args.n_shapelets}"
        / args.split
    )
    if args.restart and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output / "shapelet_event_diagnostics.log", "INFO")

    fold_table = fold_rows.copy()
    for column in ("fold_id", "selected_shapelet_count", "ridge_alpha"):
        fold_table[column] = pd.to_numeric(fold_table[column], errors="coerce")
    fold_row = _single_matching_row(
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

    shapelets, catalog = _reconstruct_shapelets(
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
    fold_matches = [fold for fold in folds if int(fold["fold_id"]) == int(args.fold_id)]
    if len(fold_matches) != 1:
        raise ValueError(
            f"Fold {args.fold_id} not found; available: {[int(fold['fold_id']) for fold in folds]}"
        )
    fold = fold_matches[0]
    train_indices = np.asarray(fold["train"], dtype=np.int64)
    validation_indices = np.asarray(fold["validation"], dtype=np.int64)
    blind_indices = np.asarray(fold["blind"], dtype=np.int64)

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
        "input_components", ["waveform"]
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
    if args.split == "validation":
        evaluation_dataset = transformed_development
        raw_evaluation_view = development_view
        evaluation_indices = validation_indices
    else:
        evaluation_dataset = transformed_blind
        raw_evaluation_view = blind_view
        evaluation_indices = blind_indices
    if evaluation_indices.size == 0:
        raise ValueError(f"The requested {args.split} split contains no events")

    x_evaluation = _materialize_difference_matrix(
        evaluation_dataset,
        evaluation_indices,
        normalization.std_mV,
        selected_features,
        chunk_size=int(args.chunk_size),
    )
    y_train = factored_correction_target_ps(transformed_development, train_indices)
    y_evaluation = factored_correction_target_ps(evaluation_dataset, evaluation_indices)

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
    model, feature_mean, feature_std = fit_fixed_alpha_ridge(
        train_shapelet_features,
        y_train,
        ridge_alpha,
    )
    standardized_evaluation = (
        evaluation_shapelet_features - feature_mean
    ) / feature_std
    coefficients = np.asarray(model.coef_, dtype=np.float64).reshape(-1)
    model_intercept = float(model.intercept_)
    linear_contribution = np.asarray(
        standardized_evaluation @ coefficients, dtype=np.float64
    ).reshape(-1)
    model_correction = model_intercept + linear_contribution
    anchor_shift = window_anchor_shift_pair_ps(evaluation_dataset, evaluation_indices)
    applied_correction = model_correction + anchor_shift
    original_led = _delta_ps(evaluation_dataset, "prepared_led", evaluation_indices)
    final_result = original_led - applied_correction
    true_tof = float(evaluation_dataset.true_tof_ps)
    if args.ranking_correction == "linear":
        ranking_correction = linear_contribution
    elif args.ranking_correction == "model":
        ranking_correction = model_correction
    else:
        ranking_correction = applied_correction

    selections = select_event_groups(ranking_correction, args.top_k)
    if not selections:
        raise RuntimeError("No finite corrections are available for plotting")
    logger.info(
        "Plotting %d events | file=%s | mode=%s | window=%s | fold=%d | K=%d | split=%s",
        len(selections),
        root_file.name,
        args.channel_mode,
        args.window_id,
        args.fold_id,
        args.n_shapelets,
        args.split,
    )

    records: list[dict[str, Any]] = []
    for selection in selections:
        local = int(selection.local_position)
        dataset_row = int(evaluation_indices[local])
        event_id = _python_scalar(evaluation_dataset.event_id[dataset_row])
        event_index = _python_scalar(evaluation_dataset.event_index[dataset_row])
        category_dir = output / selection.category
        plot_name = (
            f"{selection.category}_{selection.category_rank:03d}__"
            f"event_id_{_safe_name(event_id)}__row_{dataset_row}.png"
        )
        plot_path = category_dir / plot_name
        dominant = _plot_event(
            output_path=plot_path,
            selection=selection,
            raw_view=raw_evaluation_view,
            raw_dataset_index=dataset_row,
            x_model_event=x_evaluation[local],
            components=components,
            shapelets=shapelets,
            shapelet_metadata=catalog,
            feature_event=evaluation_shapelet_features[local],
            standardized_feature_event=standardized_evaluation[local],
            coefficients=coefficients,
            position_mode=position_mode,
            local_z_normalize=local_z_normalize,
            distance_metric=distance_metric,
            dtw_radius_points=dtw_radius_points,
            model_intercept_ps=model_intercept,
            linear_contribution_ps=float(linear_contribution[local]),
            model_correction_ps=float(model_correction[local]),
            anchor_shift_ps=float(anchor_shift[local]),
            applied_correction_ps=float(applied_correction[local]),
            original_led_ps=float(original_led[local]),
            final_result_ps=float(final_result[local]),
            true_tof_ps=true_tof,
            true_factored_correction_ps=float(y_evaluation[local]),
            event_id=event_id,
            event_index=event_index,
            dpi=int(args.dpi),
        )
        records.append(
            {
                "category": selection.category,
                "category_rank": selection.category_rank,
                "split": args.split,
                "fold_id": int(args.fold_id),
                "dataset_row_index": dataset_row,
                "event_id": event_id,
                "event_index": event_index,
                "source_file_id": _python_scalar(
                    evaluation_dataset.source_file_id[dataset_row]
                ),
                "source_run_index": _python_scalar(
                    evaluation_dataset.source_run_index[dataset_row]
                ),
                "model_intercept_ps": model_intercept,
                "linear_contribution_ps": float(linear_contribution[local]),
                "model_correction_ps": float(model_correction[local]),
                "anchor_shift_ps": float(anchor_shift[local]),
                "applied_correction_ps": float(applied_correction[local]),
                "ranking_correction_ps": float(ranking_correction[local]),
                "original_led_ps": float(original_led[local]),
                "final_result_ps": float(final_result[local]),
                "true_tof_ps": true_tof,
                "original_led_residual_ps": float(original_led[local] - true_tof),
                "final_residual_ps": float(final_result[local] - true_tof),
                "true_factored_correction_ps": float(y_evaluation[local]),
                "prediction_error_ps": float(
                    y_evaluation[local] - model_correction[local]
                ),
                **dominant,
                "plot_file": str(plot_path.relative_to(output)),
            }
        )

    output_frame = pd.DataFrame(records, columns=EVENT_OUTPUT_COLUMNS)
    output_frame.to_csv(output / "event_ranking.csv", index=False)
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
        "top_k": int(args.top_k),
        "ranking_correction": args.ranking_correction,
        "distance_metric": distance_metric,
        "dtw_radius_points": dtw_radius_points,
        "sign_convention": "final_result_ps = original_led_ps - applied_correction_ps",
        "model_intercept_ps": model_intercept,
        "linear_contribution_definition": "sum_k beta_k * standardized_shapelet_distance_ik",
        "model_correction_definition": "model_intercept_ps + linear_contribution_ps",
        "applied_correction_definition": "model_correction_ps + anchor_shift_ps",
        "output_event_count": int(len(output_frame)),
    }
    (output / "diagnostic_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    logger.info("Shapelet event diagnostics complete | output=%s", output)


if __name__ == "__main__":
    main()
