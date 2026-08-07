#!/usr/bin/env python3
"""Standalone supervised shapelet study for LED-correction information.

The script reuses the repository's preprocessing, development/blind split,
channel modes, physical windows, fold-local event masks, input transforms,
feature normalization, anchor-factored LED-correction target, and CTR metrics.
It intentionally does *not* register a new model in the main ml_pipeline study.

For each outer CV fold:

1. build the antisymmetric waveform difference s1 - s2;
2. keep one sample every ``--undersampling-factor`` points, independently in
   every input component;
3. draw candidate subsequences from large-positive, near-zero, and
   large-negative training targets only;
4. score candidates by |corr(shapelet feature, correction target)| on a
   training-only scoring subset;
5. prune redundant candidates;
6. fit Ridge models on the selected shapelet features using inner CV on the
   outer-training events;
7. evaluate on the untouched outer-validation fold and optional blind block.

The default ``--position-mode fixed`` is deliberate: waveforms are already
LED-aligned, so a shapelet extracted at a given relative time is compared at
that same relative time in all events. This preserves time localisation and is
much cheaper than classical sliding matching. ``--position-mode sliding`` is
available for a translation-tolerant comparison within the same component.

Persistent outputs are CSV/JSON summaries and plots only. No checkpoints are
written.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from ml_pipeline.common import atomic_json, canonical_hash, setup_logging
from ml_pipeline.input_transform import (
    INPUT_TRANSFORM_NORMALIZE,
    SUPPORTED_INPUT_TRANSFORMS,
    materialize_training_input_cache,
    normalize_input_transform,
)
from ml_pipeline.prediction import prediction_window_dataset_view
from ml_pipeline.study import (
    _delta_ps,
    _ensure_preprocessed,
    _extract_voltage,
    _fold_masks,
    _metrics,
    _root_id,
)
from ml_pipeline.study_config import CHANNEL_MODES, load_study_config
from ml_pipeline.torch_data import compute_normalization, factored_correction_target_ps


@dataclass(frozen=True)
class UndersampledComponent:
    name: str
    source_start: int
    source_stop: int
    selected_source_indices: np.ndarray
    output_start: int
    output_stop: int
    relative_time_ps: np.ndarray


@dataclass(frozen=True)
class ShapeletCandidate:
    candidate_id: str
    component_index: int
    component_name: str
    start_index: int
    length_points: int
    start_time_ps: float
    end_time_ps: float
    duration_ns: float
    source_event_index: int
    source_group: str
    source_target_ps: float
    values: np.ndarray


FOLD_RESULT_COLUMNS = [
    "row_key",
    "root_id",
    "root_file",
    "file_name",
    "voltage_V",
    "channel_mode",
    "input_waveforms",
    "target",
    "input_transform",
    "window_id",
    "window_start_ns",
    "window_end_ns",
    "window_length_ns",
    "fold_id",
    "undersampling_factor",
    "position_mode",
    "distance_metric",
    "dtw_radius_points",
    "candidate_count",
    "selected_shapelet_count",
    "ridge_alpha",
    "train_event_count",
    "validation_event_count",
    "blind_event_count",
    "validation_loss",
    "validation_rmse_ps",
    "validation_bias_ps",
    "validation_ctr_ps",
    "validation_baseline_ctr_ps",
    "validation_relative_improvement_pct",
    "validation_ctr_minus_led_ps",
    "blind_loss",
    "blind_rmse_ps",
    "blind_bias_ps",
    "blind_ctr_ps",
    "blind_baseline_ctr_ps",
    "blind_relative_improvement_pct",
    "blind_ctr_minus_led_ps",
    "runtime_seconds",
]


SHAPELET_COLUMNS = [
    "row_key",
    "root_id",
    "file_name",
    "voltage_V",
    "channel_mode",
    "input_transform",
    "window_id",
    "fold_id",
    "distance_metric",
    "dtw_radius_points",
    "rank",
    "candidate_id",
    "component_index",
    "component_name",
    "start_index",
    "length_points",
    "start_time_ns",
    "end_time_ns",
    "duration_ns",
    "source_event_index",
    "source_group",
    "source_target_ps",
    "score_correlation",
    "score_abs_correlation",
    "ridge_coefficient_at_max_k",
    "feature_mean_train",
    "feature_std_train",
]


SHAPELET_VALUE_COLUMNS = [
    "row_key",
    "candidate_id",
    "sample_index",
    "relative_time_ns",
    "value",
]


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _safe_name(value: object) -> str:
    text = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in str(value)
    ).strip("_")
    return text or "result"


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    return pd.read_csv(path).to_dict(orient="records")


def _filtered_values(
    requested: list[str] | None,
    available: list[str],
    label: str,
) -> list[str]:
    if not requested:
        return list(available)
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"Unknown {label}: {unknown}; available: {available}")
    requested_set = set(requested)
    return [value for value in available if value in requested_set]


def build_undersampling_plan(
    relative_time_ps: np.ndarray,
    component_lengths: Sequence[int],
    component_names: Sequence[str],
    factor: int,
) -> tuple[np.ndarray, list[UndersampledComponent]]:
    """Return component-aware one-in-``factor`` sample selection.

    Undersampling restarts at the first sample of each component. Consequently,
    no modality is shifted merely because a preceding component has a length
    that is not divisible by ``factor``.
    """

    times = np.asarray(relative_time_ps, dtype=np.float64).reshape(-1)
    lengths = [int(value) for value in component_lengths]
    names = [str(value) for value in component_names]
    if factor <= 0:
        raise ValueError("undersampling factor must be positive")
    if not lengths or any(value <= 0 for value in lengths):
        raise ValueError("component lengths must be positive")
    if sum(lengths) != times.size:
        raise ValueError("component lengths do not match the time axis")
    if len(names) != len(lengths):
        raise ValueError("component names and lengths must have equal length")

    selected_parts: list[np.ndarray] = []
    components: list[UndersampledComponent] = []
    source_cursor = 0
    output_cursor = 0
    for name, length in zip(names, lengths):
        local = np.arange(0, length, int(factor), dtype=np.int64)
        source_indices = source_cursor + local
        selected_times = times[source_indices]
        selected_parts.append(source_indices)
        components.append(
            UndersampledComponent(
                name=name,
                source_start=source_cursor,
                source_stop=source_cursor + length,
                selected_source_indices=source_indices,
                output_start=output_cursor,
                output_stop=output_cursor + local.size,
                relative_time_ps=selected_times,
            )
        )
        source_cursor += length
        output_cursor += local.size
    return np.concatenate(selected_parts), components


def _materialize_difference_matrix(
    dataset: Any,
    indices: np.ndarray,
    std_mV: float | np.ndarray,
    selected_features: np.ndarray,
    *,
    chunk_size: int,
) -> np.ndarray:
    selected_events = np.asarray(indices, dtype=np.int64)
    selected_features = np.asarray(selected_features, dtype=np.int64)
    output = np.empty(
        (selected_events.size, selected_features.size),
        dtype=np.float32,
    )
    scale = np.asarray(std_mV, dtype=np.float64)
    feature_count = int(dataset.windows_mV.shape[-1])
    if scale.ndim == 1 and scale.size != feature_count:
        raise ValueError(
            f"Normalization length {scale.size} does not match {feature_count} features"
        )
    if scale.ndim not in {0, 1} or np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("Invalid fold normalization scale")

    cursor = 0
    for start in range(0, selected_events.size, int(chunk_size)):
        block_indices = selected_events[start : start + int(chunk_size)]
        pair = np.asarray(dataset.windows_mV[block_indices], dtype=np.float64)
        difference = (pair[:, 0, :] - pair[:, 1, :]) / scale
        block = difference[:, selected_features]
        size = int(block_indices.size)
        output[cursor : cursor + size] = block.astype(np.float32, copy=False)
        cursor += size
    return output


def _safe_corr(feature: np.ndarray, target: np.ndarray) -> float:
    x = np.asarray(feature, dtype=np.float64).reshape(-1)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size < 3:
        return 0.0
    x = x - float(np.mean(x))
    y = y - float(np.mean(y))
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if not math.isfinite(denominator) or denominator <= 0.0:
        return 0.0
    value = float(np.dot(x, y) / denominator)
    return value if math.isfinite(value) else 0.0


def _local_z(values: np.ndarray, epsilon: float = 1.0e-8) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    mean = np.mean(array, axis=-1, keepdims=True)
    std = np.std(array, axis=-1, ddof=0, keepdims=True)
    std = np.where(std > epsilon, std, 1.0)
    return (array - mean) / std


def constrained_dtw_distance_batch(
    segments: np.ndarray,
    shapelet: np.ndarray,
    radius: int,
) -> np.ndarray:
    """Return path-length-normalized constrained DTW squared distance.

    ``segments`` has shape ``(events, length)`` and ``shapelet`` has the same
    length.  The Sakoe--Chiba radius is expressed in retained (undersampled)
    points.  Dynamic-programming rows are vectorized over events, keeping the
    memory cost linear in the shapelet length.
    """

    x = np.asarray(segments, dtype=np.float64)
    q = np.asarray(shapelet, dtype=np.float64).reshape(-1)
    if x.ndim != 2:
        raise ValueError("segments must have shape (events, length)")
    if x.shape[1] != q.size or q.size == 0:
        raise ValueError("DTW segments and shapelet must have the same nonzero length")
    radius = int(radius)
    if radius < 0:
        raise ValueError("DTW radius must be non-negative")

    event_count, length = x.shape
    infinity = np.inf
    previous_cost = np.full((event_count, length + 1), infinity, dtype=np.float64)
    previous_length = np.zeros((event_count, length + 1), dtype=np.int32)
    previous_cost[:, 0] = 0.0

    for i in range(1, length + 1):
        current_cost = np.full((event_count, length + 1), infinity, dtype=np.float64)
        current_length = np.zeros((event_count, length + 1), dtype=np.int32)
        j_start = max(1, i - radius)
        j_stop = min(length, i + radius)
        for j in range(j_start, j_stop + 1):
            predecessor_costs = np.stack(
                (
                    previous_cost[:, j - 1],  # diagonal; preferred on exact ties
                    previous_cost[:, j],      # vertical
                    current_cost[:, j - 1],   # horizontal
                ),
                axis=0,
            )
            choice = np.argmin(predecessor_costs, axis=0)
            best_cost = np.take_along_axis(
                predecessor_costs, choice[None, :], axis=0
            )[0]
            predecessor_lengths = np.stack(
                (
                    previous_length[:, j - 1],
                    previous_length[:, j],
                    current_length[:, j - 1],
                ),
                axis=0,
            )
            best_length = np.take_along_axis(
                predecessor_lengths, choice[None, :], axis=0
            )[0]
            local_cost = (x[:, i - 1] - q[j - 1]) ** 2
            current_cost[:, j] = best_cost + local_cost
            current_length[:, j] = best_length + 1
        previous_cost = current_cost
        previous_length = current_length

    path_length = previous_length[:, length]
    if np.any(path_length <= 0) or np.any(~np.isfinite(previous_cost[:, length])):
        raise RuntimeError(
            "No valid DTW path. Increase --dtw-radius-points or inspect the shapelet length."
        )
    return previous_cost[:, length] / path_length


def constrained_dtw_distance(
    segment: np.ndarray,
    shapelet: np.ndarray,
    radius: int,
    *,
    return_path: bool = False,
) -> float | tuple[float, list[tuple[int, int]]]:
    """Scalar constrained DTW distance, optionally with the optimal path."""

    x = np.asarray(segment, dtype=np.float64).reshape(-1)
    q = np.asarray(shapelet, dtype=np.float64).reshape(-1)
    if x.size != q.size or x.size == 0:
        raise ValueError("DTW segment and shapelet must have the same nonzero length")
    radius = int(radius)
    if radius < 0:
        raise ValueError("DTW radius must be non-negative")
    length = int(x.size)
    cost = np.full((length + 1, length + 1), np.inf, dtype=np.float64)
    steps = np.zeros((length + 1, length + 1), dtype=np.int32)
    predecessor = np.full((length + 1, length + 1), -1, dtype=np.int8)
    cost[0, 0] = 0.0
    for i in range(1, length + 1):
        for j in range(max(1, i - radius), min(length, i + radius) + 1):
            options = (cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1])
            choice = int(np.argmin(options))
            if not math.isfinite(options[choice]):
                continue
            if choice == 0:
                pi, pj = i - 1, j - 1
            elif choice == 1:
                pi, pj = i - 1, j
            else:
                pi, pj = i, j - 1
            cost[i, j] = options[choice] + (x[i - 1] - q[j - 1]) ** 2
            steps[i, j] = steps[pi, pj] + 1
            predecessor[i, j] = choice
    if steps[length, length] <= 0 or not math.isfinite(cost[length, length]):
        raise RuntimeError("No valid constrained DTW path")
    distance = float(cost[length, length] / steps[length, length])
    if not return_path:
        return distance
    path: list[tuple[int, int]] = []
    i = j = length
    while i > 0 or j > 0:
        if i <= 0 or j <= 0:
            raise RuntimeError("Invalid DTW predecessor path")
        path.append((i - 1, j - 1))
        choice = int(predecessor[i, j])
        if choice == 0:
            i, j = i - 1, j - 1
        elif choice == 1:
            i -= 1
        elif choice == 2:
            j -= 1
        else:
            raise RuntimeError("Invalid DTW predecessor code")
    path.reverse()
    return distance, path


def _distance_batch(
    segments: np.ndarray,
    shapelet: np.ndarray,
    *,
    distance_metric: str,
    dtw_radius_points: int,
) -> np.ndarray:
    metric = str(distance_metric).strip().lower()
    if metric == "mse":
        return np.mean(
            (np.asarray(segments, dtype=np.float64) - np.asarray(shapelet)[None, :]) ** 2,
            axis=1,
        )
    if metric == "dtw":
        return constrained_dtw_distance_batch(segments, shapelet, dtw_radius_points)
    raise ValueError("distance_metric must be 'dtw' or 'mse'")


def shapelet_feature(
    signals: np.ndarray,
    candidate: ShapeletCandidate,
    components: Sequence[UndersampledComponent],
    *,
    position_mode: str,
    local_z_normalize: bool,
    distance_metric: str = "dtw",
    dtw_radius_points: int = 2,
    chunk_size: int = 512,
) -> np.ndarray:
    """Compute a fixed/sliding shapelet-distance feature.

    DTW uses a small Sakoe--Chiba band and is intentionally restricted to
    fixed-position shapelets, preserving physical time localization and keeping
    computation bounded.  MSE remains available for controlled comparisons and
    for the legacy sliding mode.
    """

    values = np.asarray(signals, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("signals must have shape (events, features)")
    component = components[int(candidate.component_index)]
    component_values = values[:, component.output_start : component.output_stop]
    length = int(candidate.length_points)
    if length <= 0 or length > component_values.shape[1]:
        raise ValueError("shapelet length is invalid for its component")
    shapelet = np.asarray(candidate.values, dtype=np.float64).reshape(-1)
    if shapelet.size != length:
        raise ValueError("shapelet values do not match length_points")
    if local_z_normalize:
        shapelet = _local_z(shapelet[None, :])[0]

    output = np.empty(values.shape[0], dtype=np.float64)
    for start in range(0, values.shape[0], int(chunk_size)):
        block = np.asarray(
            component_values[start : start + int(chunk_size)], dtype=np.float64
        )
        if position_mode == "fixed":
            stop = int(candidate.start_index) + length
            segments = block[:, int(candidate.start_index) : stop]
            if local_z_normalize:
                segments = _local_z(segments)
            output[start : start + block.shape[0]] = _distance_batch(
                segments,
                shapelet,
                distance_metric=distance_metric,
                dtw_radius_points=dtw_radius_points,
            )
        elif position_mode == "sliding":
            if str(distance_metric).lower() == "dtw":
                raise ValueError(
                    "Constrained DTW is supported only with --position-mode fixed. "
                    "Use fixed-position local DTW to preserve time localization, or "
                    "select --distance-metric mse for legacy sliding shapelets."
                )
            windows = np.lib.stride_tricks.sliding_window_view(
                block,
                window_shape=length,
                axis=1,
            )
            if local_z_normalize:
                windows = _local_z(windows)
            distances = np.mean((windows - shapelet[None, None, :]) ** 2, axis=2)
            output[start : start + block.shape[0]] = np.min(distances, axis=1)
        else:
            raise ValueError("position_mode must be 'fixed' or 'sliding'")
    return output


def _shapelet_length_points(
    component: UndersampledComponent,
    duration_ns: float,
) -> int:
    times = np.asarray(component.relative_time_ps, dtype=np.float64)
    if times.size < 2:
        raise ValueError(f"Component {component.name} has fewer than two samples")
    step = float(np.median(np.abs(np.diff(times))))
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError(f"Component {component.name} has invalid sample spacing")
    points = max(2, int(round(float(duration_ns) * 1000.0 / step)) + 1)
    return min(points, int(times.size))


def _target_groups(
    target: np.ndarray,
    extreme_fraction: float,
) -> dict[str, np.ndarray]:
    values = np.asarray(target, dtype=np.float64).reshape(-1)
    count = max(1, int(math.ceil(values.size * float(extreme_fraction))))
    order = np.argsort(values)
    absolute_order = np.argsort(np.abs(values - np.median(values)))
    return {
        "negative": order[:count],
        "near_zero": absolute_order[:count],
        "positive": order[-count:],
    }


def generate_candidates(
    signals: np.ndarray,
    target: np.ndarray,
    components: Sequence[UndersampledComponent],
    durations_ns: Sequence[float],
    *,
    candidates_per_group: int,
    extreme_fraction: float,
    rng: np.random.Generator,
) -> list[ShapeletCandidate]:
    values = np.asarray(signals, dtype=np.float32)
    target_values = np.asarray(target, dtype=np.float64).reshape(-1)
    if values.shape[0] != target_values.size:
        raise ValueError("signals and target must contain the same events")
    groups = _target_groups(target_values, extreme_fraction)
    candidate_list: list[ShapeletCandidate] = []
    durations = sorted(set(float(value) for value in durations_ns))

    for group_name, group_indices in groups.items():
        if group_indices.size == 0:
            continue
        for candidate_number in range(int(candidates_per_group)):
            component_index = int(rng.integers(0, len(components)))
            component = components[component_index]
            duration = durations[int(rng.integers(0, len(durations)))]
            length = _shapelet_length_points(component, duration)
            available = int(component.output_stop - component.output_start)
            max_start = available - length
            start_index = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
            event_local_index = int(group_indices[int(rng.integers(0, group_indices.size))])
            segment_start = component.output_start + start_index
            segment_stop = segment_start + length
            segment = np.asarray(
                values[event_local_index, segment_start:segment_stop], dtype=np.float64
            ).copy()
            times = component.relative_time_ps[start_index : start_index + length]
            candidate_id = canonical_hash(
                {
                    "group": group_name,
                    "number": candidate_number,
                    "component": component.name,
                    "start": start_index,
                    "length": length,
                    "event": event_local_index,
                    "values": np.round(segment, decimals=8).tolist(),
                }
            )[:20]
            candidate_list.append(
                ShapeletCandidate(
                    candidate_id=candidate_id,
                    component_index=component_index,
                    component_name=component.name,
                    start_index=start_index,
                    length_points=length,
                    start_time_ps=float(times[0]),
                    end_time_ps=float(times[-1]),
                    duration_ns=float((times[-1] - times[0]) / 1000.0),
                    source_event_index=event_local_index,
                    source_group=group_name,
                    source_target_ps=float(target_values[event_local_index]),
                    values=segment,
                )
            )
    return candidate_list


def score_and_select_candidates(
    signals: np.ndarray,
    target: np.ndarray,
    candidates: Sequence[ShapeletCandidate],
    components: Sequence[UndersampledComponent],
    *,
    max_shapelets: int,
    redundancy_threshold: float,
    position_mode: str,
    local_z_normalize: bool,
    distance_metric: str = "dtw",
    dtw_radius_points: int = 2,
    feature_chunk_size: int = 512,
) -> tuple[list[ShapeletCandidate], np.ndarray, np.ndarray]:
    """Score candidates and greedily retain non-redundant shapelets."""

    target_values = np.asarray(target, dtype=np.float64).reshape(-1)
    features: list[np.ndarray] = []
    scores: list[float] = []
    for candidate in candidates:
        feature = shapelet_feature(
            signals,
            candidate,
            components,
            position_mode=position_mode,
            local_z_normalize=local_z_normalize,
            distance_metric=distance_metric,
            dtw_radius_points=dtw_radius_points,
            chunk_size=feature_chunk_size,
        )
        features.append(feature)
        scores.append(_safe_corr(feature, target_values))
    if not features:
        raise RuntimeError("No shapelet candidates were generated")

    feature_matrix = np.column_stack(features)
    score_array = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-np.abs(score_array))
    selected_indices: list[int] = []
    centered_cache: dict[int, np.ndarray] = {}

    for index in order:
        index = int(index)
        current = feature_matrix[:, index]
        current_centered = current - float(np.mean(current))
        current_norm = float(np.linalg.norm(current_centered))
        redundant = False
        if current_norm > 0.0:
            for chosen in selected_indices:
                chosen_centered = centered_cache[chosen]
                denominator = current_norm * float(np.linalg.norm(chosen_centered))
                correlation = (
                    abs(float(np.dot(current_centered, chosen_centered) / denominator))
                    if denominator > 0.0
                    else 0.0
                )
                if correlation >= float(redundancy_threshold):
                    redundant = True
                    break
        if redundant:
            continue
        selected_indices.append(index)
        centered_cache[index] = current_centered
        if len(selected_indices) >= int(max_shapelets):
            break

    selected = [candidates[index] for index in selected_indices]
    selected_features = feature_matrix[:, selected_indices]
    selected_scores = score_array[selected_indices]
    return selected, selected_features, selected_scores


def materialize_shapelet_features(
    signals: np.ndarray,
    shapelets: Sequence[ShapeletCandidate],
    components: Sequence[UndersampledComponent],
    *,
    position_mode: str,
    local_z_normalize: bool,
    distance_metric: str = "dtw",
    dtw_radius_points: int = 2,
    feature_chunk_size: int = 512,
) -> np.ndarray:
    if not shapelets:
        return np.empty((signals.shape[0], 0), dtype=np.float64)
    return np.column_stack(
        [
            shapelet_feature(
                signals,
                candidate,
                components,
                position_mode=position_mode,
                local_z_normalize=local_z_normalize,
                distance_metric=distance_metric,
                dtw_radius_points=dtw_radius_points,
                chunk_size=feature_chunk_size,
            )
            for candidate in shapelets
        ]
    )


def _fit_shapelet_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    alphas: Sequence[float],
    *,
    inner_folds: int,
) -> tuple[Any, np.ndarray, np.ndarray, float]:
    try:
        from sklearn.linear_model import RidgeCV
    except ImportError as exc:
        raise RuntimeError(
            "Shapelet study requires scikit-learn. Install it with "
            "'python -m pip install scikit-learn'."
        ) from exc

    feature_mean = np.mean(x_train, axis=0)
    feature_std = np.std(x_train, axis=0, ddof=0)
    feature_std = np.where(feature_std > 1.0e-12, feature_std, 1.0)
    standardized = (x_train - feature_mean) / feature_std
    effective_inner_folds = min(int(inner_folds), int(x_train.shape[0]))
    if effective_inner_folds < 2:
        raise ValueError("At least two training events are required for inner Ridge CV")
    model = RidgeCV(
        alphas=np.asarray(alphas, dtype=np.float64),
        fit_intercept=True,
        cv=effective_inner_folds,
        scoring="neg_root_mean_squared_error",
    )
    model.fit(standardized, y_train)
    return model, feature_mean, feature_std, float(model.alpha_)


def _predict_shapelet_ridge(
    model: Any,
    features: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        model.predict((features - feature_mean) / feature_std),
        dtype=np.float64,
    )


def _prediction_metrics_from_prediction(
    *,
    dataset: Any,
    indices: np.ndarray,
    prediction: np.ndarray,
    fit_config: dict[str, Any],
    loss: dict[str, Any],
    target_scale_ps: float,
) -> dict[str, float]:
    selected = np.asarray(indices, dtype=np.int64)
    target = factored_correction_target_ps(dataset, selected)
    residual = target - np.asarray(prediction, dtype=np.float64)
    true_tof = float(dataset.true_tof_ps)
    corrected = true_tof + residual
    truth = np.full(selected.size, true_tof, dtype=np.float64)
    baseline = _delta_ps(dataset, "prepared_led", selected)
    return _metrics(
        corrected,
        truth,
        baseline,
        fit_config,
        loss,
        target_scale_ps,
    )


def _metric_value(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key, float("nan"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _summary_frame(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    group_columns = [
        "root_id",
        "root_file",
        "file_name",
        "voltage_V",
        "channel_mode",
        "input_waveforms",
        "target",
        "input_transform",
        "window_id",
        "window_start_ns",
        "window_end_ns",
        "window_length_ns",
        "undersampling_factor",
        "position_mode",
        "distance_metric",
        "dtw_radius_points",
        "selected_shapelet_count",
    ]
    metric_columns = [
        "ridge_alpha",
        "candidate_count",
        "train_event_count",
        "validation_event_count",
        "blind_event_count",
        "validation_loss",
        "validation_rmse_ps",
        "validation_bias_ps",
        "validation_ctr_ps",
        "validation_baseline_ctr_ps",
        "validation_relative_improvement_pct",
        "validation_ctr_minus_led_ps",
        "blind_loss",
        "blind_rmse_ps",
        "blind_bias_ps",
        "blind_ctr_ps",
        "blind_baseline_ctr_ps",
        "blind_relative_improvement_pct",
        "blind_ctr_minus_led_ps",
        "runtime_seconds",
    ]
    records: list[dict[str, Any]] = []
    for keys, group in rows.groupby(group_columns, dropna=False, sort=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(group_columns, key_values))
        base["n_folds"] = int(group["fold_id"].nunique())
        for column in metric_columns:
            values = pd.to_numeric(group[column], errors="coerce").dropna().to_numpy(float)
            if values.size == 0:
                mean = std = sem = float("nan")
            else:
                mean = float(np.mean(values))
                std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
                sem = float(std / math.sqrt(values.size)) if values.size > 1 else 0.0
            base[f"{column}_mean"] = mean
            base[f"{column}_std"] = std
            base[f"{column}_sem"] = sem
        records.append(base)
    return pd.DataFrame.from_records(records)


def _plot_performance(summary: pd.DataFrame, output: Path, dpi: int) -> None:
    if summary.empty:
        return
    plot_root = output / "plots"
    for (file_name, mode, transform), group in summary.groupby(
        ["file_name", "channel_mode", "input_transform"], sort=True
    ):
        figure, axis = plt.subplots(figsize=(9.0, 5.5))
        for count, series in group.groupby("selected_shapelet_count", sort=True):
            series = series.sort_values(
                ["window_length_ns", "window_start_ns", "window_end_ns"]
            )
            axis.errorbar(
                series["window_length_ns"],
                series["validation_ctr_minus_led_ps_mean"],
                yerr=series["validation_ctr_minus_led_ps_sem"],
                marker="o",
                capsize=3,
                label=f"{int(count)} shapelets",
            )
        axis.axhline(0.0, linestyle="--", linewidth=1.0, label="same CTR as LED")
        axis.set_xlabel("Window length [ns]")
        axis.set_ylabel("Validation CTR − LED CTR [ps]")
        axis.set_title(
            f"Shapelet correction vs LED\n{file_name} | {mode} | {transform}"
        )
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.tight_layout()
        destination = plot_root / (
            "__".join(map(_safe_name, (file_name, mode, transform)))
            + "__validation_delta_ctr.png"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=int(dpi), bbox_inches="tight")
        plt.close(figure)


def _plot_shapelet_catalog(
    shapelets: pd.DataFrame,
    values: pd.DataFrame,
    output: Path,
    dpi: int,
    max_plots: int = 30,
) -> None:
    if shapelets.empty or values.empty:
        return
    plot_root = output / "plots" / "shapelets"
    groups = shapelets.sort_values(
        ["file_name", "channel_mode", "window_id", "fold_id", "rank"]
    ).head(int(max_plots))
    for _, metadata in groups.iterrows():
        candidate_id = str(metadata["candidate_id"])
        row_key = str(metadata["row_key"])
        series = values[
            (values["candidate_id"].astype(str) == candidate_id)
            & (values["row_key"].astype(str) == row_key)
        ].sort_values("sample_index")
        if series.empty:
            continue
        figure, axis = plt.subplots(figsize=(7.0, 4.2))
        axis.plot(series["relative_time_ns"], series["value"], marker="o", markersize=3)
        axis.set_xlabel("Relative time [ns]")
        axis.set_ylabel("Difference-signal value")
        axis.set_title(
            f"Shapelet rank {int(metadata['rank'])} | corr={float(metadata['score_correlation']):+.3f}\n"
            f"{metadata['file_name']} | {metadata['channel_mode']} | "
            f"{metadata['window_id']} | fold {int(metadata['fold_id'])}"
        )
        axis.grid(True, alpha=0.3)
        figure.tight_layout()
        stem = "__".join(
            map(
                _safe_name,
                (
                    metadata["file_name"],
                    metadata["channel_mode"],
                    metadata["window_id"],
                    f"fold_{int(metadata['fold_id'])}",
                    f"rank_{int(metadata['rank'])}",
                    candidate_id,
                ),
            )
        )
        destination = plot_root / f"{stem}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=int(dpi), bbox_inches="tight")
        plt.close(figure)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone fold-local supervised shapelet study for the anchor-factored "
            "interpolated-LED correction target."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/adhoc/shapelet_correction_study"),
    )
    parser.add_argument("--file", action="append", dest="files")
    parser.add_argument(
        "--channel-mode",
        action="append",
        dest="channel_modes",
        choices=sorted(CHANNEL_MODES),
    )
    parser.add_argument("--window-id", action="append", dest="window_ids")
    parser.add_argument(
        "--input-transform",
        default="normalize",
        choices=sorted(SUPPORTED_INPUT_TRANSFORMS),
    )
    parser.add_argument(
        "--undersampling-factor",
        type=int,
        default=4,
        help="Keep one point every N native/transformed samples, independently per component",
    )
    parser.add_argument(
        "--shapelet-length-ns",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 4.0, 8.0],
    )
    parser.add_argument(
        "--candidates-per-group",
        type=int,
        default=128,
        help="Candidates drawn from each of negative, near-zero, positive target groups",
    )
    parser.add_argument("--extreme-fraction", type=float, default=0.15)
    parser.add_argument(
        "--score-events",
        type=int,
        default=1200,
        help="Maximum training events used for candidate scoring; model fitting uses all training events",
    )
    parser.add_argument(
        "--n-shapelets",
        type=int,
        nargs="+",
        default=[5, 10, 20],
    )
    parser.add_argument("--redundancy-threshold", type=float, default=0.95)
    parser.add_argument(
        "--position-mode",
        choices=["fixed", "sliding"],
        default="fixed",
    )
    parser.add_argument(
        "--distance-metric",
        choices=["dtw", "mse"],
        default="dtw",
        help=(
            "Shapelet distance. DTW is fixed-position and constrained by "
            "--dtw-radius-points; MSE preserves the previous Euclidean behavior."
        ),
    )
    parser.add_argument(
        "--dtw-radius-points",
        type=int,
        default=2,
        help=(
            "Sakoe-Chiba half-band in retained/undersampled samples. With "
            "--undersampling-factor 4, radius 2 permits a local warp of two "
            "retained points in either direction."
        ),
    )
    parser.add_argument("--local-z-normalize", action="store_true")
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        nargs="+",
        default=[0.01, 0.1, 1.0, 10.0, 100.0],
    )
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=20260806)
    parser.add_argument("--skip-blind", action="store_true")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--rebuild-preprocessing", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--feature-chunk-size", type=int, default=512)
    parser.add_argument("--shapelet-plot-limit", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.undersampling_factor <= 0:
        raise ValueError("--undersampling-factor must be positive")
    if args.candidates_per_group <= 0:
        raise ValueError("--candidates-per-group must be positive")
    if not 0.0 < args.extreme_fraction < 0.5:
        raise ValueError("--extreme-fraction must lie in (0, 0.5)")
    if not 0.0 <= args.redundancy_threshold <= 1.0:
        raise ValueError("--redundancy-threshold must lie in [0, 1]")
    if args.dtw_radius_points < 0:
        raise ValueError("--dtw-radius-points must be non-negative")
    if args.distance_metric == "dtw" and args.position_mode != "fixed":
        raise ValueError(
            "--distance-metric dtw requires --position-mode fixed. "
            "This keeps DTW local and preserves the physical shapelet location."
        )
    shapelet_counts = sorted(set(int(value) for value in args.n_shapelets))
    if not shapelet_counts or any(value <= 0 for value in shapelet_counts):
        raise ValueError("--n-shapelets values must be positive")
    durations = sorted(set(float(value) for value in args.shapelet_length_ns))
    if not durations or any(not math.isfinite(value) or value <= 0.0 for value in durations):
        raise ValueError("--shapelet-length-ns values must be finite and positive")
    ridge_alphas = sorted(set(float(value) for value in args.ridge_alpha))
    if not ridge_alphas or any(not math.isfinite(value) or value <= 0.0 for value in ridge_alphas):
        raise ValueError("--ridge-alpha values must be finite and positive")

    config = load_study_config(args.config, PROJECT)
    output = args.output_dir if args.output_dir.is_absolute() else PROJECT / args.output_dir
    output = output.resolve()
    if args.restart and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(
        output / "shapelet_correction_study.log",
        config.get("logging", {}).get("level", "INFO"),
    )

    transform = normalize_input_transform(args.input_transform)
    root_files = [Path(value) for value in config["root_files"]]
    if args.files:
        requested = set(args.files)
        root_files = [
            path
            for path in root_files
            if str(path) in requested or path.name in requested or path.stem in requested
        ]
        if not root_files:
            raise ValueError(f"None of --file {sorted(requested)} matched configured ROOT files")
    modes = _filtered_values(args.channel_modes, config["channel_modes"], "channel modes")
    window_ids = [str(window["id"]) for window in config["windows_ns"]]
    selected_window_ids = _filtered_values(args.window_ids, window_ids, "window IDs")
    windows = [
        window for window in config["windows_ns"] if str(window["id"]) in selected_window_ids
    ]

    resolved = {
        "script": "run_shapelet_correction_study.py",
        "format_version": 2,
        "source_study_config": str(Path(args.config).resolve()),
        "source_study_config_hash": config["_config_hash"],
        "root_files": [str(path) for path in root_files],
        "channel_modes": modes,
        "windows": windows,
        "input_transform": transform,
        "undersampling_factor": int(args.undersampling_factor),
        "shapelet_lengths_ns": durations,
        "candidates_per_group": int(args.candidates_per_group),
        "extreme_fraction": float(args.extreme_fraction),
        "score_events": int(args.score_events),
        "n_shapelets": shapelet_counts,
        "redundancy_threshold": float(args.redundancy_threshold),
        "position_mode": str(args.position_mode),
        "distance_metric": str(args.distance_metric),
        "dtw_radius_points": int(args.dtw_radius_points),
        "local_z_normalize": bool(args.local_z_normalize),
        "ridge_alphas": ridge_alphas,
        "inner_folds": int(args.inner_folds),
        "random_seed": int(args.random_seed),
        "skip_blind": bool(args.skip_blind),
        "cross_validation": config["cross_validation"],
        "selection": config["selection"],
        "fit": config["fit"],
        "split": config.get("split", {}),
    }
    resolved["fingerprint"] = canonical_hash(resolved)
    resolved_path = output / "resolved_shapelet_correction_study.json"
    if resolved_path.is_file() and not args.restart:
        previous = json.loads(resolved_path.read_text(encoding="utf-8"))
        if previous.get("fingerprint") != resolved["fingerprint"]:
            raise RuntimeError(
                "Existing shapelet output was created with different settings. "
                "Use --restart or choose another --output-dir."
            )
    atomic_json(resolved_path, resolved)

    fold_path = output / "fold_results.csv"
    shapelet_path = output / "selected_shapelets.csv"
    value_path = output / "shapelet_values.csv"
    fold_rows = _load_rows(fold_path)
    shapelet_rows = _load_rows(shapelet_path)
    value_rows = _load_rows(value_path)
    completed_keys = {str(row.get("row_key", "")) for row in fold_rows}
    common_loss = {"id": "mse", "type": "mse"}

    logger.info(
        "Shapelet study start | files=%d | modes=%d | windows=%d | undersampling=%d | "
        "candidate/group=%d | shapelets=%s | position=%s | distance=%s | dtw-radius=%d",
        len(root_files),
        len(modes),
        len(windows),
        args.undersampling_factor,
        args.candidates_per_group,
        shapelet_counts,
        args.position_mode,
        args.distance_metric,
        args.dtw_radius_points,
    )

    for file_position, root_file in enumerate(root_files, start=1):
        root_id = _root_id(root_file)
        logger.info("File %d/%d | %s", file_position, len(root_files), root_file.name)
        development, blind = _ensure_preprocessed(
            config,
            root_file,
            root_id,
            output,
            rebuild=bool(args.rebuild_preprocessing),
            logger=logger,
        )
        voltage = _extract_voltage(root_file, config.get("reporting", {}))

        for mode_id in modes:
            mode = CHANNEL_MODES[mode_id]
            folds = _fold_masks(
                development,
                blind,
                mode["target"],
                config["cross_validation"],
                config["selection"],
            )
            logger.info("Mode %s | folds=%d", mode_id, len(folds))

            for window_position, window in enumerate(windows, start=1):
                logger.info(
                    "Window %d/%d | %s [%.3f, %.3f] ns",
                    window_position,
                    len(windows),
                    window["id"],
                    window["start_ns"],
                    window["end_ns"],
                )
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
                transform_cache = output / "transform_cache" / root_id / mode_id / str(window["id"])
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
                    int(args.undersampling_factor),
                )

                for fold in folds:
                    fold_id = int(fold["fold_id"])
                    required_keys = {
                        canonical_hash(
                            {
                                "fingerprint": resolved["fingerprint"],
                                "root_id": root_id,
                                "mode": mode_id,
                                "window": window["id"],
                                "fold": fold_id,
                                "shapelet_count": count,
                            }
                        )[:24]
                        for count in shapelet_counts
                    }
                    catalog_key = canonical_hash(
                        {
                            "fingerprint": resolved["fingerprint"],
                            "root_id": root_id,
                            "mode": mode_id,
                            "window": window["id"],
                            "fold": fold_id,
                            "catalog": True,
                        }
                    )[:24]
                    existing_catalog = {
                        str(row.get("row_key", "")) for row in shapelet_rows
                    }
                    need_catalog = catalog_key not in existing_catalog
                    if required_keys.issubset(completed_keys) and not need_catalog:
                        continue

                    started = time.time()
                    train_indices = np.asarray(fold["train"], dtype=np.int64)
                    validation_indices = np.asarray(fold["validation"], dtype=np.int64)
                    blind_indices = np.asarray(fold["blind"], dtype=np.int64)
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
                    x_validation = _materialize_difference_matrix(
                        transformed_development,
                        validation_indices,
                        normalization.std_mV,
                        selected_features,
                        chunk_size=int(args.chunk_size),
                    )
                    x_blind = (
                        None
                        if args.skip_blind
                        else _materialize_difference_matrix(
                            transformed_blind,
                            blind_indices,
                            normalization.std_mV,
                            selected_features,
                            chunk_size=int(args.chunk_size),
                        )
                    )
                    y_train = factored_correction_target_ps(
                        transformed_development,
                        train_indices,
                    )
                    y_validation = factored_correction_target_ps(
                        transformed_development,
                        validation_indices,
                    )
                    target_scale = max(float(np.std(y_train, ddof=0)), 1.0e-8)

                    seed = int(args.random_seed) + fold_id + 1000 * window_position + 100000 * file_position
                    rng = np.random.default_rng(seed)
                    if args.score_events > 0 and x_train.shape[0] > args.score_events:
                        score_indices = np.sort(
                            rng.choice(
                                x_train.shape[0],
                                size=int(args.score_events),
                                replace=False,
                            )
                        )
                    else:
                        score_indices = np.arange(x_train.shape[0], dtype=np.int64)
                    candidates = generate_candidates(
                        x_train,
                        y_train,
                        components,
                        durations,
                        candidates_per_group=int(args.candidates_per_group),
                        extreme_fraction=float(args.extreme_fraction),
                        rng=rng,
                    )
                    max_shapelets = max(shapelet_counts)
                    selected, _, selected_scores = score_and_select_candidates(
                        x_train[score_indices],
                        y_train[score_indices],
                        candidates,
                        components,
                        max_shapelets=max_shapelets,
                        redundancy_threshold=float(args.redundancy_threshold),
                        position_mode=str(args.position_mode),
                        local_z_normalize=bool(args.local_z_normalize),
                        distance_metric=str(args.distance_metric),
                        dtw_radius_points=int(args.dtw_radius_points),
                        feature_chunk_size=int(args.feature_chunk_size),
                    )
                    if len(selected) < min(shapelet_counts):
                        raise RuntimeError(
                            f"Only {len(selected)} non-redundant shapelets survived, but "
                            f"at least {min(shapelet_counts)} are required. Increase "
                            "--candidates-per-group or --redundancy-threshold."
                        )
                    usable_counts = [count for count in shapelet_counts if count <= len(selected)]
                    train_features_all = materialize_shapelet_features(
                        x_train,
                        selected,
                        components,
                        position_mode=str(args.position_mode),
                        local_z_normalize=bool(args.local_z_normalize),
                        distance_metric=str(args.distance_metric),
                        dtw_radius_points=int(args.dtw_radius_points),
                        feature_chunk_size=int(args.feature_chunk_size),
                    )
                    validation_features_all = materialize_shapelet_features(
                        x_validation,
                        selected,
                        components,
                        position_mode=str(args.position_mode),
                        local_z_normalize=bool(args.local_z_normalize),
                        distance_metric=str(args.distance_metric),
                        dtw_radius_points=int(args.dtw_radius_points),
                        feature_chunk_size=int(args.feature_chunk_size),
                    )
                    blind_features_all = (
                        None
                        if x_blind is None
                        else materialize_shapelet_features(
                            x_blind,
                            selected,
                            components,
                            position_mode=str(args.position_mode),
                            local_z_normalize=bool(args.local_z_normalize),
                            distance_metric=str(args.distance_metric),
                            dtw_radius_points=int(args.dtw_radius_points),
                            feature_chunk_size=int(args.feature_chunk_size),
                        )
                    )

                    max_model = None
                    max_feature_mean = None
                    max_feature_std = None
                    for count in usable_counts:
                        row_key = canonical_hash(
                            {
                                "fingerprint": resolved["fingerprint"],
                                "root_id": root_id,
                                "mode": mode_id,
                                "window": window["id"],
                                "fold": fold_id,
                                "shapelet_count": count,
                            }
                        )[:24]
                        if row_key in completed_keys and not (
                            need_catalog and count == max(usable_counts)
                        ):
                            continue
                        model, feature_mean, feature_std, chosen_alpha = _fit_shapelet_ridge(
                            train_features_all[:, :count],
                            y_train,
                            ridge_alphas,
                            inner_folds=int(args.inner_folds),
                        )
                        validation_prediction = _predict_shapelet_ridge(
                            model,
                            validation_features_all[:, :count],
                            feature_mean,
                            feature_std,
                        )
                        validation_metrics = _prediction_metrics_from_prediction(
                            dataset=transformed_development,
                            indices=validation_indices,
                            prediction=validation_prediction,
                            fit_config=config["fit"],
                            loss=common_loss,
                            target_scale_ps=target_scale,
                        )
                        if blind_features_all is None:
                            blind_metrics: dict[str, float] = {}
                        else:
                            blind_prediction = _predict_shapelet_ridge(
                                model,
                                blind_features_all[:, :count],
                                feature_mean,
                                feature_std,
                            )
                            blind_metrics = _prediction_metrics_from_prediction(
                                dataset=transformed_blind,
                                indices=blind_indices,
                                prediction=blind_prediction,
                                fit_config=config["fit"],
                                loss=common_loss,
                                target_scale_ps=target_scale,
                            )

                        fold_rows.append(
                            {
                                "row_key": row_key,
                                "root_id": root_id,
                                "root_file": str(root_file),
                                "file_name": root_file.name,
                                "voltage_V": voltage,
                                "channel_mode": mode_id,
                                "input_waveforms": mode["input_waveforms"],
                                "target": mode["target"],
                                "input_transform": transform,
                                "window_id": str(window["id"]),
                                "window_start_ns": float(window["start_ns"]),
                                "window_end_ns": float(window["end_ns"]),
                                "window_length_ns": float(window["end_ns"] - window["start_ns"]),
                                "fold_id": fold_id,
                                "undersampling_factor": int(args.undersampling_factor),
                                "position_mode": str(args.position_mode),
                                "distance_metric": str(args.distance_metric),
                                "dtw_radius_points": int(args.dtw_radius_points),
                                "candidate_count": len(candidates),
                                "selected_shapelet_count": int(count),
                                "ridge_alpha": chosen_alpha,
                                "train_event_count": int(train_indices.size),
                                "validation_event_count": int(validation_indices.size),
                                "blind_event_count": int(blind_indices.size) if not args.skip_blind else 0,
                                "validation_loss": _metric_value(validation_metrics, "loss"),
                                "validation_rmse_ps": _metric_value(validation_metrics, "rmse_ps"),
                                "validation_bias_ps": _metric_value(validation_metrics, "bias_ps"),
                                "validation_ctr_ps": _metric_value(validation_metrics, "ctr_ps"),
                                "validation_baseline_ctr_ps": _metric_value(validation_metrics, "baseline_ctr_ps"),
                                "validation_relative_improvement_pct": _metric_value(validation_metrics, "relative_improvement_pct"),
                                "validation_ctr_minus_led_ps": _metric_value(validation_metrics, "ctr_ps") - _metric_value(validation_metrics, "baseline_ctr_ps"),
                                "blind_loss": _metric_value(blind_metrics, "loss"),
                                "blind_rmse_ps": _metric_value(blind_metrics, "rmse_ps"),
                                "blind_bias_ps": _metric_value(blind_metrics, "bias_ps"),
                                "blind_ctr_ps": _metric_value(blind_metrics, "ctr_ps"),
                                "blind_baseline_ctr_ps": _metric_value(blind_metrics, "baseline_ctr_ps"),
                                "blind_relative_improvement_pct": _metric_value(blind_metrics, "relative_improvement_pct"),
                                "blind_ctr_minus_led_ps": _metric_value(blind_metrics, "ctr_ps") - _metric_value(blind_metrics, "baseline_ctr_ps"),
                                "runtime_seconds": float(time.time() - started),
                            }
                        )
                        completed_keys.add(row_key)
                        if count == max(usable_counts):
                            max_model = model
                            max_feature_mean = feature_mean
                            max_feature_std = feature_std

                    # Persist the selected catalog once per fold/window, with the
                    # coefficients from the largest requested shapelet model.
                    if need_catalog:
                        coefficients = (
                            np.asarray(max_model.coef_, dtype=np.float64).reshape(-1)
                            if max_model is not None
                            else np.full(len(selected), np.nan)
                        )
                        for rank, (candidate, score) in enumerate(
                            zip(selected, selected_scores), start=1
                        ):
                            shapelet_rows.append(
                                {
                                    "row_key": catalog_key,
                                    "root_id": root_id,
                                    "file_name": root_file.name,
                                    "voltage_V": voltage,
                                    "channel_mode": mode_id,
                                    "input_transform": transform,
                                    "window_id": str(window["id"]),
                                    "fold_id": fold_id,
                                    "distance_metric": str(args.distance_metric),
                                    "dtw_radius_points": int(args.dtw_radius_points),
                                    "rank": rank,
                                    "candidate_id": candidate.candidate_id,
                                    "component_index": candidate.component_index,
                                    "component_name": candidate.component_name,
                                    "start_index": candidate.start_index,
                                    "length_points": candidate.length_points,
                                    "start_time_ns": candidate.start_time_ps / 1000.0,
                                    "end_time_ns": candidate.end_time_ps / 1000.0,
                                    "duration_ns": candidate.duration_ns,
                                    "source_event_index": candidate.source_event_index,
                                    "source_group": candidate.source_group,
                                    "source_target_ps": candidate.source_target_ps,
                                    "score_correlation": float(score),
                                    "score_abs_correlation": abs(float(score)),
                                    "ridge_coefficient_at_max_k": (
                                        float(coefficients[rank - 1])
                                        if rank - 1 < coefficients.size
                                        else float("nan")
                                    ),
                                    "feature_mean_train": (
                                        float(max_feature_mean[rank - 1])
                                        if max_feature_mean is not None and rank - 1 < max_feature_mean.size
                                        else float("nan")
                                    ),
                                    "feature_std_train": (
                                        float(max_feature_std[rank - 1])
                                        if max_feature_std is not None and rank - 1 < max_feature_std.size
                                        else float("nan")
                                    ),
                                }
                            )
                            component = components[candidate.component_index]
                            candidate_times = component.relative_time_ps[
                                candidate.start_index : candidate.start_index + candidate.length_points
                            ]
                            for sample_index, (sample_time, sample_value) in enumerate(
                                zip(candidate_times, candidate.values)
                            ):
                                value_rows.append(
                                    {
                                        "row_key": catalog_key,
                                        "candidate_id": candidate.candidate_id,
                                        "sample_index": sample_index,
                                        "relative_time_ns": float(sample_time / 1000.0),
                                        "value": float(sample_value),
                                    }
                                )

                    _atomic_csv(fold_path, pd.DataFrame(fold_rows, columns=FOLD_RESULT_COLUMNS))
                    _atomic_csv(shapelet_path, pd.DataFrame(shapelet_rows, columns=SHAPELET_COLUMNS))
                    _atomic_csv(value_path, pd.DataFrame(value_rows, columns=SHAPELET_VALUE_COLUMNS))
                    logger.info(
                        "Completed | %s | %s | %s | fold=%d | candidates=%d | selected=%d",
                        root_file.name,
                        mode_id,
                        window["id"],
                        fold_id,
                        len(candidates),
                        len(selected),
                    )

    fold_frame = pd.DataFrame(fold_rows, columns=FOLD_RESULT_COLUMNS)
    summary = _summary_frame(fold_frame)
    _atomic_csv(output / "summary.csv", summary)
    shapelet_frame = pd.DataFrame(shapelet_rows, columns=SHAPELET_COLUMNS)
    value_frame = pd.DataFrame(value_rows, columns=SHAPELET_VALUE_COLUMNS)
    _plot_performance(summary, output, int(config.get("reporting", {}).get("dpi", 160)))
    _plot_shapelet_catalog(
        shapelet_frame,
        value_frame,
        output,
        int(config.get("reporting", {}).get("dpi", 160)),
        max_plots=int(args.shapelet_plot_limit),
    )
    logger.info(
        "Shapelet study complete | fold rows=%d | shapelets=%d | output=%s",
        len(fold_frame),
        len(shapelet_frame),
        output,
    )


if __name__ == "__main__":
    main()
