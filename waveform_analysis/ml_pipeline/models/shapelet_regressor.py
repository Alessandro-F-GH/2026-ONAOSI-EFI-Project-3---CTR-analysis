from __future__ import annotations

import csv
import hashlib
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from ..common import atomic_json, canonical_hash
from ..input_transform import component_subsampling_indices
from ..metrics import fit_times_ps
from ..training_context import TrainingContext
from ..torch_data import factored_correction_target_ps
from ..training_utils import (
    checkpoint_context,
    evaluate_model,
    make_split_loader,
    resolve_device,
)
from .spec import ModelSpec


_LOSS_ALIASES = {
    "variance": "variance",
    "rmse": "rmse",
    "variance_bias": "variance_bias",
    "variance_plus_bias": "variance_bias",
    "variance+bias": "variance_bias",
}


@dataclass(frozen=True)
class _Component:
    name: str
    output_start: int
    output_stop: int
    times_ps: np.ndarray


@dataclass(frozen=True)
class _Candidate:
    candidate_id: str
    component_index: int
    component_name: str
    local_start: int
    global_start: int
    length: int
    start_time_ps: float
    end_time_ps: float
    source_group: str
    source_target_ps: float
    values: np.ndarray


@dataclass(frozen=True)
class _ShapeletFeatureBundle:
    shapelets: tuple[_Candidate, ...]
    scores: np.ndarray
    train_features: np.ndarray
    validation_features: np.ndarray


# Small in-memory LRU caches. They avoid re-reading and re-normalizing the same
# fold/window data across shapelet hyperparameter trials without writing another
# dataset to disk. The limits keep memory bounded as the study advances.
_DIFFERENCE_CACHE: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()
_FEATURE_CACHE: OrderedDict[str, _ShapeletFeatureBundle] = OrderedDict()
_DIFFERENCE_CACHE_LIMIT = 3
_FEATURE_CACHE_LIMIT = 6


def clear_runtime_cache() -> None:
    _DIFFERENCE_CACHE.clear()
    _FEATURE_CACHE.clear()


def _lru_get(cache: OrderedDict[str, Any], key: str) -> Any | None:
    value = cache.get(key)
    if value is not None:
        cache.move_to_end(key)
    return value


def _lru_put(cache: OrderedDict[str, Any], key: str, value: Any, limit: int) -> Any:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > int(limit):
        cache.popitem(last=False)
    return value


def _selection_loss(config: dict[str, Any]) -> str:
    raw = str(config.get("loss", {}).get("type", "rmse")).strip().lower()
    try:
        return _LOSS_ALIASES[raw]
    except KeyError as exc:
        raise ValueError(
            "model.loss.type must be one of ['variance', 'rmse', 'variance_bias']"
        ) from exc


def validate_config(config: dict[str, Any]) -> None:
    if int(config.get("n_shapelets", 10)) <= 0:
        raise ValueError("n_shapelets must be positive")
    if int(config.get("candidate_pool_size", 20)) < int(config.get("n_shapelets", 10)):
        raise ValueError("candidate_pool_size must be >= n_shapelets")
    lengths = config.get("shapelet_lengths_ns", [1.0, 2.0, 4.0])
    if not isinstance(lengths, list) or not lengths:
        raise ValueError("shapelet_lengths_ns must be a non-empty list")
    if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in lengths):
        raise ValueError("shapelet_lengths_ns values must be finite and positive")
    if int(config.get("candidates_per_group", 64)) <= 0:
        raise ValueError("candidates_per_group must be positive")
    if not 0.0 < float(config.get("extreme_fraction", 0.15)) < 0.5:
        raise ValueError("extreme_fraction must lie in (0, 0.5)")
    if int(config.get("score_events", 1000)) <= 0:
        raise ValueError("score_events must be positive")
    threshold = float(config.get("redundancy_threshold", 0.95))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("redundancy_threshold must lie in [0, 1]")
    metric = str(config.get("distance_metric", "dtw")).strip().lower()
    if metric not in {"dtw", "mse"}:
        raise ValueError("distance_metric must be 'dtw' or 'mse'")
    if int(config.get("dtw_radius_points", 2)) < 0:
        raise ValueError("dtw_radius_points must be non-negative")
    alpha = float(config.get("ridge_alpha", 100.0))
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("ridge_alpha must be finite and positive")
    loss = config.get("loss", {})
    if not isinstance(loss, dict):
        raise ValueError("loss must be an object")
    _selection_loss(config)
    if float(loss.get("bias_weight", 0.0)) < 0.0:
        raise ValueError("loss.bias_weight must be non-negative")
    if str(loss.get("bias_normalization", "none")) not in {"none", "target_std"}:
        raise ValueError("loss.bias_normalization must be 'none' or 'target_std'")


def validate_training_config(config: dict[str, Any]) -> None:
    training = config["training"]
    for name in (
        "batch_size",
        "normalization_chunk_size",
        "shapelet_materialization_chunk_size",
        "shapelet_feature_chunk_size",
    ):
        if int(training.get(name, 0)) <= 0:
            raise ValueError(f"training.{name} must be positive")
    if bool(training.get("random_pair_swap", False)):
        raise ValueError(
            "training.random_pair_swap is not supported for fixed learned shapelets"
        )
    baseline_guard_metric = training.get("baseline_guard_metric")
    if baseline_guard_metric not in (None, "validation_rmse", "validation_ctr"):
        raise ValueError(
            "training.baseline_guard_metric must be null, 'validation_rmse', or 'validation_ctr'"
        )


class ShapeletPairRegressor(nn.Module):
    """Fixed-position shapelet-distance features followed by Ridge coefficients."""

    def __init__(self, config: dict[str, Any], input_length: int) -> None:
        super().__init__()
        self.input_length = int(input_length)
        count = int(config.get("_serialized_shapelet_count", config.get("n_shapelets", 1)))
        width = int(config.get("_serialized_shapelet_width", 1))
        self.distance_metric = str(config.get("distance_metric", "dtw")).lower()
        self.dtw_radius_points = int(config.get("dtw_radius_points", 2))
        self.local_z_normalize = bool(config.get("local_z_normalize", False))
        self.register_buffer("shapelet_values", torch.zeros((count, width), dtype=torch.float32))
        self.register_buffer("shapelet_lengths", torch.ones(count, dtype=torch.long))
        self.register_buffer("shapelet_starts", torch.zeros(count, dtype=torch.long))
        self.register_buffer("feature_mean", torch.zeros(count, dtype=torch.float32))
        self.register_buffer("feature_std", torch.ones(count, dtype=torch.float32))
        self.register_buffer("coefficient", torch.zeros(count, dtype=torch.float32))
        self.pair_output_bias_ps = nn.Parameter(
            torch.zeros((), dtype=torch.float32), requires_grad=False
        )

    @staticmethod
    def _local_z(values: torch.Tensor) -> torch.Tensor:
        mean = values.mean(dim=-1, keepdim=True)
        std = values.std(dim=-1, unbiased=False, keepdim=True)
        std = torch.where(std > 1.0e-8, std, torch.ones_like(std))
        return (values - mean) / std

    def _dtw(self, segments: torch.Tensor, shapelet: torch.Tensor) -> torch.Tensor:
        batch, length = segments.shape
        inf = torch.tensor(float("inf"), device=segments.device, dtype=segments.dtype)
        previous_cost = torch.full((batch, length + 1), inf, device=segments.device)
        previous_steps = torch.zeros(
            (batch, length + 1), device=segments.device, dtype=torch.long
        )
        previous_cost[:, 0] = 0.0
        radius = int(self.dtw_radius_points)
        for i in range(1, length + 1):
            current_cost = torch.full((batch, length + 1), inf, device=segments.device)
            current_steps = torch.zeros(
                (batch, length + 1), device=segments.device, dtype=torch.long
            )
            for j in range(max(1, i - radius), min(length, i + radius) + 1):
                options = torch.stack(
                    (
                        previous_cost[:, j - 1],
                        previous_cost[:, j],
                        current_cost[:, j - 1],
                    ),
                    dim=0,
                )
                best, choice = torch.min(options, dim=0)
                step_options = torch.stack(
                    (
                        previous_steps[:, j - 1],
                        previous_steps[:, j],
                        current_steps[:, j - 1],
                    ),
                    dim=0,
                )
                chosen_steps = torch.gather(step_options, 0, choice.unsqueeze(0)).squeeze(0)
                current_cost[:, j] = best + (segments[:, i - 1] - shapelet[j - 1]) ** 2
                current_steps[:, j] = chosen_steps + 1
            previous_cost = current_cost
            previous_steps = current_steps
        return previous_cost[:, length] / previous_steps[:, length].clamp_min(1).to(segments.dtype)

    def forward(self, waveform_pair: torch.Tensor) -> torch.Tensor:
        if waveform_pair.ndim != 3 or waveform_pair.shape[1] != 2:
            raise ValueError("Expected waveform pairs with shape [batch, 2, length]")
        if waveform_pair.shape[2] != self.input_length:
            raise ValueError(f"Expected waveform length {self.input_length}")
        sampled = waveform_pair[:, 0, :] - waveform_pair[:, 1, :]
        features: list[torch.Tensor] = []
        for index in range(int(self.shapelet_lengths.numel())):
            length = int(self.shapelet_lengths[index].item())
            start = int(self.shapelet_starts[index].item())
            segment = sampled[:, start : start + length]
            shapelet = self.shapelet_values[index, :length]
            if self.local_z_normalize:
                segment = self._local_z(segment)
                shapelet = self._local_z(shapelet.unsqueeze(0))[0]
            if self.distance_metric == "mse":
                distance = torch.mean((segment - shapelet.unsqueeze(0)) ** 2, dim=1)
            elif self.distance_metric == "dtw":
                distance = self._dtw(segment, shapelet)
            else:
                raise RuntimeError(f"Unsupported distance metric {self.distance_metric!r}")
            features.append(distance)
        matrix = torch.stack(features, dim=1)
        standardized = (matrix - self.feature_mean) / self.feature_std
        return standardized @ self.coefficient + self.pair_output_bias_ps


def build(config: dict[str, Any], input_length: int) -> nn.Module:
    return ShapeletPairRegressor(config, input_length)


class _ZeroCorrectionModel(nn.Module):
    apply_window_anchor_shift = False

    def forward(self, waveform_pair: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            waveform_pair.shape[0], dtype=waveform_pair.dtype, device=waveform_pair.device
        )


def _hash_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _difference_cache_key(context: TrainingContext, split_name: str) -> str:
    descriptors = []
    for dataset in context.datasets:
        indices = np.asarray(getattr(dataset, split_name), dtype=np.int64)
        descriptors.append(
            {
                "directory": str(dataset.directory),
                "fingerprint": str(dataset.manifest.get("fingerprint", "")),
                "indices": _hash_array(indices),
            }
        )
    return canonical_hash(
        {
            "split": split_name,
            "datasets": descriptors,
            "normalization": _hash_array(np.asarray(context.normalization.std_mV)),
            "input_length": context.input_length,
            "input_transform": context.input_transform,
            "subsampling_factor": int(context.subsampling_factor),
        }
    )


def _split_difference_matrix(
    context: TrainingContext, split_name: str, *, chunk_size: int
) -> tuple[np.ndarray, np.ndarray]:
    key = _difference_cache_key(context, split_name)
    cached = _lru_get(_DIFFERENCE_CACHE, key)
    if cached is not None:
        return cached
    total = sum(int(np.asarray(getattr(dataset, split_name)).size) for dataset in context.datasets)
    if total <= 0:
        raise ValueError(f"Cannot materialize empty {split_name} split")
    matrix = np.empty((total, context.input_length), dtype=np.float32)
    target = np.empty(total, dtype=np.float64)
    scale = np.asarray(context.normalization.std_mV, dtype=np.float64)
    if scale.ndim not in {0, 1}:
        raise ValueError("Invalid normalization scale")
    if scale.ndim == 1 and scale.size != context.input_length:
        raise ValueError("Featurewise normalization length mismatch")
    cursor = 0
    for dataset in context.datasets:
        indices = np.asarray(getattr(dataset, split_name), dtype=np.int64)
        for start in range(0, indices.size, int(chunk_size)):
            selected = indices[start : start + int(chunk_size)]
            pair = np.asarray(dataset.windows_mV[selected], dtype=np.float64)
            raw_lengths = dataset.manifest.get("input_component_lengths")
            lengths = (
                [int(value) for value in raw_lengths]
                if isinstance(raw_lengths, list)
                else [int(dataset.input_length)]
            )
            source_indices = component_subsampling_indices(
                lengths, context.subsampling_factor
            )
            pair = pair[..., source_indices]
            block = (pair[:, 0, :] - pair[:, 1, :]) / scale
            size = int(selected.size)
            matrix[cursor : cursor + size] = block.astype(np.float32, copy=False)
            target[cursor : cursor + size] = factored_correction_target_ps(dataset, selected)
            cursor += size
    return _lru_put(_DIFFERENCE_CACHE, key, (matrix, target), _DIFFERENCE_CACHE_LIMIT)


def _component_plan(
    dataset: Any, factor: int
) -> tuple[np.ndarray, tuple[_Component, ...]]:
    times = np.asarray(dataset.relative_time_ps, dtype=np.float64).reshape(-1)
    raw_lengths = dataset.manifest.get("input_component_lengths")
    lengths = [int(value) for value in raw_lengths] if isinstance(raw_lengths, list) else [times.size]
    raw_names = dataset.manifest.get("input_components")
    names = [str(value) for value in raw_names] if isinstance(raw_names, list) else ["waveform"]
    if len(names) != len(lengths):
        names = [f"component_{index}" for index in range(len(lengths))]
    if sum(lengths) != times.size:
        raise ValueError("Input component lengths do not match transformed time grid")
    selected_parts: list[np.ndarray] = []
    components: list[_Component] = []
    source_cursor = 0
    output_cursor = 0
    for name, length in zip(names, lengths):
        local = np.arange(0, length, int(factor), dtype=np.int64)
        source = source_cursor + local
        selected_parts.append(source)
        components.append(
            _Component(
                name=name,
                output_start=output_cursor,
                output_stop=output_cursor + local.size,
                times_ps=times[source],
            )
        )
        source_cursor += length
        output_cursor += local.size
    return np.concatenate(selected_parts), tuple(components)


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


def _local_z(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    mean = np.mean(array, axis=-1, keepdims=True)
    std = np.std(array, axis=-1, ddof=0, keepdims=True)
    std = np.where(std > 1.0e-8, std, 1.0)
    return (array - mean) / std


def _dtw_distance_batch(segments: np.ndarray, shapelet: np.ndarray, radius: int) -> np.ndarray:
    x = np.asarray(segments, dtype=np.float64)
    q = np.asarray(shapelet, dtype=np.float64).reshape(-1)
    event_count, length = x.shape
    previous_cost = np.full((event_count, length + 1), np.inf, dtype=np.float64)
    previous_steps = np.zeros((event_count, length + 1), dtype=np.int32)
    previous_cost[:, 0] = 0.0
    for i in range(1, length + 1):
        current_cost = np.full((event_count, length + 1), np.inf, dtype=np.float64)
        current_steps = np.zeros((event_count, length + 1), dtype=np.int32)
        for j in range(max(1, i - radius), min(length, i + radius) + 1):
            options = np.stack(
                (previous_cost[:, j - 1], previous_cost[:, j], current_cost[:, j - 1]),
                axis=0,
            )
            choice = np.argmin(options, axis=0)
            best = np.take_along_axis(options, choice[None, :], axis=0)[0]
            step_options = np.stack(
                (previous_steps[:, j - 1], previous_steps[:, j], current_steps[:, j - 1]),
                axis=0,
            )
            chosen_steps = np.take_along_axis(step_options, choice[None, :], axis=0)[0]
            current_cost[:, j] = best + (x[:, i - 1] - q[j - 1]) ** 2
            current_steps[:, j] = chosen_steps + 1
        previous_cost = current_cost
        previous_steps = current_steps
    steps = previous_steps[:, length]
    if np.any(steps <= 0):
        raise RuntimeError("No valid constrained DTW path")
    return previous_cost[:, length] / steps


def _distance_feature(
    signals: np.ndarray,
    candidate: _Candidate,
    *,
    metric: str,
    radius: int,
    local_z: bool,
    chunk_size: int,
) -> np.ndarray:
    output = np.empty(signals.shape[0], dtype=np.float64)
    for start in range(0, signals.shape[0], int(chunk_size)):
        block = np.asarray(signals[start : start + int(chunk_size)], dtype=np.float64)
        segment = block[:, candidate.global_start : candidate.global_start + candidate.length]
        shapelet = np.asarray(candidate.values, dtype=np.float64)
        if local_z:
            segment = _local_z(segment)
            shapelet = _local_z(shapelet[None, :])[0]
        if metric == "mse":
            values = np.mean((segment - shapelet[None, :]) ** 2, axis=1)
        else:
            values = _dtw_distance_batch(segment, shapelet, radius)
        output[start : start + block.shape[0]] = values
    return output


def _length_points(component: _Component, duration_ns: float) -> int:
    times = component.times_ps
    if times.size < 2:
        raise ValueError(f"Component {component.name} has fewer than two retained samples")
    step = float(np.median(np.abs(np.diff(times))))
    points = max(2, int(round(float(duration_ns) * 1000.0 / step)) + 1)
    return min(points, int(times.size))


def _target_groups(target: np.ndarray, fraction: float) -> dict[str, np.ndarray]:
    values = np.asarray(target, dtype=np.float64)
    count = max(1, int(math.ceil(values.size * float(fraction))))
    order = np.argsort(values)
    centered = np.argsort(np.abs(values - np.median(values)))
    return {"negative": order[:count], "near_zero": centered[:count], "positive": order[-count:]}


def _generate_candidates(
    signals: np.ndarray,
    target: np.ndarray,
    components: Sequence[_Component],
    config: dict[str, Any],
    rng: np.random.Generator,
) -> list[_Candidate]:
    groups = _target_groups(target, float(config.get("extreme_fraction", 0.15)))
    durations = [float(value) for value in config.get("shapelet_lengths_ns", [1.0, 2.0, 4.0])]
    per_group = int(config.get("candidates_per_group", 64))
    output: list[_Candidate] = []
    for group_name, group_indices in groups.items():
        for number in range(per_group):
            component_index = int(rng.integers(0, len(components)))
            component = components[component_index]
            duration = durations[int(rng.integers(0, len(durations)))]
            length = _length_points(component, duration)
            available = component.output_stop - component.output_start
            local_start = int(rng.integers(0, available - length + 1))
            source_event = int(group_indices[int(rng.integers(0, group_indices.size))])
            global_start = component.output_start + local_start
            values = np.asarray(
                signals[source_event, global_start : global_start + length], dtype=np.float64
            ).copy()
            time_values = component.times_ps[local_start : local_start + length]
            candidate_id = canonical_hash(
                {
                    "group": group_name,
                    "number": number,
                    "component": component.name,
                    "start": local_start,
                    "length": length,
                    "event": source_event,
                    "values": np.round(values, 8).tolist(),
                }
            )[:20]
            output.append(
                _Candidate(
                    candidate_id=candidate_id,
                    component_index=component_index,
                    component_name=component.name,
                    local_start=local_start,
                    global_start=global_start,
                    length=length,
                    start_time_ps=float(time_values[0]),
                    end_time_ps=float(time_values[-1]),
                    source_group=group_name,
                    source_target_ps=float(target[source_event]),
                    values=values,
                )
            )
    return output


def _feature_bundle(context: TrainingContext) -> _ShapeletFeatureBundle:
    config = context.model_config
    training = context.config["training"]
    materialization_chunk = int(training.get("shapelet_materialization_chunk_size", 1024))
    feature_chunk = int(training.get("shapelet_feature_chunk_size", 512))
    train_full, train_target = _split_difference_matrix(
        context, "train", chunk_size=materialization_chunk
    )
    validation_full, _ = _split_difference_matrix(
        context, "validation", chunk_size=materialization_chunk
    )
    factor = int(context.subsampling_factor)
    _selected_indices, components = _component_plan(context.datasets[0], factor)
    candidate_key = canonical_hash(
        {
            "train": _difference_cache_key(context, "train"),
            "validation": _difference_cache_key(context, "validation"),
            "factor": factor,
            "lengths": config.get("shapelet_lengths_ns", []),
            "candidates_per_group": int(config.get("candidates_per_group", 64)),
            "extreme_fraction": float(config.get("extreme_fraction", 0.15)),
            "score_events": int(config.get("score_events", 1000)),
            "pool": int(config.get("candidate_pool_size", 20)),
            "redundancy": float(config.get("redundancy_threshold", 0.95)),
            "metric": str(config.get("distance_metric", "dtw")),
            "radius": int(config.get("dtw_radius_points", 2)),
            "local_z": bool(config.get("local_z_normalize", False)),
            "seed": int(training.get("data_seed", training.get("seed", 12345))),
        }
    )
    cached = _lru_get(_FEATURE_CACHE, candidate_key)
    if cached is not None:
        return cached

    train_signals = train_full
    validation_signals = validation_full
    seed = int(training.get("data_seed", training.get("seed", 12345)))
    rng = np.random.default_rng(seed)
    candidates = _generate_candidates(train_signals, train_target, components, config, rng)
    score_count = min(int(config.get("score_events", 1000)), train_signals.shape[0])
    score_indices = np.sort(rng.choice(train_signals.shape[0], size=score_count, replace=False))
    score_signals = train_signals[score_indices]
    score_target = train_target[score_indices]
    metric = str(config.get("distance_metric", "dtw")).lower()
    radius = int(config.get("dtw_radius_points", 2))
    local_z = bool(config.get("local_z_normalize", False))

    candidate_features: list[np.ndarray] = []
    scores: list[float] = []
    for candidate in candidates:
        feature = _distance_feature(
            score_signals,
            candidate,
            metric=metric,
            radius=radius,
            local_z=local_z,
            chunk_size=feature_chunk,
        )
        candidate_features.append(feature)
        scores.append(_safe_corr(feature, score_target))
    matrix = np.column_stack(candidate_features)
    score_array = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-np.abs(score_array))
    selected: list[int] = []
    centered_cache: dict[int, np.ndarray] = {}
    threshold = float(config.get("redundancy_threshold", 0.95))
    pool_size = int(config.get("candidate_pool_size", 20))
    for raw_index in order:
        index = int(raw_index)
        centered = matrix[:, index] - float(np.mean(matrix[:, index]))
        norm = float(np.linalg.norm(centered))
        redundant = False
        if norm > 0.0:
            for chosen in selected:
                other = centered_cache[chosen]
                denominator = norm * float(np.linalg.norm(other))
                correlation = abs(float(np.dot(centered, other) / denominator)) if denominator > 0 else 0.0
                if correlation >= threshold:
                    redundant = True
                    break
        if redundant:
            continue
        selected.append(index)
        centered_cache[index] = centered
        if len(selected) >= pool_size:
            break
    if len(selected) < int(config.get("n_shapelets", 10)):
        raise RuntimeError(
            f"Only {len(selected)} non-redundant candidates remain; "
            f"n_shapelets={int(config.get('n_shapelets', 10))}"
        )
    shapelets = tuple(candidates[index] for index in selected)
    selected_scores = score_array[selected]
    train_features = np.column_stack(
        [
            _distance_feature(
                train_signals,
                candidate,
                metric=metric,
                radius=radius,
                local_z=local_z,
                chunk_size=feature_chunk,
            )
            for candidate in shapelets
        ]
    )
    validation_features = np.column_stack(
        [
            _distance_feature(
                validation_signals,
                candidate,
                metric=metric,
                radius=radius,
                local_z=local_z,
                chunk_size=feature_chunk,
            )
            for candidate in shapelets
        ]
    )
    bundle = _ShapeletFeatureBundle(
        shapelets=shapelets,
        scores=selected_scores,
        train_features=train_features,
        validation_features=validation_features,
    )
    return _lru_put(_FEATURE_CACHE, candidate_key, bundle, _FEATURE_CACHE_LIMIT)


def _residual_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = np.asarray(target, dtype=np.float64) - np.asarray(prediction, dtype=np.float64)
    bias = float(np.mean(residual))
    variance = float(np.mean((residual - bias) ** 2))
    return {
        "bias_ps": bias,
        "variance_ps2": variance,
        "rmse_ps": float(np.sqrt(np.mean(residual**2))),
    }


def _fitted_prediction_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    fit_config: dict[str, Any],
    label: str,
) -> dict[str, float]:
    values = _residual_metrics(target, prediction)
    residual = np.asarray(target, dtype=np.float64) - np.asarray(
        prediction, dtype=np.float64
    )
    fit = fit_times_ps(residual, label, fit_config)
    values["ctr_ps"] = float(fit.ctr_ps) if fit.success else float("nan")
    return values


def _selection_value(
    metrics: dict[str, float], *, loss_type: str, bias_weight: float, bias_scale_ps: float
) -> float:
    if loss_type == "variance":
        return float(metrics["variance_ps2"])
    if loss_type == "rmse":
        return float(metrics["rmse_ps"])
    return float(
        metrics["variance_ps2"]
        + float(bias_weight) * (metrics["bias_ps"] / max(float(bias_scale_ps), 1.0e-12)) ** 2
    )


def _write_shapelet_csv(
    path: Path,
    *,
    shapelets: Sequence[_Candidate],
    scores: np.ndarray,
    coefficients: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    mean_abs_contribution: np.ndarray,
    config: dict[str, Any],
    subsampling_factor: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "candidate_id",
        "component_index",
        "component_name",
        "local_start_index",
        "global_start_index",
        "length_points",
        "start_time_ns",
        "end_time_ns",
        "duration_ns",
        "source_group",
        "source_target_ps",
        "score_correlation",
        "ridge_coefficient",
        "feature_mean",
        "feature_std",
        "mean_abs_contribution_ps",
        "subsampling_factor",
        "distance_metric",
        "dtw_radius_points",
        "values",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for rank, candidate in enumerate(shapelets, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "candidate_id": candidate.candidate_id,
                    "component_index": candidate.component_index,
                    "component_name": candidate.component_name,
                    "local_start_index": candidate.local_start,
                    "global_start_index": candidate.global_start,
                    "length_points": candidate.length,
                    "start_time_ns": candidate.start_time_ps / 1000.0,
                    "end_time_ns": candidate.end_time_ps / 1000.0,
                    "duration_ns": (candidate.end_time_ps - candidate.start_time_ps) / 1000.0,
                    "source_group": candidate.source_group,
                    "source_target_ps": candidate.source_target_ps,
                    "score_correlation": float(scores[rank - 1]),
                    "ridge_coefficient": float(coefficients[rank - 1]),
                    "feature_mean": float(feature_mean[rank - 1]),
                    "feature_std": float(feature_std[rank - 1]),
                    "mean_abs_contribution_ps": float(mean_abs_contribution[rank - 1]),
                    "subsampling_factor": int(subsampling_factor),
                    "distance_metric": str(config.get("distance_metric", "dtw")),
                    "dtw_radius_points": int(config.get("dtw_radius_points", 2)),
                    "values": " ".join(f"{float(value):.8g}" for value in candidate.values),
                }
            )
    temporary.replace(path)


def train(context: TrainingContext) -> dict[str, Any]:
    try:
        from sklearn.linear_model import Ridge
    except ImportError as exc:
        raise RuntimeError("shapelet_regressor requires scikit-learn") from exc

    config = context.config
    model_config = context.model_config
    artifacts = dict(config.get("artifacts", {}))
    device = resolve_device(config["training"].get("device", "auto"))
    train_loader = make_split_loader(
        context.datasets, "train", context.normalization, config, device, shuffle=False,
        subsampling_factor=context.subsampling_factor,
    )
    validation_loader = make_split_loader(
        context.datasets, "validation", context.normalization, config, device, shuffle=False,
        subsampling_factor=context.subsampling_factor,
    )
    zero_model = _ZeroCorrectionModel().to(device)
    baseline_train_metrics, _, _ = evaluate_model(
        zero_model, train_loader, device, config["fit"], "Uncorrected train LED"
    )
    baseline_validation_metrics, _, _ = evaluate_model(
        zero_model, validation_loader, device, config["fit"], "Uncorrected validation LED"
    )

    bundle = _feature_bundle(context)
    count = int(model_config.get("n_shapelets", 10))
    shapelets = bundle.shapelets[:count]
    scores = bundle.scores[:count]
    train_features = bundle.train_features[:, :count]
    validation_features = bundle.validation_features[:, :count]
    _, train_target = _split_difference_matrix(
        context,
        "train",
        chunk_size=int(config["training"].get("shapelet_materialization_chunk_size", 1024)),
    )
    _, validation_target = _split_difference_matrix(
        context,
        "validation",
        chunk_size=int(config["training"].get("shapelet_materialization_chunk_size", 1024)),
    )

    feature_mean = np.mean(train_features, axis=0)
    feature_std = np.std(train_features, axis=0, ddof=0)
    feature_std = np.where(feature_std > 1.0e-12, feature_std, 1.0)
    train_standardized = (train_features - feature_mean) / feature_std
    validation_standardized = (validation_features - feature_mean) / feature_std
    alpha = float(model_config.get("ridge_alpha", 100.0))
    estimator = Ridge(alpha=alpha, fit_intercept=True, solver="auto")
    estimator.fit(train_standardized, train_target)
    coefficient = np.asarray(estimator.coef_, dtype=np.float64).reshape(-1)
    intercept = float(estimator.intercept_)
    # Explicit residual-mean calibration keeps the repository-wide bias convention.
    intercept += float(np.mean(train_target - (train_standardized @ coefficient + intercept)))
    train_prediction = train_standardized @ coefficient + intercept
    validation_prediction = validation_standardized @ coefficient + intercept

    model_config["_serialized_shapelet_count"] = count
    model_config["_serialized_shapelet_width"] = max(candidate.length for candidate in shapelets)
    model = build(model_config, context.input_length).to(device)
    assert isinstance(model, ShapeletPairRegressor)
    with torch.no_grad():
        for index, candidate in enumerate(shapelets):
            model.shapelet_values[index, : candidate.length].copy_(
                torch.from_numpy(candidate.values.astype(np.float32)).to(device)
            )
            model.shapelet_lengths[index] = int(candidate.length)
            model.shapelet_starts[index] = int(candidate.global_start)
        model.feature_mean.copy_(torch.from_numpy(feature_mean.astype(np.float32)).to(device))
        model.feature_std.copy_(torch.from_numpy(feature_std.astype(np.float32)).to(device))
        model.coefficient.copy_(torch.from_numpy(coefficient.astype(np.float32)).to(device))
        model.pair_output_bias_ps.fill_(intercept)

    # Reuse the already materialized shapelet features rather than recomputing
    # every DTW distance through the Torch inference path during training.
    train_metrics = _fitted_prediction_metrics(
        train_target,
        train_prediction,
        config["fit"],
        "Shapelet-regression train residual",
    )
    validation_metrics = _fitted_prediction_metrics(
        validation_target,
        validation_prediction,
        config["fit"],
        "Shapelet-regression validation residual",
    )

    baseline_guard_metric = config["training"].get("baseline_guard_metric")
    baseline_guard_applied = False
    if baseline_guard_metric is not None:
        key = {"validation_rmse": "rmse_ps", "validation_ctr": "ctr_ps"}[
            str(baseline_guard_metric)
        ]
        if float(validation_metrics[key]) > float(baseline_validation_metrics[key]):
            coefficient = np.zeros_like(coefficient)
            intercept = float(np.mean(train_target))
            with torch.no_grad():
                model.coefficient.zero_()
                model.pair_output_bias_ps.fill_(intercept)
            baseline_guard_applied = True
            train_prediction = np.full_like(train_target, intercept)
            validation_prediction = np.full_like(validation_target, intercept)
            train_metrics = _fitted_prediction_metrics(
                train_target,
                train_prediction,
                config["fit"],
                "Selected constant train residual",
            )
            validation_metrics = _fitted_prediction_metrics(
                validation_target,
                validation_prediction,
                config["fit"],
                "Selected constant validation residual",
            )

    loss_type = _selection_loss(model_config)
    loss_config = model_config.get("loss", {})
    bias_weight = float(loss_config.get("bias_weight", 0.0))
    bias_scale = (
        max(float(np.std(train_target, ddof=0)), float(loss_config.get("minimum_scale", 1e-8)))
        if str(loss_config.get("bias_normalization", "none")) == "target_std"
        else 1.0
    )
    validation_residual_metrics = _residual_metrics(validation_target, validation_prediction)
    selection_value = _selection_value(
        validation_residual_metrics,
        loss_type=loss_type,
        bias_weight=bias_weight,
        bias_scale_ps=bias_scale,
    )

    metadata = checkpoint_context(
        context,
        model_config=model_config,
        training_strategy=(
            "training-fold supervised fixed-position shapelet discovery, constrained "
            f"{str(model_config.get('distance_metric', 'dtw')).upper()} distances, and Ridge regression"
        ),
    )
    metadata["shapelet_regressor"] = {
        "n_shapelets": count,
        "candidate_pool_size": int(model_config.get("candidate_pool_size", 20)),
        "subsampling_factor": int(context.subsampling_factor),
        "distance_metric": str(model_config.get("distance_metric", "dtw")),
        "dtw_radius_points": int(model_config.get("dtw_radius_points", 2)),
        "ridge_alpha": alpha,
        "pair_output_bias_ps": intercept,
        "baseline_guard_selected": baseline_guard_applied,
    }
    best_path = context.checkpoint_dir / "best.pt"
    payload = {"model_state": model.state_dict(), "epoch": 0, "context": metadata}
    torch.save(payload, best_path)
    if bool(artifacts.get("save_last_checkpoint", False)):
        torch.save(payload, context.checkpoint_dir / "last.pt")

    contributions = train_standardized * coefficient[None, :]
    mean_abs_contribution = np.mean(np.abs(contributions), axis=0)
    shapelet_csv = context.output_dir / "shapelet_model.csv"
    _write_shapelet_csv(
        shapelet_csv,
        shapelets=shapelets,
        scores=scores,
        coefficients=coefficient,
        feature_mean=feature_mean,
        feature_std=feature_std,
        mean_abs_contribution=mean_abs_contribution,
        config=model_config,
        subsampling_factor=context.subsampling_factor,
    )
    atomic_json(
        context.output_dir / "shapelet_model_summary.json",
        {
            "n_shapelets": count,
            "subsampling_factor": int(context.subsampling_factor),
            "distance_metric": str(model_config.get("distance_metric", "dtw")),
            "dtw_radius_points": int(model_config.get("dtw_radius_points", 2)),
            "ridge_alpha": alpha,
            "pair_output_bias_ps": intercept,
            "shapelet_csv": str(shapelet_csv.resolve()),
        },
    )

    summary = {
        "model_type": context.model_type,
        "model_name": context.model_name,
        "best_epoch": 0,
        "best_selection_metric": loss_type,
        "best_selection_value": float(selection_value),
        "best_validation_rmse_ps": float(validation_metrics["rmse_ps"]),
        "best_validation_ctr_ps": float(validation_metrics["ctr_ps"]),
        "best_validation_bias_ps": float(validation_metrics["bias_ps"]),
        "uncorrected_led_validation_rmse_ps": float(baseline_validation_metrics["rmse_ps"]),
        "uncorrected_led_validation_ctr_ps": float(baseline_validation_metrics["ctr_ps"]),
        "uncorrected_led_validation_bias_ps": float(baseline_validation_metrics["bias_ps"]),
        "best_checkpoint": str(best_path.resolve()),
        "last_checkpoint": "",
        "train_dir": str(context.output_dir.resolve()),
        "input_length": int(context.input_length),
        "input_transform": context.input_transform,
        "subsampling_factor": int(context.subsampling_factor),
        "input_waveform_source": context.input_waveform_source,
        "prediction_target": context.prediction_target,
        "input_cache_paths": [str(path) for path in context.input_cache_dirs],
        "pair_output_bias_ps": intercept,
        "normalization": context.normalization.as_dict(),
        "training_datasets": [str(dataset.directory) for dataset in context.datasets],
        "optimizer": "scikit-learn Ridge",
        "ridge_alpha": alpha,
        "n_shapelets": count,
        "subsampling_factor": int(context.subsampling_factor),
        "distance_metric": str(model_config.get("distance_metric", "dtw")),
        "dtw_radius_points": int(model_config.get("dtw_radius_points", 2)),
        "shapelet_csv": str(shapelet_csv.resolve()),
        "final_train_rmse_ps": float(train_metrics["rmse_ps"]),
        "final_train_bias_ps": float(train_metrics["bias_ps"]),
        "data_view": dict(context.data_view),
        "data_seed": int(config["training"].get("data_seed", config["training"].get("seed", 12345))),
        "model_parameter_count": int(count + 1),
        "stored_shapelet_points": int(sum(candidate.length for candidate in shapelets)),
        "baseline_guard_metric": baseline_guard_metric,
        "baseline_guard_applied": baseline_guard_applied,
    }
    if bool(artifacts.get("save_summary", True)):
        atomic_json(context.output_dir / "training_summary.json", summary)
    # Ephemeral arrays are returned only to the in-process study runner so it
    # can score the fold without recomputing DTW features. They are deliberately
    # excluded from training_summary.json and every durable artifact.
    returned = dict(summary)
    returned["_validation_prediction_ps"] = np.asarray(
        validation_prediction, dtype=np.float64
    )
    returned["_validation_target_ps"] = np.asarray(
        validation_target, dtype=np.float64
    )
    return returned


MODEL_SPEC = ModelSpec(
    name="shapelet_regressor",
    builder=build,
    validator=validate_config,
    training_validator=validate_training_config,
    trainer=train,
    complexity_counter=lambda config, _input_length: int(config.get("n_shapelets", 10)) + 1,
)
