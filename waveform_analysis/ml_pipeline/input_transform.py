from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .dataset import PreparedDataset

INPUT_TRANSFORM_NONE = "none"

def normalize_subsampling_factor(value: Any) -> int:
    factor = 1 if value is None else int(value)
    if factor <= 0:
        raise ValueError("subsampling_factor must be a positive integer")
    return factor


def normalize_input_transform(value: Any) -> str:
    key = "none" if value is None else str(value).strip().lower()
    if key in {"", "identity"}:
        key = "none"
    if key != "none":
        raise ValueError(
            "Waveform input transforms were removed from the CTR project. "
            "Use the permanent raw/denoised preprocessing variants instead."
        )
    return INPUT_TRANSFORM_NONE


def resolve_input_transform(config: dict[str, Any]) -> str:
    top = config.get("input_transform")
    model = config.get("model") if isinstance(config.get("model"), dict) else {}
    model_value = model.get("input_transform")
    if top is not None and model_value is not None:
        if normalize_input_transform(top) != normalize_input_transform(model_value):
            raise ValueError("Conflicting input_transform settings")
    return normalize_input_transform(top if top is not None else model_value)


def transformed_component_lengths(
    component_lengths: list[int] | tuple[int, ...], input_transform: str
) -> list[int]:
    normalize_input_transform(input_transform)
    lengths = [int(v) for v in component_lengths]
    if not lengths or any(v <= 0 for v in lengths):
        raise ValueError("component_lengths must contain positive integers")
    return lengths


def component_subsampling_indices(
    component_lengths: list[int] | tuple[int, ...], subsampling_factor: int
) -> np.ndarray:
    factor = normalize_subsampling_factor(subsampling_factor)
    lengths = [int(v) for v in component_lengths]
    if not lengths or any(v <= 0 for v in lengths):
        raise ValueError("component_lengths must contain positive integers")
    parts: list[np.ndarray] = []
    cursor = 0
    for length in lengths:
        parts.append(cursor + np.arange(0, length, factor, dtype=np.int64))
        cursor += length
    return np.concatenate(parts)


def subsampled_component_lengths(
    component_lengths: list[int] | tuple[int, ...], subsampling_factor: int
) -> list[int]:
    factor = normalize_subsampling_factor(subsampling_factor)
    return [int((int(length) + factor - 1) // factor) for length in component_lengths]


def subsampled_dataset_input_length(dataset: PreparedDataset, subsampling_factor: int) -> int:
    raw = dataset.manifest.get("input_component_lengths")
    components = [int(v) for v in raw] if isinstance(raw, list) else [int(dataset.input_length)]
    if sum(components) != int(dataset.input_length):
        raise ValueError("Dataset component lengths do not match input length")
    return int(sum(subsampled_component_lengths(components, subsampling_factor)))


def transformed_subsampled_dataset_input_length(
    dataset: PreparedDataset, input_transform: str, subsampling_factor: int
) -> int:
    normalize_input_transform(input_transform)
    return subsampled_dataset_input_length(dataset, subsampling_factor)


def apply_component_subsampling(
    values: np.ndarray,
    subsampling_factor: int,
    component_lengths: list[int] | tuple[int, ...] | None = None,
) -> np.ndarray:
    array = np.asarray(values)
    length = int(array.shape[-1])
    components = [length] if component_lengths is None else [int(v) for v in component_lengths]
    if sum(components) != length:
        raise ValueError("component_lengths must match the last array dimension")
    indices = component_subsampling_indices(components, subsampling_factor)
    return np.take(array, indices, axis=-1)


def transformed_input_length(
    input_length: int,
    input_transform: str,
    component_lengths: list[int] | tuple[int, ...] | None = None,
) -> int:
    normalize_input_transform(input_transform)
    length = int(input_length)
    if length <= 0:
        raise ValueError("input_length must be positive")
    if component_lengths is not None and sum(int(v) for v in component_lengths) != length:
        raise ValueError("component_lengths must sum to input_length")
    return length


def transformed_dataset_input_length(dataset: PreparedDataset, input_transform: str) -> int:
    return transformed_input_length(dataset.input_length, input_transform)


def transform_relative_time_ps(
    relative_time_ps: np.ndarray,
    input_transform: str,
    component_lengths: list[int] | tuple[int, ...] | None = None,
) -> np.ndarray:
    normalize_input_transform(input_transform)
    return np.asarray(relative_time_ps, dtype=np.float64)


def apply_input_transform(
    windows: np.ndarray,
    input_transform: str,
    component_lengths: list[int] | tuple[int, ...] | None = None,
) -> np.ndarray:
    normalize_input_transform(input_transform)
    return np.asarray(windows)


def materialize_training_input_cache(
    dataset: PreparedDataset,
    input_transform: str,
    cache_root: str | Path,
    *,
    chunk_size: int = 2048,
    rebuild: bool = False,
    logger: Any | None = None,
) -> tuple[PreparedDataset, Path | None]:
    del cache_root, chunk_size, rebuild, logger
    normalize_input_transform(input_transform)
    # Raw/denoised choices are already permanently materialized by prepared_data;
    # no per-model representation cache is needed.
    return dataset, None
