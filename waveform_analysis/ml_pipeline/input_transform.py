from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.format import open_memmap

from .common import atomic_json, canonical_hash, read_json
from .dataset import PreparedDataset

INPUT_TRANSFORM_NONE = "none"
INPUT_TRANSFORM_DIFFERENTIATE = "differentiate"
INPUT_TRANSFORM_CONCATENATE_DIFF = "concatenate_diff"
INPUT_TRANSFORM_NORMALIZE = "normalize"
SUPPORTED_INPUT_TRANSFORMS = {
    INPUT_TRANSFORM_NONE,
    INPUT_TRANSFORM_DIFFERENTIATE,
    INPUT_TRANSFORM_CONCATENATE_DIFF,
    INPUT_TRANSFORM_NORMALIZE,
}
TRANSFORM_CACHE_FORMAT_VERSION = 3




def normalize_subsampling_factor(value: Any) -> int:
    """Return a validated integer preprocessing subsampling factor."""

    factor = 1 if value is None else int(value)
    if factor <= 0:
        raise ValueError("subsampling_factor must be a positive integer")
    return factor


def transformed_component_lengths(
    component_lengths: list[int] | tuple[int, ...],
    input_transform: str,
) -> list[int]:
    """Return component lengths after the representation transform."""

    transform = normalize_input_transform(input_transform)
    components = [int(value) for value in component_lengths]
    if not components or any(value <= 0 for value in components):
        raise ValueError("component_lengths must contain positive integers")
    if transform == INPUT_TRANSFORM_DIFFERENTIATE:
        if any(value < 2 for value in components):
            raise ValueError("Cannot differentiate a component with fewer than two samples")
        return [value - 1 for value in components]
    if transform == INPUT_TRANSFORM_CONCATENATE_DIFF:
        if any(value < 2 for value in components):
            raise ValueError(
                "Cannot concatenate a component with its derivative when it has fewer than two samples"
            )
        output: list[int] = []
        for value in components:
            output.extend((value, value - 1))
        return output
    return components


def component_subsampling_indices(
    component_lengths: list[int] | tuple[int, ...],
    subsampling_factor: int,
) -> np.ndarray:
    """Indices that keep one point every ``factor`` inside each component.

    Restarting the stride at every component prevents concatenated energy/timing
    or raw/difference representations from acquiring an arbitrary phase shift.
    """

    factor = normalize_subsampling_factor(subsampling_factor)
    lengths = [int(value) for value in component_lengths]
    if not lengths or any(value <= 0 for value in lengths):
        raise ValueError("component_lengths must contain positive integers")
    parts: list[np.ndarray] = []
    cursor = 0
    for length in lengths:
        parts.append(cursor + np.arange(0, length, factor, dtype=np.int64))
        cursor += length
    return np.concatenate(parts)


def subsampled_component_lengths(
    component_lengths: list[int] | tuple[int, ...],
    subsampling_factor: int,
) -> list[int]:
    factor = normalize_subsampling_factor(subsampling_factor)
    return [int((int(length) + factor - 1) // factor) for length in component_lengths]


def subsampled_dataset_input_length(
    dataset: PreparedDataset, subsampling_factor: int
) -> int:
    raw = dataset.manifest.get("input_component_lengths")
    components = (
        [int(value) for value in raw]
        if isinstance(raw, list)
        else [int(dataset.input_length)]
    )
    if sum(components) != int(dataset.input_length):
        raise ValueError("Dataset component lengths do not match input length")
    return int(sum(subsampled_component_lengths(components, subsampling_factor)))


def transformed_subsampled_dataset_input_length(
    dataset: PreparedDataset, input_transform: str, subsampling_factor: int
) -> int:
    raw = dataset.manifest.get("input_component_lengths")
    source = [int(value) for value in raw] if isinstance(raw, list) else [int(dataset.input_length)]
    components = transformed_component_lengths(source, input_transform)
    return int(sum(subsampled_component_lengths(components, subsampling_factor)))


def apply_component_subsampling(
    values: np.ndarray,
    subsampling_factor: int,
    component_lengths: list[int] | tuple[int, ...] | None = None,
) -> np.ndarray:
    """Apply stride subsampling along the last axis without crossing components."""

    array = np.asarray(values)
    length = int(array.shape[-1])
    components = [length] if component_lengths is None else [int(value) for value in component_lengths]
    if sum(components) != length:
        raise ValueError("component_lengths must match the last array dimension")
    indices = component_subsampling_indices(components, subsampling_factor)
    return np.take(array, indices, axis=-1)


def normalize_input_transform(value: Any) -> str:
    """Return the canonical model-input transform name.

    Older configs did not contain this option; absence therefore means ``none``.
    Migration aliases are normalized before metadata is written. The canonical
    ``concatenate_diff`` representation preserves the raw samples first and then
    appends the same first differences produced by ``differentiate``.
    """

    if value is None:
        return INPUT_TRANSFORM_NONE
    key = str(value).strip().lower()
    aliases = {
        "": INPUT_TRANSFORM_NONE,
        "identity": INPUT_TRANSFORM_NONE,
        "difference": INPUT_TRANSFORM_DIFFERENTIATE,
        "first_difference": INPUT_TRANSFORM_DIFFERENTIATE,
        "differentiate_first": INPUT_TRANSFORM_DIFFERENTIATE,
        "concat_diff": INPUT_TRANSFORM_CONCATENATE_DIFF,
        "raw_plus_diff": INPUT_TRANSFORM_CONCATENATE_DIFF,
        "zscore": INPUT_TRANSFORM_NORMALIZE,
        "z_score": INPUT_TRANSFORM_NORMALIZE,
        "standardize": INPUT_TRANSFORM_NORMALIZE,
        "standardise": INPUT_TRANSFORM_NORMALIZE,
    }
    key = aliases.get(key, key)
    if key not in SUPPORTED_INPUT_TRANSFORMS:
        raise ValueError(
            f"Unsupported input_transform {value!r}; expected one of "
            f"{sorted(SUPPORTED_INPUT_TRANSFORMS)}"
        )
    return key


def resolve_input_transform(config: dict[str, Any]) -> str:
    """Resolve input transform from a training config.

    The preferred location is the top-level ``input_transform`` key. For
    compatibility with configs that conceptually attach the option to the model,
    ``model.input_transform`` is also accepted. Supplying both with different
    values is an error.
    """

    top_level = config.get("input_transform")
    model = config.get("model")
    model_level = model.get("input_transform") if isinstance(model, dict) else None
    if top_level is not None and model_level is not None:
        resolved_top = normalize_input_transform(top_level)
        resolved_model = normalize_input_transform(model_level)
        if resolved_top != resolved_model:
            raise ValueError(
                "Conflicting input transforms: input_transform="
                f"{resolved_top!r}, model.input_transform={resolved_model!r}"
            )
        return resolved_top
    return normalize_input_transform(top_level if top_level is not None else model_level)


def transformed_input_length(
    input_length: int,
    input_transform: str,
    component_lengths: list[int] | tuple[int, ...] | None = None,
) -> int:
    transform = normalize_input_transform(input_transform)
    length = int(input_length)
    if length <= 0:
        raise ValueError("input_length must be positive")
    components = [length] if component_lengths is None else [int(v) for v in component_lengths]
    if not components or any(v <= 0 for v in components) or sum(components) != length:
        raise ValueError("component_lengths must be positive and sum to input_length")
    if transform == INPUT_TRANSFORM_DIFFERENTIATE:
        if any(v < 2 for v in components):
            raise ValueError("Cannot differentiate a component with fewer than two samples")
        return int(sum(v - 1 for v in components))
    if transform == INPUT_TRANSFORM_CONCATENATE_DIFF:
        if any(v < 2 for v in components):
            raise ValueError(
                "Cannot concatenate a component with its derivative when it has fewer than two samples"
            )
        return int(sum(2 * v - 1 for v in components))
    return length


def transformed_dataset_input_length(dataset: PreparedDataset, input_transform: str) -> int:
    raw = dataset.manifest.get("input_component_lengths")
    components = list(raw) if isinstance(raw, list) else None
    return transformed_input_length(dataset.input_length, input_transform, components)


def transform_relative_time_ps(
    relative_time_ps: np.ndarray,
    input_transform: str,
    component_lengths: list[int] | tuple[int, ...] | None = None,
) -> np.ndarray:
    transform = normalize_input_transform(input_transform)
    values = np.asarray(relative_time_ps, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("relative_time_ps must be one-dimensional")
    components = [values.size] if component_lengths is None else [int(v) for v in component_lengths]
    if sum(components) != values.size or any(v <= 0 for v in components):
        raise ValueError("component_lengths must be positive and match the time grid")
    boundaries = np.cumsum([0, *components])
    pieces = [values[boundaries[i]:boundaries[i + 1]] for i in range(len(components))]
    if transform == INPUT_TRANSFORM_DIFFERENTIATE:
        return np.concatenate([0.5 * (piece[1:] + piece[:-1]) for piece in pieces])
    if transform == INPUT_TRANSFORM_CONCATENATE_DIFF:
        output: list[np.ndarray] = []
        for piece in pieces:
            output.extend((piece, 0.5 * (piece[1:] + piece[:-1])))
        return np.concatenate(output)
    return values


def apply_input_transform(
    windows: np.ndarray,
    input_transform: str,
    component_lengths: list[int] | tuple[int, ...] | None = None,
) -> np.ndarray:
    """Apply the representation transform independently to each modality.

    Single-modality ordering remains exactly ``raw`` then ``differentiate`` for
    ``concatenate_diff``.  Combined energy/timing inputs use
    ``energy raw, energy diff, timing raw, timing diff`` and never create a
    derivative across the modality boundary.
    """

    transform = normalize_input_transform(input_transform)
    values = np.asarray(windows)
    if values.ndim < 1:
        raise ValueError("Waveform input must have at least one dimension")
    length = int(values.shape[-1])
    components = [length] if component_lengths is None else [int(v) for v in component_lengths]
    if not components or any(v <= 0 for v in components) or sum(components) != length:
        raise ValueError("component_lengths must be positive and match waveform length")
    boundaries = np.cumsum([0, *components])
    pieces = [values[..., boundaries[i]:boundaries[i + 1]] for i in range(len(components))]
    if transform == INPUT_TRANSFORM_DIFFERENTIATE:
        if any(piece.shape[-1] < 2 for piece in pieces):
            raise ValueError("Cannot differentiate a component with fewer than two samples")
        return np.concatenate([np.diff(piece, axis=-1) for piece in pieces], axis=-1)
    if transform == INPUT_TRANSFORM_CONCATENATE_DIFF:
        if any(piece.shape[-1] < 2 for piece in pieces):
            raise ValueError(
                "Cannot concatenate a component with its derivative when it has fewer than two samples"
            )
        output: list[np.ndarray] = []
        for piece in pieces:
            output.extend((piece, np.diff(piece, axis=-1)))
        return np.concatenate(output, axis=-1)
    return values


def _safe_component(value: str) -> str:
    text = "".join(character if character.isalnum() or character in "-_" else "_" for character in value)
    return text.strip("_") or "dataset"


def _cache_identity(dataset: PreparedDataset, input_transform: str) -> str:
    relative_time = np.ascontiguousarray(
        np.asarray(dataset.relative_time_ps, dtype=np.float64)
    )
    time_grid_hash = hashlib.sha256(relative_time.tobytes()).hexdigest()
    return canonical_hash(
        {
            "format_version": TRANSFORM_CACHE_FORMAT_VERSION,
            "source_fingerprint": dataset.manifest["fingerprint"],
            "source_shape": list(map(int, dataset.windows_mV.shape)),
            "source_time_grid_sha256": time_grid_hash,
            "input_transform": normalize_input_transform(input_transform),
            "prediction_view": dataset.manifest.get("prediction_view", {}),
            "input_component_lengths": dataset.manifest.get("input_component_lengths"),
            "output_dtype": str(dataset.windows_mV.dtype),
        }
    )


def materialize_training_input_cache(
    dataset: PreparedDataset,
    input_transform: str,
    cache_root: str | Path,
    *,
    chunk_size: int = 2048,
    rebuild: bool = False,
    logger: Any | None = None,
) -> tuple[PreparedDataset, Path | None]:
    """Create/reuse a model-input cache without creating another prepared dataset.

    Only transformed waveform samples, their transformed time grid, and a small
    transform manifest are written. Event metadata and split arrays continue to
    come from the canonical prepared dataset, which remains the single source of
    truth and the only object loadable through :func:`load_prepared_dataset`.
    """

    transform = normalize_input_transform(input_transform)
    # ``normalize`` is fitted fold-by-fold from training events, so it must not
    # be materialized from the complete prepared dataset. The raw dataset is
    # returned and the learned feature statistics are applied by CorrectionDataset.
    if transform in {INPUT_TRANSFORM_NONE, INPUT_TRANSFORM_NORMALIZE}:
        return dataset, None
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive")

    cache_root = Path(cache_root).resolve()
    identity = _cache_identity(dataset, transform)
    source_name = str(dataset.manifest.get("name", dataset.directory.name))
    cache_dir = cache_root / f"{_safe_component(source_name)}_{identity[:12]}"
    manifest_path = cache_dir / "transform_manifest.json"
    windows_path = cache_dir / "windows_mV.npy"
    time_path = cache_dir / "relative_time_ps.npy"

    expected_shape = (
        int(dataset.windows_mV.shape[0]),
        int(dataset.windows_mV.shape[1]),
        transformed_dataset_input_length(dataset, transform),
    )

    reusable = False
    if not rebuild and manifest_path.is_file() and windows_path.is_file() and time_path.is_file():
        try:
            manifest = read_json(manifest_path)
            reusable = (
                manifest.get("fingerprint") == identity
                and tuple(manifest.get("output_shape", [])) == expected_shape
            )
        except (OSError, ValueError, TypeError):
            reusable = False

    if not reusable:
        temporary = cache_dir.with_name(cache_dir.name + ".building")
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True, exist_ok=True)
        target = open_memmap(
            temporary / "windows_mV.npy",
            mode="w+",
            dtype=dataset.windows_mV.dtype,
            shape=expected_shape,
        )
        event_count = int(dataset.windows_mV.shape[0])
        for start in range(0, event_count, int(chunk_size)):
            stop = min(start + int(chunk_size), event_count)
            block = np.asarray(dataset.windows_mV[start:stop])
            transformed = apply_input_transform(
                block, transform, dataset.manifest.get("input_component_lengths")
            )
            target[start:stop] = transformed.astype(target.dtype, copy=False)
        target.flush()
        mmap = getattr(target, "_mmap", None)
        if mmap is not None:
            mmap.close()

        transformed_time = transform_relative_time_ps(
            dataset.relative_time_ps,
            transform,
            dataset.manifest.get("input_component_lengths"),
        )
        np.save(temporary / "relative_time_ps.npy", transformed_time.astype(np.float32))
        source_components = dataset.manifest.get("input_components", ["waveform"])
        source_lengths = dataset.manifest.get(
            "input_component_lengths", [int(dataset.input_length)]
        )
        transformed_components: list[str] = []
        transformed_lengths: list[int] = []
        for name, length in zip(source_components, source_lengths):
            length = int(length)
            if transform == INPUT_TRANSFORM_DIFFERENTIATE:
                transformed_components.append(
                    "first_difference" if name == "waveform" else f"{name}_first_difference"
                )
                transformed_lengths.append(length - 1)
            else:
                if name == "waveform":
                    transformed_components.extend(("raw_waveform", "first_difference"))
                else:
                    transformed_components.extend((f"{name}_raw", f"{name}_first_difference"))
                transformed_lengths.extend((length, length - 1))
        atomic_json(
            temporary / "transform_manifest.json",
            {
                "format_version": TRANSFORM_CACHE_FORMAT_VERSION,
                "fingerprint": identity,
                "input_transform": transform,
                "source_dataset": {
                    "path": str(dataset.directory),
                    "fingerprint": dataset.manifest["fingerprint"],
                    "input_length": int(dataset.input_length),
                    "prediction_view": dataset.manifest.get("prediction_view", {}),
                    "input_components": source_components,
                    "input_component_lengths": source_lengths,
                },
                "output_shape": list(expected_shape),
                "output_dtype": str(dataset.windows_mV.dtype),
                "sample_axis": 2,
                "component_order": transformed_components,
                "component_lengths": transformed_lengths,
                "time_grid": "component-wise native samples and/or adjacent-sample midpoints",
                "is_prepared_dataset": False,
            },
        )
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        os.replace(temporary, cache_dir)
        if logger is not None:
            logger.info(
                "Materialized model-input cache | transform=%s | source=%s | cache=%s",
                transform,
                dataset.directory,
                cache_dir,
            )
    elif logger is not None:
        logger.info(
            "Reusing model-input cache | transform=%s | source=%s | cache=%s",
            transform,
            dataset.directory,
            cache_dir,
        )

    windows = np.load(windows_path, mmap_mode="r")
    relative_time = np.load(time_path, mmap_mode="r")
    manifest = dict(dataset.manifest)
    transform_manifest = read_json(manifest_path)
    manifest["model_input"] = {
        "input_transform": transform,
        "source_fingerprint": dataset.manifest["fingerprint"],
        "cache_path": str(cache_dir),
        "input_length_before_transform": int(dataset.input_length),
        "input_length_after_transform": int(windows.shape[2]),
    }
    manifest["input_components"] = transform_manifest.get("component_order", [])
    manifest["input_component_lengths"] = transform_manifest.get(
        "component_lengths", [int(windows.shape[2])]
    )
    transformed_dataset = replace(
        dataset,
        manifest=manifest,
        windows_mV=windows,
        relative_time_ps=relative_time,
    )
    return transformed_dataset, cache_dir
