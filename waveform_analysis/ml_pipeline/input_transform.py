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
SUPPORTED_INPUT_TRANSFORMS = {
    INPUT_TRANSFORM_NONE,
    INPUT_TRANSFORM_DIFFERENTIATE,
    INPUT_TRANSFORM_CONCATENATE_DIFF,
}
TRANSFORM_CACHE_FORMAT_VERSION = 2


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


def transformed_input_length(input_length: int, input_transform: str) -> int:
    transform = normalize_input_transform(input_transform)
    length = int(input_length)
    if length <= 0:
        raise ValueError("input_length must be positive")
    if transform == INPUT_TRANSFORM_DIFFERENTIATE:
        if length < 2:
            raise ValueError("Cannot differentiate a waveform with fewer than two samples")
        return length - 1
    if transform == INPUT_TRANSFORM_CONCATENATE_DIFF:
        if length < 2:
            raise ValueError(
                "Cannot concatenate a waveform with its derivative when it has "
                "fewer than two samples"
            )
        return 2 * length - 1
    return length


def transform_relative_time_ps(
    relative_time_ps: np.ndarray,
    input_transform: str,
) -> np.ndarray:
    transform = normalize_input_transform(input_transform)
    values = np.asarray(relative_time_ps, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("relative_time_ps must be one-dimensional")
    if transform == INPUT_TRANSFORM_DIFFERENTIATE:
        if values.size < 2:
            raise ValueError("Cannot differentiate a time grid with fewer than two samples")
        return 0.5 * (values[1:] + values[:-1])
    if transform == INPUT_TRANSFORM_CONCATENATE_DIFF:
        if values.size < 2:
            raise ValueError(
                "Cannot concatenate a time grid with derivative midpoints when it "
                "has fewer than two samples"
            )
        derivative_time = 0.5 * (values[1:] + values[:-1])
        return np.concatenate((values, derivative_time), axis=0)
    return values


def apply_input_transform(windows: np.ndarray, input_transform: str) -> np.ndarray:
    """Apply the model representation transform along the sample axis."""

    transform = normalize_input_transform(input_transform)
    values = np.asarray(windows)
    if values.ndim < 1:
        raise ValueError("Waveform input must have at least one dimension")
    if transform == INPUT_TRANSFORM_DIFFERENTIATE:
        if values.shape[-1] < 2:
            raise ValueError("Cannot differentiate waveforms with fewer than two samples")
        return np.diff(values, axis=-1)
    if transform == INPUT_TRANSFORM_CONCATENATE_DIFF:
        if values.shape[-1] < 2:
            raise ValueError(
                "Cannot concatenate a waveform with its derivative when it has "
                "fewer than two samples"
            )
        derivative = np.diff(values, axis=-1)
        return np.concatenate((values, derivative), axis=-1)
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
    if transform == INPUT_TRANSFORM_NONE:
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
        transformed_input_length(dataset.input_length, transform),
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
            transformed = apply_input_transform(block, transform)
            target[start:stop] = transformed.astype(target.dtype, copy=False)
        target.flush()
        mmap = getattr(target, "_mmap", None)
        if mmap is not None:
            mmap.close()

        transformed_time = transform_relative_time_ps(dataset.relative_time_ps, transform)
        np.save(temporary / "relative_time_ps.npy", transformed_time.astype(np.float32))
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
                },
                "output_shape": list(expected_shape),
                "output_dtype": str(dataset.windows_mV.dtype),
                "sample_axis": 2,
                "expression": (
                    "windows[..., 1:] - windows[..., :-1]"
                    if transform == INPUT_TRANSFORM_DIFFERENTIATE
                    else (
                        "concatenate(windows, windows[..., 1:] - "
                        "windows[..., :-1], axis=-1)"
                    )
                ),
                "component_order": (
                    ["first_difference"]
                    if transform == INPUT_TRANSFORM_DIFFERENTIATE
                    else ["raw_waveform", "first_difference"]
                ),
                "component_lengths": (
                    [int(dataset.input_length) - 1]
                    if transform == INPUT_TRANSFORM_DIFFERENTIATE
                    else [int(dataset.input_length), int(dataset.input_length) - 1]
                ),
                "time_grid": (
                    "adjacent-sample midpoints"
                    if transform == INPUT_TRANSFORM_DIFFERENTIATE
                    else "raw sample times followed by adjacent-sample midpoints"
                ),
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
    manifest["model_input"] = {
        "input_transform": transform,
        "source_fingerprint": dataset.manifest["fingerprint"],
        "cache_path": str(cache_dir),
        "input_length_before_transform": int(dataset.input_length),
        "input_length_after_transform": int(windows.shape[2]),
    }
    transformed_dataset = replace(
        dataset,
        manifest=manifest,
        windows_mV=windows,
        relative_time_ps=relative_time,
    )
    return transformed_dataset, cache_dir
