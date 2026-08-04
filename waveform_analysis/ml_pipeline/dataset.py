from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
from numpy.lib.format import open_memmap

from .common import atomic_json, canonical_hash, read_json, write_csv_rows
if TYPE_CHECKING:
    from .data import EnergyCache, SplitData

DATASET_FORMAT_VERSION = 2
_SUPPORTED_DATASET_FORMAT_VERSIONS = {1, 2}
_ARRAY_NAMES = (
    "event_id", "event_index", "source_file_id", "source_run_index",
    "bias_voltage_V", "amplitude_mV", "noise_rms_mV", "trigger_index",
    "led_time_fs", "cfd_time_fs", "windows_mV",
)
_OPTIONAL_ARRAY_NAMES = (
    "energy_led_time_fs",
    "timing_led_time_fs",
    "timing_windows_mV",
)


@dataclass(frozen=True)
class PreparedDataset:
    directory: Path
    manifest: dict[str, Any]
    event_id: np.ndarray
    event_index: np.ndarray
    source_file_id: np.ndarray
    source_run_index: np.ndarray
    bias_voltage_V: np.ndarray
    amplitude_mV: np.ndarray
    noise_rms_mV: np.ndarray
    trigger_index: np.ndarray
    led_time_fs: np.ndarray
    cfd_time_fs: np.ndarray
    windows_mV: np.ndarray
    relative_time_ps: np.ndarray
    energy_led_time_fs: np.ndarray | None = None
    timing_led_time_fs: np.ndarray | None = None
    timing_windows_mV: np.ndarray | None = None
    timing_relative_time_ps: np.ndarray | None = None
    train: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    validation: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    test: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    evaluation: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))

    @property
    def input_length(self) -> int:
        return int(self.windows_mV.shape[2])

    @property
    def true_tof_ps(self) -> float:
        return float(self.manifest["true_tof_ps"])


def _load_array(directory: Path, name: str) -> np.ndarray:
    path = directory / f"{name}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Prepared dataset array not found: {path}")
    return np.load(path, mmap_mode="r")


def _load_optional_array(directory: Path, name: str) -> np.ndarray | None:
    path = directory / f"{name}.npy"
    return np.load(path, mmap_mode="r") if path.is_file() else None


def load_prepared_dataset(directory: str | Path) -> PreparedDataset:
    directory = Path(directory).resolve()
    manifest_path = directory / "manifest.json"
    split_path = directory / "splits.npz"
    if not manifest_path.is_file() or not split_path.is_file():
        raise FileNotFoundError(f"Not a prepared ML dataset: {directory}")
    manifest = read_json(manifest_path)
    if int(manifest.get("format_version", -1)) not in _SUPPORTED_DATASET_FORMAT_VERSIONS:
        raise ValueError(f"Unsupported prepared dataset version in {directory}")
    with np.load(split_path, allow_pickle=False) as splits:
        split_values = {
            name: splits[name].astype(np.int64)
            for name in ("train", "validation", "test", "evaluation")
        }
    return PreparedDataset(
        directory=directory,
        manifest=manifest,
        event_id=_load_array(directory, "event_id"),
        event_index=_load_array(directory, "event_index"),
        source_file_id=_load_array(directory, "source_file_id"),
        source_run_index=_load_array(directory, "source_run_index"),
        bias_voltage_V=_load_array(directory, "bias_voltage_V"),
        amplitude_mV=_load_array(directory, "amplitude_mV"),
        noise_rms_mV=_load_array(directory, "noise_rms_mV"),
        trigger_index=_load_array(directory, "trigger_index"),
        led_time_fs=_load_array(directory, "led_time_fs"),
        cfd_time_fs=_load_array(directory, "cfd_time_fs"),
        windows_mV=_load_array(directory, "windows_mV"),
        relative_time_ps=_load_array(directory, "relative_time_ps"),
        energy_led_time_fs=_load_optional_array(directory, "energy_led_time_fs"),
        timing_led_time_fs=_load_optional_array(directory, "timing_led_time_fs"),
        timing_windows_mV=_load_optional_array(directory, "timing_windows_mV"),
        timing_relative_time_ps=_load_optional_array(directory, "timing_relative_time_ps"),
        train=split_values["train"],
        validation=split_values["validation"],
        test=split_values["test"],
        evaluation=split_values["evaluation"],
    )


def load_prepared_dataset_spec(spec: str | Path | dict[str, Any]) -> PreparedDataset:
    """Load a canonical prepared dataset from a path or dataset object."""

    if isinstance(spec, (str, Path)):
        return load_prepared_dataset(spec)
    if not isinstance(spec, dict) or not str(spec.get("dataset", "")).strip():
        raise ValueError(
            "Dataset specification must be a path or an object containing 'dataset'"
        )
    return load_prepared_dataset(spec["dataset"])

def _copy_selected(source: np.ndarray, selected: np.ndarray, path: Path, chunk_size: int) -> None:
    shape = (selected.size,) + tuple(source.shape[1:])
    target = open_memmap(path, mode="w+", dtype=source.dtype, shape=shape)
    for start in range(0, selected.size, chunk_size):
        indices = selected[start : start + chunk_size]
        target[start : start + indices.size] = np.asarray(source[indices])
    target.flush()
    mmap = getattr(target, "_mmap", None)
    if mmap is not None:
        mmap.close()


def _empty_indices() -> np.ndarray:
    return np.empty(0, dtype=np.int64)


def _selected_union(groups: dict[str, np.ndarray]) -> np.ndarray:
    non_empty = [np.asarray(values, dtype=np.int64) for values in groups.values() if values.size]
    if not non_empty:
        raise RuntimeError("No selected events to materialize")
    selected = np.unique(np.concatenate(non_empty)).astype(np.int64, copy=False)
    selected.sort()
    return selected


def _materialize_subset(
    cache: "EnergyCache",
    splits: "SplitData",
    config: dict[str, Any],
    *,
    output: Path,
    name: str,
    role: str,
    source_groups: dict[str, np.ndarray],
    evaluation_source: np.ndarray,
    subset_kind: str,
    linked_dataset: dict[str, str] | None,
    rebuild: bool,
    logger: Any,
) -> PreparedDataset:
    selected_old = _selected_union(source_groups)
    source_indices_hash = hashlib.sha256(
        np.ascontiguousarray(selected_old, dtype=np.int64).tobytes()
    ).hexdigest()
    fingerprint = canonical_hash(
        {
            "format_version": DATASET_FORMAT_VERSION,
            "raw_dataset": cache.manifest["fingerprint"],
            "selection": splits.manifest["fingerprint"],
            "subset_kind": subset_kind,
            "role": role,
            "true_tof_ps": float(config["data"]["true_tof_ps"]),
            "source_event_indices_hash": source_indices_hash,
            "source_event_count": int(selected_old.size),
        }
    )
    if output.is_dir() and not rebuild:
        try:
            dataset = load_prepared_dataset(output)
            if dataset.manifest.get("fingerprint") == fingerprint:
                logger.info("Reusing prepared selected dataset: %s", output)
                return dataset
        except (FileNotFoundError, ValueError):
            pass

    remap = np.full(int(cache.event_id.shape[0]), -1, dtype=np.int64)
    remap[selected_old] = np.arange(selected_old.size, dtype=np.int64)

    def remapped(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.int64)
        if values.size == 0:
            return _empty_indices()
        mapped = remap[values]
        if np.any(mapped < 0):
            raise RuntimeError("A materialized split contains events outside the selected subset")
        return mapped.astype(np.int64, copy=False)

    train = remapped(source_groups.get("train", _empty_indices()))
    validation = remapped(source_groups.get("validation", _empty_indices()))
    test = remapped(source_groups.get("test", _empty_indices()))
    evaluation = remapped(evaluation_source)

    temporary = output.with_name(output.name + ".building")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, exist_ok=True)
    chunk_size = max(1, int(config.get("cache", {}).get("materialization_chunk_size", 2048)))
    for array_name in _ARRAY_NAMES:
        _copy_selected(
            getattr(cache, array_name),
            selected_old,
            temporary / f"{array_name}.npy",
            chunk_size,
        )
    for array_name in _OPTIONAL_ARRAY_NAMES:
        source = getattr(cache, array_name, None)
        if source is not None:
            _copy_selected(
                source, selected_old, temporary / f"{array_name}.npy", chunk_size
            )
    np.save(temporary / "relative_time_ps.npy", np.asarray(cache.relative_time_ps))
    if getattr(cache, "timing_relative_time_ps", None) is not None:
        np.save(
            temporary / "timing_relative_time_ps.npy",
            np.asarray(cache.timing_relative_time_ps),
        )
    with (temporary / "splits.npz.tmp").open("wb") as stream:
        np.savez_compressed(
            stream,
            train=train,
            validation=validation,
            test=test,
            evaluation=evaluation,
        )
    os.replace(temporary / "splits.npz.tmp", temporary / "splits.npz")

    cutflow = [
        {"stage": "raw_events", "events": int(cache.event_id.shape[0])},
        {"stage": "valid_waveform_pairs", "events": int(cache.manifest.get("valid_events", 0))},
        {"stage": "selected_train_in_source_split", "events": int(splits.train.size)},
        {"stage": "selected_validation_in_source_split", "events": int(splits.validation.size)},
        {"stage": "selected_blind_test_in_source_split", "events": int(splits.test.size)},
        {"stage": f"saved_{subset_kind}_events", "events": int(selected_old.size)},
    ]
    write_csv_rows(temporary / "cutflow.csv", cutflow)

    manifest = {
        "format_version": DATASET_FORMAT_VERSION,
        "fingerprint": fingerprint,
        "name": name,
        "role": role,
        "subset_kind": subset_kind,
        "true_tof_ps": float(config["data"]["true_tof_ps"]),
        "source_root": config["data"]["input_root"],
        "preprocess_config_path": config["_config_path"],
        "preprocess_config_hash": config["_config_hash"],
        "event_count": int(selected_old.size),
        "split_counts": {
            "train": int(train.size),
            "validation": int(validation.size),
            "test": int(test.size),
            "evaluation": int(evaluation.size),
        },
        "input_length": int(cache.windows_mV.shape[2]),
        "relative_time_ps_start": float(cache.relative_time_ps[0]),
        "relative_time_ps_stop": float(cache.relative_time_ps[-1]),
        "selection": splits.manifest,
        "raw_cache_manifest": cache.manifest,
        "ml_input_channels_one_based": cache.manifest.get(
            "energy_channels_one_based", []
        ),
        "led_timestamp_source": cache.manifest.get(
            "led_timestamp_source", "energy_channels"
        ),
        "cfd_timestamp_source": cache.manifest.get(
            "cfd_timestamp_source", "energy_channels"
        ),
        "ml_window_alignment_source": cache.manifest.get(
            "ml_window_alignment_source", "energy_channel_led"
        ),
        "timing_channel_waveforms_saved": bool(
            getattr(cache, "timing_windows_mV", None) is not None
        ),
        "available_waveform_sources": [
            "energy",
            *(
                ["timing"]
                if getattr(cache, "timing_windows_mV", None) is not None
                else []
            ),
        ],
        "available_prediction_targets": [
            "prepared_led",
            *(
                ["energy_led"]
                if getattr(cache, "energy_led_time_fs", None) is not None
                else []
            ),
            *(
                ["timing_led"]
                if getattr(cache, "timing_led_time_fs", None) is not None
                else []
            ),
        ],
        "same_event_set_for_led_cfd_and_ml": True,
        "waveform_grid": cache.manifest.get(
            "waveform_grid", "legacy_materialized_interpolation"
        ),
        "native_sample_interval_ps": cache.manifest.get(
            "native_sample_interval_ps"
        ),
        "ml_window_alignment_quantization": cache.manifest.get(
            "ml_window_alignment_quantization"
        ),
        "timing_crossing_interpolation": cache.manifest.get(
            "timing_crossing_interpolation"
        ),
        "waveform_representation": "standard",
        "is_canonical_prepared_dataset": True,
        "model_input_transform_applied": False,
        "arrays_are_post_selection": True,
        "split_frozen_before_selection_fitting": True,
        "photopeak_and_led_outlier_parameters_fit_on_training_candidate_only": True,
        "blind_test_excluded_from_training_and_validation": role == "blind",
        "linked_dataset": linked_dataset,
    }
    atomic_json(temporary / "manifest.json", manifest)
    if output.exists():
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, output)
    logger.info("Prepared %s dataset written to %s (%d events)", subset_kind, output, selected_old.size)
    return load_prepared_dataset(output)


def materialize_training_and_blind_datasets(
    cache: "EnergyCache",
    splits: "SplitData",
    config: dict[str, Any],
    *,
    rebuild: bool,
    logger: Any,
) -> tuple[PreparedDataset, PreparedDataset]:
    """Save one training/validation dataset and one physically separate blind holdout.

    The split is already frozen in ``prepare_splits``. Selection parameters are fit
    from the training candidate only and then applied unchanged to validation and
    test. This function only separates the selected arrays on disk, preventing the
    blind events from being visible to model training code.
    """

    dataset_config = config["dataset"]
    blind_config = dataset_config["blind_test"]
    training_output = Path(dataset_config["output_dir"])
    blind_output = Path(blind_config["output_dir"])
    training_name = str(dataset_config["name"])
    blind_name = str(blind_config["name"])

    training_link = {"name": blind_name, "path": str(blind_output), "relation": "held_out_blind_test"}
    blind_link = {"name": training_name, "path": str(training_output), "relation": "source_training_dataset"}

    training_dataset = _materialize_subset(
        cache,
        splits,
        config,
        output=training_output,
        name=training_name,
        role="training",
        source_groups={
            "train": splits.train,
            "validation": splits.validation,
            "test": _empty_indices(),
        },
        evaluation_source=_empty_indices(),
        subset_kind="training_validation",
        linked_dataset=training_link,
        rebuild=rebuild,
        logger=logger,
    )
    blind_dataset = _materialize_subset(
        cache,
        splits,
        config,
        output=blind_output,
        name=blind_name,
        role="blind",
        source_groups={
            "train": _empty_indices(),
            "validation": _empty_indices(),
            "test": splits.test,
        },
        evaluation_source=splits.test,
        subset_kind="blind_test",
        linked_dataset=blind_link,
        rebuild=rebuild,
        logger=logger,
    )
    return training_dataset, blind_dataset


def materialize_prepared_dataset(
    cache: "EnergyCache",
    splits: "SplitData",
    config: dict[str, Any],
    *,
    rebuild: bool,
    logger: Any,
) -> PreparedDataset:
    """Legacy single-directory materialization.

    New single-source experiments should configure ``dataset.blind_test`` and use
    :func:`materialize_training_and_blind_datasets` instead.
    """

    role = str(config["dataset"].get("role", "training"))
    evaluation_source = (
        np.concatenate([splits.train, splits.validation, splits.test]).astype(np.int64)
        if role == "blind"
        else splits.test
    )
    return _materialize_subset(
        cache,
        splits,
        config,
        output=Path(config["dataset"]["output_dir"]),
        name=str(config["dataset"]["name"]),
        role=role,
        source_groups={
            "train": splits.train,
            "validation": splits.validation,
            "test": splits.test,
        },
        evaluation_source=evaluation_source,
        subset_kind="combined_selected",
        linked_dataset=None,
        rebuild=rebuild,
        logger=logger,
    )


def window_slice_indices(
    dataset: PreparedDataset,
    before_ns: float,
    after_ns: float,
) -> tuple[int, int]:
    """Return the contiguous sample slice for ``[-before_ns, after_ns]``."""
    if before_ns < 0 or after_ns < 0:
        raise ValueError("Window bounds must be non-negative")
    times_ns = np.asarray(dataset.relative_time_ps, dtype=np.float64) / 1000.0
    selected = np.flatnonzero(
        (times_ns >= -float(before_ns) - 1e-9)
        & (times_ns <= float(after_ns) + 1e-9)
    )
    if selected.size == 0:
        raise ValueError(
            f"Requested window [-{before_ns}, {after_ns}] ns contains no samples"
        )
    start = int(selected[0])
    stop = int(selected[-1]) + 1
    if not np.array_equal(selected, np.arange(start, stop)):
        raise ValueError("Requested time window does not map to a contiguous sample slice")
    return start, stop


def prepared_dataset_view(
    dataset: PreparedDataset,
    *,
    train_indices: np.ndarray | None = None,
    validation_indices: np.ndarray | None = None,
    window_start: int = 0,
    window_stop: int | None = None,
) -> PreparedDataset:
    """Create a zero-copy experiment view over one prepared dataset.

    Fold membership and waveform windows are represented only by index arrays and
    NumPy/memmap slices; no derived dataset directory is materialized.
    """
    from dataclasses import replace

    stop = dataset.input_length if window_stop is None else int(window_stop)
    start = int(window_start)
    if not 0 <= start < stop <= dataset.input_length:
        raise ValueError(
            f"Invalid waveform slice [{start}:{stop}] for length {dataset.input_length}"
        )
    train = dataset.train if train_indices is None else np.asarray(train_indices, dtype=np.int64)
    validation = (
        dataset.validation
        if validation_indices is None
        else np.asarray(validation_indices, dtype=np.int64)
    )
    manifest = dict(dataset.manifest)
    relative = dataset.relative_time_ps[start:stop]
    manifest["input_length"] = int(stop - start)
    manifest["relative_time_ps_start"] = float(relative[0])
    manifest["relative_time_ps_stop"] = float(relative[-1])
    manifest["view"] = {
        "source_fingerprint": dataset.manifest["fingerprint"],
        "window_start_index": start,
        "window_stop_index": stop,
    }
    manifest["fingerprint"] = canonical_hash(
        {
            "source": dataset.manifest["fingerprint"],
            "window_start_index": start,
            "window_stop_index": stop,
            "prediction_view": dataset.manifest.get("prediction_view", {}),
        }
    )
    return replace(
        dataset,
        manifest=manifest,
        windows_mV=dataset.windows_mV[:, :, start:stop],
        relative_time_ps=relative,
        train=train,
        validation=validation,
        test=dataset.test,
        evaluation=dataset.evaluation,
    )
