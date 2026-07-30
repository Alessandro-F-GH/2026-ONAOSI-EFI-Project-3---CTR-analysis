from __future__ import annotations

import os
import shutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import awkward as ak
import numpy as np
from numpy.lib.format import open_memmap

from utils.photopeak import fit_photopeak, photopeak_mask
from utils.signal import INVALID_TIME_FS

from .common import atomic_json, canonical_hash, read_json, source_signature
from .energy_io import energy_event_count, iterate_energy_chunks
from .signal import extract_channel, extract_timing_reference, relative_window_grid_ps
from .splitting import contiguous_block_split

CACHE_FORMAT_VERSION = 3
SPLIT_FORMAT_VERSION = 4


@dataclass(frozen=True)
class EnergyCache:
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
    valid: np.ndarray
    relative_time_ps: np.ndarray


@dataclass(frozen=True)
class SplitData:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    manifest: dict[str, Any]


def _preprocessing_relevant(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "channels": config["channels"],
        "waveform": config["waveform"],
        "io": {
            "max_events": int(config.get("io", {}).get("max_events", 0)),
        },
    }


def dataset_fingerprint(input_path: Path, config: dict[str, Any]) -> str:
    return canonical_hash(
        {
            "format_version": CACHE_FORMAT_VERSION,
            "source": source_signature(input_path),
            "preprocessing": _preprocessing_relevant(config),
        }
    )


def _array_paths(directory: Path) -> dict[str, Path]:
    names = (
        "event_id",
        "event_index",
        "source_file_id",
        "source_run_index",
        "bias_voltage_V",
        "amplitude_mV",
        "noise_rms_mV",
        "trigger_index",
        "led_time_fs",
        "cfd_time_fs",
        "windows_mV",
        "valid",
        "relative_time_ps",
    )
    return {name: directory / f"{name}.npy" for name in names}


def load_energy_cache(directory: Path, input_path: Path, config: dict[str, Any]) -> EnergyCache:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Energy cache manifest not found: {manifest_path}")
    manifest = read_json(manifest_path)
    expected = dataset_fingerprint(input_path, config)
    if manifest.get("fingerprint") != expected:
        raise ValueError(
            "Energy cache fingerprint differs from input/preprocessing configuration; "
            "rebuild the cache"
        )
    paths = _array_paths(directory)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError("Energy cache is incomplete: " + ", ".join(missing))
    return EnergyCache(
        directory=directory,
        manifest=manifest,
        event_id=np.load(paths["event_id"], mmap_mode="r"),
        event_index=np.load(paths["event_index"], mmap_mode="r"),
        source_file_id=np.load(paths["source_file_id"], mmap_mode="r"),
        source_run_index=np.load(paths["source_run_index"], mmap_mode="r"),
        bias_voltage_V=np.load(paths["bias_voltage_V"], mmap_mode="r"),
        amplitude_mV=np.load(paths["amplitude_mV"], mmap_mode="r"),
        noise_rms_mV=np.load(paths["noise_rms_mV"], mmap_mode="r"),
        trigger_index=np.load(paths["trigger_index"], mmap_mode="r"),
        led_time_fs=np.load(paths["led_time_fs"], mmap_mode="r"),
        cfd_time_fs=np.load(paths["cfd_time_fs"], mmap_mode="r"),
        windows_mV=np.load(paths["windows_mV"], mmap_mode="r"),
        valid=np.load(paths["valid"], mmap_mode="r"),
        relative_time_ps=np.load(paths["relative_time_ps"], mmap_mode="r"),
    )


def _process_event(payload: tuple[Any, ...]) -> tuple[Any, ...]:
    (
        event_index,
        event_id,
        source_file_id,
        source_run_index,
        bias_voltage_V,
        energy_raw_a,
        energy_raw_b,
        energy_gains,
        energy_offsets,
        energy_intervals,
        energy_horizontal_offsets,
        energy_polarities,
        use_timing_channel_led,
        timing_raw_a,
        timing_raw_b,
        timing_gains,
        timing_offsets,
        timing_intervals,
        timing_horizontal_offsets,
        timing_polarities,
        waveform_config,
        relative_grid_ps,
    ) = payload

    timing_references = [None, None]
    if use_timing_channel_led:
        timing_references = []
        for channel_position, raw in enumerate((timing_raw_a, timing_raw_b)):
            timing_references.append(
                extract_timing_reference(
                    np.asarray(raw, dtype=np.int16),
                    vertical_gain_v_per_count=float(timing_gains[channel_position]),
                    vertical_offset_v=float(timing_offsets[channel_position]),
                    horizontal_interval_s=float(timing_intervals[channel_position]),
                    horizontal_offset_s=float(timing_horizontal_offsets[channel_position]),
                    polarity=int(timing_polarities[channel_position]),
                    waveform_config=waveform_config,
                )
            )

    outputs = []
    for channel_position, raw in enumerate((energy_raw_a, energy_raw_b)):
        outputs.append(
            extract_channel(
                np.asarray(raw, dtype=np.int16),
                vertical_gain_v_per_count=float(energy_gains[channel_position]),
                vertical_offset_v=float(energy_offsets[channel_position]),
                horizontal_interval_s=float(energy_intervals[channel_position]),
                horizontal_offset_s=float(energy_horizontal_offsets[channel_position]),
                polarity=int(energy_polarities[channel_position]),
                waveform_config=waveform_config,
                relative_grid_ps=relative_grid_ps,
                timing_reference=timing_references[channel_position],
            )
        )
    return (
        int(event_index),
        int(event_id),
        np.asarray(source_file_id, dtype=np.int64),
        int(source_run_index),
        float(bias_voltage_V),
        np.asarray([item.amplitude_mV for item in outputs], dtype=np.float32),
        np.asarray([item.noise_rms_mV for item in outputs], dtype=np.float32),
        np.asarray([item.trigger_index for item in outputs], dtype=np.int32),
        np.asarray([item.led_time_fs for item in outputs], dtype=np.int64),
        np.asarray([item.cfd_time_fs for item in outputs], dtype=np.int64),
        np.stack([item.window_mV for item in outputs]).astype(np.float32),
        bool(all(item.valid for item in outputs)),
    )


def _executor_map(
    payloads: list[tuple[Any, ...]], parallel: dict[str, Any]
) -> Iterable[tuple[Any, ...]]:
    workers = int(parallel.get("preprocessing_workers", 0))
    backend = str(parallel.get("preprocessing_backend", "process"))
    chunksize = max(1, int(parallel.get("preprocessing_chunksize", 8)))
    if workers <= 0 or backend == "serial":
        return map(_process_event, payloads)
    executor_class = ProcessPoolExecutor if backend == "process" else ThreadPoolExecutor
    executor = executor_class(max_workers=workers)
    # The generator owns the executor and shuts it down once consumed.
    def generate() -> Iterable[tuple[Any, ...]]:
        try:
            yield from executor.map(_process_event, payloads, chunksize=chunksize)
        finally:
            executor.shutdown(wait=True, cancel_futures=False)
    return generate()


def prepare_energy_cache(
    input_path: Path,
    directory: Path,
    config: dict[str, Any],
    *,
    rebuild: bool,
    logger: Any,
) -> EnergyCache:
    input_path = input_path.resolve()
    expected_fingerprint = dataset_fingerprint(input_path, config)
    if directory.is_dir() and not rebuild:
        try:
            cache = load_energy_cache(directory, input_path, config)
            logger.info("Reusing waveform preprocessing cache: %s", directory)
            return cache
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("Cannot reuse preprocessing cache: %s", exc)

    total_root = energy_event_count(input_path)
    max_events = int(config.get("io", {}).get("max_events", 0))
    n_events = min(total_root, max_events) if max_events > 0 else total_root
    if n_events <= 0:
        raise RuntimeError("Input ROOT file contains no events")
    relative_grid = relative_window_grid_ps(config["waveform"])

    temporary = directory.with_name(directory.name + ".building")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, exist_ok=True)
    paths = _array_paths(temporary)
    arrays = {
        "event_id": open_memmap(paths["event_id"], mode="w+", dtype=np.int64, shape=(n_events,)),
        "event_index": open_memmap(paths["event_index"], mode="w+", dtype=np.int64, shape=(n_events,)),
        "source_file_id": open_memmap(paths["source_file_id"], mode="w+", dtype=np.int64, shape=(n_events, 2)),
        "source_run_index": open_memmap(paths["source_run_index"], mode="w+", dtype=np.int32, shape=(n_events,)),
        "bias_voltage_V": open_memmap(paths["bias_voltage_V"], mode="w+", dtype=np.float64, shape=(n_events,)),
        "amplitude_mV": open_memmap(paths["amplitude_mV"], mode="w+", dtype=np.float32, shape=(n_events, 2)),
        "noise_rms_mV": open_memmap(paths["noise_rms_mV"], mode="w+", dtype=np.float32, shape=(n_events, 2)),
        "trigger_index": open_memmap(paths["trigger_index"], mode="w+", dtype=np.int32, shape=(n_events, 2)),
        "led_time_fs": open_memmap(paths["led_time_fs"], mode="w+", dtype=np.int64, shape=(n_events, 2)),
        "cfd_time_fs": open_memmap(paths["cfd_time_fs"], mode="w+", dtype=np.int64, shape=(n_events, 2)),
        "windows_mV": open_memmap(paths["windows_mV"], mode="w+", dtype=np.float32, shape=(n_events, 2, relative_grid.size)),
        "valid": open_memmap(paths["valid"], mode="w+", dtype=np.bool_, shape=(n_events,)),
    }
    np.save(paths["relative_time_ps"], relative_grid.astype(np.float32))

    energy_channels = tuple(int(item) for item in config["channels"]["energy"])
    energy_polarities = tuple(int(item) for item in config["channels"]["polarities"])
    timing_led_config = config["waveform"].get("timing_channel_led", {})
    use_timing_channel_led = bool(timing_led_config.get("enabled", False))
    timing_channels = (
        tuple(int(item) for item in config["channels"]["timing"])
        if use_timing_channel_led
        else None
    )
    timing_polarities = (
        tuple(int(item) for item in config["channels"]["timing_polarities"])
        if use_timing_channel_led
        else (1, 1)
    )
    io_config = config.get("io", {})
    parallel = config["parallelization"]
    progress_every = max(1, int(io_config.get("progress_every", 1000)))
    written = 0

    logger.info(
        "Building preprocessing cache | ML waveform branches samples_ch%d/samples_ch%d",
        energy_channels[0],
        energy_channels[1],
    )
    if use_timing_channel_led:
        assert timing_channels is not None
        logger.info(
            "Timing-channel LED mode enabled | LED and ML-window alignment from "
            "samples_ch%d/samples_ch%d | timing waveforms are not saved as ML inputs",
            timing_channels[0],
            timing_channels[1],
        )
    else:
        logger.info("Energy-channel LED mode enabled | LED/CFD and alignment from ML channels")
    denoising = config["waveform"].get("denoising", {})
    if bool(denoising.get("enabled", False)):
        logger.info(
            "Waveform denoising enabled | method %s | cutoff %.6g GHz | order %d",
            denoising.get("method", "butterworth_lowpass"),
            float(denoising["cutoff_GHz"]),
            int(denoising.get("order", 4)),
        )
    try:
        for chunk in iterate_energy_chunks(
            input_path,
            energy_channels_one_based=energy_channels,
            timing_channels_one_based=timing_channels,
            step_size=io_config.get("step_size", "128 MB"),
            entry_stop=n_events,
        ):
            payloads: list[tuple[Any, ...]] = []
            for row in range(chunk.event_id.size):
                energy_raw_a = np.asarray(
                    ak.to_numpy(chunk.samples[0][row]), dtype=np.int16
                )
                energy_raw_b = np.asarray(
                    ak.to_numpy(chunk.samples[1][row]), dtype=np.int16
                )
                if use_timing_channel_led:
                    assert chunk.timing_samples is not None
                    assert chunk.timing_vertical_gain_v_per_count is not None
                    assert chunk.timing_vertical_offset_v is not None
                    assert chunk.timing_horizontal_interval_s is not None
                    assert chunk.timing_horizontal_offset_s is not None
                    timing_raw_a = np.asarray(
                        ak.to_numpy(chunk.timing_samples[0][row]), dtype=np.int16
                    )
                    timing_raw_b = np.asarray(
                        ak.to_numpy(chunk.timing_samples[1][row]), dtype=np.int16
                    )
                    timing_gains = chunk.timing_vertical_gain_v_per_count[row]
                    timing_offsets = chunk.timing_vertical_offset_v[row]
                    timing_intervals = chunk.timing_horizontal_interval_s[row]
                    timing_horizontal_offsets = chunk.timing_horizontal_offset_s[row]
                else:
                    timing_raw_a = np.empty(0, dtype=np.int16)
                    timing_raw_b = np.empty(0, dtype=np.int16)
                    timing_gains = np.zeros(2, dtype=np.float64)
                    timing_offsets = np.zeros(2, dtype=np.float64)
                    timing_intervals = np.ones(2, dtype=np.float64)
                    timing_horizontal_offsets = np.zeros(2, dtype=np.float64)
                payloads.append(
                    (
                        chunk.event_index[row],
                        chunk.event_id[row],
                        chunk.source_file_id[row],
                        chunk.source_run_index[row],
                        chunk.bias_voltage_V[row],
                        energy_raw_a,
                        energy_raw_b,
                        chunk.vertical_gain_v_per_count[row],
                        chunk.vertical_offset_v[row],
                        chunk.horizontal_interval_s[row],
                        chunk.horizontal_offset_s[row],
                        energy_polarities,
                        use_timing_channel_led,
                        timing_raw_a,
                        timing_raw_b,
                        timing_gains,
                        timing_offsets,
                        timing_intervals,
                        timing_horizontal_offsets,
                        timing_polarities,
                        config["waveform"],
                        relative_grid,
                    )
                )
            for result in _executor_map(payloads, parallel):
                if written >= n_events:
                    break
                (
                    event_index,
                    event_id,
                    source_id,
                    source_run_index,
                    bias_voltage_V,
                    amplitude,
                    noise,
                    trigger,
                    led,
                    cfd,
                    windows,
                    valid,
                ) = result
                arrays["event_index"][written] = event_index
                arrays["event_id"][written] = event_id
                arrays["source_file_id"][written] = source_id
                arrays["source_run_index"][written] = source_run_index
                arrays["bias_voltage_V"][written] = bias_voltage_V
                arrays["amplitude_mV"][written] = amplitude
                arrays["noise_rms_mV"][written] = noise
                arrays["trigger_index"][written] = trigger
                arrays["led_time_fs"][written] = led
                arrays["cfd_time_fs"][written] = cfd
                arrays["windows_mV"][written] = windows
                arrays["valid"][written] = valid
                written += 1
                if written % progress_every == 0 or written == n_events:
                    logger.info("Preprocessed %d/%d events", written, n_events)
        if written != n_events:
            raise RuntimeError(f"Expected {n_events} events but wrote {written}")
        for array in arrays.values():
            array.flush()
        valid_events = int(np.count_nonzero(arrays["valid"]))
        manifest = {
            "format_version": CACHE_FORMAT_VERSION,
            "fingerprint": expected_fingerprint,
            "source": source_signature(input_path),
            "event_count": n_events,
            "energy_channels_one_based": list(energy_channels),
            "timing_channels_one_based": (
                list(timing_channels) if timing_channels is not None else []
            ),
            "branches_read": [
                *[f"samples_ch{channel}" for channel in energy_channels],
                *(
                    [f"samples_ch{channel}" for channel in timing_channels]
                    if timing_channels is not None
                    else []
                ),
            ],
            "ml_input_channel_branches": [
                f"samples_ch{channel}" for channel in energy_channels
            ],
            "timing_channel_branches_read": (
                [f"samples_ch{channel}" for channel in timing_channels]
                if timing_channels is not None
                else []
            ),
            "timing_channel_waveforms_saved": False,
            "led_timestamp_source": (
                "timing_channels" if use_timing_channel_led else "energy_channels"
            ),
            "cfd_timestamp_source": "energy_channels",
            "ml_window_alignment_source": (
                "timing_channel_led" if use_timing_channel_led else "energy_channel_led"
            ),
            "optional_metadata_cached": ["source_run_index", "bias_voltage_V"],
            "relative_window_points": int(relative_grid.size),
            "relative_time_ps_start": float(relative_grid[0]),
            "relative_time_ps_stop": float(relative_grid[-1]),
            "upsample_step_ps": float(config["waveform"]["upsample_step_ps"]),
            "subsample_factor": int(config["waveform"]["subsample_factor"]),
            "effective_window_step_ps": float(
                config["waveform"]["upsample_step_ps"]
            ) * int(config["waveform"]["subsample_factor"]),
            "valid_events": valid_events,
            "preprocessing": _preprocessing_relevant(config),
        }
        atomic_json(temporary / "manifest.json", manifest)
        # Close memory maps before renaming the directory (required on Windows).
        for array in arrays.values():
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()
        arrays.clear()
        if directory.exists():
            shutil.rmtree(directory)
        os.replace(temporary, directory)
    except BaseException:
        logger.exception("Waveform preprocessing failed; incomplete cache kept at %s", temporary)
        raise
    logger.info("Waveform preprocessing cache written to %s", directory)
    return load_energy_cache(directory, input_path, config)


def _source_group_split(
    groups: np.ndarray, fractions: tuple[float, float, float], seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keys = np.asarray([f"{int(a)}:{int(b)}" for a, b in groups], dtype=object)
    unique = np.unique(keys)
    if unique.size < 3:
        raise ValueError(
            "source_file split requires at least three distinct energy-channel source-file pairs"
        )
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    target = np.asarray(fractions) * keys.size
    split_lists: list[list[str]] = [[], [], []]
    counts = np.zeros(3, dtype=np.int64)
    for key in unique:
        size = int(np.count_nonzero(keys == key))
        deficits = target - counts
        destination = int(np.argmax(deficits))
        split_lists[destination].append(str(key))
        counts[destination] += size
    masks = [np.isin(keys, split_keys) for split_keys in split_lists]
    return tuple(np.flatnonzero(mask).astype(np.int64) for mask in masks)  # type: ignore[return-value]


def _event_split(
    n_events: int, fractions: tuple[float, float, float], seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_events)
    n_train = int(np.floor(fractions[0] * n_events))
    n_validation = int(np.floor(fractions[1] * n_events))
    train = order[:n_train]
    validation = order[n_train : n_train + n_validation]
    test = order[n_train + n_validation :]
    return train.astype(np.int64), validation.astype(np.int64), test.astype(np.int64)




def _stratified_event_split(
    labels: np.ndarray,
    fractions: tuple[float, float, float],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(labels, dtype=np.float64).reshape(-1)
    # Treat unavailable voltage metadata as one explicit stratum so the function
    # remains usable with ordinary single-run files.
    keys = np.where(np.isfinite(values), values, np.inf)
    rng = np.random.default_rng(seed)
    parts: list[list[np.ndarray]] = [[], [], []]
    for key in np.unique(keys):
        indices = np.flatnonzero(keys == key).astype(np.int64)
        rng.shuffle(indices)
        n_train = int(np.floor(fractions[0] * indices.size))
        n_validation = int(np.floor(fractions[1] * indices.size))
        parts[0].append(indices[:n_train])
        parts[1].append(indices[n_train : n_train + n_validation])
        parts[2].append(indices[n_train + n_validation :])
    outputs = []
    for group in parts:
        merged = np.concatenate(group) if group else np.empty(0, dtype=np.int64)
        rng.shuffle(merged)
        outputs.append(merged.astype(np.int64, copy=False))
    return outputs[0], outputs[1], outputs[2]


def _voltage_counts(cache: EnergyCache, indices: np.ndarray) -> dict[str, int]:
    values = np.asarray(cache.bias_voltage_V[indices], dtype=np.float64)
    result: dict[str, int] = {}
    for value in np.unique(values[np.isfinite(values)]):
        result[f"{float(value):g}"] = int(np.count_nonzero(np.isclose(values, value)))
    missing = int(np.count_nonzero(~np.isfinite(values)))
    if missing:
        result["unknown"] = missing
    return result


def _noise_limits(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, list):
        if len(value) != 2:
            raise ValueError("selection.energy_noise_max_mV list must contain two values")
        return float(value[0]), float(value[1])
    number = float(value)
    return number, number


def prepare_splits(
    cache: EnergyCache,
    directory: Path,
    config: dict[str, Any],
    *,
    rebuild: bool,
    logger: Any,
) -> SplitData:
    relevant = {
        "format_version": SPLIT_FORMAT_VERSION,
        "dataset_fingerprint": cache.manifest["fingerprint"],
        "split": config["split"],
        "selection": config["selection"],
        "photopeak": config["photopeak"],
    }
    fingerprint = canonical_hash(relevant)
    manifest_path = directory / "manifest.json"
    indices_path = directory / "indices.npz"
    if not rebuild and manifest_path.is_file() and indices_path.is_file():
        manifest = read_json(manifest_path)
        if manifest.get("fingerprint") == fingerprint:
            with np.load(indices_path, allow_pickle=False) as loaded:
                logger.info("Reusing frozen train/validation/test split: %s", directory)
                return SplitData(
                    train=loaded["train"].astype(np.int64),
                    validation=loaded["validation"].astype(np.int64),
                    test=loaded["test"].astype(np.int64),
                    manifest=manifest,
                )

    n_events = int(cache.event_id.shape[0])
    split_config = config["split"]
    fractions = (
        float(split_config["train_fraction"]),
        float(split_config["validation_fraction"]),
        float(split_config["test_fraction"]),
    )
    seed = int(split_config["seed"])
    strategy = str(split_config.get("strategy", "event"))
    guard_gap_events = int(split_config.get("guard_gap_events", 0))
    if strategy == "source_file":
        candidates = _source_group_split(cache.source_file_id, fractions, seed)
    elif strategy == "stratified_event":
        candidates = _stratified_event_split(cache.bias_voltage_V, fractions, seed)
    elif strategy == "contiguous_blocks":
        candidates = contiguous_block_split(
            n_events, fractions, guard_gap_events
        )
    else:
        candidates = _event_split(n_events, fractions, seed)
    train_candidate, validation_candidate, test_candidate = candidates

    # Copy the read-only memory map before combining selection masks.
    # cache.valid already guarantees finite fixed-length windows for both channels.
    base_valid = np.array(cache.valid, dtype=bool, copy=True)
    base_valid &= np.all(np.asarray(cache.led_time_fs) != INVALID_TIME_FS, axis=1)
    base_valid &= np.all(np.asarray(cache.cfd_time_fs) != INVALID_TIME_FS, axis=1)

    selection_config = config["selection"]
    trigger_range = selection_config.get("energy_trigger_index_range")
    if trigger_range is not None:
        low, high = int(trigger_range[0]), int(trigger_range[1])
        triggers = np.asarray(cache.trigger_index)
        base_valid &= np.all((triggers > low) & (triggers < high), axis=1)

    limits = _noise_limits(selection_config.get("energy_noise_max_mV"))
    if limits is not None:
        noise = np.asarray(cache.noise_rms_mV)
        base_valid &= (noise[:, 0] < limits[0]) & (noise[:, 1] < limits[1])

    # Reject gross LED-pair outliers using a center estimated from training data only.
    # The same frozen bounds are then applied to training, validation, and test, avoiding
    # validation/test leakage while keeping a common event set for all final methods.
    led_outlier_config = selection_config.get("led_outlier_rejection", {})
    led_outlier_summary: dict[str, Any] = {
        "enabled": bool(led_outlier_config.get("enabled", False))
    }
    if led_outlier_summary["enabled"]:
        max_distance_ps = float(led_outlier_config.get("max_distance_ps", 0.0))
        if not np.isfinite(max_distance_ps) or max_distance_ps <= 0.0:
            raise ValueError(
                "selection.led_outlier_rejection.max_distance_ps must be finite and > 0"
            )

        led_times_fs = np.asarray(cache.led_time_fs, dtype=np.int64)
        led_tof_ps = (
            led_times_fs[:, 0].astype(np.float64)
            - led_times_fs[:, 1].astype(np.float64)
        ) / 1000.0
        training_led_values = led_tof_ps[train_candidate[base_valid[train_candidate]]]
        training_led_values = training_led_values[np.isfinite(training_led_values)]
        if training_led_values.size < 3:
            raise RuntimeError(
                "Too few valid training LED differences to estimate the outlier-rejection median"
            )

        training_median_ps = float(np.median(training_led_values))
        lower_ps = training_median_ps - max_distance_ps
        upper_ps = training_median_ps + max_distance_ps
        led_inlier = np.isfinite(led_tof_ps) & (np.abs(led_tof_ps - training_median_ps) <= max_distance_ps)

        valid_before = base_valid.copy()
        base_valid &= led_inlier
        rejected_total = int(np.count_nonzero(valid_before & ~led_inlier))
        rejected_by_candidate = {
            "train": int(np.count_nonzero(valid_before[train_candidate] & ~led_inlier[train_candidate])),
            "validation": int(np.count_nonzero(valid_before[validation_candidate] & ~led_inlier[validation_candidate])),
            "test": int(np.count_nonzero(valid_before[test_candidate] & ~led_inlier[test_candidate])),
        }
        led_outlier_summary.update(
            {
                "center_scope": "training split only",
                "training_median_ps": training_median_ps,
                "max_distance_ps": max_distance_ps,
                "accepted_interval_ps": [lower_ps, upper_ps],
                "rejected_total": rejected_total,
                "rejected_by_candidate_split": rejected_by_candidate,
            }
        )
        logger.info(
            "LED outlier rejection | training median %.3f ps | keep |Δt - median| <= %.3f ps | "
            "rejected train/validation/test: %d/%d/%d",
            training_median_ps,
            max_distance_ps,
            rejected_by_candidate["train"],
            rejected_by_candidate["validation"],
            rejected_by_candidate["test"],
        )

    amplitudes = np.asarray(cache.amplitude_mV)
    photopeak_results = []
    if bool(config["photopeak"].get("enabled", True)):
        fit_indices = train_candidate[base_valid[train_candidate]]
        for channel_position, channel_number in enumerate(
            cache.manifest["energy_channels_one_based"]
        ):
            result = fit_photopeak(
                amplitudes[fit_indices, channel_position],
                channel=int(channel_number),
                config=config["photopeak"],
            )
            if not result.success:
                raise RuntimeError(
                    f"Training-only photopeak fit failed for channel {channel_number}: "
                    f"{result.message}"
                )
            photopeak_results.append(result)
            base_valid &= photopeak_mask(amplitudes[:, channel_position], result)

    train = train_candidate[base_valid[train_candidate]]
    validation = validation_candidate[base_valid[validation_candidate]]
    test = test_candidate[base_valid[test_candidate]]
    minimum = int(selection_config.get("minimum_events_per_split", 1))
    for name, values in (("train", train), ("validation", validation), ("test", test)):
        if values.size < minimum:
            raise RuntimeError(
                f"Only {values.size} selected events in {name}; need at least {minimum}"
            )

    directory.mkdir(parents=True, exist_ok=True)
    temporary = indices_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, train=train, validation=validation, test=test)
    os.replace(temporary, indices_path)
    manifest = {
        "format_version": SPLIT_FORMAT_VERSION,
        "fingerprint": fingerprint,
        "dataset_fingerprint": cache.manifest["fingerprint"],
        "strategy": strategy,
        "seed": seed,
        "guard_gap_events": guard_gap_events if strategy == "contiguous_blocks" else 0,
        "guard_events_excluded_total": (
            2 * guard_gap_events if strategy == "contiguous_blocks" else 0
        ),
        "fractions": {
            "train": fractions[0],
            "validation": fractions[1],
            "test": fractions[2],
        },
        "candidate_counts": {
            "train": int(train_candidate.size),
            "validation": int(validation_candidate.size),
            "test": int(test_candidate.size),
        },
        "selected_counts": {
            "train": int(train.size),
            "validation": int(validation.size),
            "test": int(test.size),
        },
        "bias_voltage_counts": {
            "train": _voltage_counts(cache, train),
            "validation": _voltage_counts(cache, validation),
            "test": _voltage_counts(cache, test),
        },
        "selection_is_energy_only": True,
        "same_event_set_for_led_cfd_and_corrected": True,
        "photopeak_fit_scope": "training split only",
        "photopeak": [result.as_dict() for result in photopeak_results],
        "led_outlier_rejection": led_outlier_summary,
        "selection": config["selection"],
    }
    atomic_json(manifest_path, manifest)
    logger.info(
        "Selected events — train: %d, validation: %d, blind test: %d",
        train.size,
        validation.size,
        test.size,
    )
    return SplitData(train=train, validation=validation, test=test, manifest=manifest)
