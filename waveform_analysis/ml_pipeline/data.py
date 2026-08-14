from __future__ import annotations

import copy
import errno
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import awkward as ak
import numpy as np
from numpy.lib.format import open_memmap

from utils.signal import INVALID_TIME_FS

from .common import atomic_json, canonical_hash, read_json, source_signature
from .energy_io import energy_event_count, energy_sampling_interval_s, iterate_energy_chunks
from .signal import (
    TimingReference,
    extract_channel,
    extract_timing_channel,
    relative_window_grid_ps,
)
from .standard_methods.cfd import select_precomputed_cfd_times

CACHE_FORMAT_VERSION = 9


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
    energy_led_time_fs: np.ndarray | None = None
    timing_led_time_fs: np.ndarray | None = None
    energy_cfd_time_fs: np.ndarray | None = None
    timing_cfd_time_fs: np.ndarray | None = None
    energy_window_anchor_time_fs: np.ndarray | None = None
    timing_aligned_energy_window_anchor_time_fs: np.ndarray | None = None
    timing_window_anchor_time_fs: np.ndarray | None = None
    timing_aligned_energy_windows_mV: np.ndarray | None = None
    timing_windows_mV: np.ndarray | None = None
    timing_relative_time_ps: np.ndarray | None = None


def _preprocessing_relevant(config: dict[str, Any]) -> dict[str, Any]:
    waveform = copy.deepcopy(config["waveform"])
    # These keys controlled the former synthetic spline grid. They are accepted
    # by config loading for migration only and no longer affect preprocessing.
    waveform.pop("upsample_step_ps", None)
    waveform.pop("subsample_factor", None)
    timing_led = waveform.get("timing_channel_led")
    if isinstance(timing_led, dict):
        timing_led.pop("upsample_step_ps", None)
    return {
        "channels": config["channels"],
        "waveform": waveform,
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
        "energy_led_time_fs",
        "timing_led_time_fs",
        "energy_cfd_time_fs",
        "timing_cfd_time_fs",
        "energy_window_anchor_time_fs",
        "timing_aligned_energy_window_anchor_time_fs",
        "timing_window_anchor_time_fs",
        "timing_aligned_energy_windows_mV",
        "timing_windows_mV",
        "timing_relative_time_ps",
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
    required = (
        "event_id", "event_index", "source_file_id", "source_run_index",
        "bias_voltage_V", "amplitude_mV", "noise_rms_mV", "trigger_index",
        "led_time_fs", "cfd_time_fs", "windows_mV", "valid",
        "relative_time_ps", "energy_led_time_fs", "energy_cfd_time_fs",
        "energy_window_anchor_time_fs",
    )
    if bool(manifest.get("timing_channel_waveforms_saved", False)):
        required += (
            "timing_led_time_fs",
            "timing_cfd_time_fs",
            "timing_aligned_energy_window_anchor_time_fs",
            "timing_window_anchor_time_fs",
            "timing_aligned_energy_windows_mV",
            "timing_windows_mV",
            "timing_relative_time_ps",
        )
    missing = [str(paths[name]) for name in required if not paths[name].is_file()]
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
        energy_led_time_fs=np.load(paths["energy_led_time_fs"], mmap_mode="r"),
        timing_led_time_fs=(
            np.load(paths["timing_led_time_fs"], mmap_mode="r")
            if paths["timing_led_time_fs"].is_file() else None
        ),
        energy_cfd_time_fs=np.load(paths["energy_cfd_time_fs"], mmap_mode="r"),
        timing_cfd_time_fs=(
            np.load(paths["timing_cfd_time_fs"], mmap_mode="r")
            if paths["timing_cfd_time_fs"].is_file() else None
        ),
        energy_window_anchor_time_fs=np.load(
            paths["energy_window_anchor_time_fs"], mmap_mode="r"
        ),
        timing_aligned_energy_window_anchor_time_fs=(
            np.load(paths["timing_aligned_energy_window_anchor_time_fs"], mmap_mode="r")
            if paths["timing_aligned_energy_window_anchor_time_fs"].is_file() else None
        ),
        timing_window_anchor_time_fs=(
            np.load(paths["timing_window_anchor_time_fs"], mmap_mode="r")
            if paths["timing_window_anchor_time_fs"].is_file() else None
        ),
        timing_aligned_energy_windows_mV=(
            np.load(paths["timing_aligned_energy_windows_mV"], mmap_mode="r")
            if paths["timing_aligned_energy_windows_mV"].is_file() else None
        ),
        timing_windows_mV=(
            np.load(paths["timing_windows_mV"], mmap_mode="r")
            if paths["timing_windows_mV"].is_file() else None
        ),
        timing_relative_time_ps=(
            np.load(paths["timing_relative_time_ps"], mmap_mode="r")
            if paths["timing_relative_time_ps"].is_file() else None
        ),
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
        native_interval_s,
        timing_relative_grid_ps,
        timing_native_interval_s,
    ) = payload

    observed_intervals = np.asarray(energy_intervals, dtype=np.float64)
    if not np.allclose(observed_intervals, float(native_interval_s), rtol=1e-9, atol=0.0):
        raise ValueError(
            "Energy-channel sampling interval changed within the input file; "
            "canonical native-grid windows require one shared interval"
        )

    timing_references: list[TimingReference | None] = [None, None]
    timing_outputs = []
    if use_timing_channel_led:
        observed_timing_intervals = np.asarray(timing_intervals, dtype=np.float64)
        if not np.allclose(
            observed_timing_intervals,
            float(timing_native_interval_s),
            rtol=1e-9,
            atol=0.0,
        ):
            raise ValueError(
                "Timing-channel sampling interval changed within the input file; "
                "canonical native-grid windows require one shared interval"
            )
        timing_outputs = []
        timing_references = []
        for channel_position, raw in enumerate((timing_raw_a, timing_raw_b)):
            item = extract_timing_channel(
                np.asarray(raw, dtype=np.int16),
                vertical_gain_v_per_count=float(timing_gains[channel_position]),
                vertical_offset_v=float(timing_offsets[channel_position]),
                horizontal_interval_s=float(timing_intervals[channel_position]),
                horizontal_offset_s=float(timing_horizontal_offsets[channel_position]),
                polarity=int(timing_polarities[channel_position]),
                waveform_config=waveform_config,
                relative_grid_ps=timing_relative_grid_ps,
            )
            timing_outputs.append(item)
            timing_references.append(
                TimingReference(
                    trigger_index=item.trigger_index,
                    led_time_fs=item.led_time_fs,
                    cfd_time_fs=item.cfd_time_fs,
                    valid=item.valid,
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
    energy_led = np.asarray([item.led_time_fs for item in outputs], dtype=np.int64)
    energy_cfd = np.asarray([item.cfd_time_fs for item in outputs], dtype=np.int64)
    timing_led = (
        np.asarray([item.led_time_fs for item in timing_outputs], dtype=np.int64)
        if use_timing_channel_led
        else None
    )
    timing_cfd = (
        np.asarray([item.cfd_time_fs for item in timing_outputs], dtype=np.int64)
        if use_timing_channel_led
        else None
    )
    energy_window_anchor = np.asarray(
        [item.window_anchor_time_fs for item in outputs], dtype=np.int64
    )
    timing_window_anchor = (
        np.asarray([item.window_anchor_time_fs for item in timing_outputs], dtype=np.int64)
        if use_timing_channel_led
        else None
    )
    timing_aligned_energy_anchor = (
        np.asarray(
            [item.reference_aligned_window_anchor_time_fs for item in outputs],
            dtype=np.int64,
        )
        if use_timing_channel_led
        else None
    )
    # ``led_time_fs`` and ``cfd_time_fs`` are the prepared standard-method
    # timestamps consumed by ml_evaluate.  Select both from the same waveform
    # family here, during preprocessing; the manifest only documents this choice.
    prepared_led = timing_led if use_timing_channel_led else energy_led
    prepared_cfd = select_precomputed_cfd_times(energy_cfd, timing_cfd)
    timing_windows = (
        np.stack([item.window_mV for item in timing_outputs]).astype(np.float32)
        if use_timing_channel_led
        else None
    )
    timing_aligned_available = bool(
        use_timing_channel_led
        and all(item.reference_aligned_window_mV is not None for item in outputs)
    )
    timing_aligned_energy_windows = (
        np.stack(
            [
                np.asarray(item.reference_aligned_window_mV, dtype=np.float32)
                for item in outputs
            ]
        ).astype(np.float32)
        if timing_aligned_available
        else (
            np.full((2, relative_grid_ps.size), np.nan, dtype=np.float32)
            if use_timing_channel_led
            else None
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
        prepared_led,
        prepared_cfd,
        energy_led,
        timing_led,
        energy_cfd,
        timing_cfd,
        energy_window_anchor,
        timing_aligned_energy_anchor,
        timing_window_anchor,
        np.stack([item.window_mV for item in outputs]).astype(np.float32),
        timing_aligned_energy_windows,
        timing_windows,
        bool(
            all(item.valid for item in outputs)
            and (not use_timing_channel_led or all(item.valid for item in timing_outputs))
            and (
                not use_timing_channel_led
                or timing_aligned_available
            )
        ),
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
    energy_channels = tuple(int(item) for item in config["channels"]["energy"])
    native_interval_s = energy_sampling_interval_s(input_path, energy_channels)
    relative_grid = relative_window_grid_ps(config["waveform"], native_interval_s)
    timing_led_config = config["waveform"].get("timing_channel_led", {})
    use_timing_channel_led = bool(timing_led_config.get("enabled", False))
    timing_channels = (
        tuple(int(item) for item in config["channels"]["timing"])
        if use_timing_channel_led else None
    )
    timing_native_interval_s = (
        energy_sampling_interval_s(input_path, timing_channels)
        if timing_channels is not None else native_interval_s
    )
    timing_relative_grid = (
        relative_window_grid_ps(
            {
                **config["waveform"],
                **{
                    key: value
                    for key, value in timing_led_config.items()
                    if key in ("ml_window_ns",)
                },
            },
            timing_native_interval_s,
        )
        if use_timing_channel_led else None
    )

    temporary = directory.with_name(directory.name + ".building")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, exist_ok=True)

    # Fail before partially allocating large memmaps. In study mode the cache may
    # contain energy-aligned energy, timing-aligned energy, and timing waveforms.
    # Their combined size is substantial, especially for long windows.
    array_specs: list[tuple[tuple[int, ...], np.dtype]] = [
        ((n_events,), np.dtype(np.int64)),
        ((n_events,), np.dtype(np.int64)),
        ((n_events, 2), np.dtype(np.int64)),
        ((n_events,), np.dtype(np.int32)),
        ((n_events,), np.dtype(np.float64)),
        ((n_events, 2), np.dtype(np.float32)),
        ((n_events, 2), np.dtype(np.float32)),
        ((n_events, 2), np.dtype(np.int32)),
        ((n_events, 2), np.dtype(np.int64)),
        ((n_events, 2), np.dtype(np.int64)),
        ((n_events, 2, int(relative_grid.size)), np.dtype(np.float32)),
        ((n_events, 2), np.dtype(np.int64)),
        ((n_events, 2), np.dtype(np.int64)),
        ((n_events, 2), np.dtype(np.int64)),
        ((n_events,), np.dtype(np.bool_)),
    ]
    if use_timing_channel_led:
        assert timing_relative_grid is not None
        array_specs.extend([
            ((n_events, 2), np.dtype(np.int64)),
            ((n_events, 2), np.dtype(np.int64)),
            ((n_events, 2), np.dtype(np.int64)),
            ((n_events, 2), np.dtype(np.int64)),
            ((n_events, 2, int(relative_grid.size)), np.dtype(np.float32)),
            ((n_events, 2, int(timing_relative_grid.size)), np.dtype(np.float32)),
        ])
    required_bytes = sum(
        int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        for shape, dtype in array_specs
    )
    free_bytes = int(shutil.disk_usage(temporary.parent).free)
    safety_bytes = max(int(required_bytes * 1.05), required_bytes + 256 * 1024**2)
    logger.info(
        "Preprocessing storage | required about %.2f GiB | free %.2f GiB",
        required_bytes / 1024**3,
        free_bytes / 1024**3,
    )
    if free_bytes < safety_bytes:
        shutil.rmtree(temporary, ignore_errors=True)
        raise OSError(
            errno.ENOSPC,
            "Insufficient disk space for preprocessing cache: "
            f"need about {safety_bytes / 1024**3:.2f} GiB including safety margin, "
            f"but only {free_bytes / 1024**3:.2f} GiB is free",
            str(directory),
        )

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
        "energy_led_time_fs": open_memmap(paths["energy_led_time_fs"], mode="w+", dtype=np.int64, shape=(n_events, 2)),
        "energy_cfd_time_fs": open_memmap(paths["energy_cfd_time_fs"], mode="w+", dtype=np.int64, shape=(n_events, 2)),
        "energy_window_anchor_time_fs": open_memmap(
            paths["energy_window_anchor_time_fs"], mode="w+", dtype=np.int64,
            shape=(n_events, 2),
        ),
        "valid": open_memmap(paths["valid"], mode="w+", dtype=np.bool_, shape=(n_events,)),
    }
    np.save(paths["relative_time_ps"], relative_grid.astype(np.float32))
    if use_timing_channel_led:
        assert timing_relative_grid is not None
        arrays["timing_led_time_fs"] = open_memmap(
            paths["timing_led_time_fs"], mode="w+", dtype=np.int64, shape=(n_events, 2)
        )
        arrays["timing_cfd_time_fs"] = open_memmap(
            paths["timing_cfd_time_fs"], mode="w+", dtype=np.int64, shape=(n_events, 2)
        )
        arrays["timing_aligned_energy_window_anchor_time_fs"] = open_memmap(
            paths["timing_aligned_energy_window_anchor_time_fs"],
            mode="w+", dtype=np.int64, shape=(n_events, 2),
        )
        arrays["timing_window_anchor_time_fs"] = open_memmap(
            paths["timing_window_anchor_time_fs"],
            mode="w+", dtype=np.int64, shape=(n_events, 2),
        )
        arrays["timing_aligned_energy_windows_mV"] = open_memmap(
            paths["timing_aligned_energy_windows_mV"],
            mode="w+",
            dtype=np.float32,
            shape=(n_events, 2, relative_grid.size),
        )
        arrays["timing_windows_mV"] = open_memmap(
            paths["timing_windows_mV"], mode="w+", dtype=np.float32,
            shape=(n_events, 2, timing_relative_grid.size),
        )
        np.save(
            paths["timing_relative_time_ps"],
            timing_relative_grid.astype(np.float32),
        )

    energy_polarities = tuple(int(item) for item in config["channels"]["polarities"])
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
        "Building preprocessing cache | ML waveform branches samples_ch%d/samples_ch%d | "
        "native sampling interval %.6g ps",
        energy_channels[0],
        energy_channels[1],
        native_interval_s * 1.0e12,
    )
    deprecated = [
        name
        for name in ("upsample_step_ps", "subsample_factor")
        if name in config["waveform"]
    ]
    timing_override = config["waveform"].get("timing_channel_led", {})
    if isinstance(timing_override, dict) and "upsample_step_ps" in timing_override:
        deprecated.append("timing_channel_led.upsample_step_ps")
    if deprecated:
        logger.warning(
            "Deprecated preprocessing option(s) ignored: %s. Saved ML windows now use "
            "the original acquisition samples.",
            ", ".join(deprecated),
        )
    if use_timing_channel_led:
        assert timing_channels is not None
        logger.info(
            "Timing-channel mode enabled | energy windows are saved with both energy-LED "
            "and timing-LED alignment | timing waveforms are saved with timing-LED alignment "
            "from samples_ch%d/samples_ch%d",
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
                        native_interval_s,
                        timing_relative_grid,
                        timing_native_interval_s,
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
                    energy_led,
                    timing_led,
                    energy_cfd,
                    timing_cfd,
                    energy_window_anchor,
                    timing_aligned_energy_anchor,
                    timing_window_anchor,
                    windows,
                    timing_aligned_energy_windows,
                    timing_windows,
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
                arrays["energy_led_time_fs"][written] = energy_led
                arrays["energy_cfd_time_fs"][written] = energy_cfd
                arrays["energy_window_anchor_time_fs"][written] = energy_window_anchor
                arrays["windows_mV"][written] = windows
                if use_timing_channel_led:
                    arrays["timing_led_time_fs"][written] = timing_led
                    arrays["timing_cfd_time_fs"][written] = timing_cfd
                    arrays["timing_aligned_energy_window_anchor_time_fs"][written] = (
                        timing_aligned_energy_anchor
                    )
                    arrays["timing_window_anchor_time_fs"][written] = timing_window_anchor
                    arrays["timing_aligned_energy_windows_mV"][written] = (
                        timing_aligned_energy_windows
                    )
                    arrays["timing_windows_mV"][written] = timing_windows
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
            "timing_channel_waveforms_saved": bool(use_timing_channel_led),
            "timing_aligned_energy_waveforms_saved": bool(use_timing_channel_led),
            "available_waveform_sources": [
                "energy", *( ["timing"] if use_timing_channel_led else [] )
            ],
            "available_prediction_targets": [
                "prepared_led", "energy_led",
                *( ["timing_led"] if use_timing_channel_led else [] ),
            ],
            "led_timestamp_source": (
                "timing_channels" if use_timing_channel_led else "energy_channels"
            ),
            "cfd_timestamp_source": (
                "timing_channels" if use_timing_channel_led else "energy_channels"
            ),
            "target_specific_cfd_timestamps_saved": True,
            "ml_window_alignment_source": "target_specific_led",
            "energy_window_alignment_sources": [
                "energy_channel_led",
                *(["timing_channel_led"] if use_timing_channel_led else []),
            ],
            "timing_window_alignment_source": (
                "timing_channel_led" if use_timing_channel_led else None
            ),
            "optional_metadata_cached": ["source_run_index", "bias_voltage_V"],
            "relative_window_points": int(relative_grid.size),
            "relative_time_ps_start": float(relative_grid[0]),
            "relative_time_ps_stop": float(relative_grid[-1]),
            "waveform_grid": "native_acquisition_samples",
            "native_sample_interval_ps": float(native_interval_s * 1.0e12),
            "timing_native_sample_interval_ps": (
                float(timing_native_interval_s * 1.0e12)
                if use_timing_channel_led else None
            ),
            "timing_relative_window_points": (
                int(timing_relative_grid.size)
                if timing_relative_grid is not None else 0
            ),
            "ml_window_alignment_quantization": "nearest_native_sample",
            "window_anchor_timestamps_saved": True,
            "correction_target_reference": "interpolated_led",
            "window_anchor_shift_factorization_supported": True,
            "timing_crossing_interpolation": "linear_between_bracketing_native_samples",
            "deprecated_preprocessing_options_ignored": deprecated,
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
