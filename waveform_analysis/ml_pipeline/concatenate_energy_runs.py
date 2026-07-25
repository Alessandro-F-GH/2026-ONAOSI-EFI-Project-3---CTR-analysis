from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import awkward as ak
import numpy as np
import uproot

from utils.photopeak import PhotopeakResult, fit_photopeak, photopeak_mask
from utils.plots import plot_energy_photopeaks

from .energy_io import energy_event_count, iterate_energy_chunks

FORMAT_VERSION = 1


@dataclass(frozen=True)
class RunSelection:
    path: Path
    run_index: int
    bias_voltage_V: float
    amplitudes_mV: np.ndarray
    selected: np.ndarray
    photopeak_results: tuple[PhotopeakResult, PhotopeakResult]

    @property
    def total_events(self) -> int:
        return int(self.selected.size)

    @property
    def selected_events(self) -> int:
        return int(np.count_nonzero(self.selected))


def load_concatenation_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    validate_concatenation_config(config)
    return config


def validate_concatenation_config(config: dict[str, Any]) -> None:
    required = ("input", "output", "channels", "waveform", "photopeak", "io")
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError("Missing configuration sections: " + ", ".join(missing))

    channels = config["channels"].get("energy")
    polarities = config["channels"].get("polarities")
    if not isinstance(channels, list) or len(channels) != 2:
        raise ValueError("channels.energy must contain exactly two one-based channel numbers")
    if len(set(int(value) for value in channels)) != 2:
        raise ValueError("channels.energy entries must be different")
    if not isinstance(polarities, list) or len(polarities) != 2:
        raise ValueError("channels.polarities must contain exactly two values")
    if any(int(value) not in (-1, 1) for value in polarities):
        raise ValueError("channels.polarities values must be +1 or -1")

    if int(config["waveform"].get("baseline_samples", 0)) <= 0:
        raise ValueError("waveform.baseline_samples must be positive")

    pattern = str(config["input"].get("pattern", "*.root")).strip()
    if not pattern:
        raise ValueError("input.pattern cannot be empty")

    minimum = int(config.get("selection", {}).get("minimum_selected_events_per_run", 1))
    if minimum < 1:
        raise ValueError("selection.minimum_selected_events_per_run must be at least 1")

    failure_policy = str(config.get("selection", {}).get("on_fit_failure", "error"))
    if failure_policy not in {"error", "skip"}:
        raise ValueError("selection.on_fit_failure must be 'error' or 'skip'")

    backend = str(config.get("parallelization", {}).get("backend", "thread"))
    if backend not in {"serial", "thread", "process"}:
        raise ValueError("parallelization.backend must be serial, thread, or process")


def discover_input_files(
    folder: Path,
    *,
    pattern: str,
    recursive: bool,
    excluded_paths: Sequence[Path] = (),
    excluded_names: Sequence[str] = (),
) -> list[Path]:
    if not folder.is_dir():
        raise NotADirectoryError(folder)
    iterator = folder.rglob(pattern) if recursive else folder.glob(pattern)
    excluded_resolved = {path.resolve() for path in excluded_paths}
    excluded_name_set = {str(name).lower() for name in excluded_names}
    files = [
        path.resolve()
        for path in iterator
        if path.is_file()
        and path.resolve() not in excluded_resolved
        and path.name.lower() not in excluded_name_set
    ]
    return sorted(set(files), key=lambda item: str(item).lower())


def extract_bias_voltage(path: Path, regex: str | None) -> float:
    if not regex:
        return float("nan")
    match = re.search(regex, str(path), flags=re.IGNORECASE)
    if match is None:
        return float("nan")
    if "bias_voltage" in match.groupdict():
        raw = match.group("bias_voltage")
    elif match.groups():
        raw = match.group(1)
    else:
        raw = match.group(0)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float("nan")


def _event_amplitudes(payload: tuple[Any, ...]) -> tuple[float, float]:
    raw_a, raw_b, gains, offsets, polarities, baseline_samples = payload
    result: list[float] = []
    for raw, gain, offset, polarity in zip(
        (raw_a, raw_b), gains, offsets, polarities, strict=True
    ):
        values = (np.asarray(raw, dtype=np.float64) * float(gain) - float(offset)) * 1000.0
        if values.ndim != 1 or values.size < 2 or np.any(~np.isfinite(values)):
            result.append(float("nan"))
            continue
        count = min(values.size, max(1, int(baseline_samples)))
        baseline = float(np.mean(values[:count]))
        corrected = int(polarity) * (values - baseline)
        result.append(float(np.max(corrected)))
    return result[0], result[1]


def _map_amplitudes(
    payloads: list[tuple[Any, ...]], parallel: dict[str, Any]
) -> Iterable[tuple[float, float]]:
    workers = int(parallel.get("workers", 0))
    backend = str(parallel.get("backend", "thread"))
    chunksize = max(1, int(parallel.get("chunksize", 16)))
    if workers <= 0 or backend == "serial":
        return map(_event_amplitudes, payloads)

    executor_type = ProcessPoolExecutor if backend == "process" else ThreadPoolExecutor
    executor = executor_type(max_workers=workers)

    def generate() -> Iterator[tuple[float, float]]:
        try:
            yield from executor.map(_event_amplitudes, payloads, chunksize=chunksize)
        finally:
            executor.shutdown(wait=True, cancel_futures=False)

    return generate()


def extract_run_amplitudes(
    path: Path,
    config: dict[str, Any],
    *,
    logger: logging.Logger,
) -> np.ndarray:
    channels = tuple(int(value) for value in config["channels"]["energy"])
    polarities = tuple(int(value) for value in config["channels"]["polarities"])
    baseline_samples = int(config["waveform"]["baseline_samples"])
    max_events = int(config["io"].get("max_events_per_file", 0))
    entry_stop = max_events if max_events > 0 else None
    total = energy_event_count(path)
    if entry_stop is not None:
        total = min(total, entry_stop)
    progress_every = max(1, int(config["io"].get("progress_every_events", 5000)))
    parallel = config.get("parallelization", {})

    parts: list[np.ndarray] = []
    processed = 0
    next_progress = progress_every
    for chunk in iterate_energy_chunks(
        path,
        energy_channels_one_based=channels,
        step_size=config["io"].get("step_size", "128 MB"),
        entry_stop=entry_stop,
    ):
        payloads: list[tuple[Any, ...]] = []
        for row in range(chunk.event_id.size):
            payloads.append(
                (
                    np.asarray(ak.to_numpy(chunk.samples[0][row]), dtype=np.int16),
                    np.asarray(ak.to_numpy(chunk.samples[1][row]), dtype=np.int16),
                    tuple(float(value) for value in chunk.vertical_gain_v_per_count[row]),
                    tuple(float(value) for value in chunk.vertical_offset_v[row]),
                    polarities,
                    baseline_samples,
                )
            )
        chunk_amplitudes = np.asarray(list(_map_amplitudes(payloads, parallel)), dtype=np.float32)
        if chunk_amplitudes.shape != (chunk.event_id.size, 2):
            raise RuntimeError(
                f"Amplitude extraction returned shape {chunk_amplitudes.shape}; "
                f"expected {(chunk.event_id.size, 2)}"
            )
        parts.append(chunk_amplitudes)
        processed += int(chunk.event_id.size)
        if processed >= next_progress or processed == total:
            logger.info("Amplitude extraction %s: %d/%d events", path.name, processed, total)
            next_progress += progress_every

    if not parts:
        return np.empty((0, 2), dtype=np.float32)
    amplitudes = np.concatenate(parts, axis=0)
    if amplitudes.shape[0] != total:
        raise RuntimeError(
            f"Read {amplitudes.shape[0]} events from {path}, expected {total}"
        )
    return amplitudes


def select_run_photopeak(
    path: Path,
    run_index: int,
    config: dict[str, Any],
    *,
    logger: logging.Logger,
) -> RunSelection | None:
    amplitudes = extract_run_amplitudes(path, config, logger=logger)
    channels = tuple(int(value) for value in config["channels"]["energy"])
    results = tuple(
        fit_photopeak(
            amplitudes[:, position],
            channel=channels[position],
            config=config["photopeak"],
        )
        for position in range(2)
    )
    if not all(result.success for result in results):
        reason = "; ".join(
            f"ch{result.channel}: {result.message}"
            for result in results
            if not result.success
        )
        policy = str(config.get("selection", {}).get("on_fit_failure", "error"))
        if policy == "skip":
            logger.warning("Skipping %s because photopeak fit failed: %s", path, reason)
            return None
        raise RuntimeError(f"Photopeak fit failed for {path}: {reason}")

    selected = photopeak_mask(amplitudes[:, 0], results[0]) & photopeak_mask(
        amplitudes[:, 1], results[1]
    )
    minimum = int(config.get("selection", {}).get("minimum_selected_events_per_run", 1))
    selected_count = int(np.count_nonzero(selected))
    if selected_count < minimum:
        message = (
            f"Run {path} has only {selected_count} events after independent "
            f"two-channel photopeak selection; minimum is {minimum}"
        )
        policy = str(config.get("selection", {}).get("on_fit_failure", "error"))
        if policy == "skip":
            logger.warning("Skipping: %s", message)
            return None
        raise RuntimeError(message)

    bias_config = config.get("metadata", {})
    bias_voltage = extract_bias_voltage(path, bias_config.get("bias_voltage_regex"))
    if bool(bias_config.get("require_bias_voltage", False)) and not np.isfinite(bias_voltage):
        raise RuntimeError(f"Cannot extract bias voltage from path: {path}")

    logger.info(
        "Run %d selected %d/%d events (%.2f%%) | "
        "ch%d %.3f±%.3f mV, cut [%.3f, %.3f] | "
        "ch%d %.3f±%.3f mV, cut [%.3f, %.3f]",
        run_index,
        selected_count,
        int(selected.size),
        100.0 * selected_count / max(1, int(selected.size)),
        results[0].channel,
        results[0].mean_mV,
        results[0].sigma_mV,
        results[0].selection_low_mV,
        results[0].selection_high_mV,
        results[1].channel,
        results[1].mean_mV,
        results[1].sigma_mV,
        results[1].selection_low_mV,
        results[1].selection_high_mV,
    )

    plotting = config.get("plotting", {})
    if bool(plotting.get("save_photopeak_plots", True)):
        plot_dir = Path(plotting.get("plot_dir", "results/concatenation_photopeaks"))
        plot_dir.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.parent.name + "_" + path.stem)
        plot_energy_photopeaks(
            amplitudes,
            np.asarray([0, 1], dtype=np.int64),
            [results[0], results[1]],
            selected,
            plot_dir / f"{run_index:04d}_{safe_name}_photopeak.png",
            dpi=int(plotting.get("dpi", 180)),
            bins=int(plotting.get("fallback_bins", 250)),
        )

    return RunSelection(
        path=path,
        run_index=run_index,
        bias_voltage_V=bias_voltage,
        amplitudes_mV=amplitudes,
        selected=selected,
        photopeak_results=(results[0], results[1]),
    )


def _event_tree_schema() -> dict[str, str]:
    return {
        "event_index": "int64",
        "event_id": "int64",
        "source_run_index": "int32",
        "bias_voltage_V": "float64",
        "original_event_index": "int64",
        "original_event_id": "int64",
        "source_file_id": "2 * int64",
        "original_source_file_id": "2 * int64",
        "sample_count": "2 * int32",
        "vertical_gain_v_per_count": "2 * float64",
        "vertical_offset_v": "2 * float64",
        "horizontal_interval_s": "2 * float64",
        "horizontal_offset_s": "2 * float64",
        "amplitude_mV": "2 * float32",
        "samples_ch1": "var * int16",
        "samples_ch2": "var * int16",
    }


def _write_selected_run(
    tree: Any,
    selection: RunSelection,
    config: dict[str, Any],
    *,
    first_global_event: int,
    logger: logging.Logger,
) -> int:
    channels = tuple(int(value) for value in config["channels"]["energy"])
    max_events = int(config["io"].get("max_events_per_file", 0))
    entry_stop = max_events if max_events > 0 else None
    processed = 0
    global_event = int(first_global_event)

    for chunk in iterate_energy_chunks(
        selection.path,
        energy_channels_one_based=channels,
        step_size=config["io"].get("step_size", "128 MB"),
        entry_stop=entry_stop,
    ):
        chunk_size = int(chunk.event_id.size)
        chunk_selected = selection.selected[processed : processed + chunk_size]
        chunk_amplitudes = selection.amplitudes_mV[processed : processed + chunk_size]
        if chunk_selected.size != chunk_size:
            raise RuntimeError(
                f"Selection length mismatch while writing {selection.path}: "
                f"got {chunk_selected.size}, expected {chunk_size}"
            )
        rows = np.flatnonzero(chunk_selected)
        processed += chunk_size
        if rows.size == 0:
            continue

        samples_a = ak.values_astype(chunk.samples[0][rows], np.int16)
        samples_b = ak.values_astype(chunk.samples[1][rows], np.int16)
        count_a = np.asarray(ak.to_numpy(ak.num(samples_a, axis=1)), dtype=np.int32)
        count_b = np.asarray(ak.to_numpy(ak.num(samples_b, axis=1)), dtype=np.int32)
        number = int(rows.size)
        new_indices = np.arange(global_event, global_event + number, dtype=np.int64)

        tree.extend(
            {
                "event_index": new_indices,
                "event_id": new_indices.copy(),
                "source_run_index": np.full(number, selection.run_index, dtype=np.int32),
                "bias_voltage_V": np.full(
                    number, selection.bias_voltage_V, dtype=np.float64
                ),
                "original_event_index": np.asarray(
                    chunk.event_index[rows], dtype=np.int64
                ),
                "original_event_id": np.asarray(chunk.event_id[rows], dtype=np.int64),
                # Assign one globally unique source pair per processed run. This makes
                # split.strategy="source_file" group complete runs instead of mixing them.
                "source_file_id": np.tile(
                    np.asarray(
                        [2 * selection.run_index, 2 * selection.run_index + 1],
                        dtype=np.int64,
                    ),
                    (number, 1),
                ),
                "original_source_file_id": np.asarray(
                    chunk.source_file_id[rows], dtype=np.int64
                ),
                "sample_count": np.stack([count_a, count_b], axis=1),
                "vertical_gain_v_per_count": np.asarray(
                    chunk.vertical_gain_v_per_count[rows], dtype=np.float64
                ),
                "vertical_offset_v": np.asarray(
                    chunk.vertical_offset_v[rows], dtype=np.float64
                ),
                "horizontal_interval_s": np.asarray(
                    chunk.horizontal_interval_s[rows], dtype=np.float64
                ),
                "horizontal_offset_s": np.asarray(
                    chunk.horizontal_offset_s[rows], dtype=np.float64
                ),
                "amplitude_mV": np.asarray(chunk_amplitudes[rows], dtype=np.float32),
                "samples_ch1": samples_a,
                "samples_ch2": samples_b,
            }
        )
        global_event += number

    expected = selection.selected_events
    written = global_event - first_global_event
    if written != expected:
        raise RuntimeError(
            f"Wrote {written} selected events from {selection.path}, expected {expected}"
        )
    logger.info("Appended run %d: %d selected events", selection.run_index, written)
    return global_event


def _write_runs_tree(root_file: Any, selections: Sequence[RunSelection]) -> None:
    schema = {
        "source_run_index": "int32",
        "bias_voltage_V": "float64",
        "total_events": "int64",
        "selected_events": "int64",
        "selected_fraction": "float64",
        "input_energy_channels": "2 * int32",
        "photopeak_mean_mV": "2 * float64",
        "photopeak_sigma_mV": "2 * float64",
        "selection_low_mV": "2 * float64",
        "selection_high_mV": "2 * float64",
        "photopeak_chi2_ndof": "2 * float64",
    }
    tree = root_file.mktree("runs", schema)
    if not selections:
        return
    tree.extend(
        {
            "source_run_index": np.asarray(
                [item.run_index for item in selections], dtype=np.int32
            ),
            "bias_voltage_V": np.asarray(
                [item.bias_voltage_V for item in selections], dtype=np.float64
            ),
            "total_events": np.asarray(
                [item.total_events for item in selections], dtype=np.int64
            ),
            "selected_events": np.asarray(
                [item.selected_events for item in selections], dtype=np.int64
            ),
            "selected_fraction": np.asarray(
                [item.selected_events / max(1, item.total_events) for item in selections],
                dtype=np.float64,
            ),
            "input_energy_channels": np.asarray(
                [[result.channel for result in item.photopeak_results] for item in selections],
                dtype=np.int32,
            ),
            "photopeak_mean_mV": np.asarray(
                [[result.mean_mV for result in item.photopeak_results] for item in selections],
                dtype=np.float64,
            ),
            "photopeak_sigma_mV": np.asarray(
                [[result.sigma_mV for result in item.photopeak_results] for item in selections],
                dtype=np.float64,
            ),
            "selection_low_mV": np.asarray(
                [
                    [result.selection_low_mV for result in item.photopeak_results]
                    for item in selections
                ],
                dtype=np.float64,
            ),
            "selection_high_mV": np.asarray(
                [
                    [result.selection_high_mV for result in item.photopeak_results]
                    for item in selections
                ],
                dtype=np.float64,
            ),
            "photopeak_chi2_ndof": np.asarray(
                [[result.chi2_ndof for result in item.photopeak_results] for item in selections],
                dtype=np.float64,
            ),
        }
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _manifest(
    input_folder: Path,
    output_path: Path,
    config: dict[str, Any],
    selections: Sequence[RunSelection],
) -> dict[str, Any]:
    return {
        "format": "energy-photopeak-concatenation",
        "format_version": FORMAT_VERSION,
        "input_folder": str(input_folder.resolve()),
        "output_root": str(output_path.resolve()),
        "number_of_runs": len(selections),
        "total_input_events": int(sum(item.total_events for item in selections)),
        "total_selected_events": int(sum(item.selected_events for item in selections)),
        "channels": config["channels"],
        "selection_rule": (
            "Independent Gaussian photopeak fit for each input ROOT file and each "
            "energy channel; output keeps the logical AND of the two per-run windows."
        ),
        "source_file_id_rule": (
            "The output source_file_id pair is unique for each input run so that the "
            "ML source_file split can keep complete runs together. The original pair "
            "is retained in original_source_file_id."
        ),
        "config": config,
        "runs": [
            {
                "source_run_index": item.run_index,
                "source_path": str(item.path),
                "source_size_bytes": item.path.stat().st_size,
                "source_mtime_ns": item.path.stat().st_mtime_ns,
                "bias_voltage_V": item.bias_voltage_V,
                "total_events": item.total_events,
                "selected_events": item.selected_events,
                "selected_fraction": item.selected_events / max(1, item.total_events),
                "photopeak": [result.as_dict() for result in item.photopeak_results],
            }
            for item in selections
        ],
    }


def concatenate_energy_photopeak_runs(
    config: dict[str, Any],
    *,
    input_folder_override: str | Path | None = None,
    output_root_override: str | Path | None = None,
    overwrite_override: bool | None = None,
    logger: logging.Logger | None = None,
) -> tuple[Path, Path]:
    validate_concatenation_config(config)
    logger = logger or logging.getLogger(__name__)

    input_folder = Path(
        input_folder_override
        if input_folder_override is not None
        else config["input"]["folder"]
    ).resolve()
    output_path = Path(
        output_root_override
        if output_root_override is not None
        else config["output"]["root_file"]
    ).resolve()
    manifest_path = Path(
        config["output"].get("manifest_json", str(output_path.with_suffix(".json")))
    ).resolve()
    overwrite = (
        bool(overwrite_override)
        if overwrite_override is not None
        else bool(config["output"].get("overwrite", False))
    )

    if not overwrite:
        existing = [str(path) for path in (output_path, manifest_path) if path.exists()]
        if existing:
            raise FileExistsError(
                "Output already exists; enable overwrite or choose another path: "
                + ", ".join(existing)
            )

    input_config = config["input"]
    files = discover_input_files(
        input_folder,
        pattern=str(input_config.get("pattern", "*.root")),
        recursive=bool(input_config.get("recursive", True)),
        excluded_paths=(output_path, output_path.with_suffix(output_path.suffix + ".partial")),
        excluded_names=input_config.get("exclude_names", []),
    )
    if not files:
        raise FileNotFoundError(
            f"No files matching {input_config.get('pattern', '*.root')!r} under {input_folder}"
        )
    logger.info("Discovered %d input ROOT files under %s", len(files), input_folder)

    selections: list[RunSelection] = []
    for discovered_index, path in enumerate(files):
        logger.info("[%d/%d] Processing %s", discovered_index + 1, len(files), path)
        selection = select_run_photopeak(
            path,
            run_index=len(selections),
            config=config,
            logger=logger,
        )
        if selection is not None:
            selections.append(selection)

    if not selections:
        raise RuntimeError("No input run passed the independent photopeak selection")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    partial_root = output_path.with_suffix(output_path.suffix + ".partial")
    partial_manifest = manifest_path.with_suffix(manifest_path.suffix + ".partial")
    for path in (partial_root, partial_manifest):
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

    try:
        with uproot.recreate(partial_root) as root_file:
            events_tree = root_file.mktree("events", _event_tree_schema())
            global_event = 0
            for selection in selections:
                global_event = _write_selected_run(
                    events_tree,
                    selection,
                    config,
                    first_global_event=global_event,
                    logger=logger,
                )
            _write_runs_tree(root_file, selections)
            metadata = root_file.mktree(
                "metadata",
                {
                    "format_version": "int32",
                    "number_of_runs": "int32",
                    "number_of_events": "int64",
                    "energy_channels_only": "bool",
                },
            )
            metadata.extend(
                {
                    "format_version": np.asarray([FORMAT_VERSION], dtype=np.int32),
                    "number_of_runs": np.asarray([len(selections)], dtype=np.int32),
                    "number_of_events": np.asarray([global_event], dtype=np.int64),
                    "energy_channels_only": np.asarray([True], dtype=np.bool_),
                }
            )

        manifest = _json_safe(_manifest(input_folder, output_path, config, selections))
        with partial_manifest.open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

        if overwrite:
            output_path.unlink(missing_ok=True)
            manifest_path.unlink(missing_ok=True)
        partial_root.replace(output_path)
        partial_manifest.replace(manifest_path)
    except Exception:
        partial_root.unlink(missing_ok=True)
        partial_manifest.unlink(missing_ok=True)
        raise

    logger.info(
        "Concatenated %d runs and %d selected events into %s",
        len(selections),
        sum(item.selected_events for item in selections),
        output_path,
    )
    logger.info("Run/photopeak manifest written to %s", manifest_path)
    return output_path, manifest_path
