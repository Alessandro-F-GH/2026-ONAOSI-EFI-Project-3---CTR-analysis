from __future__ import annotations

import copy
import hashlib
import os
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.format import open_memmap
from scipy.signal import butter, sosfiltfilt

from utils.photopeak import fit_photopeak, photopeak_mask
from utils.signal import INVALID_TIME_FS

from .common import atomic_json, canonical_hash, read_json, source_signature
if TYPE_CHECKING:
    from .data import EnergyCache
from .dataset import DATASET_FORMAT_VERSION, PreparedDataset, load_prepared_dataset
from .selection_store import load_or_compute_selection, selection_request_fingerprint

PREPARED_SELECTION_VERSION = 2

_COPY_ARRAYS = (
    "event_id",
    "event_index",
    "source_file_id",
    "source_run_index",
    "bias_voltage_V",
    "amplitude_mV",
    "noise_rms_mV",
    "trigger_index",
    "windows_mV",
    "energy_led_time_fs",
    "timing_led_time_fs",
    "energy_cfd_time_fs",
    "timing_cfd_time_fs",
    "energy_window_anchor_time_fs",
    "timing_aligned_energy_window_anchor_time_fs",
    "timing_window_anchor_time_fs",
    "timing_aligned_energy_windows_mV",
    "timing_windows_mV",
)



def _preparation_request_fingerprint(study: dict[str, Any], root_file: Path) -> str:
    """Hash only inputs that change the canonical prepared signal representation.

    Experiment windows, models, validation settings and true TOF are intentionally
    excluded.  They are runtime views/evaluation metadata and must not trigger
    another ROOT/photopeak pass.
    """
    preprocessing = copy.deepcopy(study["preprocessing"])
    for key in (
        "prepared_dir", "selection_store_dir", "cleanup_raw_cache",
        "materialization_chunk_size", "parallelization", "input_variants",
        "input_variant_by_channel", "subsampling_factors",
    ):
        preprocessing.pop(key, None)
    io = preprocessing.get("io")
    if isinstance(io, dict):
        preprocessing["io"] = {"max_events": int(io.get("max_events", 0))}
    return canonical_hash({
        "format_version": DATASET_FORMAT_VERSION,
        "selection_version": PREPARED_SELECTION_VERSION,
        "source": source_signature(root_file),
        "channels": study["data"]["channels"],
        "materialized_window_ns": preprocessing.get("materialized_window_ns"),
        "preprocessing": preprocessing,
    })

def _hash_indices(indices: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(indices, dtype=np.int64).tobytes()).hexdigest()


def _copy_selected(source: np.ndarray, selected: np.ndarray, path: Path, chunk_size: int) -> None:
    shape = (int(selected.size),) + tuple(int(v) for v in source.shape[1:])
    target = open_memmap(path, mode="w+", dtype=source.dtype, shape=shape)
    for start in range(0, selected.size, chunk_size):
        idx = selected[start : start + chunk_size]
        target[start : start + idx.size] = np.asarray(source[idx])
    target.flush()
    mmap = getattr(target, "_mmap", None)
    if mmap is not None:
        mmap.close()


def _timing_pair_ps(times_fs: np.ndarray) -> np.ndarray:
    values = np.asarray(times_fs, dtype=np.int64)
    return (values[:, 0].astype(np.float64) - values[:, 1].astype(np.float64)) / 1000.0


def _physical_photopeak_selection(
    cache: "EnergyCache", config: dict[str, Any], logger: Any
) -> tuple[np.ndarray, dict[str, Any]]:
    """Cuts that define the reusable physical/photopeak cohort only.

    Timing validity and LED mismatch rejection are deliberately excluded so LED/CFD
    threshold changes never force another photopeak fit.
    """
    amplitudes_all = np.asarray(cache.amplitude_mV, dtype=np.float64)
    noise_all = np.asarray(cache.noise_rms_mV, dtype=np.float64)
    trigger_all = np.asarray(cache.trigger_index, dtype=np.int64)
    valid = (
        np.all(np.isfinite(amplitudes_all), axis=1)
        & np.all(np.isfinite(noise_all), axis=1)
        & np.all(trigger_all >= 0, axis=1)
    )
    selection = copy.deepcopy(config.get("selection", {}))
    trigger_range = selection.get("energy_trigger_index_range")
    if trigger_range is not None:
        low, high = int(trigger_range[0]), int(trigger_range[1])
        triggers = np.asarray(cache.trigger_index)
        valid &= np.all((triggers > low) & (triggers < high), axis=1)

    noise_limit = selection.get("energy_noise_max_mV")
    if noise_limit is not None:
        if isinstance(noise_limit, (list, tuple)):
            limits = np.asarray(noise_limit, dtype=np.float64).reshape(-1)
            if limits.size != 2:
                raise ValueError("preprocessing.selection.energy_noise_max_mV must be scalar or length 2")
        else:
            limits = np.asarray([float(noise_limit), float(noise_limit)], dtype=np.float64)
        noise = np.asarray(cache.noise_rms_mV, dtype=np.float64)
        valid &= (noise[:, 0] < limits[0]) & (noise[:, 1] < limits[1])

    summary: dict[str, Any] = {
        "scope": "physical_photopeak_before_timing_and_ml",
        "valid_before_photopeak": int(np.count_nonzero(valid)),
    }
    photopeak_cfg = copy.deepcopy(config.get("photopeak", {"enabled": False}))
    photopeak_rows: list[dict[str, Any]] = []
    if bool(photopeak_cfg.get("enabled", False)):
        amplitudes = np.asarray(cache.amplitude_mV, dtype=np.float64)
        for channel_position, channel_number in enumerate(cache.manifest["energy_channels_one_based"]):
            fit_indices = np.flatnonzero(valid)
            result = fit_photopeak(
                amplitudes[fit_indices, channel_position],
                channel=int(channel_number),
                config=photopeak_cfg,
            )
            if not result.success:
                raise RuntimeError(f"Photopeak fit failed for energy channel {channel_number}: {result.message}")
            valid &= photopeak_mask(amplitudes[:, channel_position], result)
            photopeak_rows.append(result.as_dict())
        logger.info("Physical photopeak selection | retained=%d", int(np.count_nonzero(valid)))
    summary["photopeak"] = photopeak_rows
    selected = np.flatnonzero(valid).astype(np.int64)
    summary["selected_events"] = int(selected.size)
    return selected, summary


def _dataset_level_selection(
    cache: "EnergyCache",
    config: dict[str, Any],
    logger: Any,
    *,
    physical_selected: np.ndarray | None = None,
    physical_summary: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    valid = np.zeros(int(cache.event_id.size), dtype=bool)
    if physical_selected is None:
        physical_selected, physical_summary = _physical_photopeak_selection(cache, config, logger)
    valid[np.asarray(physical_selected, dtype=np.int64)] = True

    # One frozen population for all standard and ML modes. Timing validity is
    # applied after the reusable photopeak cohort because it depends on LED/CFD
    # configuration rather than the physical energy population.
    valid &= np.asarray(cache.valid, dtype=bool)
    for name in ("energy_led_time_fs", "energy_cfd_time_fs"):
        array = getattr(cache, name, None)
        if array is not None:
            valid &= np.all(np.asarray(array) != INVALID_TIME_FS, axis=1)
    if cache.timing_led_time_fs is not None:
        valid &= np.all(np.asarray(cache.timing_led_time_fs) != INVALID_TIME_FS, axis=1)
    if cache.timing_cfd_time_fs is not None:
        valid &= np.all(np.asarray(cache.timing_cfd_time_fs) != INVALID_TIME_FS, axis=1)

    selection = copy.deepcopy(config.get("selection", {}))
    summary: dict[str, Any] = {
        "scope": "complete_file_before_any_ml_split",
        "physical_selection": physical_summary or {},
        "valid_after_timing": int(np.count_nonzero(valid)),
    }

    led_cfg = selection.get("led_outlier_rejection", {}) or {}
    led_summary: dict[str, Any] = {"enabled": bool(led_cfg.get("enabled", False))}
    if led_summary["enabled"]:
        use_robust_z = "zscore_limit" in led_cfg
        zscore_limit = float(led_cfg.get("zscore_limit", 4.0))
        max_distance_ps = float(led_cfg.get("max_distance_ps", 300.0))
        families: list[tuple[str, np.ndarray]] = [("energy", np.asarray(cache.energy_led_time_fs))]
        if cache.timing_led_time_fs is not None:
            families.append(("timing", np.asarray(cache.timing_led_time_fs)))
        family_rows: list[dict[str, Any]] = []
        for name, times in families:
            delta = _timing_pair_ps(times)
            base = valid & np.isfinite(delta)
            if np.count_nonzero(base) < 3:
                raise RuntimeError(f"Too few valid {name} LED pairs for dataset-level outlier rejection")
            center = float(np.median(delta[base]))
            if use_robust_z:
                mad = float(np.median(np.abs(delta[base] - center)))
                sigma = 1.4826 * mad
                if not np.isfinite(sigma) or sigma <= 0.0:
                    sigma = float(np.std(delta[base], ddof=1))
                if not np.isfinite(sigma) or sigma <= 0.0:
                    sigma = 1.0
                distance = zscore_limit * sigma
                accepted = np.isfinite(delta) & (np.abs(delta - center) <= distance)
            else:
                sigma = float("nan")
                distance = max_distance_ps
                accepted = np.isfinite(delta) & (np.abs(delta - center) <= distance)
            rejected = int(np.count_nonzero(valid & ~accepted))
            valid &= accepted
            row = {
                "family": name, "median_ps": center, "rejected": rejected,
                "effective_max_distance_ps": float(distance),
            }
            if use_robust_z:
                row.update({"zscore_limit": zscore_limit, "robust_sigma_ps": sigma})
            else:
                row["max_distance_ps"] = max_distance_ps
            family_rows.append(row)
        led_summary["families"] = family_rows
        logger.info(
            "Dataset LED mismatch rejection | %s",
            ", ".join(f"{row['family']} rejected={row['rejected']}" for row in family_rows),
        )
    summary["led_outlier_rejection"] = led_summary

    minimum = int(selection.get("minimum_events", selection.get("minimum_events_per_split", 100)))
    selected = np.flatnonzero(valid).astype(np.int64)
    summary["selected_events"] = int(selected.size)
    if selected.size < minimum:
        raise RuntimeError(f"Only {selected.size} events remain after dataset preparation; need {minimum}")
    return selected, summary


def _denoise_windows(
    source: np.ndarray,
    destination: Path,
    *,
    relative_time_ps: np.ndarray,
    config: dict[str, Any],
    chunk_size: int,
) -> None:
    values = source
    if values.ndim != 3:
        raise ValueError("Waveform array must have shape [event, detector, sample]")
    times = np.asarray(relative_time_ps, dtype=np.float64)
    if times.size < 2:
        raise ValueError("Need at least two time samples for denoising")
    interval_s = float(np.median(np.diff(times))) * 1e-12
    fs = 1.0 / interval_s
    cutoff_hz = float(config["cutoff_GHz"]) * 1e9
    if not 0.0 < cutoff_hz < 0.5 * fs:
        raise ValueError("Denoising cutoff must be below Nyquist")
    order = int(config.get("order", 4))
    sos = butter(order, cutoff_hz, btype="lowpass", fs=fs, output="sos")
    target = open_memmap(destination, mode="w+", dtype=np.float32, shape=values.shape)
    for start in range(0, values.shape[0], chunk_size):
        stop = min(start + chunk_size, values.shape[0])
        block = np.asarray(values[start:stop], dtype=np.float64)
        zero_count = min(int(np.count_nonzero(sos[:, 2] == 0.0)), int(np.count_nonzero(sos[:, 5] == 0.0)))
        default_padlen = 3 * (2 * int(sos.shape[0]) + 1 - zero_count)
        padlen = min(default_padlen, max(0, block.shape[-1] - 1))
        filtered = sosfiltfilt(sos, block, axis=-1, padlen=padlen)
        target[start:stop] = np.asarray(filtered, dtype=np.float32)
    target.flush()
    mmap = getattr(target, "_mmap", None)
    if mmap is not None:
        mmap.close()


def _prepared_fingerprint(cache: "EnergyCache", selected: np.ndarray, config: dict[str, Any]) -> str:
    return canonical_hash({
        "format_version": DATASET_FORMAT_VERSION,
        "selection_version": PREPARED_SELECTION_VERSION,
        "raw_cache": cache.manifest["fingerprint"],
        "selected_hash": _hash_indices(selected),
        "selection": config.get("selection", {}),
        "denoising": config.get("denoising", {}),
    })


def materialize_selected_dataset(
    cache: "EnergyCache",
    *,
    output: Path,
    config: dict[str, Any],
    rebuild: bool,
    logger: Any,
) -> PreparedDataset:
    selection_store_root = Path(config.get("selection_store_dir", output.parent / "selected_events"))
    source_root = Path(config["source_root"])
    selection_fp = str(config.get("selection_request_fingerprint", ""))
    if not selection_fp:
        selection_fp = canonical_hash({
            "source": source_signature(source_root),
            "energy_channels": cache.manifest.get("energy_channels_one_based", []),
            "selection": {k: v for k, v in dict(config.get("selection", {})).items() if k != "led_outlier_rejection"},
            "photopeak": config.get("photopeak", {"enabled": False}),
        })
    physical_selected, physical_summary, physical_store = load_or_compute_selection(
        root=selection_store_root, root_file=source_root, fingerprint=selection_fp,
        rebuild=bool(config.get("rebuild_selection", False)),
        compute=lambda: _physical_photopeak_selection(cache, config, logger),
        logger=logger,
    )
    selected, selection_summary = _dataset_level_selection(
        cache, config, logger, physical_selected=physical_selected,
        physical_summary=physical_summary,
    )
    selection_summary["physical_selection_store"] = str(physical_store)
    fingerprint = _prepared_fingerprint(cache, selected, config)
    manifest_path = output / "manifest.json"
    if output.is_dir() and not rebuild and manifest_path.is_file():
        try:
            manifest = read_json(manifest_path)
            if manifest.get("fingerprint") == fingerprint:
                logger.info("Reusing permanent prepared dataset: %s", output)
                return load_prepared_dataset(output)
        except Exception:
            pass

    temporary = output.with_name(output.name + ".building")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, exist_ok=True)
    chunk_size = max(1, int(config.get("materialization_chunk_size", 2048)))

    for name in _COPY_ARRAYS:
        source = getattr(cache, name, None)
        if source is not None:
            _copy_selected(source, selected, temporary / f"{name}.npy", chunk_size)
    np.save(temporary / "relative_time_ps.npy", np.asarray(cache.relative_time_ps, dtype=np.float64))
    if cache.timing_relative_time_ps is not None:
        np.save(temporary / "timing_relative_time_ps.npy", np.asarray(cache.timing_relative_time_ps, dtype=np.float64))

    denoise_cfg = copy.deepcopy(config.get("denoising", {}))
    denoise_enabled = bool(denoise_cfg.get("enabled", False))
    if denoise_enabled:
        _denoise_windows(
            np.load(temporary / "windows_mV.npy", mmap_mode="r"),
            temporary / "denoised_windows_mV.npy",
            relative_time_ps=np.asarray(cache.relative_time_ps),
            config=denoise_cfg,
            chunk_size=chunk_size,
        )
        aligned = temporary / "timing_aligned_energy_windows_mV.npy"
        if aligned.is_file():
            _denoise_windows(
                np.load(aligned, mmap_mode="r"),
                temporary / "denoised_timing_aligned_energy_windows_mV.npy",
                relative_time_ps=np.asarray(cache.relative_time_ps),
                config=denoise_cfg,
                chunk_size=chunk_size,
            )

    manifest = {
        "format_version": DATASET_FORMAT_VERSION,
        "fingerprint": fingerprint,
        "request_fingerprint": str(config.get("request_fingerprint", "")),
        "name": str(config.get("name", output.name)),
        "role": "prepared_full_file",
        "subset_kind": "dataset_level_selected",
        "source_root": str(config["source_root"]),
        "true_tof_ps": float(config["true_tof_ps"]),
        "event_count": int(selected.size),
        "input_length": int(cache.windows_mV.shape[-1]),
        "selection": selection_summary,
        "raw_cache_manifest": cache.manifest,
        "energy_channels_one_based": cache.manifest.get("energy_channels_one_based", []),
        "timing_channel_waveforms_saved": cache.timing_windows_mV is not None,
        "timing_aligned_energy_waveforms_saved": cache.timing_aligned_energy_windows_mV is not None,
        "denoised_waveforms_saved": denoise_enabled,
        "denoised_energy_waveforms_saved": denoise_enabled,
        "denoised_timing_waveforms_saved": False,
        "denoising_scope": "energy_channels_only",
        "denoising": denoise_cfg if denoise_enabled else {"enabled": False},
        "waveform_grid": cache.manifest.get("waveform_grid", "native_samples"),
        "native_sample_interval_ps": cache.manifest.get("native_sample_interval_ps"),
        "timing_native_sample_interval_ps": cache.manifest.get("timing_native_sample_interval_ps"),
        "led_timestamp_source": "energy_channels",
        "cfd_timestamp_source": "energy_channels",
        "ml_window_alignment_source": "target_specific_led",
        "window_anchor_timestamps_saved": True,
        "correction_target_reference": "interpolated_led_direct",
        "window_anchor_shift_factored": False,
        "dataset_selection_is_independent_of_ml_split": True,
        "final_evaluation_rejects_no_additional_events": True,
        "arrays_are_post_selection": True,
        "ml_split_materialized": False,
    }
    atomic_json(temporary / "manifest.json", manifest)
    if output.exists():
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, output)
    logger.info("Permanent prepared dataset written | %s | events=%d", output, selected.size)
    return load_prepared_dataset(output)


def _raw_preprocess_config(study: dict[str, Any], root_file: Path, cache_dir: Path) -> dict[str, Any]:
    preprocessing = copy.deepcopy(study["preprocessing"])
    common = copy.deepcopy(preprocessing.get("common", {}))
    energy = copy.deepcopy(common)
    energy.update(copy.deepcopy(preprocessing.get("energy", {})))
    timing = copy.deepcopy(common)
    timing.update(copy.deepcopy(preprocessing.get("timing", {})))
    # Denoising is intentionally excluded from ROOT conversion. LED/CFD and the
    # canonical raw windows therefore never depend on an ML denoising candidate.
    energy["denoising"] = {"enabled": False}
    timing["denoising"] = {"enabled": False}
    materialized = preprocessing.get("materialized_window_ns") or {
        "before": max(float(window["before_ns"]) for window in study["windows_ns"]),
        "after": max(float(window["after_ns"]) for window in study["windows_ns"]),
    }
    energy["ml_window_ns"] = {
        "before": float(materialized["before"]), "after": float(materialized["after"])
    }
    timing["ml_window_ns"] = {
        "before": float(materialized["before"]), "after": float(materialized["after"])
    }
    timing["enabled"] = True
    energy["timing_channel_led"] = timing
    return {
        "data": {"input_root": str(root_file), "true_tof_ps": float(study["data"].get("true_tof_ps", 0.0))},
        "channels": copy.deepcopy(study["data"]["channels"]),
        "waveform": energy,
        "io": copy.deepcopy(preprocessing.get("io", {"step_size": "128 MB", "max_events": 0, "progress_every": 1000})),
        "parallelization": copy.deepcopy(preprocessing.get("parallelization", {"preprocessing_backend": "process", "preprocessing_workers": 0, "preprocessing_chunksize": 8})),
        "cache": {"raw_cache_dir": str(cache_dir)},
    }


def prepare_file_dataset(
    study: dict[str, Any],
    root_file: Path,
    *,
    rebuild: bool,
    logger: Any,
) -> PreparedDataset:
    root_id = root_file.stem
    prepared_root = Path(study["preprocessing"]["prepared_dir"])
    output = prepared_root / root_id
    request_fingerprint = _preparation_request_fingerprint(study, root_file)
    if output.is_dir() and not rebuild:
        manifest_path = output / "manifest.json"
        if manifest_path.is_file():
            try:
                manifest = read_json(manifest_path)
                if manifest.get("request_fingerprint") == request_fingerprint:
                    logger.info("Reusing permanent prepared dataset without ROOT reconversion: %s", output)
                    loaded = load_prepared_dataset(output)
                    manifest = dict(loaded.manifest)
                    manifest["true_tof_ps"] = float(study["data"].get("true_tof_ps", 0.0))
                    return replace(loaded, manifest=manifest)
            except Exception as exc:
                logger.warning("Cannot reuse permanent prepared dataset %s: %s", output, exc)

    raw_cache_dir = prepared_root / ".raw_cache" / root_id
    raw_cfg = _raw_preprocess_config(study, root_file, raw_cache_dir)
    cache_cfg = {
        "channels": raw_cfg["channels"],
        "waveform": raw_cfg["waveform"],
        "io": raw_cfg["io"],
        "parallelization": raw_cfg["parallelization"],
    }
    from .data import prepare_energy_cache

    cache = prepare_energy_cache(
        root_file,
        raw_cache_dir,
        cache_cfg,
        rebuild=rebuild,
        logger=logger,
    )
    permanent_cfg = {
        "name": root_id,
        "source_root": str(root_file),
        "request_fingerprint": request_fingerprint,
        "true_tof_ps": float(study["data"].get("true_tof_ps", 0.0)),
        "selection_store_dir": str(study["preprocessing"].get("selection_store_dir", Path(study["preprocessing"]["prepared_dir"]).parent / "selected_events")),
        "selection_request_fingerprint": selection_request_fingerprint(
            root_file=root_file, channels=study["data"]["channels"],
            preprocessing=study["preprocessing"],
        ),
        "rebuild_selection": bool(rebuild),
        "selection": copy.deepcopy(study["preprocessing"].get("selection", {})),
        "photopeak": copy.deepcopy(study["preprocessing"].get("photopeak", {"enabled": False})),
        "denoising": copy.deepcopy(study["preprocessing"].get("denoising", {"enabled": False})),
        "materialization_chunk_size": int(study["preprocessing"].get("materialization_chunk_size", 2048)),
    }
    dataset = materialize_selected_dataset(
        cache, output=output, config=permanent_cfg, rebuild=rebuild, logger=logger
    )
    if bool(study["preprocessing"].get("cleanup_raw_cache", True)):
        # Close source memmaps before deleting their directory (important on Windows).
        del cache
        shutil.rmtree(raw_cache_dir, ignore_errors=True)
    manifest = dict(dataset.manifest)
    manifest["true_tof_ps"] = float(study["data"].get("true_tof_ps", 0.0))
    return replace(dataset, manifest=manifest)


def plot_prepared_signal_examples(
    dataset: PreparedDataset,
    destination: Path,
    *,
    dpi: int = 180,
) -> None:
    if dataset.event_id.size == 0:
        return
    rows: list[tuple[str, np.ndarray, np.ndarray, np.ndarray | None]] = [
        ("Energy ch. 1", dataset.relative_time_ps, np.asarray(dataset.windows_mV[0, 0]),
         None if dataset.denoised_windows_mV is None else np.asarray(dataset.denoised_windows_mV[0, 0])),
        ("Energy ch. 2", dataset.relative_time_ps, np.asarray(dataset.windows_mV[0, 1]),
         None if dataset.denoised_windows_mV is None else np.asarray(dataset.denoised_windows_mV[0, 1])),
    ]
    if dataset.timing_windows_mV is not None and dataset.timing_relative_time_ps is not None:
        rows.extend([
            ("Timing ch. 1", dataset.timing_relative_time_ps, np.asarray(dataset.timing_windows_mV[0, 0]),
             None if dataset.denoised_timing_windows_mV is None else np.asarray(dataset.denoised_timing_windows_mV[0, 0])),
            ("Timing ch. 2", dataset.timing_relative_time_ps, np.asarray(dataset.timing_windows_mV[0, 1]),
             None if dataset.denoised_timing_windows_mV is None else np.asarray(dataset.denoised_timing_windows_mV[0, 1])),
        ])
    fig, axes = plt.subplots(len(rows), 1, figsize=(10.5, 2.7 * len(rows)), squeeze=False)
    for axis, (title, time_ps, raw, denoised) in zip(axes[:, 0], rows):
        axis.plot(np.asarray(time_ps, dtype=np.float64) / 1000.0, raw, linewidth=1.0, label="raw")
        if denoised is not None:
            axis.plot(np.asarray(time_ps, dtype=np.float64) / 1000.0, denoised, linewidth=1.0, label="denoised")
            axis.legend(loc="best")
        axis.set_title(title)
        axis.set_xlabel("Time relative to native LED anchor [ns]")
        axis.set_ylabel("Voltage [mV]")
        axis.minorticks_on()
        axis.grid(True, which="major", alpha=0.35)
        axis.grid(True, which="minor", alpha=0.15)
    fig.suptitle(f"Prepared waveform example | {Path(dataset.manifest.get('source_root', dataset.directory)).name}")
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def input_variant_dataset_view(dataset: PreparedDataset, variant: str) -> PreparedDataset:
    """Return a zero-copy raw/denoised waveform view of one prepared dataset."""
    from dataclasses import replace

    key = str(variant).strip().lower()
    if key == "raw":
        manifest = dict(dataset.manifest)
        manifest["ml_input_variant"] = "raw"
        return replace(dataset, manifest=manifest)
    if key != "denoised":
        raise ValueError("ML input variant must be 'raw' or 'denoised'")
    if dataset.denoised_windows_mV is None:
        raise ValueError(
            f"Dataset {dataset.directory} has no materialized denoised waveforms"
        )
    manifest = dict(dataset.manifest)
    manifest["ml_input_variant"] = "denoised"
    return replace(
        dataset,
        manifest=manifest,
        windows_mV=dataset.denoised_windows_mV,
        timing_aligned_energy_windows_mV=(
            dataset.denoised_timing_aligned_energy_windows_mV
            if dataset.denoised_timing_aligned_energy_windows_mV is not None
            else dataset.timing_aligned_energy_windows_mV
        ),
        timing_windows_mV=(
            dataset.denoised_timing_windows_mV
            if dataset.denoised_timing_windows_mV is not None
            else dataset.timing_windows_mV
        ),
    )
