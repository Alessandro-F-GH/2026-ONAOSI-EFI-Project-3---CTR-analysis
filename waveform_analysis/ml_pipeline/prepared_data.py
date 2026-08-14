from __future__ import annotations

import copy
import hashlib
import os
import shutil
from pathlib import Path
from typing import Any, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.format import open_memmap

from utils.signal import INVALID_TIME_FS

from .common import atomic_json, canonical_hash, read_json, source_signature
if TYPE_CHECKING:
    from .data import EnergyCache
from .dataset import DATASET_FORMAT_VERSION, PreparedDataset, load_prepared_dataset

PREPARED_SELECTION_VERSION = 5

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
    "raw_energy_led_time_fs",
    "raw_timing_led_time_fs",
    "raw_energy_cfd_time_fs",
    "raw_timing_cfd_time_fs",
    "raw_energy_window_anchor_time_fs",
    "raw_timing_aligned_energy_window_anchor_time_fs",
    "raw_timing_window_anchor_time_fs",
    "raw_energy_windows_mV",
    "raw_timing_aligned_energy_windows_mV",
    "raw_timing_windows_mV",
)



def _preparation_request_fingerprint(study: dict[str, Any], root_file: Path) -> str:
    """Hash only inputs that can change the permanent prepared dataset.

    This can be computed without opening the ROOT file, so a valid permanent
    dataset is reusable even when the transient raw conversion cache was deleted.
    """
    preprocessing = copy.deepcopy(study["preprocessing"])
    for key in ("prepared_dir", "cleanup_raw_cache", "materialization_chunk_size", "parallelization"):
        preprocessing.pop(key, None)
    # I/O chunk size/progress do not change data, while max_events does.
    io = preprocessing.get("io")
    if isinstance(io, dict):
        preprocessing["io"] = {"max_events": int(io.get("max_events", 0))}
    return canonical_hash({
        "format_version": DATASET_FORMAT_VERSION,
        "selection_version": PREPARED_SELECTION_VERSION,
        "source": source_signature(root_file),
        "channels": study["data"]["channels"],
        "true_tof_ps": float(study["data"].get("true_tof_ps", 0.0)),
        "windows_ns": study["windows_ns"],
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


def robust_led_zscore_mask(
    delta_ps: np.ndarray,
    *,
    zscore_limit: float,
    base_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float, str]:
    """Return a robust LED-pair outlier mask and its fitted location/scale.

    The z-score is |delta - median| / (1.4826 * MAD).  Standard deviation is
    used only when MAD is degenerate.  ``base_mask`` defines the population used
    to fit median/scale, while the returned mask is defined for every element.
    """
    values = np.asarray(delta_ps, dtype=np.float64).reshape(-1)
    if not np.isfinite(zscore_limit) or zscore_limit <= 0.0:
        raise ValueError("zscore_limit must be finite and > 0")
    base = np.isfinite(values)
    if base_mask is not None:
        supplied = np.asarray(base_mask, dtype=bool).reshape(-1)
        if supplied.shape != values.shape:
            raise ValueError("base_mask shape does not match LED differences")
        base &= supplied
    if np.count_nonzero(base) < 3:
        raise RuntimeError("Too few valid LED pairs for robust z-score estimation")
    sample = values[base]
    center = float(np.median(sample))
    mad = float(np.median(np.abs(sample - center)))
    scale = 1.482602218505602 * mad
    source = "mad"
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.std(sample, ddof=0))
        source = "std_fallback"
    if not np.isfinite(scale) or scale <= 0.0:
        z = np.zeros(values.shape, dtype=np.float64)
        z[~np.isfinite(values)] = np.inf
        scale = 0.0
        source = "degenerate"
    else:
        z = np.abs(values - center) / scale
    return np.isfinite(values) & (z <= float(zscore_limit)), center, scale, source


def _dataset_level_selection(cache: "EnergyCache", config: dict[str, Any], logger: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply only timing-validity/mismatch cuts after photopeak preselection.

    Raw-energy trigger/noise/photopeak selection has already happened before any
    denoising, LED/CFD extraction, or window materialization.  Repeating it here
    would both waste work and risk selecting a different population from the
    canonical preprocessed amplitudes.
    """
    valid = np.asarray(cache.valid, dtype=bool).copy()
    # Standard LED/CFD and all requested ML modes use one frozen event population.
    for name in ("energy_led_time_fs", "energy_cfd_time_fs"):
        array = getattr(cache, name, None)
        if array is not None:
            valid &= np.all(np.asarray(array) != INVALID_TIME_FS, axis=1)
    if cache.timing_led_time_fs is not None:
        valid &= np.all(np.asarray(cache.timing_led_time_fs) != INVALID_TIME_FS, axis=1)
    if cache.timing_cfd_time_fs is not None:
        valid &= np.all(np.asarray(cache.timing_cfd_time_fs) != INVALID_TIME_FS, axis=1)

    selection = copy.deepcopy(config.get("selection", {}))
    preselection = copy.deepcopy(cache.manifest.get("photopeak_preselection", {}))
    selection_summary: dict[str, Any] = {
        "scope": "photopeak_first_then_timing_cleanup_before_any_ml_split",
        "photopeak_preselection": preselection,
        "photopeak": copy.deepcopy(preselection.get("photopeak", [])),
        "photopeak_selected_events": int(cache.manifest.get("event_count", valid.size)),
        "valid_after_timing_extraction": int(np.count_nonzero(valid)),
    }

    # Optional gross LED mismatch rejection remains a dataset-preparation
    # operation. It is deliberately later than photopeak because LED timestamps
    # do not exist during the cheap first scan.
    led_cfg = selection.get("led_outlier_rejection", {}) or {}
    led_summary: dict[str, Any] = {"enabled": bool(led_cfg.get("enabled", False))}
    if led_summary["enabled"]:
        z_limit = float(led_cfg.get("zscore_limit", 6.0))
        if not np.isfinite(z_limit) or z_limit <= 0.0:
            raise ValueError("led_outlier_rejection.zscore_limit must be finite and > 0")
        families: list[tuple[str, np.ndarray]] = [("energy", np.asarray(cache.energy_led_time_fs))]
        if cache.timing_led_time_fs is not None:
            families.append(("timing", np.asarray(cache.timing_led_time_fs)))
        family_rows: list[dict[str, Any]] = []
        for name, times in families:
            delta = _timing_pair_ps(times)
            base = valid & np.isfinite(delta)
            if np.count_nonzero(base) < 3:
                raise RuntimeError(f"Too few valid {name} LED pairs for dataset-level outlier rejection")
            accepted, center, scale, scale_source = robust_led_zscore_mask(
                delta, zscore_limit=z_limit, base_mask=valid
            )
            rejected = int(np.count_nonzero(valid & ~accepted))
            if scale > 0.0:
                z_score = np.abs(delta - center) / scale
                max_kept_z = float(np.max(z_score[valid & accepted])) if np.any(valid & accepted) else float("nan")
            else:
                max_kept_z = 0.0
            valid &= accepted
            family_rows.append({
                "family": name,
                "median_ps": center,
                "robust_sigma_ps": scale,
                "scale_source": scale_source,
                "zscore_limit": z_limit,
                "max_kept_zscore": max_kept_z,
                "rejected": rejected,
            })
        led_summary["estimator"] = "abs(delta_led - median) / (1.4826 * MAD)"
        led_summary["families"] = family_rows
        logger.info(
            "Dataset LED robust-z mismatch rejection | %s",
            ", ".join(
                f"{row['family']} sigma={row['robust_sigma_ps']:.3f} ps "
                f"z<={row['zscore_limit']:.2f} rejected={row['rejected']}"
                for row in family_rows
            ),
        )
    selection_summary["led_outlier_rejection"] = led_summary
    selection_summary["selected_events"] = int(np.count_nonzero(valid))

    minimum = int(selection.get("minimum_events", selection.get("minimum_events_per_split", 100)))
    selected = np.flatnonzero(valid).astype(np.int64)
    if selected.size < minimum:
        raise RuntimeError(f"Only {selected.size} events remain after dataset preparation; need {minimum}")
    return selected, selection_summary


def _prepared_fingerprint(cache: "EnergyCache", selected: np.ndarray, config: dict[str, Any]) -> str:
    return canonical_hash({
        "format_version": DATASET_FORMAT_VERSION,
        "selection_version": PREPARED_SELECTION_VERSION,
        "raw_cache": cache.manifest["fingerprint"],
        "selected_hash": _hash_indices(selected),
        "true_tof_ps": float(config["true_tof_ps"]),
        "selection": config.get("selection", {}),
        "photopeak": config.get("photopeak", {}),
        "denoising": config.get("denoising", {}),
        "input_variant_by_channel": config.get("input_variant_by_channel", {}),
    })


def materialize_selected_dataset(
    cache: "EnergyCache",
    *,
    output: Path,
    config: dict[str, Any],
    rebuild: bool,
    logger: Any,
) -> PreparedDataset:
    selected, selection_summary = _dataset_level_selection(cache, config, logger)
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

    variant_by_channel = {
        "energy": "raw",
        "timing": "raw",
        **copy.deepcopy(config.get("input_variant_by_channel", {})),
    }
    denoise_cfg = copy.deepcopy(config.get("denoising", {}))
    # Denoising is part of channel preprocessing, not a post-materialization ML
    # transform. ``windows_mV``/``timing_windows_mV`` and their LED/CFD labels
    # already come from the configured signal representation in the raw-cache
    # extraction pass.  Keeping a second denoised copy here would both duplicate
    # storage and, more importantly, decouple the target LED from the waveform
    # seen by the model.
    denoise_energy = variant_by_channel["energy"] == "denoised"
    denoise_timing = variant_by_channel["timing"] == "denoised"
    denoise_enabled = denoise_energy or denoise_timing

    manifest = {
        "format_version": DATASET_FORMAT_VERSION,
        "fingerprint": fingerprint,
        "request_fingerprint": str(config.get("request_fingerprint", "")),
        "name": str(config.get("name", output.name)),
        "role": "prepared_full_file",
        "subset_kind": "photopeak_preselected_then_timing_cleaned",
        "photopeak_selection_before_expensive_preprocessing": True,
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
        "denoised_energy_waveforms_saved": denoise_energy,
        "denoised_timing_waveforms_saved": denoise_timing,
        "alternate_posthoc_denoised_arrays_saved": False,
        "input_variant_by_channel": variant_by_channel,
        "denoising": denoise_cfg if denoise_enabled else {"enabled": False},
        "waveform_grid": cache.manifest.get("waveform_grid", "native_samples"),
        "native_sample_interval_ps": cache.manifest.get("native_sample_interval_ps"),
        "timing_native_sample_interval_ps": cache.manifest.get("timing_native_sample_interval_ps"),
        "led_timestamp_source": "energy_channels",
        "cfd_timestamp_source": "energy_channels",
        "denoising_stage": "before_led_cfd_and_window_extraction",
        "energy_led_signal_variant": variant_by_channel["energy"],
        "timing_led_signal_variant": variant_by_channel["timing"],
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
    variants = {
        "energy": "raw",
        "timing": "raw",
        **copy.deepcopy(preprocessing.get("input_variant_by_channel", {})),
    }
    denoising = copy.deepcopy(preprocessing.get("denoising", {}))

    # Channel preprocessing is frozen before any LED/CFD extraction.  Therefore
    # the timestamp used as a target and the waveform later consumed by ML are
    # derived from the same signal representation.
    energy["denoising"] = (
        {**denoising, "enabled": True}
        if variants["energy"] == "denoised"
        else {"enabled": False}
    )
    timing["denoising"] = (
        {**denoising, "enabled": True}
        if variants["timing"] == "denoised"
        else {"enabled": False}
    )
    max_before = max(float(window["before_ns"]) for window in study["windows_ns"])
    max_after = max(float(window["after_ns"]) for window in study["windows_ns"])
    energy["ml_window_ns"] = {"before": max_before, "after": max_after}
    timing["ml_window_ns"] = {"before": max_before, "after": max_after}
    timing["enabled"] = True
    energy["timing_channel_led"] = timing
    return {
        "data": {"input_root": str(root_file), "true_tof_ps": float(study["data"].get("true_tof_ps", 0.0))},
        "channels": copy.deepcopy(study["data"]["channels"]),
        "waveform": energy,
        "selection": copy.deepcopy(preprocessing.get("selection", {})),
        "photopeak": copy.deepcopy(preprocessing.get("photopeak", {"enabled": False})),
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
                    return load_prepared_dataset(output)
            except Exception as exc:
                logger.warning("Cannot reuse permanent prepared dataset %s: %s", output, exc)

    raw_cache_dir = prepared_root / ".raw_cache" / root_id
    raw_cfg = _raw_preprocess_config(study, root_file, raw_cache_dir)
    cache_cfg = {
        "channels": raw_cfg["channels"],
        "waveform": raw_cfg["waveform"],
        "selection": raw_cfg["selection"],
        "photopeak": raw_cfg["photopeak"],
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
        "selection": copy.deepcopy(study["preprocessing"].get("selection", {})),
        "photopeak": copy.deepcopy(study["preprocessing"].get("photopeak", {"enabled": False})),
        "denoising": copy.deepcopy(study["preprocessing"].get("denoising", {"enabled": False})),
        "input_variant_by_channel": copy.deepcopy(
            study["preprocessing"].get(
                "input_variant_by_channel", {"energy": "raw", "timing": "raw"}
            )
        ),
        "materialization_chunk_size": int(study["preprocessing"].get("materialization_chunk_size", 2048)),
    }
    dataset = materialize_selected_dataset(
        cache, output=output, config=permanent_cfg, rebuild=rebuild, logger=logger
    )
    if bool(study["preprocessing"].get("cleanup_raw_cache", True)):
        # Close source memmaps before deleting their directory (important on Windows).
        del cache
        shutil.rmtree(raw_cache_dir, ignore_errors=True)
    return dataset


def plot_prepared_signal_examples(
    dataset: PreparedDataset,
    destination: Path,
    *,
    dpi: int = 180,
) -> None:
    if dataset.event_id.size == 0:
        return
    variants = dataset.manifest.get("input_variant_by_channel", {})
    energy_variant = str(variants.get("energy", "raw"))
    timing_variant = str(variants.get("timing", "raw"))
    rows: list[tuple[str, np.ndarray, np.ndarray, str]] = [
        ("Energy ch. 1", dataset.relative_time_ps, np.asarray(dataset.windows_mV[0, 0]), energy_variant),
        ("Energy ch. 2", dataset.relative_time_ps, np.asarray(dataset.windows_mV[0, 1]), energy_variant),
    ]
    if dataset.timing_windows_mV is not None and dataset.timing_relative_time_ps is not None:
        rows.extend([
            ("Timing ch. 1", dataset.timing_relative_time_ps, np.asarray(dataset.timing_windows_mV[0, 0]), timing_variant),
            ("Timing ch. 2", dataset.timing_relative_time_ps, np.asarray(dataset.timing_windows_mV[0, 1]), timing_variant),
        ])
    fig, axes = plt.subplots(len(rows), 1, figsize=(10.5, 2.7 * len(rows)), squeeze=False)
    for axis, (title, time_ps, waveform, variant) in zip(axes[:, 0], rows):
        axis.plot(
            np.asarray(time_ps, dtype=np.float64) / 1000.0,
            waveform,
            linewidth=1.0,
            label=f"ML preprocessing: {variant}",
        )
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


def input_channel_variant_dataset_view(
    dataset: PreparedDataset, channel_type: str, variant: str
) -> PreparedDataset:
    """Validate and expose the already-preprocessed channel representation.

    Denoising is no longer an ML-time alternate array.  It is applied before
    LED/CFD extraction and windowing during permanent preprocessing, so each
    channel family has exactly one canonical representation in the prepared
    dataset.  This helper remains as a zero-copy compatibility layer for the
    study runner.
    """
    from dataclasses import replace

    channel = str(channel_type).strip().lower()
    key = str(variant).strip().lower()
    if channel not in {"energy", "timing"}:
        raise ValueError("ML input channel_type must be 'energy' or 'timing'")
    if key not in {"raw", "denoised"}:
        raise ValueError("ML input variant must be 'raw' or 'denoised'")

    manifest = dict(dataset.manifest)
    manifest["ml_input_channel"] = channel
    manifest["ml_input_variant"] = key
    configured = dataset.manifest.get("input_variant_by_channel", {})
    if not configured:
        # Read-only compatibility for old prepared datasets used by standalone
        # tools/tests. New study datasets (format v7+) are always materialized
        # with one canonical preprocessing variant per channel.
        if key == "raw":
            return replace(dataset, manifest=manifest)
        if channel == "energy" and dataset.denoised_windows_mV is not None:
            return replace(
                dataset,
                manifest=manifest,
                windows_mV=dataset.denoised_windows_mV,
                timing_aligned_energy_windows_mV=(
                    dataset.denoised_timing_aligned_energy_windows_mV
                    if dataset.denoised_timing_aligned_energy_windows_mV is not None
                    else dataset.timing_aligned_energy_windows_mV
                ),
            )
        if channel == "timing" and dataset.denoised_timing_windows_mV is not None:
            return replace(
                dataset,
                manifest=manifest,
                timing_windows_mV=dataset.denoised_timing_windows_mV,
            )
    configured_key = str(configured.get(channel, "raw")).strip().lower()
    if key != configured_key:
        raise ValueError(
            f"Prepared dataset {dataset.directory} contains {channel}={configured_key}, "
            f"but the experiment requested {key}. Rebuild preprocessing with the "
            "desired input_variant_by_channel policy."
        )
    return replace(dataset, manifest=manifest)


def raw_dataset_view(dataset: PreparedDataset) -> PreparedDataset:
    """Return the canonical raw waveform view.

    This intentionally has no variant argument: multithreshold must never be
    able to request a denoised representation by configuration or accident.
    """
    from dataclasses import replace

    manifest = dict(dataset.manifest)
    manifest["ml_input_variant"] = "raw"
    manifest["ml_input_channel"] = "raw_multithreshold"
    energy_windows = (
        dataset.raw_energy_windows_mV
        if dataset.raw_energy_windows_mV is not None
        else dataset.windows_mV
    )
    energy_led = (
        dataset.raw_energy_led_time_fs
        if dataset.raw_energy_led_time_fs is not None
        else dataset.energy_led_time_fs
        if dataset.energy_led_time_fs is not None
        else dataset.led_time_fs
    )
    energy_cfd = (
        dataset.raw_energy_cfd_time_fs
        if dataset.raw_energy_cfd_time_fs is not None
        else dataset.energy_cfd_time_fs
        if dataset.energy_cfd_time_fs is not None
        else dataset.cfd_time_fs
    )
    energy_anchor = (
        dataset.raw_energy_window_anchor_time_fs
        if dataset.raw_energy_window_anchor_time_fs is not None
        else dataset.energy_window_anchor_time_fs
    )
    timing_windows = (
        dataset.raw_timing_windows_mV
        if dataset.raw_timing_windows_mV is not None
        else dataset.timing_windows_mV
    )
    timing_led = (
        dataset.raw_timing_led_time_fs
        if dataset.raw_timing_led_time_fs is not None
        else dataset.timing_led_time_fs
    )
    timing_cfd = (
        dataset.raw_timing_cfd_time_fs
        if dataset.raw_timing_cfd_time_fs is not None
        else dataset.timing_cfd_time_fs
    )
    timing_anchor = (
        dataset.raw_timing_window_anchor_time_fs
        if dataset.raw_timing_window_anchor_time_fs is not None
        else dataset.timing_window_anchor_time_fs
    )
    timing_aligned_energy = (
        dataset.raw_timing_aligned_energy_windows_mV
        if dataset.raw_timing_aligned_energy_windows_mV is not None
        else dataset.timing_aligned_energy_windows_mV
    )
    timing_aligned_energy_anchor = (
        dataset.raw_timing_aligned_energy_window_anchor_time_fs
        if dataset.raw_timing_aligned_energy_window_anchor_time_fs is not None
        else dataset.timing_aligned_energy_window_anchor_time_fs
    )
    if energy_led is None or energy_cfd is None:
        raise ValueError(f"Dataset {dataset.directory} lacks raw energy timing arrays")
    manifest["input_variant_by_channel"] = {"energy": "raw", "timing": "raw"}
    manifest["energy_led_signal_variant"] = "raw"
    manifest["timing_led_signal_variant"] = "raw"
    return replace(
        dataset,
        manifest=manifest,
        windows_mV=energy_windows,
        led_time_fs=energy_led,
        cfd_time_fs=energy_cfd,
        energy_led_time_fs=energy_led,
        timing_led_time_fs=timing_led,
        energy_cfd_time_fs=energy_cfd,
        timing_cfd_time_fs=timing_cfd,
        energy_window_anchor_time_fs=energy_anchor,
        timing_aligned_energy_window_anchor_time_fs=timing_aligned_energy_anchor,
        timing_window_anchor_time_fs=timing_anchor,
        window_anchor_time_fs=energy_anchor,
        timing_aligned_energy_windows_mV=timing_aligned_energy,
        timing_windows_mV=timing_windows,
    )
