from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.lib.format import open_memmap

from .common import atomic_json, canonical_hash, read_json
if TYPE_CHECKING:
    from .data import EnergyCache, SplitData


CATCH22_FEATURE_NAMES = [
    "DN_HistogramMode_5",
    "DN_HistogramMode_10",
    "CO_f1ecac",
    "CO_FirstMin_ac",
    "CO_HistogramAMI_even_2_5",
    "CO_trev_1_num",
    "MD_hrv_classic_pnn40",
    "SB_BinaryStats_mean_longstretch1",
    "SB_TransitionMatrix_3ac_sumdiagcov",
    "PD_PeriodicityWang_th0_01",
    "CO_Embed2_Dist_tau_d_expfit_meandiff",
    "IN_AutoMutualInfoStats_40_gaussian_fmmi",
    "FC_LocalSimple_mean1_tauresrat",
    "DN_OutlierInclude_p_001_mdrmd",
    "DN_OutlierInclude_n_001_mdrmd",
    "SP_Summaries_welch_rect_area_5_1",
    "SB_BinaryStats_diff_longstretch0",
    "SB_MotifThree_quantile_hh",
    "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1",
    "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1",
    "SP_Summaries_welch_rect_centroid",
    "FC_LocalSimple_mean3_stderr",
]

CATCH22_SHORT_NAMES = [
    "mode_5",
    "mode_10",
    "acf_timescale",
    "acf_first_min",
    "ami2",
    "trev",
    "high_fluctuation",
    "stretch_high",
    "transition_matrix",
    "periodicity",
    "embedding_dist",
    "ami_timescale",
    "whiten_timescale",
    "outlier_timing_pos",
    "outlier_timing_neg",
    "centroid_freq",
    "stretch_decreasing",
    "entropy_pairs",
    "rs_range",
    "dfa",
    "low_freq_power",
    "forecast_error",
]

FEATURE_CACHE_FORMAT_VERSION = 3


@dataclass(frozen=True)
class Catch22FeatureCache:
    directory: Path
    features: np.ndarray
    feature_names: list[str]
    short_names: list[str]
    manifest: dict[str, Any]


def _feature_config(model_config: dict[str, Any]) -> dict[str, Any]:
    section = model_config.get("features", {})
    return {
        "implementation": str(section.get("implementation", "aeon")).lower(),
        "catch24": bool(section.get("catch24", True)),
        "outlier_norm": bool(section.get("outlier_norm", True)),
        "use_pycatch22": bool(section.get("use_pycatch22", True)),
        "replace_non_finite": bool(section.get("replace_non_finite", True)),
        "chunk_events": int(section.get("chunk_events", 2048)),
        "checkpoint_every_chunks": int(section.get("checkpoint_every_chunks", 4)),
        "n_jobs": int(section.get("n_jobs", 1)),
        "parallel_backend": section.get("parallel_backend", "threading"),
    }


def _aeon_version() -> str:
    try:
        return str(version("aeon"))
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "The catch22 random-forest model requires aeon. Install the ML "
            "dependencies with `python -m pip install -r requirements_ml.txt`."
        ) from exc


def _pycatch22_version() -> str | None:
    try:
        return str(version("pycatch22"))
    except PackageNotFoundError:
        return None


def _selected_indices(splits: SplitData) -> np.ndarray:
    """Return the sorted union of the already-selected train/validation/test events.

    ``prepare_splits`` applies the energy-only quality cuts and the training-fitted
    photopeak selection before constructing these arrays.  Restricting Catch24
    extraction to their union therefore avoids spending time on events that can
    never be used by the random-forest model.
    """

    selected = np.unique(
        np.concatenate(
            [
                np.asarray(splits.train, dtype=np.int64),
                np.asarray(splits.validation, dtype=np.int64),
                np.asarray(splits.test, dtype=np.int64),
            ]
        )
    )
    if selected.size == 0:
        raise ValueError("Cannot build Catch24 features for an empty selected dataset")
    return selected


def feature_cache_fingerprint(
    cache: EnergyCache,
    splits: SplitData,
    model_config: dict[str, Any],
) -> str:
    config = _feature_config(model_config)
    # Runtime parallelism and chunk size affect speed only, not feature values.
    value_relevant = {
        key: config[key]
        for key in (
            "implementation",
            "catch24",
            "outlier_norm",
            "replace_non_finite",
            "use_pycatch22",
        )
    }
    value_relevant["aeon_version"] = _aeon_version()
    if bool(config["use_pycatch22"]):
        value_relevant["pycatch22_version"] = _pycatch22_version()
    return canonical_hash(
        {
            "format_version": FEATURE_CACHE_FORMAT_VERSION,
            "energy_dataset_fingerprint": cache.manifest["fingerprint"],
            "split_fingerprint": splits.manifest["fingerprint"],
            "features": value_relevant,
        }
    )


def feature_cache_directory(
    cache: EnergyCache,
    splits: SplitData,
    model_config: dict[str, Any],
) -> Path:
    fingerprint = feature_cache_fingerprint(cache, splits, model_config)
    return cache.directory / "catch22_features" / fingerprint[:20]


def _feature_names(catch24: bool) -> tuple[list[str], list[str]]:
    full = list(CATCH22_FEATURE_NAMES)
    short = list(CATCH22_SHORT_NAMES)
    if catch24:
        full.extend(["Mean", "StandardDeviation"])
        short.extend(["mean", "standard_deviation"])
    return full, short


def _load_aeon_transformer(config: dict[str, Any]) -> Any:
    try:
        from aeon.transformations.collection.feature_based import Catch22
    except ImportError as exc:
        raise RuntimeError(
            "The catch22 random-forest model requires aeon. Install the ML "
            "dependencies with `python -m pip install -r requirements_ml.txt`."
        ) from exc

    if bool(config["use_pycatch22"]) and _pycatch22_version() is None:
        raise RuntimeError(
            "features.use_pycatch22=true requires pycatch22. Install it with "
            "`python -m pip install pycatch22==0.4.5`, then restart extraction."
        )

    return Catch22(
        features="all",
        catch24=bool(config["catch24"]),
        outlier_norm=bool(config["outlier_norm"]),
        replace_nans=bool(config["replace_non_finite"]),
        use_pycatch22=bool(config["use_pycatch22"]),
        n_jobs=int(config["n_jobs"]),
        parallel_backend=config["parallel_backend"],
    )


def _extract_chunk(
    windows: np.ndarray,
    valid: np.ndarray,
    config: dict[str, Any],
    feature_count: int,
    transformer: Any,
) -> np.ndarray:
    event_count = int(windows.shape[0])
    result = np.zeros((event_count, 2, feature_count), dtype=np.float32)
    usable = np.asarray(valid, dtype=bool) & np.all(np.isfinite(windows), axis=(1, 2))
    usable_indices = np.flatnonzero(usable)
    if usable_indices.size == 0:
        return result

    cases = np.asarray(windows[usable_indices], dtype=np.float64).reshape(
        usable_indices.size * 2, 1, windows.shape[2]
    )
    transformed = np.asarray(transformer.transform(cases), dtype=np.float64)
    if transformed.shape != (usable_indices.size * 2, feature_count):
        raise RuntimeError(
            "Unexpected Catch22 output shape: "
            f"{transformed.shape}; expected {(usable_indices.size * 2, feature_count)}"
        )
    if bool(config["replace_non_finite"]):
        transformed = np.nan_to_num(
            transformed, nan=0.0, posinf=0.0, neginf=0.0
        )
    elif not np.all(np.isfinite(transformed)):
        raise RuntimeError(
            "Catch22 produced NaN/inf values. Enable features.replace_non_finite."
        )
    result[usable_indices] = transformed.reshape(usable_indices.size, 2, feature_count)
    return result


def prepare_catch22_feature_cache(
    cache: EnergyCache,
    splits: SplitData,
    model_config: dict[str, Any],
    *,
    logger: Any,
) -> Catch22FeatureCache:
    config = _feature_config(model_config)
    if config["implementation"] != "aeon":
        raise ValueError("features.implementation currently supports only 'aeon'")
    if config["chunk_events"] <= 0:
        raise ValueError("features.chunk_events must be positive")
    if config["checkpoint_every_chunks"] <= 0:
        raise ValueError("features.checkpoint_every_chunks must be positive")
    if config["n_jobs"] == 0 or config["n_jobs"] < -1:
        raise ValueError("features.n_jobs must be -1 or a positive integer")

    selected_indices = _selected_indices(splits)
    feature_family = "Catch24" if bool(config["catch24"]) else "Catch22"
    fingerprint = feature_cache_fingerprint(cache, splits, model_config)
    directory = feature_cache_directory(cache, splits, model_config)
    manifest_path = directory / "manifest.json"
    features_path = directory / "features.npy"
    full_names, short_names = _feature_names(bool(config["catch24"]))
    feature_count = len(full_names)
    event_count = int(cache.windows_mV.shape[0])
    selected_count = int(selected_indices.size)
    expected_shape = (event_count, 2, feature_count)

    if manifest_path.is_file() and features_path.is_file():
        manifest = read_json(manifest_path)
        if manifest.get("fingerprint") != fingerprint:
            raise RuntimeError("Catch22 cache fingerprint mismatch")
        if bool(manifest.get("complete", False)):
            features = np.load(features_path, mmap_mode="r")
            if tuple(features.shape) != expected_shape:
                raise RuntimeError("Catch22 cache has an unexpected shape")
            logger.info("Reusing Catch22 feature cache: %s", directory)
            return Catch22FeatureCache(
                directory=directory,
                features=features,
                feature_names=full_names,
                short_names=short_names,
                manifest=manifest,
            )
        next_selected_position = int(manifest.get("next_selected_position", 0))
        features_map = open_memmap(features_path, mode="r+", dtype=np.float32)
        if tuple(features_map.shape) != expected_shape:
            raise RuntimeError("Incomplete Catch22 cache has an unexpected shape")
        logger.info(
            "Resuming %s feature extraction at selected event %d/%d",
            feature_family,
            next_selected_position,
            selected_count,
        )
    else:
        directory.mkdir(parents=True, exist_ok=True)
        features_map = open_memmap(
            features_path,
            mode="w+",
            dtype=np.float32,
            shape=expected_shape,
        )
        features_map[:] = 0.0
        features_map.flush()
        next_selected_position = 0
        manifest = {
            "format_version": FEATURE_CACHE_FORMAT_VERSION,
            "fingerprint": fingerprint,
            "energy_dataset_fingerprint": cache.manifest["fingerprint"],
            "split_fingerprint": splits.manifest["fingerprint"],
            "feature_config": config,
            "feature_names": full_names,
            "feature_short_names": short_names,
            "shape": list(expected_shape),
            "selected_event_count": selected_count,
            "selection_scope": (
                "union of frozen train/validation/test events after energy-only "
                "quality cuts and training-fitted photopeak selection"
            ),
            "next_selected_position": 0,
            "complete": False,
        }
        atomic_json(manifest_path, manifest)
        logger.info(
            "Building %s feature cache for %d selected photopeak events "
            "(%d total cached events): %s",
            feature_family,
            selected_count,
            event_count,
            directory,
        )

    chunk_events = int(config["chunk_events"])
    checkpoint_every_chunks = int(config["checkpoint_every_chunks"])
    aeon_version = _aeon_version()
    pycatch22_version = _pycatch22_version() if bool(config["use_pycatch22"]) else None

    transformer = _load_aeon_transformer(config)
    fit_indices = selected_indices[: min(2, selected_count)]
    fit_windows = np.asarray(cache.windows_mV[fit_indices], dtype=np.float64).reshape(
        fit_indices.size * 2, 1, cache.windows_mV.shape[2]
    )
    transformer.fit(fit_windows)

    logger.info(
        "%s extraction backend | pycatch22=%s | n_jobs=%s | backend=%s | "
        "chunk_events=%d | checkpoint_every_chunks=%d",
        feature_family,
        bool(config["use_pycatch22"]),
        config["n_jobs"],
        config["parallel_backend"],
        chunk_events,
        checkpoint_every_chunks,
    )

    extraction_start = time.perf_counter()
    progress_origin = next_selected_position
    chunk_number = 0

    for start in range(next_selected_position, selected_count, chunk_events):
        stop = min(start + chunk_events, selected_count)
        chunk_indices = selected_indices[start:stop]
        windows = np.asarray(cache.windows_mV[chunk_indices], dtype=np.float32)
        valid = np.asarray(cache.valid[chunk_indices], dtype=bool)
        chunk_started = time.perf_counter()
        features_map[chunk_indices] = _extract_chunk(
            windows, valid, config, feature_count, transformer
        )
        chunk_number += 1

        must_checkpoint = (
            chunk_number % checkpoint_every_chunks == 0 or stop == selected_count
        )
        if must_checkpoint:
            features_map.flush()
            manifest.update(
                {
                    "next_selected_position": stop,
                    "complete": stop == selected_count,
                    "aeon_version": aeon_version,
                    "pycatch22_version": pycatch22_version,
                }
            )
            atomic_json(manifest_path, manifest)

        now = time.perf_counter()
        elapsed = max(now - extraction_start, 1e-9)
        processed = stop - progress_origin
        rate = processed / elapsed
        remaining = selected_count - stop
        eta_seconds = remaining / rate if rate > 0 else float("inf")
        chunk_rate = (stop - start) / max(now - chunk_started, 1e-9)
        logger.info(
            "%s features: selected events %d-%d/%d | chunk %.1f events/s | "
            "average %.1f events/s | ETA %.1f min%s",
            feature_family,
            start + 1,
            stop,
            selected_count,
            chunk_rate,
            rate,
            eta_seconds / 60.0,
            " | checkpoint saved" if must_checkpoint else "",
        )

    features = np.load(features_path, mmap_mode="r")
    return Catch22FeatureCache(
        directory=directory,
        features=features,
        feature_names=full_names,
        short_names=short_names,
        manifest=read_json(manifest_path),
    )
