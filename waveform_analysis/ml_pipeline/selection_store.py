from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .common import atomic_json, canonical_hash, read_json, source_signature

# Version 2 invalidates selections produced before baseline-RMSE filtering was
# moved after photopeak selection and made population-derived.
SELECTION_STORE_VERSION = 2


def _hash_indices(indices: np.ndarray) -> str:
    values = np.ascontiguousarray(indices, dtype=np.int64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def selection_request_fingerprint(
    *,
    root_file: Path,
    channels: dict[str, Any],
    preprocessing: dict[str, Any],
) -> str:
    """Fingerprint only the physical/photopeak cohort definition.
    LED/CFD thresholds, ML windows, denoising, true TOF, validation and model
    settings are intentionally absent: changing any of them must not refit the
    photopeak population.
    """
    common = dict(preprocessing.get("common", {}) or {})
    energy = dict(common)
    energy.update(dict(preprocessing.get("energy", {}) or {}))
    selection = dict(preprocessing.get("selection", {}) or {})
    # LED mismatch rejection is timing-dependent and is applied later when the
    # canonical prepared dataset is materialized.  It is not a photopeak cut.
    selection.pop("led_outlier_rejection", None)
    io = preprocessing.get("io", {}) or {}
    descriptor = {
        "version": SELECTION_STORE_VERSION,
        "source": source_signature(root_file),
        "energy_channels": list(channels.get("energy", [])),
        "energy_polarities": list(channels.get("polarities", [])),
        "baseline_samples": int(energy.get("baseline_samples", 500)),
        "search_trigger_threshold_mV": float(energy.get("search_trigger_threshold_mV", 50.0)),
        "selection": selection,
        "photopeak": preprocessing.get("photopeak", {"enabled": False}),
        "max_events": int(io.get("max_events", 0)),
    }
    return canonical_hash(descriptor)


def store_directory(root: Path, root_file: Path, fingerprint: str) -> Path:
    return Path(root) / root_file.stem / fingerprint[:16]


def load_or_compute_selection(
    *,
    root: Path,
    root_file: Path,
    fingerprint: str,
    rebuild: bool,
    compute: Callable[[], tuple[np.ndarray, dict[str, Any]]],
    logger: Any,
) -> tuple[np.ndarray, dict[str, Any], Path]:
    directory = store_directory(root, root_file, fingerprint)
    manifest_path = directory / "manifest.json"
    indices_path = directory / "selected_indices.npy"
    if not rebuild and manifest_path.is_file() and indices_path.is_file():
        manifest = read_json(manifest_path)
        if (
            int(manifest.get("selection_store_version", -1)) == SELECTION_STORE_VERSION
            and manifest.get("fingerprint") == fingerprint
        ):
            indices = np.asarray(np.load(indices_path, allow_pickle=False), dtype=np.int64)
            logger.info(
                "Reusing permanent physical/photopeak selection | %s | events=%d",
                directory,
                indices.size,
            )
            return indices, dict(manifest.get("selection_summary", {})), directory
    indices, summary = compute()
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    temporary = directory.with_name(directory.name + ".building")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, exist_ok=True)
    np.save(temporary / "selected_indices.npy", indices)
    manifest = {
        "selection_store_version": SELECTION_STORE_VERSION,
        "fingerprint": fingerprint,
        "source_root": str(root_file.resolve()),
        "selected_count": int(indices.size),
        "selected_indices_sha256": _hash_indices(indices),
        "selection_summary": summary,
    }
    atomic_json(temporary / "manifest.json", manifest)
    directory.parent.mkdir(parents=True, exist_ok=True)
    if directory.exists():
        shutil.rmtree(directory)
    os.replace(temporary, directory)
    logger.info(
        "Permanent physical/photopeak selection written | %s | events=%d",
        directory,
        indices.size,
    )
    return indices, summary, directory
