from __future__ import annotations

import sys

import copy
import csv
import gc
import hashlib
import itertools
import json
import os
import re
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from utils_fit import FitResult
from .common import atomic_json, canonical_hash
from .dataset import PreparedDataset
from .metrics import FWHM_PER_SIGMA, ctr_bootstrap_uncertainty, fit_times_ps, residual_metrics
from .models import validate_model, validate_model_training
from .prediction import prediction_window_dataset_view
from .prepared_data import (
    input_variant_dataset_view,
    plot_prepared_signal_examples,
    prepare_file_dataset,
)
from .study_config import CHANNEL_MODES, candidate_overrides, discover_root_files, set_nested
from .experiment_config import cfd_enabled
from .torch_data import Normalization, compute_normalization
from .training import train_model
from .training_utils import make_split_loader, predict_loader, resolve_device
from .validation import outer_splits, random_dev_blind, selection_splits, nested_inner_validation
from .reporting import (
    plot_correction_examples, plot_correction_matrix, plot_ctr_vs_voltage,
    plot_final_bars, plot_result_distribution, plot_selection_vs_blind,
    plot_window_scan_bars,
    write_csv as write_report_csv, write_summary_results,
)
_STAGE_OOF = 0
_STAGE_BLIND = 1
_MODEL_LED = "led"
_MODEL_CFD = "cfd"
_MODEL_MULTITHRESHOLD = "multithreshold_svr"


def _seed_for(base: int, *parts: Any) -> int:
    payload = "|".join(str(v) for v in (base, *parts)).encode("utf-8")
    import hashlib
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little") & 0x7FFFFFFF



def _fit_early_split(train_pool: np.ndarray, *, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(train_pool, dtype=np.int64)
    if not 0.0 < fraction < 0.5:
        raise ValueError("early-stop fraction must be in (0, 0.5)")
    order = np.random.default_rng(seed).permutation(values)
    n_early = max(1, int(round(values.size * fraction)))
    n_early = min(n_early, values.size - 1)
    early = np.sort(order[:n_early])
    fit = np.sort(order[n_early:])
    return fit, early

def _voltage_from_name(name: str, pattern: str) -> float:
    match = re.search(pattern, name)
    if not match:
        return float("nan")
    try:
        return float(match.group("voltage"))
    except (IndexError, KeyError):
        return float(match.group(1))

def _pending_csv_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.pending{path.suffix}")


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    logger: Any | None = None,
    retries: int = 6,
    retry_delay_s: float = 0.20,
) -> bool:
    """Atomically publish a CSV without letting a Windows file lock kill a run.

    Windows refuses ``os.replace`` when the destination is open in applications
    such as Excel.  Candidate/model caches are the authoritative resume state,
    while this CSV is a progressive human-readable view.  We therefore retry
    transient locks and, if the destination remains locked, preserve the newest
    complete snapshot as ``<stem>.pending.csv``.  A later flush/resume merges the
    pending snapshot back into the canonical CSV.

    Returns True when ``path`` was successfully published, False when the latest
    snapshot had to be parked in the pending sidecar.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        if fields:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    last_error: PermissionError | None = None
    for attempt in range(max(0, int(retries)) + 1):
        try:
            os.replace(temporary, path)
            pending = _pending_csv_path(path)
            try:
                pending.unlink(missing_ok=True)
                for stale in path.parent.glob(f"{path.stem}.pending.*{path.suffix}"):
                    stale.unlink(missing_ok=True)
            except OSError:
                pass
            return True
        except PermissionError as exc:
            last_error = exc
            if attempt < int(retries):
                time.sleep(float(retry_delay_s) * (attempt + 1))

    pending = _pending_csv_path(path)
    try:
        os.replace(temporary, pending)
    except PermissionError:
        # Very unusual: both canonical and pending files are locked. Keep a
        # uniquely named complete snapshot rather than failing the experiment.
        stamp = int(time.time() * 1000)
        pending = path.with_name(f"{path.stem}.pending.{stamp}{path.suffix}")
        os.replace(temporary, pending)

    if logger is not None:
        logger.warning(
            "Could not update %s because it is locked by another process; "
            "latest progressive snapshot saved to %s. Training will continue. "
            "Close the CSV in Excel/preview to allow the next flush to publish it.",
            path,
            pending,
        )
    return False


# Increment this if the on-disk resume artifact semantics change.
_RESUME_CACHE_VERSION = 1
_RESULT_KEY_FIELDS = ("stage", "file_id", "mode_id", "model_id", "candidate_id")


def _row_int(value: Any, default: int = -10**9) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _result_row_key(row: dict[str, Any]) -> tuple[int, int, int, int, int]:
    return tuple(_row_int(row.get(name)) for name in _RESULT_KEY_FIELDS)  # type: ignore[return-value]


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _merge_result_rows(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge progressive snapshots with later groups taking precedence."""
    merged: list[dict[str, Any]] = []
    positions: dict[tuple[int, int, int, int, int], int] = {}
    for group in groups:
        for row in group:
            key = _result_row_key(row)
            if key in positions:
                merged[positions[key]] = row
            else:
                positions[key] = len(merged)
                merged.append(row)
    return merged


def _read_progressive_rows(path: Path) -> list[dict[str, Any]]:
    """Read canonical results plus any newer sidecar left by a file lock."""
    groups = [_read_csv_rows(path)]
    pending = _pending_csv_path(path)
    if pending.is_file():
        groups.append(_read_csv_rows(pending))
    # Also recover rare uniquely named pending snapshots. Lexicographic order is
    # chronological because their suffix is a millisecond timestamp.
    for candidate in sorted(path.parent.glob(f"{path.stem}.pending.*{path.suffix}")):
        groups.append(_read_csv_rows(candidate))
    return _merge_result_rows(*groups)


class _ProgressiveResultRows(list[dict[str, Any]]):
    """Result rows that atomically refresh results.csv after every completed step.

    ``append`` is intentionally an upsert keyed by stage/file/mode/model/candidate.
    This makes an interrupted run idempotent: a resumed candidate replaces its
    previous row instead of creating duplicates.
    """

    def __init__(
        self,
        path: Path,
        initial: list[dict[str, Any]] | None = None,
        *,
        logger: Any | None = None,
    ):
        super().__init__(initial or [])
        self.path = path
        self.logger = logger

    def append(self, row: dict[str, Any]) -> None:  # type: ignore[override]
        key = _result_row_key(row)
        for index, existing in enumerate(self):
            if _result_row_key(existing) == key:
                super().__setitem__(index, row)
                self.flush()
                return
        super().append(row)
        self.flush()

    def flush(self) -> None:
        _write_csv(self.path, list(self), logger=self.logger)


def _flush_progress_rows(rows: list[dict[str, Any]] | None) -> None:
    if isinstance(rows, _ProgressiveResultRows):
        rows.flush()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype=np.int64)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _splits_signature(
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> list[dict[str, Any]]:
    return [
        {
            "train_n": int(len(train)),
            "train_sha256": _array_sha256(np.asarray(train, dtype=np.int64)),
            "score_n": int(len(score)),
            "score_sha256": _array_sha256(np.asarray(score, dtype=np.int64)),
        }
        for train, score in splits
    ]


def _dataset_resume_token(dataset: PreparedDataset) -> dict[str, Any]:
    manifest = dataset.manifest if isinstance(dataset.manifest, dict) else {}
    return {
        "fingerprint": manifest.get("fingerprint"),
        "source_root": manifest.get("source_root"),
        "event_count": int(dataset.event_id.size),
        "event_id_sha256": hashlib.sha256(
            np.ascontiguousarray(dataset.event_id, dtype=np.int64).tobytes()
        ).hexdigest(),
        "true_tof_ps": float(dataset.true_tof_ps),
    }


def _selection_cache_payload(
    study: dict[str, Any],
    dataset: PreparedDataset,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    mode: str,
    descriptor: dict[str, Any],
    space: dict[str, Any] | None,
) -> dict[str, Any]:
    """Candidate-specific scientific fingerprint.

    Deliberately does not hash global reporting configuration or unrelated model
    spaces. Consequently adding plots/XAI or another model cannot invalidate a
    completed candidate from this model.
    """
    return {
        "schema_version": _RESUME_CACHE_VERSION,
        "dataset": _dataset_resume_token(dataset),
        "mode": mode,
        "descriptor": copy.deepcopy(descriptor),
        "base_train_config": (
            None if space is None else copy.deepcopy(space["base_train_config"])
        ),
        "validation_seed": int(study["validation"]["seed"]),
        "splits": _splits_signature(splits),
    }


def _candidate_cache_paths(
    study: dict[str, Any],
    *,
    file_id: int,
    mode_id: int,
    model_name: str,
    cache_key: str,
    kind: str = "selection",
) -> tuple[Path, Path]:
    root = (
        Path(study["experiment"]["output_dir"])
        / "artifacts"
        / f"{kind}_cache_v{_RESUME_CACHE_VERSION}"
        / f"f{file_id}_m{mode_id}"
        / _artifact_model_key(model_name)
    )
    return root / f"{cache_key}.npz", root / f"{cache_key}.json"


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _save_selection_cache(
    npz_path: Path,
    json_path: Path,
    *,
    cache_key: str,
    metrics: dict[str, Any],
    fold_ctrs: list[float],
    score_indices: np.ndarray,
    residual: np.ndarray,
) -> None:
    _atomic_npz(
        npz_path,
        score_indices=np.asarray(score_indices, dtype=np.int64),
        residual=np.asarray(residual, dtype=np.float64),
    )
    atomic_json(
        json_path,
        {
            "schema_version": _RESUME_CACHE_VERSION,
            "cache_key": cache_key,
            "metrics": copy.deepcopy(metrics),
            "fold_ctrs": [float(value) for value in fold_ctrs],
        },
    )


def _restore_cached_metrics(values: Any) -> dict[str, Any]:
    """Restore numeric metric semantics after strict JSON serialization.

    ``atomic_json`` writes non-finite floating-point values as JSON ``null``.
    Selection/final caches legitimately contain NaN diagnostics (for example
    ``ctr_err_ps`` for a one-split holdout).  On reload those values therefore
    become ``None`` and must be converted back to NaN before report code calls
    ``float(...)``.  This keeps existing v1 caches reusable.
    """
    if not isinstance(values, dict):
        raise ValueError("Cached metrics must be a JSON object")
    restored = dict(values)
    numeric_keys = {
        "ctr_ps",
        "ctr_err_ps",
        "mean_ps",
        "std_ps",
        "rmse_ps",
        "bias_ps",
        "dev_ndof",
        "bin_ps",
        "phase_ps",
        "phase_ctr_std_ps",
    }
    for key in numeric_keys:
        if key not in restored:
            continue
        value = restored[key]
        if value is None or value == "":
            restored[key] = float("nan")
        else:
            restored[key] = float(value)
    if "n" in restored:
        restored["n"] = int(restored["n"])
    return restored


def _load_selection_cache(
    npz_path: Path,
    json_path: Path,
    *,
    cache_key: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], list[float]] | None:
    if not npz_path.is_file() or not json_path.is_file():
        return None
    try:
        with json_path.open("r", encoding="utf-8") as stream:
            meta = json.load(stream)
        if (
            int(meta.get("schema_version", -1)) != _RESUME_CACHE_VERSION
            or str(meta.get("cache_key", "")) != cache_key
        ):
            return None
        with np.load(npz_path) as arrays:
            score_indices = np.asarray(arrays["score_indices"], dtype=np.int64)
            residual = np.asarray(arrays["residual"], dtype=np.float64)
        metrics = _restore_cached_metrics(meta["metrics"])
        fold_ctrs = [float(value) for value in meta.get("fold_ctrs", []) if value is not None]
        if score_indices.size != residual.size:
            return None
        return score_indices, residual, metrics, fold_ctrs
    except Exception:
        return None


def _final_cache_payload(
    study: dict[str, Any],
    dataset: PreparedDataset,
    *,
    mode: str,
    space: dict[str, Any],
    chosen: dict[str, Any],
    development: np.ndarray,
    blind: np.ndarray,
) -> dict[str, Any]:
    return {
        "schema_version": _RESUME_CACHE_VERSION,
        "dataset": _dataset_resume_token(dataset),
        "mode": mode,
        "space_id": str(space["id"]),
        "base_train_config": copy.deepcopy(space["base_train_config"]),
        "window": copy.deepcopy(chosen["window"]),
        "variant": str(chosen["variant"]),
        "subsampling": int(chosen["subsampling"]),
        "overrides": copy.deepcopy(chosen["overrides"]),
        "candidate_id": int(chosen["candidate_id"]),
        "development_sha256": _array_sha256(np.asarray(development, dtype=np.int64)),
        "blind_sha256": _array_sha256(np.asarray(blind, dtype=np.int64)),
        "validation_seed": int(study["validation"]["seed"]),
    }


def _save_final_cache(
    npz_path: Path,
    json_path: Path,
    *,
    cache_key: str,
    residual: np.ndarray,
    train_residual: np.ndarray,
    metrics: dict[str, Any],
    meta: dict[str, Any],
    xai_profile: tuple[np.ndarray, np.ndarray] | None,
) -> None:
    arrays: dict[str, np.ndarray] = {
        "blind_residual": np.asarray(residual, dtype=np.float64),
        "train_residual": np.asarray(train_residual, dtype=np.float64),
    }
    if xai_profile is not None:
        arrays["xai_time_ps"] = np.asarray(xai_profile[0], dtype=np.float64)
        arrays["xai_importance"] = np.asarray(xai_profile[1], dtype=np.float64)
    _atomic_npz(npz_path, **arrays)
    atomic_json(
        json_path,
        {
            "schema_version": _RESUME_CACHE_VERSION,
            "cache_key": cache_key,
            "metrics": copy.deepcopy(metrics),
            "meta": copy.deepcopy(meta),
        },
    )


def _load_final_cache(
    npz_path: Path,
    json_path: Path,
    *,
    cache_key: str,
) -> tuple[
    np.ndarray,
    dict[str, Any],
    dict[str, Any],
    tuple[np.ndarray, np.ndarray] | None,
    np.ndarray,
] | None:
    if not npz_path.is_file() or not json_path.is_file():
        return None
    try:
        with json_path.open("r", encoding="utf-8") as stream:
            meta_file = json.load(stream)
        if (
            int(meta_file.get("schema_version", -1)) != _RESUME_CACHE_VERSION
            or str(meta_file.get("cache_key", "")) != cache_key
        ):
            return None
        with np.load(npz_path) as arrays:
            residual = np.asarray(arrays["blind_residual"], dtype=np.float64)
            train_residual = np.asarray(arrays["train_residual"], dtype=np.float64)
            xai_profile = None
            if "xai_time_ps" in arrays and "xai_importance" in arrays:
                xai_profile = (
                    np.asarray(arrays["xai_time_ps"], dtype=np.float64),
                    np.asarray(arrays["xai_importance"], dtype=np.float64),
                )
        return (
            residual,
            _restore_cached_metrics(meta_file["metrics"]),
            dict(meta_file["meta"]),
            xai_profile,
            train_residual,
        )
    except Exception:
        return None


def _resume_state_hash(config: dict[str, Any]) -> str:
    return str(config.get("_core_hash") or config.get("_config_hash") or "")


def _check_or_create_run_state(
    config: dict[str, Any],
    output: Path,
    *,
    resume: bool,
    logger: Any,
) -> Path:
    """Protect progressive results from being mixed across scientific configs."""
    path = output / "run_state.json"
    current = _resume_state_hash(config)
    if resume and path.is_file():
        with path.open("r", encoding="utf-8") as stream:
            previous = json.load(stream)
        old = str(previous.get("core_hash", ""))
        if old and current and old != current:
            raise RuntimeError(
                "Partial results exist for a different training/core configuration. "
                "Use a different experiment output directory or --restart."
            )
        logger.info(
            "Partial resume state found | status=%s | progressive rows/candidate caches will be reused",
            previous.get("status", "unknown"),
        )
    elif resume and (output / "results.csv").is_file() and not path.is_file():
        raise RuntimeError(
            "results.csv exists without run_state.json/manifest.json, so partial-result "
            "compatibility cannot be proven. Preserve the directory and start a new "
            "output, or use --restart if those partial rows may be discarded."
        )
    atomic_json(
        path,
        {
            "schema_version": 1,
            "experiment": config["experiment"]["name"],
            "core_hash": current,
            "artifact_hash": config.get("_artifact_hash"),
            "config_hash": config.get("_config_hash"),
            "status": "running",
        },
    )
    return path


def _fit_row(
    values_ps: np.ndarray, *, method: str, fit_config: dict[str, Any]
) -> tuple[FitResult | None, dict[str, Any]]:
    """Study-level all-event CTR metrics without clipping or fit-based selection.
    The model/training pipeline is the working uploaded implementation.  Only
    experiment-level evaluation uses the newer CTR convention requested for the
    studies: 2.355 times the sample standard deviation over every event.
    A Gaussian fit is attempted only as optional diagnostics and never changes
    the reported/selected CTR.
    """
    values = np.asarray(values_ps, dtype=np.float64).reshape(-1)
    if values.size == 0 or np.any(~np.isfinite(values)):
        raise RuntimeError(f"{method}: evaluation requires one finite value for every event")
    simple = residual_metrics(values)
    fit: FitResult | None = None
    try:
        fit = fit_times_ps(values, method, fit_config)
    except Exception:
        fit = None
    return fit, {
        "n": int(simple["n"]),
        "ctr_ps": float(simple["ctr_ps"]),
        "ctr_err_ps": float("nan"),
        "mean_ps": float(simple["mean_ps"]),
        "std_ps": float(simple["std_ps"]),
        "rmse_ps": float(simple["rmse_ps"]),
        "bias_ps": float(simple["bias_ps"]),
        "dev_ndof": float(fit.chi2_ndof) if fit is not None and fit.success else float("nan"),
        "bin_ps": float(fit.bin_width_ps) if fit is not None and fit.success else float("nan"),
        "phase_ps": float(fit.bin_phase_ps) if fit is not None and fit.success else float("nan"),
        "phase_ctr_std_ps": float(fit.phase_ctr_std_ps) if fit is not None and fit.success else float("nan"),
    }

def _target_deltas(dataset: PreparedDataset, mode: str, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    _input, target = CHANNEL_MODES[mode]
    if target == "energy_led":
        led = dataset.energy_led_time_fs
        cfd = dataset.energy_cfd_time_fs
    else:
        led = dataset.timing_led_time_fs
        cfd = dataset.timing_cfd_time_fs
    if led is None:
        raise ValueError(f"Prepared dataset lacks LED timing arrays required by mode {mode}")
    idx = np.asarray(indices, dtype=np.int64)
    led_ps = (np.asarray(led[idx, 0], dtype=np.float64) - np.asarray(led[idx, 1], dtype=np.float64)) / 1000.0
    cfd_ps = None if cfd is None else (np.asarray(cfd[idx, 0], dtype=np.float64) - np.asarray(cfd[idx, 1], dtype=np.float64)) / 1000.0
    return led_ps, cfd_ps

def _require_cfd(values: np.ndarray | None, mode: str) -> np.ndarray:
    if values is None:
        raise ValueError(f"CFD is enabled for {mode}, but the prepared dataset has no CFD timestamps")
    return values

def _require_cfd_metric(values: dict[str, Any] | None, mode: str) -> dict[str, Any]:
    if values is None:
        raise ValueError(f"CFD metrics requested for disabled/unavailable mode {mode}")
    return values

def _apply_overrides(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for dotted, value in overrides.items():
        set_nested(result, dotted, copy.deepcopy(value))
    # Every search-space point is already a complete candidate.  In particular,
    # the supported Linear-SVR space supplies a singleton epsilon_values list, so
    # the trainer cannot perform a hidden second model-selection step.
    return result

def _candidate_training_config(
    study: dict[str, Any],
    space: dict[str, Any],
    overrides: dict[str, Any],
    *,
    mode: str,
    subsampling: int,
    train_dir: Path,
    seed: int,
    final: bool,
) -> dict[str, Any]:
    cfg = _apply_overrides(space["base_train_config"], overrides)
    cfg["fit"] = copy.deepcopy(study["fit"])
    input_waveforms, target = CHANNEL_MODES[mode]
    cfg["prediction"] = {"input_waveforms": input_waveforms, "target": target}
    cfg["input_transform"] = "none"
    cfg.setdefault("preprocessing", {})["subsampling_factor"] = int(subsampling)
    cfg.setdefault("output", {})["train_dir"] = str(train_dir)
    cfg.setdefault("plotting", {})["dpi"] = int(study["reporting"]["dpi"])
    training = cfg.setdefault("training", {})
    training["seed"] = int(seed)
    training["data_seed"] = int(seed)
    training["initialization_seed"] = int(seed)
    # Early stopping should be stable and cheap; pooled OOF CTR, not the early-
    # stop metric, performs scientific candidate selection.
    if cfg["model"]["type"] in {"cnn_regressor", "constructive_mlp_encoder"}:
        training["selection_metric"] = "validation_rmse"
        training["fit_interval_epochs"] = 0
        training["fit_train_during_training"] = False
        training["fit_validation_during_training"] = False
    training["baseline_guard_metric"] = None
    artifacts = cfg.setdefault("artifacts", {})
    artifacts.update({
        "save_config": False,
        "save_history": False,
        "save_plots": False,
        "save_summary": False,
        "save_model_artifacts": False,
        "save_last_checkpoint": False,
        "save_best_checkpoint": bool(final),
        "perform_internal_gaussian_fit": False,
    })
    model_cfg = dict(cfg["model"])
    model_type = str(model_cfg.pop("type"))
    model_cfg.pop("name", None)
    validate_model(model_type, model_cfg)
    validate_model_training(model_type, cfg)
    return cfg

def _train_in_memory(
    cfg: dict[str, Any],
    view: PreparedDataset,
    *,
    logger: Any,
    data_view: dict[str, Any],
    normalization_override: Normalization | None = None,
) -> tuple[torch.nn.Module, Normalization, dict[str, Any]]:
    summary = train_model(
        cfg,
        restart=True,
        logger=logger,
        prepared_datasets=[view],
        data_view=data_view,
        normalization_override=normalization_override,
    )
    model = summary.get("_trained_model")
    if not isinstance(model, torch.nn.Module):
        raise RuntimeError("Trainer did not return its selected in-memory model")
    normalization = Normalization.from_dict(summary["normalization"])
    return model, normalization, summary

def _predict_indices(
    model: torch.nn.Module,
    normalization: Normalization,
    cfg: dict[str, Any],
    view: PreparedDataset,
    indices: np.ndarray,
) -> np.ndarray:
    evaluation_view = replace(view, evaluation=np.asarray(indices, dtype=np.int64))
    device = resolve_device(cfg["training"].get("device", "auto"))
    model = model.to(device)
    loader = make_split_loader(
        [evaluation_view], "evaluation", normalization, cfg, device,
        shuffle=False, subsampling_factor=int(cfg["preprocessing"]["subsampling_factor"]),
    )
    prediction = predict_loader(model, loader, device)
    residual = np.asarray(prediction["residual_ps"], dtype=np.float64)
    if residual.size != len(indices):
        raise RuntimeError("Prediction count differs from requested evaluation population")
    return residual

def _cleanup_training(model: torch.nn.Module, directory: Path, *, keep_best: Path | None = None) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if keep_best is not None:
        source = directory / "checkpoints" / "best.pt"
        if not source.is_file():
            raise RuntimeError(f"Expected final checkpoint was not produced: {source}")
        keep_best.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, keep_best)
    shutil.rmtree(directory, ignore_errors=True)

def _early_fraction(study: dict[str, Any], candidate_cfg: dict[str, Any]) -> float:
    value = float(candidate_cfg.get("training", {}).get(
        "early_stop_fraction", study["validation"]["early_stop_fraction"]
    ))
    if not 0.0 < value < 0.5:
        raise ValueError("training.early_stop_fraction must be in (0, 0.5)")
    return value

def _normalization_for_fit_subset(
    view: PreparedDataset,
    fit_indices: np.ndarray,
    *,
    subsampling: int,
    cache: dict[str, Normalization],
) -> Normalization:
    """Reuse only tiny train-derived normalization stats, never waveform/model data."""
    indices = np.ascontiguousarray(fit_indices, dtype=np.int64)
    import hashlib
    index_hash = hashlib.sha256(indices.tobytes()).hexdigest()
    descriptor = {
        "dataset": view.manifest.get("fingerprint", str(view.directory)),
        "variant": view.manifest.get("ml_input_variant", "raw"),
        "prediction": view.manifest.get("prediction_view", {}),
        "window_before_ns": view.manifest.get("window_before_ns"),
        "window_after_ns": view.manifest.get("window_after_ns"),
        "subsampling": int(subsampling),
        "fit_indices_sha256": index_hash,
    }
    key = canonical_hash(descriptor)
    if key not in cache:
        cache[key] = compute_normalization(
            [(view, indices)],
            chunk_size=4096,
            featurewise=False,
            subsampling_factor=int(subsampling),
        )
    return cache[key]

def _waveform_candidate_combinations(
    study: dict[str, Any],
    space: dict[str, Any],
    *,
    mode: str,
    seed: int,
) -> list[tuple[dict[str, Any], str, int, dict[str, Any]]]:
    """Combine model and preprocessing choices without multiplying random searches.
    Grid model spaces remain exhaustive. Random model spaces use ``n_trials`` as
    the total experiment budget and deterministically cycle through a shuffled
    list of window/input-variant/subsampling choices, so preprocessing options
    are covered without multiplying expensive neural-network trials.
    """
    model_candidates = candidate_overrides(space, seed=seed)
    variant_by_channel = study["preprocessing"].get("input_variant_by_channel")
    if isinstance(variant_by_channel, dict):
        input_family = CHANNEL_MODES[mode][0]
        variants = [str(variant_by_channel.get(input_family, "raw"))]
    else:
        variants = list(study["preprocessing"]["input_variants"])
    prep = [
        (window, variant, int(factor))
        for window in study["windows_ns"]
        for variant in variants
        for factor in study["preprocessing"]["subsampling_factors"]
    ]
    if str(space["search"].get("method", "grid")) == "grid":
        return [(window, variant, factor, overrides) for window, variant, factor in prep for overrides in model_candidates]
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(len(prep)))
    output: list[tuple[dict[str, Any], str, int, dict[str, Any]]] = []
    for i, overrides in enumerate(model_candidates):
        if i > 0 and i % len(prep) == 0:
            order = list(rng.permutation(len(prep)))
        window, variant, factor = prep[order[i % len(prep)]]
        output.append((window, variant, factor, overrides))
    return output
def _integrated_gradient_profile(
    model: torch.nn.Module,
    normalization: Normalization,
    cfg: dict[str, Any],
    view: PreparedDataset,
    indices: np.ndarray,
    *,
    max_events: int,
    steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean absolute integrated-gradient attribution for the pair correction.
    Inputs are normalized exactly as during training. A zero baseline therefore
    corresponds to the training-set mean waveform for global/feature z-scoring.
    The two detector attributions are pooled only after taking absolute values.
    """
    if max_events <= 0 or steps <= 0:
        raise ValueError("XAI max_events and steps must be positive")
    chosen = np.asarray(indices, dtype=np.int64)[: min(len(indices), max_events)]
    eval_view = replace(view, evaluation=chosen)
    device = resolve_device(cfg["training"].get("device", "auto"))
    model = model.to(device)
    model.eval()
    loader = make_split_loader(
        [eval_view], "evaluation", normalization, cfg, device,
        shuffle=False, subsampling_factor=int(cfg["preprocessing"]["subsampling_factor"]),
    )
    total: np.ndarray | None = None
    count = 0
    alphas = torch.linspace(0.0, 1.0, steps, device=device)
    for batch in loader:
        x = batch[0].to(device, non_blocking=True)
        batch_grad = torch.zeros_like(x)
        for alpha in alphas:
            interpolated = (x * alpha).detach().requires_grad_(True)
            output = model(interpolated)
            grad = torch.autograd.grad(output.sum(), interpolated, retain_graph=False, create_graph=False)[0]
            batch_grad += grad.detach()
        attribution = x * (batch_grad / float(steps))
        profile = torch.sum(torch.abs(attribution), dim=(0, 1)).detach().cpu().numpy().astype(np.float64)
        total = profile if total is None else total + profile
        count += int(x.shape[0] * x.shape[1])
    if total is None or count == 0:
        raise RuntimeError("Cannot compute XAI profile on an empty event sample")
    importance = total / count
    peak = float(np.max(importance))
    if peak > 0.0:
        importance /= peak
    factor = int(cfg["preprocessing"]["subsampling_factor"])
    time_ps = np.asarray(view.relative_time_ps, dtype=np.float64)[::factor]
    if time_ps.size != importance.size:
        # The retained modes each use one waveform family, so this should only
        # occur if a future input transform changes component length semantics.
        time_ps = np.linspace(float(view.relative_time_ps[0]), float(view.relative_time_ps[-1]), importance.size)
    return time_ps, importance
def _threshold_crossing_matrix(view: PreparedDataset, thresholds_mV: np.ndarray, *, chunk_size: int = 2048) -> np.ndarray:
    """Raw-window threshold pair differences relative to the target LED.
    Prepared windows are baseline-corrected, polarity-oriented raw native samples.
    For each threshold we take the final rising crossing before the pulse maximum,
    interpolate between native samples, and express the pair crossing difference
    relative to the exact target LED pair. No denoising is consulted.
    """
    waves = view.windows_mV
    times = np.asarray(view.relative_time_ps, dtype=np.float64)
    anchors = view.window_anchor_time_fs
    leds = view.led_time_fs
    if anchors is None:
        raise ValueError("Multithreshold extraction requires saved native window anchors")
    n = int(waves.shape[0])
    thresholds = np.asarray(thresholds_mV, dtype=np.float64)
    out = np.full((n, thresholds.size), np.nan, dtype=np.float64)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        block = np.asarray(waves[start:stop], dtype=np.float64)
        for local in range(block.shape[0]):
            event = start + local
            detector_crossings: list[np.ndarray] = []
            for det in range(2):
                y = block[local, det]
                peak = int(np.nanargmax(y))
                crossing_values = np.full(thresholds.size, np.nan, dtype=np.float64)
                if peak > 0:
                    y0, y1 = y[:peak], y[1:peak + 1]
                    finite = np.isfinite(y0) & np.isfinite(y1)
                    for j, threshold in enumerate(thresholds):
                        crossing = finite & (y0 < threshold) & (y1 >= threshold)
                        loc = np.flatnonzero(crossing)
                        if loc.size == 0:
                            continue
                        i = int(loc[-1])
                        if y1[i] == y0[i]:
                            continue
                        fraction = (threshold - y0[i]) / (y1[i] - y0[i])
                        sample_rel_anchor_ps = times[i] + fraction * (times[i + 1] - times[i])
                        anchor_rel_led_ps = (float(anchors[event, det]) - float(leds[event, det])) / 1000.0
                        crossing_values[j] = sample_rel_anchor_ps + anchor_rel_led_ps
                detector_crossings.append(crossing_values)
            out[event] = detector_crossings[0] - detector_crossings[1]
    return out

def _multithreshold_candidates(config: dict[str, Any], threshold_count: int) -> list[dict[str, Any]]:
    minimum = int(config["min_thresholds"])
    maximum = min(int(config["max_thresholds"]), threshold_count)
    indices = range(threshold_count)
    combos = [combo for m in range(minimum, maximum + 1) for combo in itertools.combinations(indices, m)]
    output: list[dict[str, Any]] = []
    for combo in combos:
        for kernel in config["kernels"]:
            gammas = config["gamma_values"] if str(kernel) == "rbf" else ["scale"]
            for c, epsilon, gamma in itertools.product(config["C_values"], config["epsilon_values_ps"], gammas):
                output.append({
                    "threshold_indices": list(combo), "kernel": str(kernel),
                    "C": float(c), "epsilon_ps": float(epsilon), "gamma": gamma,
                })
    return output


def _selection_metric_summary(
    residual_parts: list[np.ndarray],
) -> tuple[np.ndarray, dict[str, Any], list[float]]:
    if not residual_parts:
        raise RuntimeError("Selection procedure produced no score residuals")
    parts = [
        np.asarray(values, dtype=np.float64).reshape(-1)
        for values in residual_parts
    ]
    if any(
        values.size == 0 or np.any(~np.isfinite(values))
        for values in parts
    ):
        raise RuntimeError(
            "Selection procedure produced empty/non-finite score residuals"
        )
    fold_ctrs = [
        float(residual_metrics(values)["ctr_ps"])
        for values in parts
    ]
    combined = np.concatenate(parts)
    simple = residual_metrics(combined)
    selection_ctr = (
        fold_ctrs[0]
        if len(fold_ctrs) == 1
        else float(np.mean(fold_ctrs))
    )
    fold_std = (
        float(np.std(np.asarray(fold_ctrs, dtype=np.float64), ddof=1))
        if len(fold_ctrs) > 1
        else float("nan")
    )
    metrics = {
        "n": int(simple["n"]),
        "ctr_ps": float(selection_ctr),
        "ctr_err_ps": fold_std,
        "mean_ps": float(simple["mean_ps"]),
        "std_ps": float(selection_ctr / FWHM_PER_SIGMA),
        "rmse_ps": float(simple["rmse_ps"]),
        "bias_ps": float(simple["bias_ps"]),
        "dev_ndof": float("nan"),
        "bin_ps": float("nan"),
        "phase_ps": float("nan"),
        "phase_ctr_std_ps": fold_std,
    }
    return combined, metrics, fold_ctrs


def _waveform_selection_candidate(
    study: dict[str, Any],
    dataset: PreparedDataset,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    file_id: int,
    mode: str,
    window: dict[str, Any],
    variant: str,
    subsampling: int,
    space: dict[str, Any],
    overrides: dict[str, Any],
    candidate_id: int,
    work_root: Path,
    logger: Any,
    normalization_cache: dict[str, Normalization],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], list[float]]:
    """Evaluate one candidate with holdout or K-fold selection only."""
    source = input_variant_dataset_view(dataset, variant)
    input_waveforms, target = CHANNEL_MODES[mode]
    view = prediction_window_dataset_view(
        source,
        input_waveforms=input_waveforms,
        target=target,
        before_ns=float(window["before_ns"]),
        after_ns=float(window["after_ns"]),
    )
    residual_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []
    base_seed = int(study["validation"]["seed"])
    for split_index, (train_pool, score_idx) in enumerate(splits):
        candidate_seed = _seed_for(
            base_seed,
            file_id,
            mode,
            window["id"],
            variant,
            subsampling,
            space["id"],
            candidate_id,
            split_index,
        )
        preview_cfg = _candidate_training_config(
            study,
            space,
            overrides,
            mode=mode,
            subsampling=subsampling,
            train_dir=work_root / f"s{split_index}",
            seed=candidate_seed,
            final=False,
        )
        if preview_cfg["model"]["type"] == "linear_svr":
            fit_idx = np.asarray(train_pool, dtype=np.int64)
            early_idx = fit_idx
        else:
            fit_idx, early_idx = _fit_early_split(
                np.asarray(train_pool, dtype=np.int64),
                fraction=_early_fraction(study, preview_cfg),
                seed=_seed_for(
                    base_seed,
                    file_id,
                    mode,
                    "early",
                    split_index,
                ),
            )
        fold_view = replace(
            view,
            train=fit_idx,
            validation=early_idx,
            evaluation=np.asarray(score_idx, dtype=np.int64),
        )
        cached_normalization = _normalization_for_fit_subset(
            fold_view,
            fit_idx,
            subsampling=subsampling,
            cache=normalization_cache,
        )
        model, normalization, _summary = _train_in_memory(
            preview_cfg,
            fold_view,
            logger=logger,
            data_view={
                "stage": "selection",
                "split": split_index,
                "candidate_id": candidate_id,
            },
            normalization_override=cached_normalization,
        )
        residual = _predict_indices(
            model,
            normalization,
            preview_cfg,
            fold_view,
            np.asarray(score_idx, dtype=np.int64),
        )
        residual_parts.append(np.asarray(residual, dtype=np.float64))
        score_parts.append(np.asarray(score_idx, dtype=np.int64))
        _cleanup_training(
            model,
            Path(preview_cfg["output"]["train_dir"]),
        )
    combined, metrics, fold_ctrs = _selection_metric_summary(residual_parts)
    return np.concatenate(score_parts), combined, metrics, fold_ctrs


def _waveform_evaluate_selected(
    study: dict[str, Any],
    dataset: PreparedDataset,
    train_pool: np.ndarray,
    evaluation: np.ndarray,
    *,
    file_id: int,
    mode: str,
    window: dict[str, Any],
    variant: str,
    subsampling: int,
    space: dict[str, Any],
    overrides: dict[str, Any],
    candidate_id: int,
    work_dir: Path,
    logger: Any,
    normalization_cache: dict[str, Normalization],
    checkpoint_path: Path | None = None,
    resume_model_path: Path | None = None,
    compute_xai: bool = False,
    return_train_residual: bool = False,
) -> tuple[
    np.ndarray,
    dict[str, Any],
    dict[str, Any],
    tuple[np.ndarray, np.ndarray] | None,
    np.ndarray | None,
]:
    """Fit the selected final model once and evaluate requested populations.

    ``return_train_residual`` adds one extra inference pass on the development
    population.  It never performs a second fit and is used only for final
    train/development diagnostics and centered correction statistics.
    """
    source = input_variant_dataset_view(dataset, variant)
    input_waveforms, target = CHANNEL_MODES[mode]
    view = prediction_window_dataset_view(
        source,
        input_waveforms=input_waveforms,
        target=target,
        before_ns=float(window["before_ns"]),
        after_ns=float(window["after_ns"]),
    )
    seed = _seed_for(
        int(study["validation"]["seed"]),
        file_id,
        mode,
        space["id"],
        candidate_id,
        "evaluation",
        int(np.asarray(evaluation).size),
    )
    cfg = _candidate_training_config(
        study,
        space,
        overrides,
        mode=mode,
        subsampling=subsampling,
        train_dir=work_dir,
        seed=seed,
        final=checkpoint_path is not None,
    )
    train_pool = np.asarray(train_pool, dtype=np.int64)
    evaluation = np.asarray(evaluation, dtype=np.int64)
    if cfg["model"]["type"] == "linear_svr":
        fit_idx = train_pool
        early_idx = train_pool
    else:
        fit_idx, early_idx = _fit_early_split(
            train_pool,
            fraction=_early_fraction(study, cfg),
            seed=_seed_for(
                int(study["validation"]["seed"]),
                file_id,
                mode,
                "early",
                "eval",
            ),
        )
    eval_view = replace(
        view,
        train=fit_idx,
        validation=early_idx,
        evaluation=evaluation,
    )
    cached_normalization = _normalization_for_fit_subset(
        eval_view,
        fit_idx,
        subsampling=subsampling,
        cache=normalization_cache,
    )
    model, normalization, summary = _train_in_memory(
        cfg,
        eval_view,
        logger=logger,
        data_view={"stage": "evaluation", "candidate_id": candidate_id},
        normalization_override=cached_normalization,
    )
    residual = _predict_indices(
        model,
        normalization,
        cfg,
        eval_view,
        evaluation,
    )
    _fit, metrics = _fit_row(
        residual,
        method=f"Evaluation {space['id']}",
        fit_config=study["fit"],
    )

    train_residual: np.ndarray | None = None
    if return_train_residual:
        train_residual = _predict_indices(
            model,
            normalization,
            cfg,
            eval_view,
            train_pool,
        )

    xai_profile = None
    if compute_xai:
        xai_cfg = study.get("reporting", {}).get("xai", {}) or {}
        if bool(xai_cfg.get("enabled", False)):
            xai_profile = _integrated_gradient_profile(
                model,
                normalization,
                cfg,
                eval_view,
                evaluation,
                max_events=int(xai_cfg.get("max_events", 512)),
                steps=int(xai_cfg.get("integrated_gradient_steps", 16)),
            )
    meta = {
        "best_epoch": int(summary.get("best_epoch", 0)),
        "normalization": summary.get("normalization", {}),
        "model_type": cfg["model"]["type"],
    }
    if resume_model_path is not None:
        resume_model_path.parent.mkdir(parents=True, exist_ok=True)
        # Trusted local artifact used only to enrich the same experiment later
        # (e.g. XAI enabled after the original run).
        torch.save(model.to("cpu"), resume_model_path)
    _cleanup_training(model, work_dir, keep_best=checkpoint_path)
    return residual, metrics, meta, xai_profile, train_residual


def _resolved_hyperparameters(
    space: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    cfg = _apply_overrides(space["base_train_config"], overrides)
    return {
        "model": copy.deepcopy(cfg.get("model", {})),
        "optimizer": copy.deepcopy(cfg.get("optimizer", {})),
        "training": {
            key: copy.deepcopy(value)
            for key, value in cfg.get("training", {}).items()
            if key
            in {
                "batch_size",
                "epochs",
                "epochs_per_unit",
                "early_stopping_patience",
                "unit_early_stopping_patience",
                "min_unit_improvement_ps",
                "min_relative_unit_improvement",
                "early_stop_fraction",
            }
        },
    }


def _candidate_descriptor(
    space: dict[str, Any],
    mode: str,
    window: dict[str, Any],
    variant: str,
    subsampling: int,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    return {
        "family": space["id"],
        "mode": mode,
        "window": window["id"],
        "variant": variant,
        "subsampling": int(subsampling),
        "overrides": overrides,
        "resolved_hyperparameters": _resolved_hyperparameters(space, overrides),
    }


def _compact_candidate_params(overrides: dict[str, Any]) -> str:
    """Short hyperparameter string for INFO logs only."""
    aliases = {
        "epsilon_values": "eps",
        "hidden_units": "hidden",
        "latent_units": "latent",
        "learning_rate": "lr",
        "weight_decay": "wd",
        "batch_size": "batch",
        "epochs_per_unit": "ep/unit",
        "early_stopping_patience": "patience",
    }
    parts: list[str] = []
    for key, value in overrides.items():
        short_key = aliases.get(key.rsplit(".", 1)[-1], key.rsplit(".", 1)[-1])
        if isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
        value_text = f"{value:g}" if isinstance(value, float) else str(value)
        parts.append(f"{short_key}={value_text}")
    return " | ".join(parts) if parts else "default params"



def _select_waveform_space(
    study: dict[str, Any],
    dataset: PreparedDataset,
    selection_indices: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    file_id: int,
    mode: str,
    mode_id: int,
    space: dict[str, Any],
    model_id: int,
    codebooks: dict[str, dict[str, int]],
    candidate_ids: dict[str, int],
    candidate_manifest: dict[str, Any],
    work_root: Path,
    logger: Any,
    normalization_cache: dict[str, Normalization],
    voltage: float,
    result_rows: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Select one waveform model with candidate-level persistent resume.

    Every completed candidate stores its score residual and metrics in a small
    scientific cache. On a later --resume the candidate is reconstructed from
    that cache and training is skipped entirely.
    """
    combinations = _waveform_candidate_combinations(
        study,
        space,
        mode=mode,
        seed=_seed_for(
            int(study["validation"]["seed"]),
            file_id,
            mode,
            space["id"],
            "search",
        ),
    )
    best: dict[str, Any] | None = None
    total = len(combinations)
    last_scope: tuple[str, str, int] | None = None

    for sequence, (window, variant, subsampling, overrides) in enumerate(
        combinations,
        start=1,
    ):
        scope = (str(window["id"]), str(variant), int(subsampling))
        if scope != last_scope:
            start_ns = float(window.get("start_ns", float("nan")))
            end_ns = float(window.get("end_ns", float("nan")))
            if np.isfinite(start_ns) and np.isfinite(end_ns):
                window_text = f"[{start_ns:g},{end_ns:+g}] ns"
            else:
                window_text = str(window["id"])
            logger.info(
                "%s search | window=%s | input=%s | subsampling=%d | candidates=%d",
                space["id"],
                window_text,
                variant,
                int(subsampling),
                total,
            )
            last_scope = scope

        descriptor = _candidate_descriptor(
            space,
            mode,
            window,
            variant,
            subsampling,
            overrides,
        )
        candidate_descriptor_hash = canonical_hash(descriptor)
        candidate_id = candidate_ids.setdefault(
            candidate_descriptor_hash,
            len(candidate_ids),
        )
        candidate_manifest[str(candidate_id)] = descriptor

        cache_payload = _selection_cache_payload(
            study,
            dataset,
            splits,
            mode=mode,
            descriptor=descriptor,
            space=space,
        )
        cache_key = canonical_hash(cache_payload)
        cache_npz, cache_json = _candidate_cache_paths(
            study,
            file_id=file_id,
            mode_id=mode_id,
            model_name=space["id"],
            cache_key=cache_key,
            kind="selection",
        )
        cached = _load_selection_cache(
            cache_npz,
            cache_json,
            cache_key=cache_key,
        )

        if cached is not None:
            score_idx, residual, metrics, fold_ctrs = cached
            logger.info(
                "Candidate %d/%d | REUSED | %s | s-CTR %.1f ps",
                sequence,
                total,
                _compact_candidate_params(overrides),
                float(metrics["ctr_ps"]),
            )
        else:
            candidate_work = (
                work_root
                / f"select_f{file_id}_m{mode_id}_model{model_id}_c{candidate_id}"
            )
            try:
                score_idx, residual, metrics, fold_ctrs = _waveform_selection_candidate(
                    study,
                    dataset,
                    splits,
                    file_id=file_id,
                    mode=mode,
                    window=window,
                    variant=variant,
                    subsampling=int(subsampling),
                    space=space,
                    overrides=overrides,
                    candidate_id=candidate_id,
                    work_root=candidate_work,
                    logger=logger,
                    normalization_cache=normalization_cache,
                )
                _save_selection_cache(
                    cache_npz,
                    cache_json,
                    cache_key=cache_key,
                    metrics=metrics,
                    fold_ctrs=fold_ctrs,
                    score_indices=score_idx,
                    residual=residual,
                )
            finally:
                shutil.rmtree(candidate_work, ignore_errors=True)
            logger.info(
                "Candidate %d/%d | %s | s-CTR %.1f ps",
                sequence,
                total,
                _compact_candidate_params(overrides),
                float(metrics["ctr_ps"]),
            )

        if result_rows is not None:
            result_rows.append(
                {
                    "stage": _STAGE_OOF,
                    "file_id": file_id,
                    "mode_id": mode_id,
                    "model_id": model_id,
                    "candidate_id": candidate_id,
                    "window_id": codebooks["window"][window["id"]],
                    "variant_id": codebooks["variant"][variant],
                    "subsampling": int(subsampling),
                    "selected": 0,
                    "coverage": 1.0,
                    "voltage_V": voltage,
                    **metrics,
                }
            )

        item = {
            "candidate_id": candidate_id,
            "window": window,
            "variant": variant,
            "subsampling": int(subsampling),
            "overrides": overrides,
            "score_indices": score_idx,
            "score_residual": residual,
            "metrics": metrics,
            "fold_ctrs": fold_ctrs,
            "resume_cache_key": cache_key,
        }
        if best is None or float(metrics["ctr_ps"]) < float(best["metrics"]["ctr_ps"]):
            best = item

    if best is None:
        raise RuntimeError(f"No successful candidate for {space['id']} | mode={mode}")

    if result_rows is not None:
        for row in result_rows:
            if (
                int(float(row.get("stage", -1))) == _STAGE_OOF
                and int(float(row.get("file_id", -1))) == file_id
                and int(float(row.get("mode_id", -1))) == mode_id
                and int(float(row.get("model_id", -1))) == model_id
            ):
                row["selected"] = int(
                    int(float(row.get("candidate_id", -2)))
                    == int(best["candidate_id"])
                )
        _flush_progress_rows(result_rows)

    return best


def _multithreshold_feature_cache(
    study: dict[str, Any],
    dataset: PreparedDataset,
    *,
    mode: str,
    window: dict[str, Any],
    cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    key = f"{mode}:{window['id']}"
    if key in cache:
        return cache[key]
    cfg = study["multithreshold"]
    raw = input_variant_dataset_view(dataset, "raw")
    input_waveforms, target = CHANNEL_MODES[mode]
    view = prediction_window_dataset_view(
        raw,
        input_waveforms=input_waveforms,
        target=target,
        before_ns=float(window["before_ns"]),
        after_ns=float(window["after_ns"]),
    )
    thresholds = np.asarray(cfg["thresholds_mV"], dtype=np.float64)
    features = _threshold_crossing_matrix(
        view,
        thresholds,
        chunk_size=int(cfg.get("chunk_size", 2048)),
    )
    led_ps, _ = _target_deltas(
        dataset,
        mode,
        np.arange(dataset.event_id.size),
    )
    entry = {
        "features": features,
        "thresholds": thresholds,
        "led_ps": led_ps,
        "target_correction": led_ps - float(dataset.true_tof_ps),
        "view": view,
    }
    cache[key] = entry
    return entry



def _select_multithreshold(
    study: dict[str, Any],
    dataset: PreparedDataset,
    selection_indices: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    file_id: int,
    mode: str,
    mode_id: int,
    window: dict[str, Any],
    model_id: int,
    codebooks: dict[str, dict[str, int]],
    candidate_ids: dict[str, int],
    candidate_manifest: dict[str, Any],
    voltage: float,
    logger: Any,
    result_rows: list[dict[str, Any]] | None,
    feature_cache: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Select multithreshold SVR with persistent candidate-level resume."""
    if not bool(study["multithreshold"].get("enabled", False)):
        return None

    cfg = study["multithreshold"]
    data = _multithreshold_feature_cache(
        study,
        dataset,
        mode=mode,
        window=window,
        cache=feature_cache,
    )
    thresholds = data["thresholds"]
    features = data["features"]
    led_ps = data["led_ps"]
    target = data["target_correction"]
    best: dict[str, Any] | None = None
    candidates = _multithreshold_candidates(cfg, thresholds.size)

    for sequence, params in enumerate(candidates, start=1):
        columns = params["threshold_indices"]
        valid = np.all(np.isfinite(features[:, columns]), axis=1)
        if not np.all(valid[np.asarray(selection_indices, dtype=np.int64)]):
            continue

        descriptor = {
            "family": _MODEL_MULTITHRESHOLD,
            "mode": mode,
            "window": window["id"],
            "thresholds_mV": thresholds[columns].tolist(),
            **{
                key: value
                for key, value in params.items()
                if key != "threshold_indices"
            },
        }
        descriptor_hash = canonical_hash(descriptor)
        candidate_id = candidate_ids.setdefault(
            descriptor_hash,
            len(candidate_ids),
        )
        candidate_manifest[str(candidate_id)] = descriptor

        cache_payload = _selection_cache_payload(
            study,
            dataset,
            splits,
            mode=mode,
            descriptor={
                **descriptor,
                "multithreshold_config": copy.deepcopy(cfg),
            },
            space=None,
        )
        cache_key = canonical_hash(cache_payload)
        cache_npz, cache_json = _candidate_cache_paths(
            study,
            file_id=file_id,
            mode_id=mode_id,
            model_name=_MODEL_MULTITHRESHOLD,
            cache_key=cache_key,
            kind="selection",
        )
        cached = _load_selection_cache(
            cache_npz,
            cache_json,
            cache_key=cache_key,
        )

        if cached is not None:
            score_indices, combined, metrics, fold_ctrs = cached
            if sequence == 1 or sequence % max(1, len(candidates) // 10) == 0:
                logger.info(
                    "MT-SVR candidate %d/%d | REUSED | s-CTR %.1f ps",
                    sequence,
                    len(candidates),
                    float(metrics["ctr_ps"]),
                )
        else:
            residual_parts: list[np.ndarray] = []
            score_parts: list[np.ndarray] = []
            for train_pool, score_idx in splits:
                estimator = make_pipeline(
                    StandardScaler(),
                    SVR(
                        kernel=params["kernel"],
                        C=params["C"],
                        epsilon=params["epsilon_ps"],
                        gamma=params["gamma"],
                    ),
                )
                estimator.fit(
                    features[np.ix_(train_pool, columns)],
                    target[train_pool],
                )
                correction = estimator.predict(
                    features[np.ix_(score_idx, columns)]
                )
                residual_parts.append(
                    led_ps[score_idx]
                    - correction
                    - float(dataset.true_tof_ps)
                )
                score_parts.append(np.asarray(score_idx, dtype=np.int64))

            combined, metrics, fold_ctrs = _selection_metric_summary(residual_parts)
            score_indices = np.concatenate(score_parts)
            _save_selection_cache(
                cache_npz,
                cache_json,
                cache_key=cache_key,
                metrics=metrics,
                fold_ctrs=fold_ctrs,
                score_indices=score_indices,
                residual=combined,
            )

        if result_rows is not None:
            result_rows.append(
                {
                    "stage": _STAGE_OOF,
                    "file_id": file_id,
                    "mode_id": mode_id,
                    "model_id": model_id,
                    "candidate_id": candidate_id,
                    "window_id": codebooks["window"][window["id"]],
                    "variant_id": 0,
                    "subsampling": 1,
                    "selected": 0,
                    "coverage": 1.0,
                    "voltage_V": voltage,
                    **metrics,
                }
            )

        item = {
            "candidate_id": candidate_id,
            "window": window,
            "params": params,
            "features": features,
            "led_ps": led_ps,
            "target_correction": target,
            "score_indices": score_indices,
            "score_residual": combined,
            "metrics": metrics,
            "fold_ctrs": fold_ctrs,
            "window_id": codebooks["window"][window["id"]],
            "resume_cache_key": cache_key,
        }
        if best is None or float(metrics["ctr_ps"]) < float(best["metrics"]["ctr_ps"]):
            best = item

    if best is None:
        logger.warning(
            "No multithreshold candidate covers the complete selection population | mode=%s",
            mode,
        )
        return None

    if result_rows is not None:
        for row in result_rows:
            if (
                int(float(row.get("stage", -1))) == _STAGE_OOF
                and int(float(row.get("file_id", -1))) == file_id
                and int(float(row.get("mode_id", -1))) == mode_id
                and int(float(row.get("model_id", -1))) == model_id
            ):
                row["selected"] = int(
                    int(float(row.get("candidate_id", -2)))
                    == int(best["candidate_id"])
                )
        _flush_progress_rows(result_rows)

    logger.info(
        "Selected multithreshold SVR | mode=%s | s-CTR %.1f ps | thresholds=%s",
        mode,
        float(best["metrics"]["ctr_ps"]),
        thresholds[best["params"]["threshold_indices"]].tolist(),
    )
    return best



def _multithreshold_evaluate(
    study: dict[str, Any],
    dataset: PreparedDataset,
    train_pool: np.ndarray,
    evaluation: np.ndarray,
    selected: dict[str, Any],
    *,
    return_train_residual: bool = False,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray | None]:
    params = selected["params"]
    columns = params["threshold_indices"]
    features = selected["features"]
    valid = np.all(np.isfinite(features[:, columns]), axis=1)
    train_pool = np.asarray(train_pool, dtype=np.int64)
    evaluation = np.asarray(evaluation, dtype=np.int64)
    if not np.all(valid[train_pool]) or not np.all(valid[evaluation]):
        raise RuntimeError(
            "Selected multithreshold model lacks a threshold crossing "
            "for at least one evaluation event"
        )
    estimator = make_pipeline(
        StandardScaler(),
        SVR(
            kernel=params["kernel"],
            C=params["C"],
            epsilon=params["epsilon_ps"],
            gamma=params["gamma"],
        ),
    )
    estimator.fit(
        features[np.ix_(train_pool, columns)],
        selected["target_correction"][train_pool],
    )
    correction = estimator.predict(features[np.ix_(evaluation, columns)])
    residual = (
        selected["led_ps"][evaluation]
        - correction
        - float(dataset.true_tof_ps)
    )
    _fit, metrics = _fit_row(
        residual,
        method="Evaluation multithreshold SVR",
        fit_config=study["fit"],
    )
    train_residual: np.ndarray | None = None
    if return_train_residual:
        train_correction = estimator.predict(
            features[np.ix_(train_pool, columns)]
        )
        train_residual = (
            selected["led_ps"][train_pool]
            - train_correction
            - float(dataset.true_tof_ps)
        )
    return residual, metrics, train_residual


def _report_base(
    *,
    root_file: Path,
    file_id: int,
    voltage: float,
    mode: str,
    model: str,
    stage_name: str,
    metrics: dict[str, Any],
    selected: int = 1,
) -> dict[str, Any]:
    return {
        "file": root_file.name,
        "file_id": file_id,
        "voltage_V": voltage,
        "mode": mode,
        "model": model,
        "stage_name": stage_name,
        "selected": int(selected),
        "n": int(metrics.get("n", 0)),
        "mean_ps": float(metrics.get("mean_ps", float("nan"))),
        "std_ps": float(metrics.get("std_ps", float("nan"))),
        "ctr_ps": float(metrics.get("ctr_ps", float("nan"))),
        "rmse_ps": float(metrics.get("rmse_ps", float("nan"))),
        "bias_ps": float(metrics.get("bias_ps", float("nan"))),
    }


def _report_model_details(
    row: dict[str, Any],
    *,
    chosen: dict[str, Any],
    space: dict[str, Any] | None,
    strategy: str,
) -> dict[str, Any]:
    row["candidate_id"] = int(chosen["candidate_id"])
    window = chosen["window"]
    row["window_id"] = window["id"]
    row["window_before_ns"] = float(window["before_ns"])
    row["window_after_ns"] = float(window["after_ns"])
    row["validation_strategy"] = strategy
    if space is None:
        params = chosen["params"]
        row["variant"] = "raw"
        row["subsampling"] = 1
        row["hyperparameters_json"] = json.dumps(
            {
                "thresholds_mV": np.asarray(
                    chosen.get("threshold_values", []),
                    dtype=float,
                ).tolist(),
                **{
                    key: value
                    for key, value in params.items()
                    if key != "threshold_indices"
                },
            },
            sort_keys=True,
        )
    else:
        row["variant"] = chosen["variant"]
        row["subsampling"] = int(chosen["subsampling"])
        row["hyperparameters_json"] = json.dumps(
            _resolved_hyperparameters(space, chosen["overrides"]),
            sort_keys=True,
        )
    return row


def _plot_inclusion(
    ctr: float,
    led_ctr: float,
    ratio_limit: float,
) -> tuple[float, int]:
    ratio = (
        float(ctr / led_ctr)
        if np.isfinite(ctr)
        and np.isfinite(led_ctr)
        and led_ctr > 0
        else float("nan")
    )
    # Non-finite model performance is not useful in aggregate figures.
    included = int(np.isfinite(ratio) and ratio <= float(ratio_limit))
    return ratio, included


def _artifact_model_key(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(name)).strip("_") or "model"


def _save_evaluation_artifact(
    output: Path,
    *,
    file_id: int,
    mode_id: int,
    development: np.ndarray,
    blind: np.ndarray,
    development_led_residual: np.ndarray,
    blind_led_residual: np.ndarray,
    development_cfd: np.ndarray | None,
    blind_cfd: np.ndarray | None,
    true_tof_ps: float,
    final_candidates: list[tuple[str, dict[str, Any], dict[str, Any] | None, np.ndarray, np.ndarray, dict[str, Any]]],
) -> None:
    root = output / "artifacts" / "evaluations"
    root.mkdir(parents=True, exist_ok=True)
    stem = f"f{file_id}_m{mode_id}"
    arrays: dict[str, np.ndarray] = {
        "development": np.asarray(development, dtype=np.int64),
        "blind": np.asarray(blind, dtype=np.int64),
        "train_led": np.asarray(development_led_residual, dtype=np.float64),
        "blind_led": np.asarray(blind_led_residual, dtype=np.float64),
    }
    if development_cfd is not None:
        arrays["train_cfd"] = np.asarray(development_cfd, dtype=np.float64) - float(true_tof_ps)
    if blind_cfd is not None:
        arrays["blind_cfd"] = np.asarray(blind_cfd, dtype=np.float64) - float(true_tof_ps)
    model_keys: dict[str, str] = {}
    for name, _chosen, _space, blind_residual, train_residual, _report in final_candidates:
        key = _artifact_model_key(name)
        model_keys[name] = key
        arrays[f"train__{key}"] = np.asarray(train_residual, dtype=np.float64)
        arrays[f"blind__{key}"] = np.asarray(blind_residual, dtype=np.float64)
    np.savez_compressed(root / f"{stem}.npz", **arrays)
    atomic_json(root / f"{stem}.json", {"models": model_keys})


def _plot_xai_profile(path: Path, *, time_ps: np.ndarray, importance: np.ndarray, title: str, dpi: int) -> None:
    fig, axis = plt.subplots(figsize=(8.0, 3.6))
    axis.plot(np.asarray(time_ps) / 1000.0, importance)
    axis.set_xlabel("Relative time [ns]")
    axis.set_ylabel("Normalized |attribution|")
    axis.set_title(title)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _repair_additive_artifacts(
    config: dict[str, Any],
    output: Path,
    manifest: dict[str, Any],
    *,
    logger: Any,
) -> dict[str, Any]:
    """Generate only missing report/XAI artifacts from a compatible completed run.

    Numeric result CSVs and trained checkpoints are never overwritten here.
    Evaluation arrays saved by the original run are sufficient for standard
    plots. XAI loads the trusted local serialized final model and therefore does
    not retrain the selected model.
    """
    codebooks = manifest.get("codebooks", {})
    file_codes = codebooks.get("file", {})
    mode_codes = codebooks.get("mode", {})
    dpi = int(config["reporting"]["dpi"])
    ratio_limit = float(config["reporting"]["max_ctr_to_led_ratio"])
    bootstrap_samples = int(config["reporting"]["ctr_uncertainty_bootstrap_samples"])
    base_seed = int(config["validation"]["seed"])
    plots_root = output / "plots"
    repaired: list[str] = []

    root_by_name = {path.name: path for path in discover_root_files(config)}
    spaces = {str(space["id"]): space for space in config.get("_model_spaces", [])}

    for file_name, file_id_raw in file_codes.items():
        file_id = int(file_id_raw)
        root_file = root_by_name.get(file_name)
        if root_file is None:
            continue
        dataset: PreparedDataset | None = None
        for mode in config["channel_modes"]:
            if mode not in mode_codes:
                continue
            mode_id = int(mode_codes[mode])
            stem = f"f{file_id}_m{mode_id}"
            npz_path = output / "artifacts" / "evaluations" / f"{stem}.npz"
            map_path = output / "artifacts" / "evaluations" / f"{stem}.json"
            if not npz_path.is_file() or not map_path.is_file():
                logger.warning("Resume artifact cache missing for %s/%s; existing numeric results are preserved", file_name, mode)
                continue
            arrays = np.load(npz_path)
            with map_path.open("r", encoding="utf-8") as stream:
                model_map = json.load(stream).get("models", {})
            train_methods: dict[str, np.ndarray] = {_MODEL_LED: arrays["train_led"]}
            blind_methods: dict[str, np.ndarray] = {_MODEL_LED: arrays["blind_led"]}
            if cfd_enabled(config, mode) and "train_cfd" in arrays and "blind_cfd" in arrays:
                train_methods[_MODEL_CFD] = arrays["train_cfd"]
                blind_methods[_MODEL_CFD] = arrays["blind_cfd"]
            for model_name, key in model_map.items():
                if f"train__{key}" in arrays and f"blind__{key}" in arrays:
                    train_methods[model_name] = arrays[f"train__{key}"]
                    blind_methods[model_name] = arrays[f"blind__{key}"]

            for split_name, methods, seed_tag in (
                ("train_distributions", train_methods, "train_distribution_bootstrap"),
                ("blind_distributions", blind_methods, "blind_distribution_bootstrap"),
            ):
                path = plots_root / split_name / f"{Path(file_name).stem}__{mode}.png"
                if not path.is_file():
                    plot_result_distribution(
                        path, mode=mode, methods=methods, dpi=dpi,
                        ratio_limit=ratio_limit, bootstrap_samples=bootstrap_samples,
                        seed=_seed_for(base_seed, file_id, mode, seed_tag),
                        split_label=("Train / development" if split_name.startswith("train") else "Blind"),
                    )
                    repaired.append(str(path.relative_to(output)))

            # TOP/WORST can be rebuilt from stored residuals + prepared waveforms.
            requested_top = int(config["reporting"].get("top_corrections_k", 3))
            requested_worst = int(config["reporting"].get("worst_corrections_k", 3))
            if (requested_top > 0 or requested_worst > 0) and model_map:
                candidates = []
                for model_name, key in model_map.items():
                    if f"blind__{key}" in arrays:
                        candidates.append((float(residual_metrics(arrays[f"blind__{key}"])["ctr_ps"]), model_name, key))
                if candidates:
                    _ctr, best_name, best_key = min(candidates)
                    if dataset is None:
                        dataset = prepare_file_dataset(config, root_file, rebuild=False, logger=logger)
                    meta_match = None
                    for meta in manifest.get("final_models", {}).values():
                        if int(meta.get("file_id", -1)) == file_id and meta.get("mode") == mode and meta.get("model") == best_name:
                            meta_match = meta
                            break
                    if meta_match is not None:
                        variant = str(meta_match["variant"])
                        source = input_variant_dataset_view(dataset, variant)
                        input_waveforms, target = CHANNEL_MODES[mode]
                        materialized = config["preprocessing"]["materialized_window_ns"]
                        full_view = prediction_window_dataset_view(
                            source, input_waveforms=input_waveforms, target=target,
                            before_ns=float(materialized["before"]), after_ns=float(materialized["after"]),
                        )
                        kwargs = dict(
                            time_ps=np.asarray(full_view.relative_time_ps, dtype=np.float64),
                            waveforms=np.asarray(full_view.windows_mV[arrays["blind"]], dtype=np.float32),
                            led_residual=arrays["blind_led"],
                            corrected_residual=arrays[f"blind__{best_key}"],
                            led_center_ps=float(meta_match["reporting_led_center_ps"]),
                            correction_center_ps=float(meta_match["reporting_correction_center_ps"]),
                            model=best_name, mode=mode, dpi=dpi,
                            window_before_ns=float(meta_match["window"]["before_ns"]),
                            window_after_ns=float(meta_match["window"]["after_ns"]),
                            event_ids=np.asarray(dataset.event_id[arrays["blind"]]),
                        )
                        for selection, k in (("top", requested_top), ("worst", requested_worst)):
                            if k <= 0:
                                continue
                            path = plots_root / f"{selection}_corrections" / f"{Path(file_name).stem}__{mode}.png"
                            if not path.is_file():
                                plot_correction_examples(path, selection=selection, k=k, **kwargs)
                                repaired.append(str(path.relative_to(output)))

            xai_cfg = config.get("reporting", {}).get("xai", {}) or {}
            if bool(xai_cfg.get("enabled", False)):
                if dataset is None:
                    dataset = prepare_file_dataset(config, root_file, rebuild=False, logger=logger)
                blind = np.asarray(arrays["blind"], dtype=np.int64)
                for meta in manifest.get("final_models", {}).values():
                    if int(meta.get("file_id", -1)) != file_id or meta.get("mode") != mode:
                        continue
                    model_name = str(meta.get("model", ""))
                    if model_name not in spaces:
                        continue
                    xai_path = plots_root / "xai" / f"{Path(file_name).stem}__{mode}__{model_name}.png"
                    if xai_path.is_file():
                        continue
                    model_path = output / str(meta.get("resume_model", ""))
                    if not model_path.is_file():
                        logger.warning("Cannot repair XAI for %s/%s/%s: resume model missing", file_name, mode, model_name)
                        continue
                    try:
                        model = torch.load(model_path, map_location="cpu", weights_only=False)
                    except TypeError:  # PyTorch < 2.6
                        model = torch.load(model_path, map_location="cpu")
                    space = spaces[model_name]
                    window = dict(meta["window"])
                    variant = str(meta["variant"])
                    subsampling = int(meta["subsampling"])
                    source = input_variant_dataset_view(dataset, variant)
                    input_waveforms, target = CHANNEL_MODES[mode]
                    view = prediction_window_dataset_view(
                        source, input_waveforms=input_waveforms, target=target,
                        before_ns=float(window["before_ns"]), after_ns=float(window["after_ns"]),
                    )
                    cfg = _candidate_training_config(
                        config, space, dict(meta.get("overrides", {})), mode=mode,
                        subsampling=subsampling, train_dir=output / ".resume_xai",
                        seed=_seed_for(base_seed, file_id, mode, model_name, "resume_xai"), final=False,
                    )
                    normalization = Normalization.from_dict(meta["normalization"])
                    time_ps, importance = _integrated_gradient_profile(
                        model, normalization, cfg, view, blind,
                        max_events=int(xai_cfg.get("max_events", 512)),
                        steps=int(xai_cfg.get("integrated_gradient_steps", 16)),
                    )
                    _plot_xai_profile(xai_path, time_ps=time_ps, importance=importance, title=f"{model_name} · {mode}", dpi=dpi)
                    repaired.append(str(xai_path.relative_to(output)))
                    del model

    manifest["config_hash"] = config.get("_config_hash")
    manifest["core_hash"] = config.get("_core_hash")
    manifest["artifact_hash"] = config.get("_artifact_hash")
    manifest["modes"] = copy.deepcopy(config.get("modes", {}))
    manifest["config_sources"] = copy.deepcopy(config.get("_config_sources", []))
    manifest["reporting"] = copy.deepcopy(config.get("reporting", {}))
    manifest["last_resume_repaired"] = repaired
    atomic_json(output / "manifest.json", manifest)
    atomic_json(
        output / "config_resolved.json",
        {k: v for k, v in config.items() if not str(k).startswith("_")},
    )
    atomic_json(
        output / "run_state.json",
        {
            "schema_version": 1,
            "experiment": config["experiment"]["name"],
            "core_hash": _resume_state_hash(config),
            "artifact_hash": config.get("_artifact_hash"),
            "config_hash": config.get("_config_hash"),
            "status": "complete",
            "row_count": int(manifest.get("row_count", 0)),
        },
    )
    logger.info("Additive resume | repaired %d missing artifacts; numeric results preserved", len(repaired))
    return {"output_dir": str(output), "row_count": int(manifest.get("row_count", 0)), "resumed": True, "repaired_artifacts": repaired}


def run_study(
    config: dict[str, Any],
    *,
    dry_run: bool,
    resume: bool,
    restart: bool,
    rebuild_preprocessing: bool,
    logger: Any,
) -> dict[str, Any]:
    output = Path(config["experiment"]["output_dir"])
    if restart and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    results_path = output / "results.csv"
    summary_path = output / "summary_results.csv"
    nested_path = output / "nested_results.csv"
    manifest_path = output / "manifest.json"

    # Recover the newest complete progressive snapshot before deciding whether
    # this is a completed additive-resume run. This matters on Windows when a
    # previous process finished while results.csv was open in Excel: the newest
    # snapshot then lives in results.pending.csv rather than the locked file.
    if resume:
        pending_candidates = [
            _pending_csv_path(results_path),
            *sorted(output.glob(f"{results_path.stem}.pending.*{results_path.suffix}")),
        ]
        if any(path.is_file() for path in pending_candidates):
            recovered_rows = _read_progressive_rows(results_path)
            if recovered_rows:
                published = _write_csv(
                    results_path,
                    recovered_rows,
                    logger=logger,
                )
                if published:
                    logger.info(
                        "Recovered pending progressive results into %s",
                        results_path,
                    )

    if resume and results_path.is_file() and manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        previous_core = manifest.get("core_hash")
        current_core = config.get("_core_hash")
        if previous_core is None:
            # Backward compatibility: an older run can still resume unchanged,
            # but additive reporting needs the new core fingerprint/artifact cache.
            if manifest.get("config_hash") == config.get("_config_hash"):
                logger.info("Legacy complete result set already exists; reuse %s", output)
                return {"output_dir": str(output), "row_count": int(manifest.get("row_count", 0)), "resumed": True}
            raise RuntimeError(
                "Existing results predate additive-resume fingerprints. The scientific configuration cannot be proven compatible; use --restart once with the new code."
            )
        if previous_core != current_core:
            raise RuntimeError(
                "Existing results have a different training/core configuration. "
                "Reporting, XAI, TOP/WORST and per-mode CFD flags may be changed additively, "
                "but preprocessing/windows/validation/models/enabled modes require a new experiment or --restart."
            )
        return _repair_additive_artifacts(config, output, manifest, logger=logger)

    root_files = discover_root_files(config)
    strategy = str(config["validation"]["strategy"])
    logger.info(
        "Study %s | files=%d | validation=%s | working ML/model implementation preserved",
        config["experiment"]["name"],
        len(root_files),
        strategy,
    )
    if dry_run:
        return {
            "output_dir": str(output),
            "row_count": 0,
            "dry_run": True,
            "files": [str(value) for value in root_files],
            "models": [value["id"] for value in config["_model_spaces"]],
            "multithreshold": bool(
                config["multithreshold"].get("enabled", False)
            ),
            "prepared_dir": config["preprocessing"]["prepared_dir"],
            "selection_store_dir": config["preprocessing"]["selection_store_dir"],
            "validation_strategy": strategy,
        }
    if not root_files:
        raise FileNotFoundError(
            f"No ROOT files match {config['data']['root_glob']} "
            f"in {config['data']['root_folder']}"
        )

    run_state_path = _check_or_create_run_state(
        config,
        output,
        resume=resume,
        logger=logger,
    )
    if resume and (results_path.is_file() or _pending_csv_path(results_path).is_file()):
        logger.info(
            "Partial run resume | loading progressive results from %s%s",
            results_path,
            (" + pending snapshot" if _pending_csv_path(results_path).is_file() else ""),
        )

    codebooks = {
        "file": {path.name: index for index, path in enumerate(root_files)},
        "mode": {
            name: index for index, name in enumerate(config["channel_modes"])
        },
        "model": {
            _MODEL_LED: 0,
            _MODEL_CFD: 1,
            **{
                space["id"]: index + 2
                for index, space in enumerate(config["_model_spaces"])
            },
        },
        "window": {
            window["id"]: index
            for index, window in enumerate(config["windows_ns"])
        },
        "variant": {
            name: index
            for index, name in enumerate(config["preprocessing"]["input_variants"])
        },
    }
    if bool(config["multithreshold"].get("enabled", False)):
        codebooks["model"][_MODEL_MULTITHRESHOLD] = (
            max(codebooks["model"].values()) + 1
        )

    candidate_ids: dict[str, int] = {}
    candidate_manifest: dict[str, Any] = {}
    initial_rows = _read_progressive_rows(results_path) if resume else []
    rows: list[dict[str, Any]] = _ProgressiveResultRows(
        results_path,
        initial_rows,
        logger=logger,
    )
    report_rows: list[dict[str, Any]] = []
    nested_rows: list[dict[str, Any]] = []
    normalization_cache: dict[str, Normalization] = {}
    final_metadata: dict[str, Any] = {}

    dpi = int(config["reporting"]["dpi"])
    base_seed = int(config["validation"]["seed"])
    ratio_limit = float(config["reporting"]["max_ctr_to_led_ratio"])
    bootstrap_samples = int(
        config["reporting"]["ctr_uncertainty_bootstrap_samples"]
    )
    work_root = output / ".work"
    checkpoint_root = output / "models"
    signal_plot_root = output / "preprocessing_examples"
    plots_root = output / "plots"

    for root_file in root_files:
        file_id = codebooks["file"][root_file.name]
        logger.info(
            "File %d/%d | %s",
            file_id + 1,
            len(root_files),
            root_file.name,
        )
        dataset = prepare_file_dataset(
            config,
            root_file,
            rebuild=rebuild_preprocessing,
            logger=logger,
        )
        plot_prepared_signal_examples(
            dataset,
            signal_plot_root / f"{root_file.stem}.png",
            dpi=dpi,
        )
        development, blind = random_dev_blind(
            int(dataset.event_id.size),
            blind_fraction=float(config["validation"]["blind_fraction"]),
            seed=_seed_for(base_seed, file_id, "devblind"),
        )
        voltage = _voltage_from_name(
            root_file.name,
            str(config["reporting"]["voltage_pattern"]),
        )
        mt_feature_cache: dict[str, dict[str, Any]] = {}

        for mode in config["channel_modes"]:
            mode_id = codebooks["mode"][mode]
            use_cfd = cfd_enabled(config, mode)
            mt_window = None
            if bool(config["multithreshold"].get("enabled", False)):
                mt_window = next(
                    window
                    for window in config["windows_ns"]
                    if window["id"]
                    == str(config["multithreshold"]["window_id"])
                )

            # ---------------- optional nested pipeline evaluation ----------------
            if strategy == "nested":
                outer = outer_splits(
                    development,
                    config["validation"],
                    seed=_seed_for(base_seed, file_id, mode, "outer"),
                )
                outer_model_metrics: dict[str, list[float]] = {
                    _MODEL_LED: [],
                    **{space["id"]: [] for space in config["_model_spaces"]},
                }
                if use_cfd:
                    outer_model_metrics[_MODEL_CFD] = []
                if mt_window is not None:
                    outer_model_metrics[_MODEL_MULTITHRESHOLD] = []
                inner_validation = nested_inner_validation(config["validation"])

                for outer_index, (outer_train, outer_test) in enumerate(outer):
                    inner_splits = selection_splits(
                        outer_train,
                        inner_validation,
                        seed=_seed_for(
                            base_seed,
                            file_id,
                            mode,
                            "outer",
                            outer_index,
                            "inner",
                        ),
                    )
                    led_outer, cfd_outer = _target_deltas(
                        dataset,
                        mode,
                        outer_test,
                    )
                    baseline_pairs = [(_MODEL_LED, led_outer - float(dataset.true_tof_ps))]
                    if use_cfd:
                        cfd_outer = _require_cfd(cfd_outer, mode)
                        baseline_pairs.append((_MODEL_CFD, cfd_outer - float(dataset.true_tof_ps)))
                    for model_name, residual in baseline_pairs:
                        metrics = residual_metrics(residual)
                        outer_model_metrics[model_name].append(
                            float(metrics["ctr_ps"])
                        )
                        nested_rows.append(
                            {
                                "file": root_file.name,
                                "file_id": file_id,
                                "voltage_V": voltage,
                                "mode": mode,
                                "outer_fold": outer_index,
                                "model": model_name,
                                "candidate_id": -1,
                                "window_id": "",
                                "window_before_ns": "",
                                "window_after_ns": "",
                                "variant": "",
                                "subsampling": "",
                                "hyperparameters_json": "",
                                **metrics,
                            }
                        )

                    for space in config["_model_spaces"]:
                        model_id = codebooks["model"][space["id"]]
                        chosen = _select_waveform_space(
                            config,
                            dataset,
                            outer_train,
                            inner_splits,
                            file_id=file_id,
                            mode=mode,
                            mode_id=mode_id,
                            space=space,
                            model_id=model_id,
                            codebooks=codebooks,
                            candidate_ids=candidate_ids,
                            candidate_manifest=candidate_manifest,
                            work_root=work_root / f"nested_o{outer_index}",
                            logger=logger,
                            normalization_cache=normalization_cache,
                            voltage=voltage,
                            result_rows=None,
                        )
                        outer_dir = (
                            work_root
                            / f"outer_eval_f{file_id}_m{mode_id}_o{outer_index}_model{model_id}"
                        )
                        (
                            residual,
                            metrics,
                            _meta,
                            _xai,
                            _train_residual,
                        ) = _waveform_evaluate_selected(
                            config,
                            dataset,
                            outer_train,
                            outer_test,
                            file_id=file_id,
                            mode=mode,
                            window=chosen["window"],
                            variant=chosen["variant"],
                            subsampling=chosen["subsampling"],
                            space=space,
                            overrides=chosen["overrides"],
                            candidate_id=chosen["candidate_id"],
                            work_dir=outer_dir,
                            logger=logger,
                            normalization_cache=normalization_cache,
                        )
                        outer_model_metrics[space["id"]].append(
                            float(metrics["ctr_ps"])
                        )
                        nested_rows.append(
                            {
                                "file": root_file.name,
                                "file_id": file_id,
                                "voltage_V": voltage,
                                "mode": mode,
                                "outer_fold": outer_index,
                                "model": space["id"],
                                "candidate_id": chosen["candidate_id"],
                                "window_id": chosen["window"]["id"],
                                "window_before_ns": chosen["window"]["before_ns"],
                                "window_after_ns": chosen["window"]["after_ns"],
                                "variant": chosen["variant"],
                                "subsampling": chosen["subsampling"],
                                "hyperparameters_json": json.dumps(
                                    _resolved_hyperparameters(
                                        space,
                                        chosen["overrides"],
                                    ),
                                    sort_keys=True,
                                ),
                                **metrics,
                            }
                        )

                    if mt_window is not None:
                        selected_mt_outer = _select_multithreshold(
                            config,
                            dataset,
                            outer_train,
                            inner_splits,
                            file_id=file_id,
                            mode=mode,
                            mode_id=mode_id,
                            window=mt_window,
                            model_id=codebooks["model"][_MODEL_MULTITHRESHOLD],
                            codebooks=codebooks,
                            candidate_ids=candidate_ids,
                            candidate_manifest=candidate_manifest,
                            voltage=voltage,
                            logger=logger,
                            result_rows=None,
                            feature_cache=mt_feature_cache,
                        )
                        if selected_mt_outer is not None:
                            (
                                residual,
                                metrics,
                                _train_residual,
                            ) = _multithreshold_evaluate(
                                config,
                                dataset,
                                outer_train,
                                outer_test,
                                selected_mt_outer,
                            )
                            outer_model_metrics[_MODEL_MULTITHRESHOLD].append(
                                float(metrics["ctr_ps"])
                            )
                            params = selected_mt_outer["params"]
                            threshold_values = np.asarray(
                                config["multithreshold"]["thresholds_mV"],
                                float,
                            )[params["threshold_indices"]].tolist()
                            nested_rows.append(
                                {
                                    "file": root_file.name,
                                    "file_id": file_id,
                                    "voltage_V": voltage,
                                    "mode": mode,
                                    "outer_fold": outer_index,
                                    "model": _MODEL_MULTITHRESHOLD,
                                    "candidate_id": selected_mt_outer[
                                        "candidate_id"
                                    ],
                                    "window_id": mt_window["id"],
                                    "window_before_ns": mt_window["before_ns"],
                                    "window_after_ns": mt_window["after_ns"],
                                    "variant": "raw",
                                    "subsampling": 1,
                                    "hyperparameters_json": json.dumps(
                                        {
                                            "thresholds_mV": threshold_values,
                                            **{
                                                key: value
                                                for key, value in params.items()
                                                if key != "threshold_indices"
                                            },
                                        },
                                        sort_keys=True,
                                    ),
                                    **metrics,
                                }
                            )

                nested_led = float(
                    np.mean(outer_model_metrics[_MODEL_LED])
                )
                for model_name, ctr_values in outer_model_metrics.items():
                    if not ctr_values:
                        continue
                    ctr = float(np.mean(ctr_values))
                    spread = (
                        float(np.std(ctr_values, ddof=1))
                        if len(ctr_values) > 1
                        else float("nan")
                    )
                    row = {
                        "file": root_file.name,
                        "file_id": file_id,
                        "voltage_V": voltage,
                        "mode": mode,
                        "model": model_name,
                        "stage_name": "nested",
                        "selected": 1,
                        "n": len(ctr_values),
                        "mean_ps": float("nan"),
                        "std_ps": ctr / FWHM_PER_SIGMA,
                        "ctr_ps": ctr,
                        "ctr_fold_std_ps": spread,
                        "ctr_uncertainty_ps": spread,
                        "rmse_ps": float("nan"),
                        "bias_ps": float("nan"),
                        "led_ctr_ps": nested_led,
                    }
                    ratio, included = _plot_inclusion(
                        ctr,
                        nested_led,
                        ratio_limit,
                    )
                    row["ctr_over_led"] = ratio
                    row["plot_included"] = included
                    report_rows.append(row)

            # -------------------- final development selection --------------------
            final_selection_cfg = (
                nested_inner_validation(config["validation"])
                if strategy == "nested"
                else config["validation"]
            )
            final_splits = selection_splits(
                development,
                final_selection_cfg,
                seed=_seed_for(
                    base_seed,
                    file_id,
                    mode,
                    "final_selection",
                ),
            )
            validation_score_indices = np.unique(
                np.concatenate(
                    [
                        np.asarray(score, dtype=np.int64)
                        for _train, score in final_splits
                    ]
                )
            )
            led_val, cfd_val = _target_deltas(
                dataset,
                mode,
                validation_score_indices,
            )
            led_val_res = led_val - float(dataset.true_tof_ps)
            led_val_metrics = residual_metrics(led_val_res)
            cfd_val_metrics = None
            validation_baselines = [(_MODEL_LED, led_val_metrics)]
            if use_cfd:
                cfd_val = _require_cfd(cfd_val, mode)
                cfd_val_metrics = residual_metrics(cfd_val - float(dataset.true_tof_ps))
                validation_baselines.append((_MODEL_CFD, cfd_val_metrics))
            logger.info(
                "Validation baseline | mode=%s | n=%d | LED s-CTR %.1f ps%s",
                mode, int(led_val_metrics["n"]), float(led_val_metrics["ctr_ps"]),
                (f" | CFD s-CTR {float(cfd_val_metrics['ctr_ps']):.1f} ps" if cfd_val_metrics is not None else " | CFD disabled"),
            )
            selection_stage = "validation"
            for model_name, metrics in validation_baselines:
                report_row = _report_base(
                    root_file=root_file,
                    file_id=file_id,
                    voltage=voltage,
                    mode=mode,
                    model=model_name,
                    stage_name=selection_stage,
                    metrics=metrics,
                )
                report_row["validation_strategy"] = (
                    str(config["validation"]["nested"]["inner_strategy"])
                    if strategy == "nested"
                    else strategy
                )
                ratio, included = _plot_inclusion(
                    float(metrics["ctr_ps"]),
                    float(led_val_metrics["ctr_ps"]),
                    ratio_limit,
                )
                report_row["led_ctr_ps"] = float(led_val_metrics["ctr_ps"])
                report_row["ctr_over_led"] = ratio
                report_row["plot_included"] = included
                report_rows.append(report_row)

            selected_waveforms: list[
                tuple[dict[str, Any], int, dict[str, Any]]
            ] = []
            validation_corrections: dict[
                str,
                tuple[np.ndarray, np.ndarray],
            ] = {}
            for space in config["_model_spaces"]:
                model_id = codebooks["model"][space["id"]]
                chosen = _select_waveform_space(
                    config,
                    dataset,
                    development,
                    final_splits,
                    file_id=file_id,
                    mode=mode,
                    mode_id=mode_id,
                    space=space,
                    model_id=model_id,
                    codebooks=codebooks,
                    candidate_ids=candidate_ids,
                    candidate_manifest=candidate_manifest,
                    work_root=work_root,
                    logger=logger,
                    normalization_cache=normalization_cache,
                    voltage=voltage,
                    result_rows=rows,
                )
                selected_waveforms.append((space, model_id, chosen))
                score_led, _ = _target_deltas(
                    dataset,
                    mode,
                    chosen["score_indices"],
                )
                score_led_res = score_led - float(dataset.true_tof_ps)
                validation_corrections[space["id"]] = (
                    chosen["score_indices"],
                    score_led_res - chosen["score_residual"],
                )
                metrics = chosen["metrics"]
                report_row = _report_base(
                    root_file=root_file,
                    file_id=file_id,
                    voltage=voltage,
                    mode=mode,
                    model=space["id"],
                    stage_name=selection_stage,
                    metrics=metrics,
                )
                report_row = _report_model_details(
                    report_row,
                    chosen=chosen,
                    space=space,
                    strategy=(
                        str(config["validation"]["nested"]["inner_strategy"])
                        if strategy == "nested"
                        else strategy
                    ),
                )
                report_row["ctr_fold_std_ps"] = float(
                    metrics.get("ctr_err_ps", float("nan"))
                )
                report_row["ctr_uncertainty_ps"] = float(
                    metrics.get("ctr_err_ps", float("nan"))
                )
                ratio, included = _plot_inclusion(
                    float(metrics["ctr_ps"]),
                    float(led_val_metrics["ctr_ps"]),
                    ratio_limit,
                )
                report_row["led_ctr_ps"] = float(led_val_metrics["ctr_ps"])
                report_row["ctr_over_led"] = ratio
                report_row["plot_included"] = included
                report_rows.append(report_row)

            selected_mt = None
            if mt_window is not None:
                selected_mt = _select_multithreshold(
                    config,
                    dataset,
                    development,
                    final_splits,
                    file_id=file_id,
                    mode=mode,
                    mode_id=mode_id,
                    window=mt_window,
                    model_id=codebooks["model"][_MODEL_MULTITHRESHOLD],
                    codebooks=codebooks,
                    candidate_ids=candidate_ids,
                    candidate_manifest=candidate_manifest,
                    voltage=voltage,
                    logger=logger,
                    result_rows=rows,
                    feature_cache=mt_feature_cache,
                )
                if selected_mt is not None:
                    params = selected_mt["params"]
                    selected_mt["threshold_values"] = np.asarray(
                        config["multithreshold"]["thresholds_mV"],
                        float,
                    )[params["threshold_indices"]]
                    score_led, _ = _target_deltas(
                        dataset,
                        mode,
                        selected_mt["score_indices"],
                    )
                    score_led_res = score_led - float(dataset.true_tof_ps)
                    validation_corrections[_MODEL_MULTITHRESHOLD] = (
                        selected_mt["score_indices"],
                        score_led_res - selected_mt["score_residual"],
                    )
                    metrics = selected_mt["metrics"]
                    report_row = _report_base(
                        root_file=root_file,
                        file_id=file_id,
                        voltage=voltage,
                        mode=mode,
                        model=_MODEL_MULTITHRESHOLD,
                        stage_name=selection_stage,
                        metrics=metrics,
                    )
                    report_row = _report_model_details(
                        report_row,
                        chosen=selected_mt,
                        space=None,
                        strategy=(
                            str(
                                config["validation"]["nested"][
                                    "inner_strategy"
                                ]
                            )
                            if strategy == "nested"
                            else strategy
                        ),
                    )
                    report_row["ctr_fold_std_ps"] = float(
                        metrics.get("ctr_err_ps", float("nan"))
                    )
                    report_row["ctr_uncertainty_ps"] = float(
                        metrics.get("ctr_err_ps", float("nan"))
                    )
                    ratio, included = _plot_inclusion(
                        float(metrics["ctr_ps"]),
                        float(led_val_metrics["ctr_ps"]),
                        ratio_limit,
                    )
                    report_row["led_ctr_ps"] = float(
                        led_val_metrics["ctr_ps"]
                    )
                    report_row["ctr_over_led"] = ratio
                    report_row["plot_included"] = included
                    report_rows.append(report_row)

            plot_correction_matrix(
                plots_root
                / "correction_correlations"
                / f"{root_file.stem}__{mode}__validation.png",
                corrections=validation_corrections,
                dpi=dpi,
                title=f"{root_file.stem} · {mode} · validation corrections",
            )

            # --------------------------- blind once ---------------------------
            development_led, development_cfd = _target_deltas(
                dataset,
                mode,
                development,
            )
            development_led_residual = (
                development_led - float(dataset.true_tof_ps)
            )
            train_methods: dict[str, np.ndarray] = {_MODEL_LED: development_led_residual}
            if use_cfd:
                development_cfd = _require_cfd(development_cfd, mode)
                train_methods[_MODEL_CFD] = development_cfd - float(dataset.true_tof_ps)

            led_blind, cfd_blind = _target_deltas(dataset, mode, blind)
            led_residual = led_blind - float(dataset.true_tof_ps)
            blind_methods: dict[str, np.ndarray] = {_MODEL_LED: led_residual}
            cfd_residual = None
            if use_cfd:
                cfd_blind = _require_cfd(cfd_blind, mode)
                cfd_residual = cfd_blind - float(dataset.true_tof_ps)
                blind_methods[_MODEL_CFD] = cfd_residual
            blind_corrections: dict[
                str,
                tuple[np.ndarray, np.ndarray],
            ] = {}
            led_blind_metrics = residual_metrics(led_residual)
            blind_uncertainties: dict[str, float] = {
                _MODEL_LED: ctr_bootstrap_uncertainty(
                    led_residual, bootstrap_samples,
                    _seed_for(base_seed, file_id, mode, "led_bootstrap"),
                )
            }
            blind_baselines = [(_MODEL_LED, led_residual)]
            if use_cfd:
                assert cfd_residual is not None
                blind_uncertainties[_MODEL_CFD] = ctr_bootstrap_uncertainty(
                    cfd_residual, bootstrap_samples,
                    _seed_for(base_seed, file_id, mode, "cfd_bootstrap"),
                )
                blind_baselines.append((_MODEL_CFD, cfd_residual))
            for model_name, residual in blind_baselines:
                _fit, metrics = _fit_row(
                    residual,
                    method=f"Blind {model_name} {mode}",
                    fit_config=config["fit"],
                )
                rows.append(
                    {
                        "stage": _STAGE_BLIND,
                        "file_id": file_id,
                        "mode_id": mode_id,
                        "model_id": codebooks["model"][model_name],
                        "candidate_id": -1,
                        "window_id": -1,
                        "variant_id": -1,
                        "subsampling": 1,
                        "selected": 1,
                        "coverage": 1.0,
                        "voltage_V": voltage,
                        **metrics,
                    }
                )
                report_row = _report_base(
                    root_file=root_file,
                    file_id=file_id,
                    voltage=voltage,
                    mode=mode,
                    model=model_name,
                    stage_name="blind",
                    metrics=metrics,
                )
                report_row.update(
                    {
                        "candidate_id": -1,
                        "window_id": "",
                        "window_before_ns": "",
                        "window_after_ns": "",
                        "variant": "",
                        "subsampling": "",
                        "hyperparameters_json": "",
                        "validation_strategy": strategy,
                        "validation_ctr_ps": float(
                            led_val_metrics["ctr_ps"]
                            if model_name == _MODEL_LED
                            else _require_cfd_metric(cfd_val_metrics, mode)["ctr_ps"]
                        ),
                        "validation_ctr_uncertainty_ps": float("nan"),
                        "ctr_uncertainty_ps": blind_uncertainties[model_name],
                        "led_ctr_ps": float(led_blind_metrics["ctr_ps"]),
                    }
                )
                ratio, included = _plot_inclusion(
                    float(metrics["ctr_ps"]),
                    float(led_blind_metrics["ctr_ps"]),
                    ratio_limit,
                )
                report_row["ctr_over_led"] = ratio
                report_row["plot_included"] = included
                report_rows.append(report_row)

            # name, chosen, model-space, blind residual, train residual, report row
            final_candidates: list[
                tuple[
                    str,
                    dict[str, Any],
                    dict[str, Any] | None,
                    np.ndarray,
                    np.ndarray,
                    dict[str, Any],
                ]
            ] = []

            for space, model_id, chosen in selected_waveforms:
                final_dir = (
                    work_root
                    / f"final_f{file_id}_m{mode_id}_model{model_id}"
                )
                checkpoint = (
                    checkpoint_root
                    / f"f{file_id}_m{mode_id}_model{model_id}.pt"
                )
                resume_model = (
                    checkpoint_root
                    / f"f{file_id}_m{mode_id}_model{model_id}__resume_model.pt"
                )
                final_cache_payload = _final_cache_payload(
                    config,
                    dataset,
                    mode=mode,
                    space=space,
                    chosen=chosen,
                    development=development,
                    blind=blind,
                )
                final_cache_key = canonical_hash(final_cache_payload)
                final_cache_npz, final_cache_json = _candidate_cache_paths(
                    config,
                    file_id=file_id,
                    mode_id=mode_id,
                    model_name=space["id"],
                    cache_key=final_cache_key,
                    kind="final",
                )
                cached_final = _load_final_cache(
                    final_cache_npz,
                    final_cache_json,
                    cache_key=final_cache_key,
                )
                # A reusable final cache is only considered complete when the
                # persistent model artifacts also exist. This guarantees that a
                # later XAI-only resume can enrich the experiment without fit.
                if cached_final is not None and checkpoint.is_file() and resume_model.is_file():
                    (
                        residual,
                        metrics,
                        meta,
                        xai_profile,
                        train_residual,
                    ) = cached_final
                    logger.info(
                        "Final model REUSED | %s | mode=%s | candidate=%d",
                        space["id"],
                        mode,
                        int(chosen["candidate_id"]),
                    )

                    xai_cfg = config.get("reporting", {}).get("xai", {}) or {}
                    if bool(xai_cfg.get("enabled", False)) and xai_profile is None:
                        try:
                            saved_model = torch.load(
                                resume_model,
                                map_location="cpu",
                                weights_only=False,
                            )
                        except TypeError:
                            saved_model = torch.load(
                                resume_model,
                                map_location="cpu",
                            )
                        source = input_variant_dataset_view(
                            dataset,
                            chosen["variant"],
                        )
                        input_waveforms, target = CHANNEL_MODES[mode]
                        xai_view = prediction_window_dataset_view(
                            source,
                            input_waveforms=input_waveforms,
                            target=target,
                            before_ns=float(chosen["window"]["before_ns"]),
                            after_ns=float(chosen["window"]["after_ns"]),
                        )
                        xai_cfg_model = _candidate_training_config(
                            config,
                            space,
                            chosen["overrides"],
                            mode=mode,
                            subsampling=chosen["subsampling"],
                            train_dir=output / ".resume_xai",
                            seed=_seed_for(
                                base_seed,
                                file_id,
                                mode,
                                space["id"],
                                "partial_resume_xai",
                            ),
                            final=False,
                        )
                        normalization = Normalization.from_dict(
                            meta["normalization"]
                        )
                        xai_profile = _integrated_gradient_profile(
                            saved_model,
                            normalization,
                            xai_cfg_model,
                            xai_view,
                            blind,
                            max_events=int(xai_cfg.get("max_events", 512)),
                            steps=int(
                                xai_cfg.get(
                                    "integrated_gradient_steps",
                                    16,
                                )
                            ),
                        )
                        del saved_model
                        _save_final_cache(
                            final_cache_npz,
                            final_cache_json,
                            cache_key=final_cache_key,
                            residual=residual,
                            train_residual=train_residual,
                            metrics=metrics,
                            meta=meta,
                            xai_profile=xai_profile,
                        )
                else:
                    (
                        residual,
                        metrics,
                        meta,
                        xai_profile,
                        train_residual,
                    ) = _waveform_evaluate_selected(
                        config,
                        dataset,
                        development,
                        blind,
                        file_id=file_id,
                        mode=mode,
                        window=chosen["window"],
                        variant=chosen["variant"],
                        subsampling=chosen["subsampling"],
                        space=space,
                        overrides=chosen["overrides"],
                        candidate_id=chosen["candidate_id"],
                        work_dir=final_dir,
                        logger=logger,
                        normalization_cache=normalization_cache,
                        checkpoint_path=checkpoint,
                        resume_model_path=resume_model,
                        compute_xai=True,
                        return_train_residual=True,
                    )
                    if train_residual is None:
                        raise RuntimeError(
                            f"Final {space['id']} evaluation did not return train residuals"
                        )
                    _save_final_cache(
                        final_cache_npz,
                        final_cache_json,
                        cache_key=final_cache_key,
                        residual=residual,
                        train_residual=train_residual,
                        metrics=metrics,
                        meta=meta,
                        xai_profile=xai_profile,
                    )
                if train_residual is None:
                    raise RuntimeError(
                        f"Final {space['id']} evaluation did not return train residuals"
                    )
                rows.append(
                    {
                        "stage": _STAGE_BLIND,
                        "file_id": file_id,
                        "mode_id": mode_id,
                        "model_id": model_id,
                        "candidate_id": chosen["candidate_id"],
                        "window_id": codebooks["window"][chosen["window"]["id"]],
                        "variant_id": codebooks["variant"][chosen["variant"]],
                        "subsampling": chosen["subsampling"],
                        "selected": 1,
                        "coverage": 1.0,
                        "voltage_V": voltage,
                        **metrics,
                    }
                )

                correction_center_ps = float(
                    np.mean(train_residual - development_led_residual)
                )
                led_center_ps = float(np.mean(development_led_residual))
                final_metadata[f"{file_id}:{mode_id}:{model_id}"] = {
                    "checkpoint": str(checkpoint.relative_to(output)),
                    "resume_model": str(resume_model.relative_to(output)),
                    "file_id": int(file_id),
                    "mode_id": int(mode_id),
                    "model_id": int(model_id),
                    "mode": mode,
                    "model": space["id"],
                    "candidate_id": int(chosen["candidate_id"]),
                    "window": copy.deepcopy(chosen["window"]),
                    "variant": str(chosen["variant"]),
                    "subsampling": int(chosen["subsampling"]),
                    "overrides": copy.deepcopy(chosen["overrides"]),
                    "reporting_led_center_ps": led_center_ps,
                    "reporting_correction_center_ps": correction_center_ps,
                    **meta,
                }

                train_methods[space["id"]] = train_residual
                blind_methods[space["id"]] = residual
                # Correlation convention retained from current main: amount
                # subtracted from LED, not final-minus-LED.
                blind_corrections[space["id"]] = (
                    blind,
                    led_residual - residual,
                )
                uncertainty = ctr_bootstrap_uncertainty(
                    residual,
                    bootstrap_samples,
                    _seed_for(
                        base_seed,
                        file_id,
                        mode,
                        space["id"],
                        "bootstrap",
                    ),
                )
                blind_uncertainties[space["id"]] = uncertainty
                report_row = _report_base(
                    root_file=root_file,
                    file_id=file_id,
                    voltage=voltage,
                    mode=mode,
                    model=space["id"],
                    stage_name="blind",
                    metrics=metrics,
                )
                report_row = _report_model_details(
                    report_row,
                    chosen=chosen,
                    space=space,
                    strategy=strategy,
                )
                report_row.update(
                    {
                        "validation_ctr_ps": float(
                            chosen["metrics"]["ctr_ps"]
                        ),
                        "validation_ctr_uncertainty_ps": float(
                            chosen["metrics"].get(
                                "ctr_err_ps",
                                float("nan"),
                            )
                        ),
                        "ctr_uncertainty_ps": uncertainty,
                        "led_ctr_ps": float(led_blind_metrics["ctr_ps"]),
                    }
                )
                ratio, included = _plot_inclusion(
                    float(metrics["ctr_ps"]),
                    float(led_blind_metrics["ctr_ps"]),
                    ratio_limit,
                )
                report_row["ctr_over_led"] = ratio
                report_row["plot_included"] = included
                report_rows.append(report_row)
                final_candidates.append(
                    (
                        space["id"],
                        chosen,
                        space,
                        residual,
                        train_residual,
                        report_row,
                    )
                )

                if xai_profile is not None:
                    time_ps, importance = xai_profile
                    fig, axis = plt.subplots(figsize=(8.0, 3.6))
                    axis.plot(np.asarray(time_ps) / 1000.0, importance)
                    axis.set_xlabel("Relative time [ns]")
                    axis.set_ylabel("Normalized |attribution|")
                    axis.set_title(f"{space['id']} · {mode}")
                    axis.grid(alpha=0.2)
                    fig.tight_layout()
                    xai_path = (
                        plots_root
                        / "xai"
                        / f"{root_file.stem}__{mode}__{space['id']}.png"
                    )
                    xai_path.parent.mkdir(parents=True, exist_ok=True)
                    fig.savefig(
                        xai_path,
                        dpi=dpi,
                        bbox_inches="tight",
                    )
                    plt.close(fig)

            if selected_mt is not None:
                residual, metrics, train_residual = _multithreshold_evaluate(
                    config,
                    dataset,
                    development,
                    blind,
                    selected_mt,
                    return_train_residual=True,
                )
                if train_residual is None:
                    raise RuntimeError(
                        "Final multithreshold evaluation did not return train residuals"
                    )
                mt_model_id = codebooks["model"][_MODEL_MULTITHRESHOLD]
                rows.append(
                    {
                        "stage": _STAGE_BLIND,
                        "file_id": file_id,
                        "mode_id": mode_id,
                        "model_id": mt_model_id,
                        "candidate_id": selected_mt["candidate_id"],
                        "window_id": selected_mt["window_id"],
                        "variant_id": codebooks["variant"].get("raw", 0),
                        "subsampling": 1,
                        "selected": 1,
                        "coverage": 1.0,
                        "voltage_V": voltage,
                        **metrics,
                    }
                )
                train_methods[_MODEL_MULTITHRESHOLD] = train_residual
                blind_methods[_MODEL_MULTITHRESHOLD] = residual
                blind_corrections[_MODEL_MULTITHRESHOLD] = (
                    blind,
                    led_residual - residual,
                )
                uncertainty = ctr_bootstrap_uncertainty(
                    residual,
                    bootstrap_samples,
                    _seed_for(
                        base_seed,
                        file_id,
                        mode,
                        _MODEL_MULTITHRESHOLD,
                        "bootstrap",
                    ),
                )
                report_row = _report_base(
                    root_file=root_file,
                    file_id=file_id,
                    voltage=voltage,
                    mode=mode,
                    model=_MODEL_MULTITHRESHOLD,
                    stage_name="blind",
                    metrics=metrics,
                )
                report_row = _report_model_details(
                    report_row,
                    chosen=selected_mt,
                    space=None,
                    strategy=strategy,
                )
                report_row.update(
                    {
                        "validation_ctr_ps": float(
                            selected_mt["metrics"]["ctr_ps"]
                        ),
                        "validation_ctr_uncertainty_ps": float(
                            selected_mt["metrics"].get(
                                "ctr_err_ps",
                                float("nan"),
                            )
                        ),
                        "ctr_uncertainty_ps": uncertainty,
                        "led_ctr_ps": float(led_blind_metrics["ctr_ps"]),
                    }
                )
                ratio, included = _plot_inclusion(
                    float(metrics["ctr_ps"]),
                    float(led_blind_metrics["ctr_ps"]),
                    ratio_limit,
                )
                report_row["ctr_over_led"] = ratio
                report_row["plot_included"] = included
                report_rows.append(report_row)
                final_candidates.append(
                    (
                        _MODEL_MULTITHRESHOLD,
                        selected_mt,
                        None,
                        residual,
                        train_residual,
                        report_row,
                    )
                )

            # Train/development and blind distributions share exactly the same
            # reporting implementation and CTR/bootstrap logic.
            plot_result_distribution(
                plots_root
                / "train_distributions"
                / f"{root_file.stem}__{mode}.png",
                mode=mode,
                methods=train_methods,
                dpi=dpi,
                ratio_limit=ratio_limit,
                bootstrap_samples=bootstrap_samples,
                seed=_seed_for(
                    base_seed,
                    file_id,
                    mode,
                    "train_distribution_bootstrap",
                ),
                split_label="Train / development",
            )
            plot_result_distribution(
                plots_root
                / "blind_distributions"
                / f"{root_file.stem}__{mode}.png",
                mode=mode,
                methods=blind_methods,
                dpi=dpi,
                ratio_limit=ratio_limit,
                bootstrap_samples=bootstrap_samples,
                seed=_seed_for(
                    base_seed,
                    file_id,
                    mode,
                    "blind_distribution_bootstrap",
                ),
                split_label="Blind",
            )
            plot_correction_matrix(
                plots_root
                / "correction_correlations"
                / f"{root_file.stem}__{mode}__blind.png",
                corrections=blind_corrections,
                dpi=dpi,
                title=f"{root_file.stem} · {mode} · blind corrections",
            )

            # TOP/WORST examples from the single best validation-selected ML family.
            eligible_final = [
                item
                for item in final_candidates
                if int(item[5].get("plot_included", 1)) == 1
            ]
            if eligible_final:
                (
                    best_name,
                    best_chosen,
                    best_space,
                    best_residual,
                    best_train_residual,
                    _best_report_row,
                ) = min(
                    eligible_final,
                    key=lambda item: float(item[1]["metrics"]["ctr_ps"]),
                )
                variant = (
                    "raw"
                    if best_space is None
                    else best_chosen["variant"]
                )
                source = input_variant_dataset_view(dataset, variant)
                input_waveforms, target = CHANNEL_MODES[mode]
                materialized = config["preprocessing"]["materialized_window_ns"]
                full_view = prediction_window_dataset_view(
                    source,
                    input_waveforms=input_waveforms,
                    target=target,
                    before_ns=float(materialized["before"]),
                    after_ns=float(materialized["after"]),
                )

                # Both centers are learned from development only.  The large
                # global LED/TOF offset and the model intercept therefore do not
                # dominate event ranking or displayed correction signs.
                led_center_ps = float(np.mean(development_led_residual))
                correction_center_ps = float(
                    np.mean(
                        best_train_residual
                        - development_led_residual
                    )
                )
                common_correction_args = {
                    "time_ps": np.asarray(
                        full_view.relative_time_ps,
                        dtype=np.float64,
                    ),
                    "waveforms": np.asarray(
                        full_view.windows_mV[blind],
                        dtype=np.float32,
                    ),
                    "led_residual": led_residual,
                    "corrected_residual": best_residual,
                    "led_center_ps": led_center_ps,
                    "correction_center_ps": correction_center_ps,
                    "model": best_name,
                    "mode": mode,
                    "dpi": dpi,
                    "window_before_ns": float(
                        best_chosen["window"]["before_ns"]
                    ),
                    "window_after_ns": float(
                        best_chosen["window"]["after_ns"]
                    ),
                    "event_ids": np.asarray(dataset.event_id[blind]),
                }

                top_k = int(config["reporting"].get("top_corrections_k", 3))
                if top_k > 0:
                    plot_correction_examples(
                        plots_root
                        / "top_corrections"
                        / f"{root_file.stem}__{mode}.png",
                        selection="top",
                        k=top_k,
                        **common_correction_args,
                    )

                worst_k = int(
                    config["reporting"].get("worst_corrections_k", 3)
                )
                if worst_k > 0:
                    plot_correction_examples(
                        plots_root
                        / "worst_corrections"
                        / f"{root_file.stem}__{mode}.png",
                        selection="worst",
                        k=worst_k,
                        **common_correction_args,
                    )

            _save_evaluation_artifact(
                output,
                file_id=file_id,
                mode_id=mode_id,
                development=development,
                blind=blind,
                development_led_residual=development_led_residual,
                blind_led_residual=led_residual,
                development_cfd=development_cfd,
                blind_cfd=cfd_blind,
                true_tof_ps=float(dataset.true_tof_ps),
                final_candidates=final_candidates,
            )

        normalization_cache.clear()
        mt_feature_cache.clear()

    shutil.rmtree(work_root, ignore_errors=True)
    _write_csv(results_path, list(rows), logger=logger)
    if nested_rows:
        write_report_csv(nested_path, nested_rows)
    write_summary_results(summary_path, report_rows)
    write_report_csv(output / "report_results.csv", report_rows)

    plot_ctr_vs_voltage(
        plots_root / "validation_ctr_vs_voltage.png",
        rows=report_rows,
        stage="validation",
        dpi=dpi,
        ratio_limit=ratio_limit,
        title="Validation CTR vs voltage",
    )
    plot_ctr_vs_voltage(
        plots_root / "blind_ctr_vs_voltage.png",
        rows=report_rows,
        stage="blind",
        dpi=dpi,
        ratio_limit=ratio_limit,
        title="Blind CTR vs voltage",
    )
    plot_final_bars(
        plots_root / "blind_ctr_bar_by_voltage.png",
        rows=report_rows,
        dpi=dpi,
    )
    if bool(config["reporting"].get("window_scan_bars", False)):
        plot_window_scan_bars(
            plots_root / "window_scan",
            candidate_rows=rows,
            report_rows=report_rows,
            codebooks=codebooks,
            windows=config["windows_ns"],
            dpi=dpi,
            ratio_limit=ratio_limit,
        )
    plot_selection_vs_blind(
        plots_root / "validation_vs_blind_ctr.png",
        rows=report_rows,
        selection_stage="validation",
        dpi=dpi,
    )
    if strategy == "nested":
        plot_ctr_vs_voltage(
            plots_root / "nested_ctr_vs_voltage.png",
            rows=report_rows,
            stage="nested",
            dpi=dpi,
            ratio_limit=ratio_limit,
            title="Nested pipeline CTR vs voltage",
        )
        plot_selection_vs_blind(
            plots_root / "nested_vs_blind_ctr.png",
            rows=report_rows,
            selection_stage="nested",
            dpi=dpi,
        )

    manifest = {
        "schema_version": 5,
        "experiment": config["experiment"]["name"],
        "config_hash": config["_config_hash"],
        "core_hash": config.get("_core_hash"),
        "artifact_hash": config.get("_artifact_hash"),
        "modes": copy.deepcopy(config.get("modes", {})),
        "config_sources": copy.deepcopy(config.get("_config_sources", [])),
        "reporting": copy.deepcopy(config.get("reporting", {})),
        "config_path": config["_config_path"],
        "prepared_dir": config["preprocessing"]["prepared_dir"],
        "selection_store_dir": config["preprocessing"]["selection_store_dir"],
        "materialized_window_ns": config["preprocessing"]["materialized_window_ns"],
        "row_count": len(rows),
        "results_csv_pending": _pending_csv_path(results_path).is_file(),
        "codebooks": codebooks,
        "candidate_parameters": candidate_manifest,
        "final_models": final_metadata,
        "fit": config["fit"],
        "validation": config["validation"],
        "protocol": {
            "model_pipeline": "working model implementation preserved",
            "target": "direct antisymmetric LED correction target from torch_data.py",
            "normalization": "train-derived shared waveform normalization",
            "selection": "holdout/CV/nested wrapper only; no model implementation replacement",
            "nested": "outer K-fold pipeline evaluation with configured inner selection",
            "blind": "single untouched blind partition opened only after final development selection",
            "ctr": "2*sqrt(2*ln(2))*sample standard deviation over all evaluation events",
            "evaluation_rejection": "none after permanent prepared population; pathological models are hidden from figures only",
            "photopeak_cache": "physical/photopeak indices persisted independently from ML windows/models/validation",
            "multithreshold": "raw native-grid relative threshold implementation; candidates cannot drop events",
            "correction_ranking": (
                "TOP/WORST uses development-centered LED residual and "
                "development-centered final-minus-LED correction; blind statistics "
                "never define centering"
            ),
            "train_distribution": (
                "final fitted model evaluated on complete development population; "
                "diagnostic only"
            ),
        },
        "result_columns": {
            "stage": {
                "0": "development selection candidate",
                "1": "blind",
            },
            "candidate_id": (
                "parameters stored once in candidate_parameters; "
                "-1 for fixed LED/CFD"
            ),
        },
    }
    atomic_json(manifest_path, manifest)
    atomic_json(
        output / "config_resolved.json",
        {k: v for k, v in config.items() if not str(k).startswith("_")},
    )
    atomic_json(
        run_state_path,
        {
            "schema_version": 1,
            "experiment": config["experiment"]["name"],
            "core_hash": _resume_state_hash(config),
            "artifact_hash": config.get("_artifact_hash"),
            "config_hash": config.get("_config_hash"),
            "status": "complete",
            "row_count": len(rows),
        },
    )
    _flush_progress_rows(rows)
    logger.info("Study complete | rows=%d | %s", len(rows), output)
    return {
        "output_dir": str(output),
        "row_count": len(rows),
        "results": str(results_path),
        "summary_results": str(summary_path),
        "nested_results": str(nested_path) if nested_rows else None,
    }
