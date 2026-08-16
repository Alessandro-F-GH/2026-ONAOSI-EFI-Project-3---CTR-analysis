from __future__ import annotations

import copy
import csv
import gc
import itertools
import json
import math
import os
import re
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .common import atomic_json, canonical_hash
from .dataset import PreparedDataset
from .models import validate_model, validate_model_training
from .prediction import prediction_window_dataset_view
from .prepared_data import (
    input_channel_variant_dataset_view,
    raw_dataset_view,
    plot_prepared_signal_examples,
    prepare_file_dataset,
)
from .study_config import CHANNEL_MODES, candidate_overrides, discover_root_files, set_nested
from .torch_data import Normalization, compute_normalization
from .training import train_model
from .training_utils import make_split_loader, predict_loader, resolve_device

_STAGE_OOF = 0
_STAGE_BLIND = 1
_MODEL_LED = "led"
_MODEL_CFD = "cfd"
_MODEL_MULTITHRESHOLD = "multithreshold_svr"
_TARGET_PROTOCOL_VERSION = "channel_preprocessing_led_consistent_v3"


def _seed_for(base: int, *parts: Any) -> int:
    payload = "|".join(str(v) for v in (base, *parts)).encode("utf-8")
    import hashlib
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little") & 0x7FFFFFFF


def _random_dev_blind(n: int, *, blind_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if n < 5:
        raise RuntimeError("Need at least five prepared events")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n).astype(np.int64)
    n_blind = int(round(n * blind_fraction))
    n_blind = min(max(1, n_blind), n - 2)
    blind = np.sort(order[:n_blind])
    development = np.sort(order[n_blind:])
    return development, blind


def _kfold(indices: np.ndarray, *, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    values = np.asarray(indices, dtype=np.int64)
    if values.size < n_splits:
        raise RuntimeError(f"Only {values.size} development events for {n_splits}-fold CV")
    order = np.random.default_rng(seed).permutation(values)
    score_folds = [np.sort(v.astype(np.int64)) for v in np.array_split(order, n_splits)]
    output: list[tuple[np.ndarray, np.ndarray]] = []
    for i, score in enumerate(score_folds):
        train_pool = np.sort(np.concatenate([v for j, v in enumerate(score_folds) if j != i]))
        output.append((train_pool, score))
    return output


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




def _ml_input_variant_for_mode(study: dict[str, Any], mode: str) -> str:
    """Resolve the single waveform variant used by one channel mode.

    Variant selection is a fixed channel policy, never a search dimension.
    Multithreshold intentionally does not call this helper and always uses raw.
    """
    input_waveforms, _target = CHANNEL_MODES[mode]
    variants = study["preprocessing"]["input_variant_by_channel"]
    if input_waveforms not in {"energy", "timing"}:
        raise ValueError(f"No single-channel variant policy for input family {input_waveforms!r}")
    return str(variants[input_waveforms])

def _voltage_from_name(name: str, pattern: str) -> float:
    match = re.search(pattern, name)
    if not match:
        return float("nan")
    try:
        return float(match.group("voltage"))
    except (IndexError, KeyError):
        return float(match.group(1))


def _write_csv(path, rows, *, logger=None, retries=8, retry_delay_s=0.5):
    import csv
    import os
    import time
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = list(rows)
    if not rows:
        return

    temporary = path.with_suffix(path.suffix + ".tmp")

    fieldnames = list(rows[0].keys())

    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

        handle.flush()
        os.fsync(handle.fileno())

    for attempt in range(retries):
        try:
            os.replace(temporary, path)
            return

        except PermissionError:
            if attempt + 1 < retries:
                time.sleep(retry_delay_s)
                continue

    # Do NOT kill a long experiment because the user has results.csv open.
    if logger is not None:
        logger.warning(
            "Could not update %s because it is locked by another process. "
            "Keeping the pending update in %s; training will continue. "
            "Close the CSV so the next persistence point can update it.",
            path,
            temporary,
        )


_RESULT_INT_FIELDS = {
    "stage", "file_id", "mode_id", "model_id", "candidate_id", "window_id",
    "variant_id", "subsampling", "selected", "n",
}
_RESULT_FLOAT_FIELDS = {
    "coverage", "voltage_V", "mean_ps", "std_ps", "ctr_ps",
    "ctr_fold_std_ps", "rmse_ps", "rmse_fold_std_ps",
}


def _read_results_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        for raw in csv.DictReader(stream):
            row: dict[str, Any] = {}
            for key, value in raw.items():
                if value is None or value == "":
                    continue
                if key in _RESULT_INT_FIELDS:
                    row[key] = int(float(value))
                elif key in _RESULT_FLOAT_FIELDS:
                    row[key] = float(value)
                else:
                    row[key] = value
            rows.append(row)
    return rows


def _result_key(row: dict[str, Any]) -> tuple[int, int, int, int, int]:
    return (
        int(row["stage"]), int(row["file_id"]), int(row["mode_id"]),
        int(row["model_id"]), int(row["candidate_id"]),
    )


def _find_result_row(
    rows: list[dict[str, Any]], *, stage: int, file_id: int, mode_id: int,
    model_id: int, candidate_id: int,
) -> dict[str, Any] | None:
    key = (int(stage), int(file_id), int(mode_id), int(model_id), int(candidate_id))
    for row in rows:
        if _result_key(row) == key:
            return row
    return None


def _upsert_result_row(rows: list[dict[str, Any]], new_row: dict[str, Any]) -> None:
    key = _result_key(new_row)
    for index, row in enumerate(rows):
        if _result_key(row) == key:
            rows[index] = new_row
            return
    rows.append(new_row)


def _candidate_id_for_key(candidate_ids: dict[str, int], key: str) -> int:
    if key in candidate_ids:
        return int(candidate_ids[key])
    next_id = max((int(value) for value in candidate_ids.values()), default=-1) + 1
    candidate_ids[key] = next_id
    return next_id


_FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))


def _distribution_stats(values_ps: np.ndarray, *, method: str) -> dict[str, Any]:
    """Classical all-event timing statistics for one fitted model/output population.

    No Gaussian fit and no event rejection are performed here.  CTR is reported
    as the Gaussian-equivalent FWHM corresponding to the ordinary sample
    standard deviation: ``CTR = 2*sqrt(2 ln 2) * s``.
    """
    values = np.asarray(values_ps, dtype=np.float64).reshape(-1)
    if values.size < 2 or np.any(~np.isfinite(values)):
        raise RuntimeError(f"{method}: evaluation requires at least two finite values and keeps every event")
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    return {
        "n": int(values.size),
        "mean_ps": mean,
        "std_ps": std,
        "ctr_ps": float(_FWHM_PER_SIGMA * std),
        "rmse_ps": float(np.sqrt(np.mean(values * values))),
    }


def _aggregate_fold_stats(fold_metrics: list[dict[str, Any]], *, method: str) -> dict[str, Any]:
    """Summarize independent score-fold metrics without pooling model outputs.

    Each score fold is evaluated on predictions from its own fitted model.  We
    only take arithmetic summaries of the fold-level statistics; predictions
    from different models are never concatenated for CTR estimation.
    """
    if not fold_metrics:
        raise RuntimeError(f"{method}: no CV score-fold metrics were produced")

    def _mean(key: str) -> float:
        return float(np.mean([float(row[key]) for row in fold_metrics]))

    def _std(key: str) -> float:
        values = np.asarray([float(row[key]) for row in fold_metrics], dtype=np.float64)
        return float(np.std(values, ddof=1)) if values.size > 1 else 0.0

    return {
        "n": int(sum(int(row["n"]) for row in fold_metrics)),
        "mean_ps": _mean("mean_ps"),
        "std_ps": _mean("std_ps"),
        "ctr_ps": _mean("ctr_ps"),
        "ctr_fold_std_ps": _std("ctr_ps"),
        "rmse_ps": _mean("rmse_ps"),
        "rmse_fold_std_ps": _std("rmse_ps"),
    }


def _target_deltas(dataset: PreparedDataset, mode: str, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _input, target = CHANNEL_MODES[mode]
    if target == "energy_led":
        led = dataset.energy_led_time_fs
        cfd = dataset.energy_cfd_time_fs
    else:
        led = dataset.timing_led_time_fs
        cfd = dataset.timing_cfd_time_fs
    if led is None or cfd is None:
        raise ValueError(f"Prepared dataset lacks timing arrays required by mode {mode}")
    idx = np.asarray(indices, dtype=np.int64)
    led_ps = (np.asarray(led[idx, 0], dtype=np.float64) - np.asarray(led[idx, 1], dtype=np.float64)) / 1000.0
    cfd_ps = (np.asarray(cfd[idx, 0], dtype=np.float64) - np.asarray(cfd[idx, 1], dtype=np.float64)) / 1000.0
    return led_ps, cfd_ps


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
    # Early stopping remains an optimization criterion only. Scientific model
    # selection is performed later from independent score-fold CTR estimates.
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
        "early_stop_fraction", study["cross_validation"]["early_stop_fraction"]
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
    """Combine model and preprocessing choices without duplicating raw/denoised runs.

    Each channel mode has exactly one waveform variant resolved from
    ``preprocessing.input_variant_by_channel``. Grid model spaces remain
    exhaustive over windows/subsampling/model parameters only.
    """
    model_candidates = candidate_overrides(space, seed=seed)
    variant = _ml_input_variant_for_mode(study, mode)
    prep = [
        (window, variant, int(factor))
        for window in study["windows_ns"]
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

def _waveform_oof_candidate(
    study: dict[str, Any],
    dataset: PreparedDataset,
    development: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
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
) -> dict[str, Any]:
    input_waveforms, target = CHANNEL_MODES[mode]
    source = input_channel_variant_dataset_view(dataset, input_waveforms, variant)
    view = prediction_window_dataset_view(
        source,
        input_waveforms=input_waveforms,
        target=target,
        before_ns=float(window["before_ns"]),
        after_ns=float(window["after_ns"]),
    )
    fold_metrics: list[dict[str, Any]] = []
    base_seed = int(study["cross_validation"]["seed"])
    for fold_index, (train_pool, score_idx) in enumerate(folds):
        candidate_seed = _seed_for(base_seed, file_id, mode, window["id"], variant, subsampling, space["id"], candidate_id, fold_index)
        preview_cfg = _candidate_training_config(
            study, space, overrides, mode=mode, subsampling=subsampling,
            train_dir=work_root / f"f{fold_index}", seed=candidate_seed, final=False,
        )
        if preview_cfg["model"]["type"] in {"linear_svr"}:
            fit_idx, early_idx = np.asarray(train_pool), np.asarray(train_pool)
        else:
            fraction = _early_fraction(study, preview_cfg)
            fit_idx, early_idx = _fit_early_split(
                train_pool,
                fraction=fraction,
                seed=_seed_for(base_seed, file_id, "early", fold_index),
            )
        logger.info(
            "CV fold %d/%d | fit=%d | early_stop=%d | score=%d",
            fold_index + 1, len(folds), len(fit_idx), len(early_idx), len(score_idx),
        )
        fold_view = replace(view, train=fit_idx, validation=early_idx, evaluation=score_idx)
        cached_normalization = _normalization_for_fit_subset(
            fold_view, fit_idx, subsampling=subsampling, cache=normalization_cache
        )
        model, normalization, _summary = _train_in_memory(
            preview_cfg, fold_view, logger=logger,
            data_view={"stage": "oof", "fold": fold_index, "candidate_id": candidate_id},
            normalization_override=cached_normalization,
        )
        fold_values = _predict_indices(model, normalization, preview_cfg, fold_view, score_idx)
        fold_metrics.append(
            _distribution_stats(
                fold_values,
                method=f"CV {space['id']} fold {fold_index + 1}",
            )
        )
        _cleanup_training(model, Path(preview_cfg["output"]["train_dir"]))
    return _aggregate_fold_stats(fold_metrics, method=f"CV {space['id']}")




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

def _waveform_final(
    study: dict[str, Any],
    dataset: PreparedDataset,
    development: np.ndarray,
    blind: np.ndarray,
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
    checkpoint_path: Path,
    logger: Any,
    normalization_cache: dict[str, Normalization],
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any], tuple[np.ndarray, np.ndarray] | None]:
    input_waveforms, target = CHANNEL_MODES[mode]
    source = input_channel_variant_dataset_view(dataset, input_waveforms, variant)
    view = prediction_window_dataset_view(
        source, input_waveforms=input_waveforms, target=target,
        before_ns=float(window["before_ns"]), after_ns=float(window["after_ns"]),
    )
    seed = _seed_for(int(study["cross_validation"]["seed"]), file_id, mode, space["id"], candidate_id, "final")
    cfg = _candidate_training_config(
        study, space, overrides, mode=mode, subsampling=subsampling,
        train_dir=work_dir, seed=seed, final=True,
    )
    if cfg["model"]["type"] in {"linear_svr"}:
        fit_idx, early_idx = development, development
    else:
        fit_idx, early_idx = _fit_early_split(
            development,
            fraction=_early_fraction(study, cfg),
            seed=_seed_for(int(study["cross_validation"]["seed"]), file_id, "early", "final"),
        )
    final_view = replace(view, train=fit_idx, validation=early_idx, evaluation=blind)
    cached_normalization = _normalization_for_fit_subset(
        final_view, fit_idx, subsampling=subsampling, cache=normalization_cache
    )
    model, normalization, summary = _train_in_memory(
        cfg, final_view, logger=logger,
        data_view={"stage": "final", "candidate_id": candidate_id},
        normalization_override=cached_normalization,
    )
    residual = _predict_indices(model, normalization, cfg, final_view, blind)
    metrics = _distribution_stats(residual, method=f"Blind {space['id']}")
    # XAI can be computed before the model is released; caller receives only compact metadata.
    final_meta = {
        "best_epoch": int(summary.get("best_epoch", 0)),
        "normalization": summary.get("normalization", {}),
        "model_type": cfg["model"]["type"],
    }
    xai_profile = None
    xai_cfg = study.get("reporting", {}).get("xai", {}) or {}
    if (
        bool(xai_cfg.get("enabled", True))
    ):
        xai_profile = _integrated_gradient_profile(
            model, normalization, cfg, final_view, blind,
            max_events=int(xai_cfg.get("max_events", 512)),
            steps=int(xai_cfg.get("integrated_gradient_steps", 16)),
        )
    elif bool(xai_cfg.get("enabled", True)):
        logger.info("Integrated gradients skipped | model=multirocket_hydra")
    _cleanup_training(model, work_dir, keep_best=checkpoint_path)
    return residual, metrics, final_meta, xai_profile


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


def _multithreshold_oof_select(
    study: dict[str, Any],
    dataset: PreparedDataset,
    development: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    *,
    file_id: int,
    mode: str,
    mode_id: int,
    window: dict[str, Any],
    window_id: int,
    model_id: int,
    candidate_ids: dict[str, int],
    candidate_manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    voltage: float,
    logger: Any,
    persist_progress: Callable[[], None] | None = None,
) -> dict[str, Any] | None:
    """Select multithreshold-SVR settings using development OOF predictions only."""
    cfg = study["multithreshold"]
    if not bool(cfg.get("enabled", False)):
        return None

    # Hard invariant: threshold crossings are extracted only from the canonical
    # raw native-sample representation.  Denoised arrays cannot enter this path.
    raw = raw_dataset_view(dataset)
    input_waveforms, target = CHANNEL_MODES[mode]
    view = prediction_window_dataset_view(
        raw, input_waveforms=input_waveforms, target=target,
        before_ns=float(window["before_ns"]), after_ns=float(window["after_ns"]),
    )
    thresholds = np.asarray(cfg["thresholds_mV"], dtype=np.float64)
    features_all = _threshold_crossing_matrix(
        view, thresholds, chunk_size=int(cfg.get("chunk_size", 2048))
    )
    led_ps, _ = _target_deltas(raw, mode, np.arange(dataset.event_id.size))
    target_correction = led_ps - float(dataset.true_tof_ps)

    best: tuple[float, dict[str, Any], int] | None = None
    for params in _multithreshold_candidates(cfg, thresholds.size):
        key = canonical_hash({
            "family": _MODEL_MULTITHRESHOLD, "mode": mode,
            "window": window["id"], **params,
        })
        candidate_id = _candidate_id_for_key(candidate_ids, key)
        candidate_manifest[str(candidate_id)] = {
            "family": _MODEL_MULTITHRESHOLD,
            "mode": mode,
            "window": window["id"],
            "thresholds_mV": thresholds[params["threshold_indices"]].tolist(),
            **{k: v for k, v in params.items() if k != "threshold_indices"},
        }
        cols = params["threshold_indices"]
        valid = np.all(np.isfinite(features_all[:, cols]), axis=1)
        # A candidate that cannot represent every prepared development event is
        # not comparable: missing threshold crossings never become event cuts.
        if not np.all(valid[development]):
            continue

        existing = _find_result_row(
            rows, stage=_STAGE_OOF, file_id=file_id, mode_id=mode_id,
            model_id=model_id, candidate_id=candidate_id,
        )
        if existing is not None:
            metrics = existing
            logger.info(
                "CV resume | mode=%s | model=multithreshold_svr | candidate=%d | CTR=%.3f ps",
                mode, candidate_id, float(metrics["ctr_ps"]),
            )
        else:
            fold_metrics: list[dict[str, Any]] = []
            for fold_index, (train_pool, score_idx) in enumerate(folds):
                estimator = make_pipeline(
                    StandardScaler(),
                    SVR(
                        kernel=params["kernel"], C=params["C"],
                        epsilon=params["epsilon_ps"], gamma=params["gamma"],
                    ),
                )
                estimator.fit(features_all[np.ix_(train_pool, cols)], target_correction[train_pool])
                correction = estimator.predict(features_all[np.ix_(score_idx, cols)])
                fold_values = led_ps[score_idx] - correction - float(dataset.true_tof_ps)
                fold_metrics.append(
                    _distribution_stats(
                        fold_values,
                        method=f"CV multithreshold SVR fold {fold_index + 1}",
                    )
                )
            metrics = _aggregate_fold_stats(fold_metrics, method="CV multithreshold SVR")
            _upsert_result_row(rows, {
                "stage": _STAGE_OOF, "file_id": file_id, "mode_id": mode_id,
                "model_id": model_id, "candidate_id": candidate_id,
                "voltage_V": voltage, "window_id": window_id,
                "variant_id": 0, "subsampling": 1, "selected": 0,
                "coverage": 1.0, **metrics,
            })
            if persist_progress is not None:
                persist_progress()
        if best is None or float(metrics["ctr_ps"]) < best[0]:
            best = (float(metrics["ctr_ps"]), params, candidate_id)

    if best is None:
        logger.warning(
            "No multithreshold candidate covers every development event | file=%s mode=%s",
            dataset.directory.name, mode,
        )
        return None

    _, params, selected_candidate = best
    for row in rows:
        if (
            int(row["stage"]) == _STAGE_OOF
            and int(row["file_id"]) == file_id
            and int(row["mode_id"]) == mode_id
            and int(row["model_id"]) == model_id
        ):
            row["selected"] = int(int(row["candidate_id"]) == selected_candidate)
    if persist_progress is not None:
        persist_progress()
    return {
        "params": params,
        "candidate_id": selected_candidate,
        "features": features_all,
        "led_ps": led_ps,
        "target_correction": target_correction,
        "window_id": window_id,
    }


def _multithreshold_final(
    study: dict[str, Any],
    dataset: PreparedDataset,
    development: np.ndarray,
    blind: np.ndarray,
    selected: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit the selected multithreshold SVR on development and open blind once."""
    params = selected["params"]
    cols = params["threshold_indices"]
    features_all = selected["features"]
    valid = np.all(np.isfinite(features_all[:, cols]), axis=1)
    if not np.all(valid[development]) or not np.all(valid[blind]):
        raise RuntimeError(
            "Selected multithreshold model lacks a threshold crossing for at least one "
            "prepared blind event. The final fit never drops events; lower the configured "
            "threshold range and rerun the experiment."
        )
    estimator = make_pipeline(
        StandardScaler(),
        SVR(
            kernel=params["kernel"], C=params["C"],
            epsilon=params["epsilon_ps"], gamma=params["gamma"],
        ),
    )
    estimator.fit(
        features_all[np.ix_(development, cols)],
        selected["target_correction"][development],
    )
    correction = estimator.predict(features_all[np.ix_(blind, cols)])
    residual = selected["led_ps"][blind] - correction - float(dataset.true_tof_ps)
    metrics = _distribution_stats(residual, method="Blind multithreshold SVR")
    return residual, metrics

def _plot_final_file(
    destination: Path,
    panels: dict[str, dict[str, np.ndarray]],
    *,
    dpi: int,
) -> None:
    if not panels:
        return

    def _robust_bounds(values: np.ndarray) -> tuple[float, float]:
        """Return a wide, outlier-resistant display interval for one method.

        This is reporting-only: it never changes evaluation arrays or metrics.
        Eight robust standard deviations keep the complete central distribution
        visible while preventing isolated catastrophic residuals from setting the
        plot scale.
        """
        center = float(np.median(values))
        mad = float(np.median(np.abs(values - center)))
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale <= 0.0:
            q25, q75 = np.percentile(values, [25.0, 75.0])
            scale = float((q75 - q25) / 1.349)
        if not np.isfinite(scale) or scale <= 0.0:
            scale = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        return center - 8.0 * scale, center + 8.0 * scale

    modes = list(panels)
    fig, axes = plt.subplots(len(modes), 1, figsize=(10, 3.6 * len(modes)), squeeze=False)
    for ax, mode in zip(axes[:, 0], modes):
        methods = panels[mode]
        prepared: dict[str, np.ndarray] = {}
        method_bounds: list[tuple[float, float]] = []
        for label, raw_values in methods.items():
            values = np.asarray(raw_values, dtype=np.float64).reshape(-1)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            prepared[label] = values
            method_bounds.append(_robust_bounds(values))

        if not prepared:
            ax.set_visible(False)
            continue

        # One common display range per mode, wide enough to contain the robust
        # central region of every method.  Only gross tails outside this union
        # are hidden from the visualization.
        x_min = min(bounds[0] for bounds in method_bounds)
        x_max = max(bounds[1] for bounds in method_bounds)
        span = x_max - x_min
        if not np.isfinite(span) or span <= 0.0:
            span = 2.0
            midpoint = 0.5 * (x_min + x_max)
            x_min, x_max = midpoint - 1.0, midpoint + 1.0
        padding = 0.05 * span
        x_min -= padding
        x_max += padding
        bins = np.linspace(x_min, x_max, 81)

        for label, values in prepared.items():
            visible = (values >= x_min) & (values <= x_max)
            outside_count = int(values.size - np.count_nonzero(visible))
            if np.any(visible):
                ax.hist(
                    values[visible],
                    bins=bins,
                    histtype="step",
                    density=True,
                    label=f"{label} (outside={outside_count})",
                )

        ax.set_xlim(x_min, x_max)
        ax.set_title(mode)
        ax.set_xlabel("Residual timing error [ps]")
        ax.set_ylabel("Density")
        ax.minorticks_on(); ax.grid(True, which="major", alpha=0.3); ax.grid(True, which="minor", alpha=0.12)
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(fig)




def _plot_xai_file(
    destination: Path,
    profiles: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    corrections: dict[str, dict[str, np.ndarray]],
    *,
    dpi: int,
) -> None:
    modes = [m for m in profiles if profiles[m] or corrections.get(m)]
    if not modes:
        return
    fig, axes = plt.subplots(len(modes), 2, figsize=(13, 3.8 * len(modes)), squeeze=False)
    for row_index, mode in enumerate(modes):
        ax = axes[row_index, 0]
        for label, (time_ps, importance) in profiles.get(mode, {}).items():
            ax.plot(np.asarray(time_ps) / 1000.0, importance, label=label)
        ax.set_title(f"{mode} | integrated-gradient importance")
        ax.set_xlabel("Relative time [ns]"); ax.set_ylabel("Normalized |attribution|")
        ax.minorticks_on(); ax.grid(True, which="major", alpha=0.3); ax.grid(True, which="minor", alpha=0.12)
        if profiles.get(mode):
            ax.legend(loc="best", fontsize=8)

        axc = axes[row_index, 1]
        corr_data = corrections.get(mode, {})
        labels = list(corr_data)
        if len(labels) >= 2:
            matrix = np.corrcoef(np.stack([np.asarray(corr_data[label], dtype=np.float64) for label in labels], axis=0))
            image = axc.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
            axc.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
            axc.set_yticks(range(len(labels)), labels)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    axc.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
            fig.colorbar(image, ax=axc, fraction=0.046, pad=0.04)
        else:
            axc.text(0.5, 0.5, "Need ≥2 model corrections", transform=axc.transAxes, ha="center", va="center")
            axc.set_xticks([]); axc.set_yticks([])
        axc.set_title(f"{mode} | blind correction correlation")
    fig.tight_layout(); destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=dpi, bbox_inches="tight"); plt.close(fig)

def _plot_ctr_vs_voltage(
    destination: Path,
    rows: list[dict[str, Any]],
    codebooks: dict[str, dict[str, int]],
    *,
    dpi: int,
) -> None:
    reverse_mode = {v: k for k, v in codebooks["mode"].items()}
    reverse_model = {v: k for k, v in codebooks["model"].items()}
    blind = [r for r in rows if int(r["stage"]) == _STAGE_BLIND and math.isfinite(float(r["voltage_V"]))]
    if not blind:
        return
    modes = sorted({int(r["mode_id"]) for r in blind if int(r["mode_id"]) >= 0})
    fig, axes = plt.subplots(len(modes), 1, figsize=(9.5, 3.8 * max(1, len(modes))), squeeze=False)
    for ax, mode_id in zip(axes[:, 0], modes):
        subset = [r for r in blind if int(r["mode_id"]) == mode_id]
        for model_id in sorted({int(r["model_id"]) for r in subset}):
            points = sorted((r for r in subset if int(r["model_id"]) == model_id), key=lambda r: float(r["voltage_V"]))
            ax.plot(
                [float(r["voltage_V"]) for r in points],
                [float(r["ctr_ps"]) for r in points],
                marker="o", label=reverse_model.get(model_id, str(model_id)),
            )
        ax.set_title(reverse_mode.get(mode_id, str(mode_id)))
        ax.set_xlabel("Bias voltage [V]"); ax.set_ylabel("CTR FWHM [ps]")
        ax.minorticks_on(); ax.grid(True, which="major", alpha=0.3); ax.grid(True, which="minor", alpha=0.12)
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout(); destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=dpi, bbox_inches="tight"); plt.close(fig)


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
    manifest_path = output / "manifest.json"

    root_files = discover_root_files(config)
    logger.info("Study %s | files=%d | random CV only", config["experiment"]["name"], len(root_files))
    if dry_run:
        return {
            "output_dir": str(output), "row_count": 0, "dry_run": True,
            "files": [str(v) for v in root_files],
            "models": [v["id"] for v in config["_model_spaces"]],
            "multithreshold": bool(config["multithreshold"].get("enabled", False)),
            "prepared_dir": config["preprocessing"]["prepared_dir"],
        }
    if not root_files:
        raise FileNotFoundError(
            f"No ROOT files match {config['data']['root_glob']} in {config['data']['root_folder']}"
        )

    codebooks = {
        "file": {path.name: i for i, path in enumerate(root_files)},
        "mode": {name: i for i, name in enumerate(config["channel_modes"])},
        "model": {
            _MODEL_LED: 0, _MODEL_CFD: 1,
            **{space["id"]: i + 2 for i, space in enumerate(config["_model_spaces"])},
        },
        "window": {w["id"]: i for i, w in enumerate(config["windows_ns"])},
        "variant": {
            name: i
            for i, name in enumerate(
                ["raw"]
                + (["denoised"] if "denoised" in config["preprocessing"]["input_variant_by_channel"].values() else [])
            )
        },
    }
    if bool(config["multithreshold"].get("enabled", False)):
        codebooks["model"][_MODEL_MULTITHRESHOLD] = max(codebooks["model"].values()) + 1

    candidate_ids: dict[str, int] = {}
    candidate_manifest: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    final_metadata: dict[str, Any] = {}
    completed_files: set[int] = set()
    prepared_fingerprints: dict[str, str] = {}

    if resume:
        if not manifest_path.is_file():
            if results_path.is_file():
                raise RuntimeError(
                    "Cannot safely resume: results.csv exists but manifest.json is missing. "
                    "Use --restart."
                )
        else:
            with manifest_path.open("r", encoding="utf-8") as stream:
                previous = json.load(stream)
            if previous.get("config_hash") != config.get("_config_hash"):
                raise RuntimeError(
                    "Existing results were produced by a different configuration. "
                    "Use --restart or a different experiment.output_dir."
                )
            if previous.get("target_protocol_version") != _TARGET_PROTOCOL_VERSION:
                raise RuntimeError(
                    "Existing results use an older ML target definition. The native-alignment "
                    "target changed; restart this study once with --restart."
                )
            if previous.get("codebooks") != codebooks:
                raise RuntimeError(
                    "Input files/modes/models differ from the persisted run. Use --restart."
                )
            if previous.get("status") == "complete" and results_path.is_file():
                logger.info("Complete compact result set already exists; reuse %s", output)
                return {
                    "output_dir": str(output),
                    "row_count": int(previous.get("row_count", 0)),
                    "resumed": True,
                }
            rows = _read_results_csv(results_path)
            candidate_manifest = dict(previous.get("candidate_parameters", {}))
            candidate_ids = {
                str(key): int(value)
                for key, value in previous.get("candidate_key_to_id", {}).items()
            }
            final_metadata = dict(previous.get("final_models", {}))
            completed_files = {int(value) for value in previous.get("completed_files", [])}
            prepared_fingerprints = {
                str(key): str(value)
                for key, value in previous.get("prepared_fingerprints", {}).items()
            }
            logger.info(
                "Resuming partial study | completed_rows=%d | completed_files=%d",
                len(rows), len(completed_files),
            )
    elif results_path.is_file() or manifest_path.is_file():
        raise RuntimeError(
            "An existing study state is present. Use --resume to continue it or --restart "
            "to discard it."
        )

    normalization_cache: dict[str, Normalization] = {}
    dpi = int(config["reporting"]["dpi"])
    base_seed = int(config["cross_validation"]["seed"])
    work_root = output / ".work"
    checkpoint_root = output / "models"
    signal_plot_root = output / "preprocessing_examples"

    def _progress_manifest(status: str) -> dict[str, Any]:
        return {
            "schema_version": 5,
            "status": status,
            "target_protocol_version": _TARGET_PROTOCOL_VERSION,
            "experiment": config["experiment"]["name"],
            "config_hash": config["_config_hash"],
            "config_path": config["_config_path"],
            "prepared_dir": config["preprocessing"]["prepared_dir"],
            "prepared_fingerprints": dict(prepared_fingerprints),
            "row_count": len(rows),
            "completed_files": sorted(completed_files),
            "codebooks": codebooks,
            "candidate_key_to_id": dict(candidate_ids),
            "candidate_parameters": candidate_manifest,
            "final_models": final_metadata,
            "ctr_estimator": {
                "distribution": "ordinary sample mean and sample standard deviation (ddof=1), all events",
                "ctr_definition": "2*sqrt(2*ln(2))*sample_std",
                "cv_selection": "lowest arithmetic mean of independent score-fold CTR values",
            },
            "target_definition": {
                "waveform_ml": (
                    "g(s1)-g(s2) = Delta t_LED - Delta(shift_needed - shift_applied) - true TOF; "
                    "the LED and native anchor are extracted after the configured channel preprocessing "
                    "from the same signal representation consumed by ML; no LED-derived anchor/alignment "
                    "term is added analytically at inference"
                ),
                "multithreshold": "unchanged raw threshold-crossing SVR target",
            },
            "protocol": {
                "split": "single deterministic random development/blind split per file",
                "cv": "random K-fold on development; each score fold is evaluated separately and model outputs are never pooled",
                "early_stop": "neural fit set + disjoint early-stop subset carved only from K-1 training folds",
                "final": "all candidate selections are frozen before blind is opened; selected configurations are retrained from scratch on development fit/early-stop splits and blind is evaluated once",
                "evaluation_rejection": "none; every prepared event in the requested split is included",
                "multithreshold_denoising": False,
                "resume": "candidate summaries are persisted atomically; an interrupted candidate is retrained, completed candidates are skipped",
            },
            "result_columns": {
                "stage": {"0": "cv_candidate_summary", "1": "blind"},
                "candidate_id": "parameters stored once in candidate_parameters; -1 for fixed LED/CFD",
            },
        }

    def persist_progress(status: str = "in_progress") -> None:
        _write_csv(results_path, rows)
        atomic_json(manifest_path, _progress_manifest(status))

    # Create progress files before the first expensive candidate so the run is
    # resumable even if it is interrupted very early.
    persist_progress("in_progress")

    for root_file in root_files:
        file_id = codebooks["file"][root_file.name]
        logger.info("File %d/%d | %s", file_id + 1, len(root_files), root_file.name)
        dataset = prepare_file_dataset(config, root_file, rebuild=rebuild_preprocessing, logger=logger)
        prepared_fingerprint = str(
            dataset.manifest.get("request_fingerprint", dataset.manifest.get("fingerprint", ""))
        )
        previous_fingerprint = prepared_fingerprints.get(str(file_id))
        if previous_fingerprint and previous_fingerprint != prepared_fingerprint:
            raise RuntimeError(
                f"Prepared dataset changed for {root_file.name} while resuming. "
                "Use --restart so CV results are not mixed across datasets."
            )
        prepared_fingerprints[str(file_id)] = prepared_fingerprint
        persist_progress("in_progress")
        if file_id in completed_files:
            logger.info("File already complete; skipping | %s", root_file.name)
            continue
        plot_prepared_signal_examples(dataset, signal_plot_root / f"{root_file.stem}.png", dpi=dpi)
        development, blind = _random_dev_blind(
            int(dataset.event_id.size),
            blind_fraction=float(config["cross_validation"]["blind_fraction"]),
            seed=_seed_for(base_seed, file_id, "devblind"),
        )
        folds = _kfold(
            development, n_splits=int(config["cross_validation"]["n_splits"]),
            seed=_seed_for(base_seed, file_id, "folds"),
        )
        voltage = _voltage_from_name(root_file.name, str(config["reporting"]["voltage_pattern"]))
        file_panels: dict[str, dict[str, np.ndarray]] = {}
        file_xai: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        file_corrections: dict[str, dict[str, np.ndarray]] = {}

        for mode in config["channel_modes"]:
            mode_id = codebooks["mode"][mode]
            logger.info(
                "Mode | %s | development=%d | blind=%d | folds=%d",
                mode, len(development), len(blind), len(folds),
            )
            file_panels[mode] = {}
            file_xai[mode] = {}
            file_corrections[mode] = {}

            # ------------------------- DEVELOPMENT ONLY -------------------------
            # Complete every candidate search before any blind timing value or
            # blind waveform is passed to an evaluator/model.
            selected_waveform: list[tuple[dict[str, Any], int, dict[str, Any]]] = []
            for space in config["_model_spaces"]:
                model_id = codebooks["model"][space["id"]]
                selection: tuple[float, dict[str, Any]] | None = None
                combinations = _waveform_candidate_combinations(
                    config, space, mode=mode,
                    seed=_seed_for(base_seed, file_id, mode, space["id"], "search"),
                )
                logger.info(
                    "CV search | mode=%s | model=%s | candidates=%d",
                    mode, space["id"], len(combinations),
                )
                for candidate_position, (window, variant, subsampling, overrides) in enumerate(combinations, start=1):
                    descriptor = {
                        "family": space["id"], "mode": mode, "window": window["id"],
                        "variant": variant, "subsampling": int(subsampling), "overrides": overrides,
                    }
                    key = canonical_hash(descriptor)
                    candidate_id = _candidate_id_for_key(candidate_ids, key)
                    candidate_manifest[str(candidate_id)] = descriptor
                    candidate_work = work_root / f"f{file_id}_m{mode_id}_model{model_id}_c{candidate_id}"
                    logger.info(
                        "CV candidate %d/%d | mode=%s | model=%s | window=%s [%.3g, %.3g] ns | input=%s:%s | subsampling=%d",
                        candidate_position, len(combinations), mode, space["id"], window["id"],
                        -float(window["before_ns"]), float(window["after_ns"]),
                        CHANNEL_MODES[mode][0], variant, int(subsampling),
                    )
                    existing = _find_result_row(
                        rows, stage=_STAGE_OOF, file_id=file_id, mode_id=mode_id,
                        model_id=model_id, candidate_id=candidate_id,
                    )
                    if existing is not None:
                        metrics = existing
                        logger.info(
                            "CV resume | mode=%s | model=%s | window=%s | candidate=%d | CTR=%.3f ps",
                            mode, space["id"], window["id"], candidate_id, float(metrics["ctr_ps"]),
                        )
                    else:
                        try:
                            metrics = _waveform_oof_candidate(
                                config, dataset, development, folds,
                                file_id=file_id, mode=mode, window=window, variant=variant,
                                subsampling=int(subsampling), space=space, overrides=overrides,
                                candidate_id=candidate_id, work_root=candidate_work, logger=logger,
                                normalization_cache=normalization_cache,
                            )
                        finally:
                            shutil.rmtree(candidate_work, ignore_errors=True)
                        _upsert_result_row(rows, {
                            "stage": _STAGE_OOF, "file_id": file_id, "mode_id": mode_id,
                            "model_id": model_id, "candidate_id": candidate_id,
                            "window_id": codebooks["window"][window["id"]],
                            "variant_id": codebooks["variant"][variant], "subsampling": int(subsampling),
                            "selected": 0, "coverage": 1.0, "voltage_V": voltage, **metrics,
                        })
                        persist_progress("in_progress")
                        logger.info(
                            "CV result | mode=%s | model=%s | window=%s | CTR=%.3f +/- %.3f ps | std=%.3f ps | RMSE=%.3f ps",
                            mode, space["id"], window["id"], metrics["ctr_ps"],
                            metrics["ctr_fold_std_ps"], metrics["std_ps"], metrics["rmse_ps"],
                        )
                    if selection is None or float(metrics["ctr_ps"]) < selection[0]:
                        selection = (float(metrics["ctr_ps"]), {
                            "candidate_id": candidate_id, "window": window, "variant": variant,
                            "subsampling": int(subsampling), "overrides": overrides,
                        })
                if selection is None:
                    raise RuntimeError(
                        f"No successful candidate for {space['id']} | {root_file.name} | {mode}"
                    )
                chosen = selection[1]
                logger.info(
                    "CV selected | mode=%s | model=%s | window=%s | input=%s | subsampling=%d | mean fold CTR=%.3f ps",
                    mode, space["id"], chosen["window"]["id"], chosen["variant"],
                    int(chosen["subsampling"]), selection[0],
                )
                for row in rows:
                    if (
                        int(row["stage"]) == _STAGE_OOF and int(row["file_id"]) == file_id
                        and int(row["mode_id"]) == mode_id and int(row["model_id"]) == model_id
                    ):
                        row["selected"] = int(
                            int(row["candidate_id"]) == int(chosen["candidate_id"])
                        )
                persist_progress("in_progress")
                selected_waveform.append((space, model_id, chosen))

            selected_mt: dict[str, Any] | None = None
            mt_window: dict[str, Any] | None = None
            if bool(config["multithreshold"].get("enabled", False)):
                mt_window_id = str(config["multithreshold"]["window_id"])
                mt_window = next(w for w in config["windows_ns"] if w["id"] == mt_window_id)
                logger.info(
                    "CV search | mode=%s | model=multithreshold_svr | window=%s [%.3g, %.3g] ns | input=raw",
                    mode, mt_window["id"], -float(mt_window["before_ns"]), float(mt_window["after_ns"]),
                )
                selected_mt = _multithreshold_oof_select(
                    config, dataset, development, folds,
                    file_id=file_id, mode=mode, mode_id=mode_id,
                    window=mt_window, window_id=codebooks["window"][mt_window["id"]],
                    model_id=codebooks["model"][_MODEL_MULTITHRESHOLD],
                    candidate_ids=candidate_ids, candidate_manifest=candidate_manifest,
                    rows=rows, voltage=voltage, logger=logger,
                    persist_progress=lambda: persist_progress("in_progress"),
                )

            # --------------------------- BLIND PHASE ----------------------------
            # From this point onward candidate selection is frozen.  Blind is used
            # only to report the final LED/CFD and selected-model performance.
            led_blind, cfd_blind = _target_deltas(dataset, mode, blind)
            led_residual = led_blind - float(dataset.true_tof_ps)
            cfd_residual = cfd_blind - float(dataset.true_tof_ps)
            for label, residual in ((_MODEL_LED, led_residual), (_MODEL_CFD, cfd_residual)):
                metrics = _distribution_stats(residual, method=f"Blind {label} {mode}")
                _upsert_result_row(rows, {
                    "stage": _STAGE_BLIND, "file_id": file_id, "mode_id": mode_id,
                    "model_id": codebooks["model"][label], "candidate_id": -1,
                    "window_id": -1, "variant_id": -1, "subsampling": 1,
                    "selected": 1, "coverage": 1.0, "voltage_V": voltage, **metrics,
                })
                persist_progress("in_progress")
                file_panels[mode][label.upper()] = residual

            for space, model_id, chosen in selected_waveform:
                final_dir = work_root / f"final_f{file_id}_m{mode_id}_model{model_id}"
                checkpoint = checkpoint_root / f"f{file_id}_m{mode_id}_model{model_id}.pt"
                residual, metrics, meta, xai_profile = _waveform_final(
                    config, dataset, development, blind,
                    file_id=file_id, mode=mode, window=chosen["window"], variant=chosen["variant"],
                    subsampling=chosen["subsampling"], space=space, overrides=chosen["overrides"],
                    candidate_id=chosen["candidate_id"], work_dir=final_dir,
                    checkpoint_path=checkpoint, logger=logger,
                    normalization_cache=normalization_cache,
                )
                _upsert_result_row(rows, {
                    "stage": _STAGE_BLIND, "file_id": file_id, "mode_id": mode_id,
                    "model_id": model_id, "candidate_id": chosen["candidate_id"],
                    "window_id": codebooks["window"][chosen["window"]["id"]],
                    "variant_id": codebooks["variant"][chosen["variant"]],
                    "subsampling": chosen["subsampling"], "selected": 1, "coverage": 1.0,
                    "voltage_V": voltage, **metrics,
                })
                final_metadata[f"{file_id}:{mode_id}:{model_id}"] = {
                    "checkpoint": str(checkpoint.relative_to(output)), **meta,
                }
                persist_progress("in_progress")
                file_panels[mode][space["id"]] = residual
                file_corrections[mode][space["id"]] = led_residual - residual
                if xai_profile is not None:
                    file_xai[mode][space["id"]] = xai_profile

            if selected_mt is not None and mt_window is not None:
                residual, metrics = _multithreshold_final(
                    config, dataset, development, blind, selected_mt
                )
                mt_model_id = codebooks["model"][_MODEL_MULTITHRESHOLD]
                _upsert_result_row(rows, {
                    "stage": _STAGE_BLIND, "file_id": file_id, "mode_id": mode_id,
                    "model_id": mt_model_id, "candidate_id": selected_mt["candidate_id"],
                    "window_id": selected_mt["window_id"], "variant_id": 0,
                    "subsampling": 1, "selected": 1, "coverage": 1.0,
                    "voltage_V": voltage, **metrics,
                })
                persist_progress("in_progress")
                file_panels[mode][_MODEL_MULTITHRESHOLD] = residual
                file_corrections[mode][_MODEL_MULTITHRESHOLD] = led_residual - residual

        if bool(config["reporting"].get("save_final_fit_plots", True)):
            _plot_final_file(output / "final_distributions" / f"{root_file.stem}.png", file_panels, dpi=dpi)
        if bool(config.get("reporting", {}).get("xai", {}).get("enabled", True)):
            _plot_xai_file(output / "xai" / f"{root_file.stem}.png", file_xai, file_corrections, dpi=dpi)
        normalization_cache.clear()
        completed_files.add(file_id)
        persist_progress("in_progress")

    shutil.rmtree(work_root, ignore_errors=True)
    _plot_ctr_vs_voltage(output / "ctr_vs_voltage.png", rows, codebooks, dpi=dpi)
    persist_progress("complete")
    logger.info("Compact study complete | rows=%d | %s", len(rows), output)
    return {"output_dir": str(output), "row_count": len(rows), "results": str(results_path)}
