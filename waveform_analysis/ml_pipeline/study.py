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
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from utils.fit import FitResult

from .common import atomic_json, canonical_hash
from .dataset import PreparedDataset
from .metrics import fit_times_ps
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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _fit_row(values_ps: np.ndarray, *, method: str, fit_config: dict[str, Any]) -> tuple[FitResult, dict[str, Any]]:
    values = np.asarray(values_ps, dtype=np.float64).reshape(-1)
    if values.size == 0 or np.any(~np.isfinite(values)):
        raise RuntimeError(f"{method}: final evaluation requires one finite value for every event")
    fit = fit_times_ps(values, method, fit_config)
    if not fit.success:
        raise RuntimeError(f"{method}: Gaussian fit failed: {fit.message}")
    return fit, {
        "n": int(values.size),
        "ctr_ps": float(fit.ctr_ps),
        "ctr_err_ps": float(fit.ctr_error_ps),
        "mean_ps": float(fit.mean_ps),
        "rmse_ps": float(np.sqrt(np.mean(values * values))),
        "bias_ps": float(np.mean(values)),
        "dev_ndof": float(fit.chi2_ndof),
        "bin_ps": float(fit.bin_width_ps),
        "phase_ps": float(fit.bin_phase_ps),
        "phase_ctr_std_ps": float(fit.phase_ctr_std_ps),
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
) -> tuple[np.ndarray, FitResult, dict[str, Any]]:
    input_waveforms, target = CHANNEL_MODES[mode]
    source = input_channel_variant_dataset_view(dataset, input_waveforms, variant)
    view = prediction_window_dataset_view(
        source,
        input_waveforms=input_waveforms,
        target=target,
        before_ns=float(window["before_ns"]),
        after_ns=float(window["after_ns"]),
    )
    oof = np.full(dataset.event_id.size, np.nan, dtype=np.float64)
    base_seed = int(study["cross_validation"]["seed"])
    for fold_index, (train_pool, score_idx) in enumerate(folds):
        candidate_seed = _seed_for(base_seed, file_id, mode, window["id"], variant, subsampling, space["id"], candidate_id, fold_index)
        preview_cfg = _candidate_training_config(
            study, space, overrides, mode=mode, subsampling=subsampling,
            train_dir=work_root / f"f{fold_index}", seed=candidate_seed, final=False,
        )
        if preview_cfg["model"]["type"] == "linear_svr":
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
        oof[score_idx] = _predict_indices(model, normalization, preview_cfg, fold_view, score_idx)
        _cleanup_training(model, Path(preview_cfg["output"]["train_dir"]))
    values = oof[development]
    if np.any(~np.isfinite(values)):
        missing = int(np.count_nonzero(~np.isfinite(values)))
        raise RuntimeError(f"Candidate {candidate_id} has {missing} missing OOF predictions")
    fit, metrics = _fit_row(values, method=f"OOF {space['id']}", fit_config=study["fit"])
    return values, fit, metrics




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
) -> tuple[np.ndarray, FitResult, dict[str, Any], dict[str, Any], tuple[np.ndarray, np.ndarray] | None]:
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
    if cfg["model"]["type"] == "linear_svr":
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
    fit, metrics = _fit_row(residual, method=f"Blind {space['id']}", fit_config=study["fit"])
    # XAI can be computed before the model is released; caller receives only compact metadata.
    final_meta = {
        "best_epoch": int(summary.get("best_epoch", 0)),
        "normalization": summary.get("normalization", {}),
        "model_type": cfg["model"]["type"],
    }
    xai_profile = None
    xai_cfg = study.get("reporting", {}).get("xai", {}) or {}
    if bool(xai_cfg.get("enabled", True)):
        xai_profile = _integrated_gradient_profile(
            model, normalization, cfg, final_view, blind,
            max_events=int(xai_cfg.get("max_events", 512)),
            steps=int(xai_cfg.get("integrated_gradient_steps", 16)),
        )
    _cleanup_training(model, work_dir, keep_best=checkpoint_path)
    return residual, fit, metrics, final_meta, xai_profile


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
    led_ps, _ = _target_deltas(dataset, mode, np.arange(dataset.event_id.size))
    target_correction = led_ps - float(dataset.true_tof_ps)

    best: tuple[float, dict[str, Any], int] | None = None
    for params in _multithreshold_candidates(cfg, thresholds.size):
        key = canonical_hash({
            "family": _MODEL_MULTITHRESHOLD, "mode": mode,
            "window": window["id"], **params,
        })
        candidate_id = candidate_ids.setdefault(key, len(candidate_ids))
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

        oof = np.full(dataset.event_id.size, np.nan, dtype=np.float64)
        for train_pool, score_idx in folds:
            estimator = make_pipeline(
                StandardScaler(),
                SVR(
                    kernel=params["kernel"], C=params["C"],
                    epsilon=params["epsilon_ps"], gamma=params["gamma"],
                ),
            )
            estimator.fit(features_all[np.ix_(train_pool, cols)], target_correction[train_pool])
            correction = estimator.predict(features_all[np.ix_(score_idx, cols)])
            oof[score_idx] = led_ps[score_idx] - correction - float(dataset.true_tof_ps)
        values = oof[development]
        if np.any(~np.isfinite(values)):
            raise RuntimeError("Multithreshold OOF prediction is incomplete")
        _fit, metrics = _fit_row(
            values, method="OOF multithreshold SVR", fit_config=study["fit"]
        )
        rows.append({
            "stage": _STAGE_OOF, "file_id": file_id, "mode_id": mode_id,
            "model_id": model_id, "candidate_id": candidate_id,
            "voltage_V": voltage, "window_id": window_id,
            "variant_id": 0, "subsampling": 1, "selected": 0,
            "coverage": 1.0, **metrics,
        })
        if best is None or metrics["ctr_ps"] < best[0]:
            best = (metrics["ctr_ps"], params, candidate_id)

    if best is None:
        logger.warning(
            "No multithreshold candidate covers every development event | file=%s mode=%s",
            dataset.directory.name, mode,
        )
        return None

    _, params, selected_candidate = best
    for row in rows:
        if (
            row["stage"] == _STAGE_OOF
            and row["file_id"] == file_id
            and row["mode_id"] == mode_id
            and row["model_id"] == model_id
            and row["candidate_id"] == selected_candidate
        ):
            row["selected"] = 1
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
) -> tuple[np.ndarray, FitResult, dict[str, Any]]:
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
    fit, metrics = _fit_row(
        residual, method="Blind multithreshold SVR", fit_config=study["fit"]
    )
    return residual, fit, metrics

def _plot_final_file(
    destination: Path,
    panels: dict[str, dict[str, np.ndarray]],
    *,
    dpi: int,
) -> None:
    if not panels:
        return
    modes = list(panels)
    fig, axes = plt.subplots(len(modes), 1, figsize=(10, 3.6 * len(modes)), squeeze=False)
    for ax, mode in zip(axes[:, 0], modes):
        methods = panels[mode]
        for label, values in methods.items():
            values = np.asarray(values, dtype=np.float64)
            # Reporting follows the evaluation invariant: display every prepared
            # blind event rather than clipping tails by quantiles.
            ax.hist(values, bins=80, histtype="step", density=True, label=label)
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
            ax.errorbar(
                [float(r["voltage_V"]) for r in points],
                [float(r["ctr_ps"]) for r in points],
                yerr=[float(r["ctr_err_ps"]) for r in points],
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
    if resume and results_path.is_file() and manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        if manifest.get("config_hash") != config.get("_config_hash"):
            raise RuntimeError(
                "Existing compact results were produced by a different configuration. "
                "Use --restart or a different experiment.output_dir."
            )
        logger.info("Complete compact result set already exists; reuse %s", output)
        return {"output_dir": str(output), "row_count": int(manifest.get("row_count", 0)), "resumed": True}

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
        raise FileNotFoundError(f"No ROOT files match {config['data']['root_glob']} in {config['data']['root_folder']}")

    codebooks = {
        "file": {path.name: i for i, path in enumerate(root_files)},
        "mode": {name: i for i, name in enumerate(config["channel_modes"])},
        "model": {
            _MODEL_LED: 0, _MODEL_CFD: 1,
            **{space["id"]: i + 2 for i, space in enumerate(config["_model_spaces"])},
        },
        "window": {w["id"]: i for i, w in enumerate(config["windows_ns"])},
        # Keep raw=0 permanently because multithreshold is hard-wired to raw.
        # Denoised is present only when at least one ML channel requests it.
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
    normalization_cache: dict[str, Normalization] = {}
    final_metadata: dict[str, Any] = {}
    dpi = int(config["reporting"]["dpi"])
    base_seed = int(config["cross_validation"]["seed"])
    work_root = output / ".work"
    checkpoint_root = output / "models"
    signal_plot_root = output / "preprocessing_examples"

    for root_file in root_files:
        file_id = codebooks["file"][root_file.name]
        logger.info("File %d/%d | %s", file_id + 1, len(root_files), root_file.name)
        dataset = prepare_file_dataset(config, root_file, rebuild=rebuild_preprocessing, logger=logger)
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
                    candidate_id = candidate_ids.setdefault(key, len(candidate_ids))
                    candidate_manifest[str(candidate_id)] = descriptor
                    candidate_work = work_root / f"f{file_id}_m{mode_id}_model{model_id}_c{candidate_id}"
                    logger.info(
                        "CV candidate %d/%d | mode=%s | model=%s | window=%s [%.3g, %.3g] ns | input=%s:%s | subsampling=%d",
                        candidate_position, len(combinations), mode, space["id"], window["id"],
                        -float(window["before_ns"]), float(window["after_ns"]),
                        CHANNEL_MODES[mode][0], variant, int(subsampling),
                    )
                    try:
                        _values, _fit, metrics = _waveform_oof_candidate(
                            config, dataset, development, folds,
                            file_id=file_id, mode=mode, window=window, variant=variant,
                            subsampling=int(subsampling), space=space, overrides=overrides,
                            candidate_id=candidate_id, work_root=candidate_work, logger=logger,
                            normalization_cache=normalization_cache,
                        )
                    finally:
                        shutil.rmtree(candidate_work, ignore_errors=True)
                    rows.append({
                        "stage": _STAGE_OOF, "file_id": file_id, "mode_id": mode_id,
                        "model_id": model_id, "candidate_id": candidate_id,
                        "window_id": codebooks["window"][window["id"]],
                        "variant_id": codebooks["variant"][variant], "subsampling": int(subsampling),
                        "selected": 0, "coverage": 1.0, "voltage_V": voltage, **metrics,
                    })
                    logger.info(
                        "OOF result | mode=%s | model=%s | window=%s | CTR=%.3f ps | RMSE=%.3f ps | bias=%.3f ps",
                        mode, space["id"], window["id"], metrics["ctr_ps"],
                        metrics["rmse_ps"], metrics["bias_ps"],
                    )
                    if selection is None or metrics["ctr_ps"] < selection[0]:
                        selection = (metrics["ctr_ps"], {
                            "candidate_id": candidate_id, "window": window, "variant": variant,
                            "subsampling": int(subsampling), "overrides": overrides,
                        })
                if selection is None:
                    raise RuntimeError(
                        f"No successful candidate for {space['id']} | {root_file.name} | {mode}"
                    )
                chosen = selection[1]
                logger.info(
                    "CV selected | mode=%s | model=%s | window=%s | input=%s | subsampling=%d | OOF CTR=%.3f ps",
                    mode, space["id"], chosen["window"]["id"], chosen["variant"],
                    int(chosen["subsampling"]), selection[0],
                )
                for row in rows:
                    if (
                        row["stage"] == _STAGE_OOF and row["file_id"] == file_id
                        and row["mode_id"] == mode_id and row["model_id"] == model_id
                        and row["candidate_id"] == chosen["candidate_id"]
                    ):
                        row["selected"] = 1
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
                )

            # --------------------------- BLIND PHASE ----------------------------
            # From this point onward candidate selection is frozen.  Blind is used
            # only to report the final LED/CFD and selected-model performance.
            led_blind, cfd_blind = _target_deltas(dataset, mode, blind)
            led_residual = led_blind - float(dataset.true_tof_ps)
            cfd_residual = cfd_blind - float(dataset.true_tof_ps)
            for label, residual in ((_MODEL_LED, led_residual), (_MODEL_CFD, cfd_residual)):
                _fit, metrics = _fit_row(
                    residual, method=f"Blind {label} {mode}", fit_config=config["fit"]
                )
                rows.append({
                    "stage": _STAGE_BLIND, "file_id": file_id, "mode_id": mode_id,
                    "model_id": codebooks["model"][label], "candidate_id": -1,
                    "window_id": -1, "variant_id": -1, "subsampling": 1,
                    "selected": 1, "coverage": 1.0, "voltage_V": voltage, **metrics,
                })
                file_panels[mode][label.upper()] = residual

            for space, model_id, chosen in selected_waveform:
                final_dir = work_root / f"final_f{file_id}_m{mode_id}_model{model_id}"
                checkpoint = checkpoint_root / f"f{file_id}_m{mode_id}_model{model_id}.pt"
                residual, _fit, metrics, meta, xai_profile = _waveform_final(
                    config, dataset, development, blind,
                    file_id=file_id, mode=mode, window=chosen["window"], variant=chosen["variant"],
                    subsampling=chosen["subsampling"], space=space, overrides=chosen["overrides"],
                    candidate_id=chosen["candidate_id"], work_dir=final_dir,
                    checkpoint_path=checkpoint, logger=logger,
                    normalization_cache=normalization_cache,
                )
                rows.append({
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
                file_panels[mode][space["id"]] = residual
                file_corrections[mode][space["id"]] = led_residual - residual
                if xai_profile is not None:
                    file_xai[mode][space["id"]] = xai_profile

            if selected_mt is not None and mt_window is not None:
                residual, _fit, metrics = _multithreshold_final(
                    config, dataset, development, blind, selected_mt
                )
                mt_model_id = codebooks["model"][_MODEL_MULTITHRESHOLD]
                rows.append({
                    "stage": _STAGE_BLIND, "file_id": file_id, "mode_id": mode_id,
                    "model_id": mt_model_id, "candidate_id": selected_mt["candidate_id"],
                    "window_id": selected_mt["window_id"], "variant_id": 0,
                    "subsampling": 1, "selected": 1, "coverage": 1.0,
                    "voltage_V": voltage, **metrics,
                })
                file_panels[mode][_MODEL_MULTITHRESHOLD] = residual
                file_corrections[mode][_MODEL_MULTITHRESHOLD] = led_residual - residual

        if bool(config["reporting"].get("save_final_fit_plots", True)):
            _plot_final_file(output / "final_distributions" / f"{root_file.stem}.png", file_panels, dpi=dpi)
        if bool(config.get("reporting", {}).get("xai", {}).get("enabled", True)):
            _plot_xai_file(output / "xai" / f"{root_file.stem}.png", file_xai, file_corrections, dpi=dpi)
        normalization_cache.clear()

    shutil.rmtree(work_root, ignore_errors=True)
    _write_csv(results_path, rows)
    _plot_ctr_vs_voltage(output / "ctr_vs_voltage.png", rows, codebooks, dpi=dpi)
    manifest = {
        "schema_version": 3,
        "experiment": config["experiment"]["name"],
        "config_hash": config["_config_hash"],
        "config_path": config["_config_path"],
        "prepared_dir": config["preprocessing"]["prepared_dir"],
        "row_count": len(rows),
        "codebooks": codebooks,
        "candidate_parameters": candidate_manifest,
        "final_models": final_metadata,
        "fit": config["fit"],
        "protocol": {
            "split": "single deterministic random development/blind split per file",
            "cv": "random K-fold on development; pooled OOF CTR selects candidates",
            "early_stop": "neural fit set + disjoint early-stop subset carved only from K-1 training folds",
            "final": "all candidate selections are frozen before blind is opened; selected configurations are retrained from scratch on development fit/early-stop splits and blind is evaluated once",
            "evaluation_rejection": "none; every prepared event in the requested split is included",
            "multithreshold_denoising": False,
        },
        "result_columns": {
            "stage": {"0": "pooled_oof", "1": "blind"},
            "candidate_id": "parameters stored once in candidate_parameters; -1 for fixed LED/CFD",
        },
    }
    atomic_json(manifest_path, manifest)
    logger.info("Compact study complete | rows=%d | %s", len(rows), output)
    return {"output_dir": str(output), "row_count": len(rows), "results": str(results_path)}
