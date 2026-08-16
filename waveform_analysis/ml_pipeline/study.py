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
from .metrics import FWHM_PER_SIGMA, ctr_bootstrap_uncertainty, fit_times_ps, residual_metrics
from .models import validate_model, validate_model_training
from .prediction import prediction_window_dataset_view
from .prepared_data import (
    input_variant_dataset_view,
    plot_prepared_signal_examples,
    prepare_file_dataset,
)
from .study_config import CHANNEL_MODES, candidate_overrides, discover_root_files, set_nested
from .torch_data import Normalization, compute_normalization
from .training import train_model
from .training_utils import make_split_loader, predict_loader, resolve_device
from .validation import outer_splits, random_dev_blind, selection_splits, nested_inner_validation
from .reporting import (
    plot_blind_distribution, plot_correction_matrix, plot_ctr_vs_voltage,
    plot_final_bars, plot_selection_vs_blind, plot_top_corrections,
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
    source = input_variant_dataset_view(dataset, variant)
    input_waveforms, target = CHANNEL_MODES[mode]
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
    source = input_variant_dataset_view(dataset, variant)
    input_waveforms, target = CHANNEL_MODES[mode]
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
    raw = input_variant_dataset_view(dataset, "raw")
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


def _selection_metric_summary(
    residual_parts: list[np.ndarray],
) -> tuple[np.ndarray, dict[str, Any], list[float]]:
    if not residual_parts:
        raise RuntimeError("Selection procedure produced no score residuals")
    parts = [np.asarray(values, dtype=np.float64).reshape(-1) for values in residual_parts]
    if any(values.size == 0 or np.any(~np.isfinite(values)) for values in parts):
        raise RuntimeError("Selection procedure produced empty/non-finite score residuals")
    fold_ctrs = [float(residual_metrics(values)["ctr_ps"]) for values in parts]
    combined = np.concatenate(parts)
    simple = residual_metrics(combined)
    selection_ctr = fold_ctrs[0] if len(fold_ctrs) == 1 else float(np.mean(fold_ctrs))
    fold_std = (
        float(np.std(np.asarray(fold_ctrs, dtype=np.float64), ddof=1))
        if len(fold_ctrs) > 1 else float("nan")
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
    """Evaluate one candidate with either one holdout split or K CV folds.

    All model construction, target definition, normalization and fitting are the
    unchanged working implementation supplied by the user. This wrapper changes
    only how train/score indices are generated and how experiment-level CTR is
    aggregated.
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
    residual_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []
    base_seed = int(study["validation"]["seed"])
    for split_index, (train_pool, score_idx) in enumerate(splits):
        candidate_seed = _seed_for(
            base_seed, file_id, mode, window["id"], variant, subsampling,
            space["id"], candidate_id, split_index,
        )
        preview_cfg = _candidate_training_config(
            study, space, overrides, mode=mode, subsampling=subsampling,
            train_dir=work_root / f"s{split_index}", seed=candidate_seed, final=False,
        )
        if preview_cfg["model"]["type"] == "linear_svr":
            fit_idx = np.asarray(train_pool, dtype=np.int64)
            early_idx = fit_idx
        else:
            fit_idx, early_idx = _fit_early_split(
                np.asarray(train_pool, dtype=np.int64),
                fraction=_early_fraction(study, preview_cfg),
                seed=_seed_for(base_seed, file_id, mode, "early", split_index),
            )
        fold_view = replace(
            view, train=fit_idx, validation=early_idx,
            evaluation=np.asarray(score_idx, dtype=np.int64),
        )
        cached_normalization = _normalization_for_fit_subset(
            fold_view, fit_idx, subsampling=subsampling, cache=normalization_cache
        )
        model, normalization, _summary = _train_in_memory(
            preview_cfg, fold_view, logger=logger,
            data_view={"stage": "selection", "split": split_index, "candidate_id": candidate_id},
            normalization_override=cached_normalization,
        )
        residual = _predict_indices(
            model, normalization, preview_cfg, fold_view,
            np.asarray(score_idx, dtype=np.int64),
        )
        residual_parts.append(np.asarray(residual, dtype=np.float64))
        score_parts.append(np.asarray(score_idx, dtype=np.int64))
        _cleanup_training(model, Path(preview_cfg["output"]["train_dir"]))
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
    compute_xai: bool = False,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any], tuple[np.ndarray, np.ndarray] | None]:
    source = input_variant_dataset_view(dataset, variant)
    input_waveforms, target = CHANNEL_MODES[mode]
    view = prediction_window_dataset_view(
        source, input_waveforms=input_waveforms, target=target,
        before_ns=float(window["before_ns"]), after_ns=float(window["after_ns"]),
    )
    seed = _seed_for(
        int(study["validation"]["seed"]), file_id, mode, space["id"],
        candidate_id, "evaluation", int(np.asarray(evaluation).size),
    )
    cfg = _candidate_training_config(
        study, space, overrides, mode=mode, subsampling=subsampling,
        train_dir=work_dir, seed=seed, final=checkpoint_path is not None,
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
            seed=_seed_for(int(study["validation"]["seed"]), file_id, mode, "early", "eval"),
        )
    eval_view = replace(view, train=fit_idx, validation=early_idx, evaluation=evaluation)
    cached_normalization = _normalization_for_fit_subset(
        eval_view, fit_idx, subsampling=subsampling, cache=normalization_cache
    )
    model, normalization, summary = _train_in_memory(
        cfg, eval_view, logger=logger,
        data_view={"stage": "evaluation", "candidate_id": candidate_id},
        normalization_override=cached_normalization,
    )
    residual = _predict_indices(model, normalization, cfg, eval_view, evaluation)
    _fit, metrics = _fit_row(
        residual, method=f"Evaluation {space['id']}", fit_config=study["fit"]
    )
    xai_profile = None
    if compute_xai:
        xai_cfg = study.get("reporting", {}).get("xai", {}) or {}
        if bool(xai_cfg.get("enabled", False)):
            xai_profile = _integrated_gradient_profile(
                model, normalization, cfg, eval_view, evaluation,
                max_events=int(xai_cfg.get("max_events", 512)),
                steps=int(xai_cfg.get("integrated_gradient_steps", 16)),
            )
    meta = {
        "best_epoch": int(summary.get("best_epoch", 0)),
        "normalization": summary.get("normalization", {}),
        "model_type": cfg["model"]["type"],
    }
    _cleanup_training(model, work_dir, keep_best=checkpoint_path)
    return residual, metrics, meta, xai_profile


def _resolved_hyperparameters(space: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    cfg = _apply_overrides(space["base_train_config"], overrides)
    return {
        "model": copy.deepcopy(cfg.get("model", {})),
        "optimizer": copy.deepcopy(cfg.get("optimizer", {})),
        "training": {
            key: copy.deepcopy(value)
            for key, value in cfg.get("training", {}).items()
            if key in {
                "batch_size", "epochs", "epochs_per_unit", "early_stopping_patience",
                "unit_early_stopping_patience", "min_unit_improvement_ps",
                "min_relative_unit_improvement", "early_stop_fraction",
            }
        },
    }


def _candidate_descriptor(
    space: dict[str, Any], mode: str, window: dict[str, Any], variant: str,
    subsampling: int, overrides: dict[str, Any],
) -> dict[str, Any]:
    return {
        "family": space["id"], "mode": mode, "window": window["id"],
        "variant": variant, "subsampling": int(subsampling), "overrides": overrides,
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
        short_key = key.rsplit(".", 1)[-1]
        short_key = aliases.get(short_key, short_key)
        if isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
        if isinstance(value, float):
            value_text = f"{value:g}"
        else:
            value_text = str(value)
        parts.append(f"{short_key}={value_text}")
    return " | ".join(parts) if parts else "default params"


def _select_waveform_space(
    study: dict[str, Any], dataset: PreparedDataset, selection_indices: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]], *, file_id: int, mode: str,
    mode_id: int, space: dict[str, Any], model_id: int, codebooks: dict[str, dict[str, int]],
    candidate_ids: dict[str, int], candidate_manifest: dict[str, Any],
    work_root: Path, logger: Any, normalization_cache: dict[str, Normalization],
    voltage: float, result_rows: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    combinations = _waveform_candidate_combinations(
        study, space, mode=mode,
        seed=_seed_for(int(study["validation"]["seed"]), file_id, mode, space["id"], "search"),
    )
    best: dict[str, Any] | None = None
    total = len(combinations)
    last_scope: tuple[str, str, int] | None = None
    for sequence, (window, variant, subsampling, overrides) in enumerate(combinations, start=1):
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
                space["id"], window_text, variant, int(subsampling), total,
            )
            last_scope = scope
        descriptor = _candidate_descriptor(space, mode, window, variant, subsampling, overrides)
        key = canonical_hash(descriptor)
        candidate_id = candidate_ids.setdefault(key, len(candidate_ids))
        candidate_manifest[str(candidate_id)] = descriptor
        candidate_work = work_root / f"select_f{file_id}_m{mode_id}_model{model_id}_c{candidate_id}"
        try:
            score_idx, residual, metrics, fold_ctrs = _waveform_selection_candidate(
                study, dataset, splits, file_id=file_id, mode=mode, window=window,
                variant=variant, subsampling=int(subsampling), space=space,
                overrides=overrides, candidate_id=candidate_id, work_root=candidate_work,
                logger=logger, normalization_cache=normalization_cache,
            )
        finally:
            shutil.rmtree(candidate_work, ignore_errors=True)
        logger.info(
            "Candidate %d/%d | %s | s-CTR %.1f ps",
            sequence, total, _compact_candidate_params(overrides), float(metrics["ctr_ps"]),
        )
        if result_rows is not None:
            result_rows.append({
                "stage": _STAGE_OOF, "file_id": file_id, "mode_id": mode_id,
                "model_id": model_id, "candidate_id": candidate_id,
                "window_id": codebooks["window"][window["id"]],
                "variant_id": codebooks["variant"][variant],
                "subsampling": int(subsampling), "selected": 0, "coverage": 1.0,
                "voltage_V": voltage, **metrics,
            })
        item = {
            "candidate_id": candidate_id, "window": window, "variant": variant,
            "subsampling": int(subsampling), "overrides": overrides,
            "score_indices": score_idx, "score_residual": residual,
            "metrics": metrics, "fold_ctrs": fold_ctrs,
        }
        if best is None or float(metrics["ctr_ps"]) < float(best["metrics"]["ctr_ps"]):
            best = item
    if best is None:
        raise RuntimeError(f"No successful candidate for {space['id']} | mode={mode}")
    if result_rows is not None:
        for row in result_rows:
            if (
                int(row.get("stage", -1)) == _STAGE_OOF
                and int(row.get("file_id", -1)) == file_id
                and int(row.get("mode_id", -1)) == mode_id
                and int(row.get("model_id", -1)) == model_id
                and int(row.get("candidate_id", -2)) == int(best["candidate_id"])
            ):
                row["selected"] = 1
    return best


def _multithreshold_feature_cache(
    study: dict[str, Any], dataset: PreparedDataset, *, mode: str, window: dict[str, Any],
    cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    key = f"{mode}:{window['id']}"
    if key in cache:
        return cache[key]
    cfg = study["multithreshold"]
    raw = input_variant_dataset_view(dataset, "raw")
    input_waveforms, target = CHANNEL_MODES[mode]
    view = prediction_window_dataset_view(
        raw, input_waveforms=input_waveforms, target=target,
        before_ns=float(window["before_ns"]), after_ns=float(window["after_ns"]),
    )
    thresholds = np.asarray(cfg["thresholds_mV"], dtype=np.float64)
    features = _threshold_crossing_matrix(
        view, thresholds, chunk_size=int(cfg.get("chunk_size", 2048))
    )
    led_ps, _ = _target_deltas(dataset, mode, np.arange(dataset.event_id.size))
    entry = {
        "features": features, "thresholds": thresholds, "led_ps": led_ps,
        "target_correction": led_ps - float(dataset.true_tof_ps), "view": view,
    }
    cache[key] = entry
    return entry


def _select_multithreshold(
    study: dict[str, Any], dataset: PreparedDataset, selection_indices: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]], *, file_id: int, mode: str, mode_id: int,
    window: dict[str, Any], model_id: int, codebooks: dict[str, dict[str, int]],
    candidate_ids: dict[str, int], candidate_manifest: dict[str, Any],
    voltage: float, logger: Any, result_rows: list[dict[str, Any]] | None,
    feature_cache: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    if not bool(study["multithreshold"].get("enabled", False)):
        return None
    cfg = study["multithreshold"]
    data = _multithreshold_feature_cache(study, dataset, mode=mode, window=window, cache=feature_cache)
    thresholds = data["thresholds"]
    features = data["features"]
    led_ps = data["led_ps"]
    target = data["target_correction"]
    best: dict[str, Any] | None = None
    candidates = _multithreshold_candidates(cfg, thresholds.size)
    for sequence, params in enumerate(candidates, start=1):
        cols = params["threshold_indices"]
        valid = np.all(np.isfinite(features[:, cols]), axis=1)
        # Same population for every candidate: never improve CTR by dropping hard events.
        if not np.all(valid[np.asarray(selection_indices, dtype=np.int64)]):
            continue
        descriptor = {
            "family": _MODEL_MULTITHRESHOLD, "mode": mode, "window": window["id"],
            "thresholds_mV": thresholds[cols].tolist(),
            **{k: v for k, v in params.items() if k != "threshold_indices"},
        }
        key = canonical_hash(descriptor)
        candidate_id = candidate_ids.setdefault(key, len(candidate_ids))
        candidate_manifest[str(candidate_id)] = descriptor
        residual_parts: list[np.ndarray] = []
        score_parts: list[np.ndarray] = []
        for train_pool, score_idx in splits:
            estimator = make_pipeline(
                StandardScaler(),
                SVR(kernel=params["kernel"], C=params["C"],
                    epsilon=params["epsilon_ps"], gamma=params["gamma"]),
            )
            estimator.fit(features[np.ix_(train_pool, cols)], target[train_pool])
            correction = estimator.predict(features[np.ix_(score_idx, cols)])
            residual_parts.append(led_ps[score_idx] - correction - float(dataset.true_tof_ps))
            score_parts.append(np.asarray(score_idx, dtype=np.int64))
        combined, metrics, fold_ctrs = _selection_metric_summary(residual_parts)
        if result_rows is not None:
            result_rows.append({
                "stage": _STAGE_OOF, "file_id": file_id, "mode_id": mode_id,
                "model_id": model_id, "candidate_id": candidate_id,
                "window_id": codebooks["window"][window["id"]], "variant_id": 0,
                "subsampling": 1, "selected": 0, "coverage": 1.0,
                "voltage_V": voltage, **metrics,
            })
        item = {
            "candidate_id": candidate_id, "window": window, "params": params,
            "features": features, "led_ps": led_ps, "target_correction": target,
            "score_indices": np.concatenate(score_parts), "score_residual": combined,
            "metrics": metrics, "fold_ctrs": fold_ctrs,
            "window_id": codebooks["window"][window["id"]],
        }
        if best is None or float(metrics["ctr_ps"]) < float(best["metrics"]["ctr_ps"]):
            best = item
    if best is None:
        logger.warning("No multithreshold candidate covers the complete selection population | mode=%s", mode)
        return None
    if result_rows is not None:
        for row in result_rows:
            if (
                int(row.get("stage", -1)) == _STAGE_OOF
                and int(row.get("file_id", -1)) == file_id
                and int(row.get("mode_id", -1)) == mode_id
                and int(row.get("model_id", -1)) == model_id
                and int(row.get("candidate_id", -2)) == int(best["candidate_id"])
            ):
                row["selected"] = 1
    logger.info(
        "Selected multithreshold SVR | mode=%s | s-CTR %.1f ps | thresholds=%s",
        mode, float(best["metrics"]["ctr_ps"]),
        thresholds[best["params"]["threshold_indices"]].tolist(),
    )
    return best


def _multithreshold_evaluate(
    study: dict[str, Any], dataset: PreparedDataset, train_pool: np.ndarray,
    evaluation: np.ndarray, selected: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    params = selected["params"]
    cols = params["threshold_indices"]
    features = selected["features"]
    valid = np.all(np.isfinite(features[:, cols]), axis=1)
    train_pool = np.asarray(train_pool, dtype=np.int64)
    evaluation = np.asarray(evaluation, dtype=np.int64)
    if not np.all(valid[train_pool]) or not np.all(valid[evaluation]):
        raise RuntimeError(
            "Selected multithreshold model lacks a threshold crossing for at least one evaluation event"
        )
    estimator = make_pipeline(
        StandardScaler(),
        SVR(kernel=params["kernel"], C=params["C"],
            epsilon=params["epsilon_ps"], gamma=params["gamma"]),
    )
    estimator.fit(features[np.ix_(train_pool, cols)], selected["target_correction"][train_pool])
    correction = estimator.predict(features[np.ix_(evaluation, cols)])
    residual = selected["led_ps"][evaluation] - correction - float(dataset.true_tof_ps)
    _fit, metrics = _fit_row(residual, method="Evaluation multithreshold SVR", fit_config=study["fit"])
    return residual, metrics


def _report_base(
    *, root_file: Path, file_id: int, voltage: float, mode: str, model: str,
    stage_name: str, metrics: dict[str, Any], selected: int = 1,
) -> dict[str, Any]:
    return {
        "file": root_file.name, "file_id": file_id, "voltage_V": voltage,
        "mode": mode, "model": model, "stage_name": stage_name,
        "selected": int(selected), "n": int(metrics.get("n", 0)),
        "mean_ps": float(metrics.get("mean_ps", float("nan"))),
        "std_ps": float(metrics.get("std_ps", float("nan"))),
        "ctr_ps": float(metrics.get("ctr_ps", float("nan"))),
        "rmse_ps": float(metrics.get("rmse_ps", float("nan"))),
        "bias_ps": float(metrics.get("bias_ps", float("nan"))),
    }


def _report_model_details(
    row: dict[str, Any], *, chosen: dict[str, Any], space: dict[str, Any] | None,
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
        row["hyperparameters_json"] = json.dumps({
            "thresholds_mV": np.asarray(chosen.get("threshold_values", []), dtype=float).tolist(),
            **{k: v for k, v in params.items() if k != "threshold_indices"},
        }, sort_keys=True)
    else:
        row["variant"] = chosen["variant"]
        row["subsampling"] = int(chosen["subsampling"])
        row["hyperparameters_json"] = json.dumps(
            _resolved_hyperparameters(space, chosen["overrides"]), sort_keys=True
        )
    return row


def _plot_inclusion(ctr: float, led_ctr: float, ratio_limit: float) -> tuple[float, int]:
    ratio = float(ctr / led_ctr) if np.isfinite(ctr) and np.isfinite(led_ctr) and led_ctr > 0 else float("nan")
    included = int(not np.isfinite(ratio) or ratio <= float(ratio_limit))
    return ratio, included


def run_study(
    config: dict[str, Any], *, dry_run: bool, resume: bool, restart: bool,
    rebuild_preprocessing: bool, logger: Any,
) -> dict[str, Any]:
    output = Path(config["experiment"]["output_dir"])
    if restart and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "results.csv"
    summary_path = output / "summary_results.csv"
    nested_path = output / "nested_results.csv"
    manifest_path = output / "manifest.json"
    if resume and results_path.is_file() and manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        if manifest.get("config_hash") != config.get("_config_hash"):
            raise RuntimeError(
                "Existing compact results were produced by a different configuration. Use --restart."
            )
        logger.info("Complete compact result set already exists; reuse %s", output)
        return {"output_dir": str(output), "row_count": int(manifest.get("row_count", 0)), "resumed": True}

    root_files = discover_root_files(config)
    strategy = str(config["validation"]["strategy"])
    logger.info(
        "Study %s | files=%d | validation=%s | working ML/model implementation preserved",
        config["experiment"]["name"], len(root_files), strategy,
    )
    if dry_run:
        return {
            "output_dir": str(output), "row_count": 0, "dry_run": True,
            "files": [str(v) for v in root_files],
            "models": [v["id"] for v in config["_model_spaces"]],
            "multithreshold": bool(config["multithreshold"].get("enabled", False)),
            "prepared_dir": config["preprocessing"]["prepared_dir"],
            "selection_store_dir": config["preprocessing"]["selection_store_dir"],
            "validation_strategy": strategy,
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
        "variant": {name: i for i, name in enumerate(config["preprocessing"]["input_variants"])},
    }
    if bool(config["multithreshold"].get("enabled", False)):
        codebooks["model"][_MODEL_MULTITHRESHOLD] = max(codebooks["model"].values()) + 1

    candidate_ids: dict[str, int] = {}
    candidate_manifest: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    nested_rows: list[dict[str, Any]] = []
    normalization_cache: dict[str, Normalization] = {}
    final_metadata: dict[str, Any] = {}
    dpi = int(config["reporting"]["dpi"])
    base_seed = int(config["validation"]["seed"])
    ratio_limit = float(config["reporting"]["max_ctr_to_led_ratio"])
    bootstrap_samples = int(config["reporting"]["ctr_uncertainty_bootstrap_samples"])
    work_root = output / ".work"
    checkpoint_root = output / "models"
    signal_plot_root = output / "preprocessing_examples"
    plots_root = output / "plots"

    for root_file in root_files:
        file_id = codebooks["file"][root_file.name]
        logger.info("File %d/%d | %s", file_id + 1, len(root_files), root_file.name)
        dataset = prepare_file_dataset(config, root_file, rebuild=rebuild_preprocessing, logger=logger)
        plot_prepared_signal_examples(dataset, signal_plot_root / f"{root_file.stem}.png", dpi=dpi)
        development, blind = random_dev_blind(
            int(dataset.event_id.size),
            blind_fraction=float(config["validation"]["blind_fraction"]),
            seed=_seed_for(base_seed, file_id, "devblind"),
        )
        voltage = _voltage_from_name(root_file.name, str(config["reporting"]["voltage_pattern"]))
        mt_feature_cache: dict[str, dict[str, Any]] = {}

        for mode in config["channel_modes"]:
            mode_id = codebooks["mode"][mode]
            mt_window = None
            if bool(config["multithreshold"].get("enabled", False)):
                mt_window = next(
                    w for w in config["windows_ns"]
                    if w["id"] == str(config["multithreshold"]["window_id"])
                )

            # -------------------- optional nested pipeline evaluation --------------------
            if strategy == "nested":
                outer = outer_splits(
                    development, config["validation"],
                    seed=_seed_for(base_seed, file_id, mode, "outer"),
                )
                outer_model_metrics: dict[str, list[float]] = {
                    _MODEL_LED: [], _MODEL_CFD: [],
                    **{space["id"]: [] for space in config["_model_spaces"]},
                }
                if mt_window is not None:
                    outer_model_metrics[_MODEL_MULTITHRESHOLD] = []
                inner_validation = nested_inner_validation(config["validation"])
                for outer_index, (outer_train, outer_test) in enumerate(outer):
                    inner_splits = selection_splits(
                        outer_train, inner_validation,
                        seed=_seed_for(base_seed, file_id, mode, "outer", outer_index, "inner"),
                    )
                    led_outer, cfd_outer = _target_deltas(dataset, mode, outer_test)
                    for model_name, residual in (
                        (_MODEL_LED, led_outer - float(dataset.true_tof_ps)),
                        (_MODEL_CFD, cfd_outer - float(dataset.true_tof_ps)),
                    ):
                        metrics = residual_metrics(residual)
                        outer_model_metrics[model_name].append(float(metrics["ctr_ps"]))
                        nested_rows.append({
                            "file": root_file.name, "file_id": file_id, "voltage_V": voltage,
                            "mode": mode, "outer_fold": outer_index, "model": model_name,
                            "candidate_id": -1, "window_id": "", "window_before_ns": "",
                            "window_after_ns": "", "variant": "", "subsampling": "",
                            "hyperparameters_json": "", **metrics,
                        })
                    for space in config["_model_spaces"]:
                        model_id = codebooks["model"][space["id"]]
                        chosen = _select_waveform_space(
                            config, dataset, outer_train, inner_splits,
                            file_id=file_id, mode=mode, mode_id=mode_id, space=space,
                            model_id=model_id, codebooks=codebooks,
                            candidate_ids=candidate_ids, candidate_manifest=candidate_manifest,
                            work_root=work_root / f"nested_o{outer_index}", logger=logger,
                            normalization_cache=normalization_cache, voltage=voltage,
                            result_rows=None,
                        )
                        outer_dir = work_root / f"outer_eval_f{file_id}_m{mode_id}_o{outer_index}_model{model_id}"
                        residual, metrics, _meta, _xai = _waveform_evaluate_selected(
                            config, dataset, outer_train, outer_test,
                            file_id=file_id, mode=mode, window=chosen["window"],
                            variant=chosen["variant"], subsampling=chosen["subsampling"],
                            space=space, overrides=chosen["overrides"],
                            candidate_id=chosen["candidate_id"], work_dir=outer_dir,
                            logger=logger, normalization_cache=normalization_cache,
                        )
                        outer_model_metrics[space["id"]].append(float(metrics["ctr_ps"]))
                        nested_rows.append({
                            "file": root_file.name, "file_id": file_id, "voltage_V": voltage,
                            "mode": mode, "outer_fold": outer_index, "model": space["id"],
                            "candidate_id": chosen["candidate_id"],
                            "window_id": chosen["window"]["id"],
                            "window_before_ns": chosen["window"]["before_ns"],
                            "window_after_ns": chosen["window"]["after_ns"],
                            "variant": chosen["variant"], "subsampling": chosen["subsampling"],
                            "hyperparameters_json": json.dumps(_resolved_hyperparameters(space, chosen["overrides"]), sort_keys=True),
                            **metrics,
                        })
                    if mt_window is not None:
                        selected_mt_outer = _select_multithreshold(
                            config, dataset, outer_train, inner_splits,
                            file_id=file_id, mode=mode, mode_id=mode_id, window=mt_window,
                            model_id=codebooks["model"][_MODEL_MULTITHRESHOLD], codebooks=codebooks,
                            candidate_ids=candidate_ids, candidate_manifest=candidate_manifest,
                            voltage=voltage, logger=logger, result_rows=None,
                            feature_cache=mt_feature_cache,
                        )
                        if selected_mt_outer is not None:
                            residual, metrics = _multithreshold_evaluate(
                                config, dataset, outer_train, outer_test, selected_mt_outer
                            )
                            outer_model_metrics[_MODEL_MULTITHRESHOLD].append(float(metrics["ctr_ps"]))
                            thresholds = selected_mt_outer["features"]
                            del thresholds
                            params = selected_mt_outer["params"]
                            threshold_values = np.asarray(config["multithreshold"]["thresholds_mV"], float)[params["threshold_indices"]].tolist()
                            nested_rows.append({
                                "file": root_file.name, "file_id": file_id, "voltage_V": voltage,
                                "mode": mode, "outer_fold": outer_index, "model": _MODEL_MULTITHRESHOLD,
                                "candidate_id": selected_mt_outer["candidate_id"],
                                "window_id": mt_window["id"], "window_before_ns": mt_window["before_ns"],
                                "window_after_ns": mt_window["after_ns"], "variant": "raw", "subsampling": 1,
                                "hyperparameters_json": json.dumps({
                                    "thresholds_mV": threshold_values,
                                    **{k: v for k, v in params.items() if k != "threshold_indices"},
                                }, sort_keys=True), **metrics,
                            })
                nested_led = float(np.mean(outer_model_metrics[_MODEL_LED]))
                for model_name, ctr_values in outer_model_metrics.items():
                    if not ctr_values:
                        continue
                    ctr = float(np.mean(ctr_values))
                    spread = float(np.std(ctr_values, ddof=1)) if len(ctr_values) > 1 else float("nan")
                    row = {
                        "file": root_file.name, "file_id": file_id, "voltage_V": voltage,
                        "mode": mode, "model": model_name, "stage_name": "nested", "selected": 1,
                        "n": len(ctr_values), "mean_ps": float("nan"),
                        "std_ps": ctr / FWHM_PER_SIGMA, "ctr_ps": ctr,
                        "ctr_fold_std_ps": spread, "ctr_uncertainty_ps": spread,
                        "rmse_ps": float("nan"), "bias_ps": float("nan"),
                        "led_ctr_ps": nested_led,
                    }
                    ratio, included = _plot_inclusion(ctr, nested_led, ratio_limit)
                    row["ctr_over_led"] = ratio; row["plot_included"] = included
                    report_rows.append(row)

            # -------------------- final development selection --------------------
            final_selection_cfg = (
                nested_inner_validation(config["validation"])
                if strategy == "nested" else config["validation"]
            )
            final_splits = selection_splits(
                development, final_selection_cfg,
                seed=_seed_for(base_seed, file_id, mode, "final_selection"),
            )
            validation_score_indices = np.unique(
                np.concatenate([np.asarray(score, dtype=np.int64) for _train, score in final_splits])
            )
            led_val, cfd_val = _target_deltas(dataset, mode, validation_score_indices)
            led_val_res = led_val - float(dataset.true_tof_ps)
            cfd_val_res = cfd_val - float(dataset.true_tof_ps)
            led_val_metrics = residual_metrics(led_val_res)
            cfd_val_metrics = residual_metrics(cfd_val_res)
            logger.info(
                "Validation baseline | mode=%s | n=%d | LED s-CTR %s | CFD s-CTR %s",
                mode,
                int(led_val_metrics["n"]),
                (f"{float(led_val_metrics['ctr_ps']):.1f} ps" if np.isfinite(led_val_metrics["ctr_ps"]) else "nan"),
                (f"{float(cfd_val_metrics['ctr_ps']):.1f} ps" if np.isfinite(cfd_val_metrics["ctr_ps"]) else "nan"),
            )
            selection_stage = "validation"
            for model_name, metrics in ((_MODEL_LED, led_val_metrics), (_MODEL_CFD, cfd_val_metrics)):
                rr = _report_base(
                    root_file=root_file, file_id=file_id, voltage=voltage, mode=mode,
                    model=model_name, stage_name=selection_stage, metrics=metrics,
                )
                rr["validation_strategy"] = (
                    str(config["validation"]["nested"]["inner_strategy"])
                    if strategy == "nested" else strategy
                )
                ratio, included = _plot_inclusion(float(metrics["ctr_ps"]), float(led_val_metrics["ctr_ps"]), ratio_limit)
                rr["led_ctr_ps"] = float(led_val_metrics["ctr_ps"])
                rr["ctr_over_led"] = ratio; rr["plot_included"] = included
                report_rows.append(rr)

            selected_waveforms: list[tuple[dict[str, Any], int, dict[str, Any]]] = []
            validation_corrections: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for space in config["_model_spaces"]:
                model_id = codebooks["model"][space["id"]]
                chosen = _select_waveform_space(
                    config, dataset, development, final_splits,
                    file_id=file_id, mode=mode, mode_id=mode_id, space=space,
                    model_id=model_id, codebooks=codebooks,
                    candidate_ids=candidate_ids, candidate_manifest=candidate_manifest,
                    work_root=work_root, logger=logger,
                    normalization_cache=normalization_cache, voltage=voltage,
                    result_rows=rows,
                )
                selected_waveforms.append((space, model_id, chosen))
                score_led, _ = _target_deltas(dataset, mode, chosen["score_indices"])
                score_led_res = score_led - float(dataset.true_tof_ps)
                validation_corrections[space["id"]] = (
                    chosen["score_indices"], score_led_res - chosen["score_residual"]
                )
                metrics = chosen["metrics"]
                rr = _report_base(
                    root_file=root_file, file_id=file_id, voltage=voltage, mode=mode,
                    model=space["id"], stage_name=selection_stage, metrics=metrics,
                )
                rr = _report_model_details(
                    rr, chosen=chosen, space=space,
                    strategy=(str(config["validation"]["nested"]["inner_strategy"]) if strategy == "nested" else strategy),
                )
                rr["ctr_fold_std_ps"] = float(metrics.get("ctr_err_ps", float("nan")))
                rr["ctr_uncertainty_ps"] = float(metrics.get("ctr_err_ps", float("nan")))
                ratio, included = _plot_inclusion(float(metrics["ctr_ps"]), float(led_val_metrics["ctr_ps"]), ratio_limit)
                rr["led_ctr_ps"] = float(led_val_metrics["ctr_ps"])
                rr["ctr_over_led"] = ratio; rr["plot_included"] = included
                report_rows.append(rr)

            selected_mt = None
            if mt_window is not None:
                selected_mt = _select_multithreshold(
                    config, dataset, development, final_splits,
                    file_id=file_id, mode=mode, mode_id=mode_id, window=mt_window,
                    model_id=codebooks["model"][_MODEL_MULTITHRESHOLD], codebooks=codebooks,
                    candidate_ids=candidate_ids, candidate_manifest=candidate_manifest,
                    voltage=voltage, logger=logger, result_rows=rows,
                    feature_cache=mt_feature_cache,
                )
                if selected_mt is not None:
                    params = selected_mt["params"]
                    selected_mt["threshold_values"] = np.asarray(
                        config["multithreshold"]["thresholds_mV"], float
                    )[params["threshold_indices"]]
                    score_led, _ = _target_deltas(dataset, mode, selected_mt["score_indices"])
                    score_led_res = score_led - float(dataset.true_tof_ps)
                    validation_corrections[_MODEL_MULTITHRESHOLD] = (
                        selected_mt["score_indices"], score_led_res - selected_mt["score_residual"]
                    )
                    metrics = selected_mt["metrics"]
                    rr = _report_base(
                        root_file=root_file, file_id=file_id, voltage=voltage, mode=mode,
                        model=_MODEL_MULTITHRESHOLD, stage_name=selection_stage, metrics=metrics,
                    )
                    rr = _report_model_details(
                        rr, chosen=selected_mt, space=None,
                        strategy=(str(config["validation"]["nested"]["inner_strategy"]) if strategy == "nested" else strategy),
                    )
                    rr["ctr_fold_std_ps"] = float(metrics.get("ctr_err_ps", float("nan")))
                    rr["ctr_uncertainty_ps"] = float(metrics.get("ctr_err_ps", float("nan")))
                    ratio, included = _plot_inclusion(float(metrics["ctr_ps"]), float(led_val_metrics["ctr_ps"]), ratio_limit)
                    rr["led_ctr_ps"] = float(led_val_metrics["ctr_ps"])
                    rr["ctr_over_led"] = ratio; rr["plot_included"] = included
                    report_rows.append(rr)

            plot_correction_matrix(
                plots_root / "correction_correlations" / f"{root_file.stem}__{mode}__validation.png",
                corrections=validation_corrections, dpi=dpi,
                title=f"{root_file.stem} · {mode} · validation corrections",
            )

            # --------------------------- blind once ---------------------------
            led_blind, cfd_blind = _target_deltas(dataset, mode, blind)
            led_residual = led_blind - float(dataset.true_tof_ps)
            cfd_residual = cfd_blind - float(dataset.true_tof_ps)
            blind_methods: dict[str, np.ndarray] = {
                _MODEL_LED: led_residual, _MODEL_CFD: cfd_residual,
            }
            blind_corrections: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            led_blind_metrics = residual_metrics(led_residual)
            blind_uncertainties: dict[str, float] = {
                _MODEL_LED: ctr_bootstrap_uncertainty(
                    led_residual, bootstrap_samples,
                    _seed_for(base_seed, file_id, mode, "led_bootstrap"),
                ),
                _MODEL_CFD: ctr_bootstrap_uncertainty(
                    cfd_residual, bootstrap_samples,
                    _seed_for(base_seed, file_id, mode, "cfd_bootstrap"),
                ),
            }
            for model_name, residual in ((_MODEL_LED, led_residual), (_MODEL_CFD, cfd_residual)):
                _fit, metrics = _fit_row(residual, method=f"Blind {model_name} {mode}", fit_config=config["fit"])
                rows.append({
                    "stage": _STAGE_BLIND, "file_id": file_id, "mode_id": mode_id,
                    "model_id": codebooks["model"][model_name], "candidate_id": -1,
                    "window_id": -1, "variant_id": -1, "subsampling": 1,
                    "selected": 1, "coverage": 1.0, "voltage_V": voltage, **metrics,
                })
                rr = _report_base(
                    root_file=root_file, file_id=file_id, voltage=voltage, mode=mode,
                    model=model_name, stage_name="blind", metrics=metrics,
                )
                rr.update({
                    "candidate_id": -1, "window_id": "", "window_before_ns": "",
                    "window_after_ns": "", "variant": "", "subsampling": "",
                    "hyperparameters_json": "", "validation_strategy": strategy,
                    "validation_ctr_ps": float(led_val_metrics["ctr_ps"] if model_name == _MODEL_LED else cfd_val_metrics["ctr_ps"]),
                    "validation_ctr_uncertainty_ps": float("nan"),
                    "ctr_uncertainty_ps": blind_uncertainties[model_name],
                    "led_ctr_ps": float(led_blind_metrics["ctr_ps"]),
                })
                ratio, included = _plot_inclusion(float(metrics["ctr_ps"]), float(led_blind_metrics["ctr_ps"]), ratio_limit)
                rr["ctr_over_led"] = ratio; rr["plot_included"] = included
                report_rows.append(rr)

            final_candidates: list[tuple[str, dict[str, Any], dict[str, Any] | None, np.ndarray, dict[str, Any]]] = []
            for space, model_id, chosen in selected_waveforms:
                final_dir = work_root / f"final_f{file_id}_m{mode_id}_model{model_id}"
                checkpoint = checkpoint_root / f"f{file_id}_m{mode_id}_model{model_id}.pt"
                residual, metrics, meta, xai_profile = _waveform_evaluate_selected(
                    config, dataset, development, blind,
                    file_id=file_id, mode=mode, window=chosen["window"], variant=chosen["variant"],
                    subsampling=chosen["subsampling"], space=space, overrides=chosen["overrides"],
                    candidate_id=chosen["candidate_id"], work_dir=final_dir, logger=logger,
                    normalization_cache=normalization_cache, checkpoint_path=checkpoint,
                    compute_xai=True,
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
                blind_methods[space["id"]] = residual
                blind_corrections[space["id"]] = (blind, led_residual - residual)
                uncertainty = ctr_bootstrap_uncertainty(
                    residual, bootstrap_samples,
                    _seed_for(base_seed, file_id, mode, space["id"], "bootstrap"),
                )
                blind_uncertainties[space["id"]] = uncertainty
                rr = _report_base(
                    root_file=root_file, file_id=file_id, voltage=voltage, mode=mode,
                    model=space["id"], stage_name="blind", metrics=metrics,
                )
                rr = _report_model_details(rr, chosen=chosen, space=space, strategy=strategy)
                rr.update({
                    "validation_ctr_ps": float(chosen["metrics"]["ctr_ps"]),
                    "validation_ctr_uncertainty_ps": float(chosen["metrics"].get("ctr_err_ps", float("nan"))),
                    "ctr_uncertainty_ps": uncertainty,
                    "led_ctr_ps": float(led_blind_metrics["ctr_ps"]),
                })
                ratio, included = _plot_inclusion(float(metrics["ctr_ps"]), float(led_blind_metrics["ctr_ps"]), ratio_limit)
                rr["ctr_over_led"] = ratio; rr["plot_included"] = included
                report_rows.append(rr)
                final_candidates.append((space["id"], chosen, space, residual, rr))
                if xai_profile is not None:
                    # Preserve the working model XAI data without mixing it into the
                    # clean correlation figure; one compact per-model profile file.
                    time_ps, importance = xai_profile
                    fig, ax = plt.subplots(figsize=(8.0, 3.6))
                    ax.plot(np.asarray(time_ps) / 1000.0, importance)
                    ax.set_xlabel("Relative time [ns]"); ax.set_ylabel("Normalized |attribution|")
                    ax.set_title(f"{space['id']} · {mode}"); ax.grid(alpha=0.2)
                    fig.tight_layout(); xai_path = plots_root / "xai" / f"{root_file.stem}__{mode}__{space['id']}.png"
                    xai_path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(xai_path, dpi=dpi, bbox_inches="tight"); plt.close(fig)

            if selected_mt is not None:
                residual, metrics = _multithreshold_evaluate(config, dataset, development, blind, selected_mt)
                mt_model_id = codebooks["model"][_MODEL_MULTITHRESHOLD]
                rows.append({
                    "stage": _STAGE_BLIND, "file_id": file_id, "mode_id": mode_id,
                    "model_id": mt_model_id, "candidate_id": selected_mt["candidate_id"],
                    "window_id": selected_mt["window_id"], "variant_id": codebooks["variant"].get("raw", 0),
                    "subsampling": 1, "selected": 1, "coverage": 1.0,
                    "voltage_V": voltage, **metrics,
                })
                blind_methods[_MODEL_MULTITHRESHOLD] = residual
                blind_corrections[_MODEL_MULTITHRESHOLD] = (blind, led_residual - residual)
                uncertainty = ctr_bootstrap_uncertainty(
                    residual, bootstrap_samples,
                    _seed_for(base_seed, file_id, mode, _MODEL_MULTITHRESHOLD, "bootstrap"),
                )
                rr = _report_base(
                    root_file=root_file, file_id=file_id, voltage=voltage, mode=mode,
                    model=_MODEL_MULTITHRESHOLD, stage_name="blind", metrics=metrics,
                )
                rr = _report_model_details(rr, chosen=selected_mt, space=None, strategy=strategy)
                rr.update({
                    "validation_ctr_ps": float(selected_mt["metrics"]["ctr_ps"]),
                    "validation_ctr_uncertainty_ps": float(selected_mt["metrics"].get("ctr_err_ps", float("nan"))),
                    "ctr_uncertainty_ps": uncertainty,
                    "led_ctr_ps": float(led_blind_metrics["ctr_ps"]),
                })
                ratio, included = _plot_inclusion(float(metrics["ctr_ps"]), float(led_blind_metrics["ctr_ps"]), ratio_limit)
                rr["ctr_over_led"] = ratio; rr["plot_included"] = included
                report_rows.append(rr)
                final_candidates.append((_MODEL_MULTITHRESHOLD, selected_mt, None, residual, rr))

            plot_blind_distribution(
                plots_root / "blind_distributions" / f"{root_file.stem}__{mode}.png",
                mode=mode, methods=blind_methods, dpi=dpi, ratio_limit=ratio_limit,
                bootstrap_samples=bootstrap_samples,
                seed=_seed_for(base_seed, file_id, mode, "distribution_bootstrap"),
            )
            plot_correction_matrix(
                plots_root / "correction_correlations" / f"{root_file.stem}__{mode}__blind.png",
                corrections=blind_corrections, dpi=dpi,
                title=f"{root_file.stem} · {mode} · blind corrections",
            )

            # Top-k examples from the single best validation-selected ML family.
            eligible_final = [
                item for item in final_candidates if int(item[4].get("plot_included", 1)) == 1
            ]
            if eligible_final and int(config["reporting"].get("top_corrections_k", 3)) > 0:
                best_name, best_chosen, best_space, best_residual, _best_rr = min(
                    eligible_final, key=lambda item: float(item[1]["metrics"]["ctr_ps"])
                )
                variant = "raw" if best_space is None else best_chosen["variant"]
                source = input_variant_dataset_view(dataset, variant)
                input_waveforms, target = CHANNEL_MODES[mode]
                materialized = config["preprocessing"]["materialized_window_ns"]
                full_view = prediction_window_dataset_view(
                    source, input_waveforms=input_waveforms, target=target,
                    before_ns=float(materialized["before"]), after_ns=float(materialized["after"]),
                )
                development_led, _ = _target_deltas(dataset, mode, development)
                calibration_offset = float(np.mean(development_led - float(dataset.true_tof_ps)))
                plot_top_corrections(
                    plots_root / "top_corrections" / f"{root_file.stem}__{mode}.png",
                    time_ps=np.asarray(full_view.relative_time_ps, dtype=np.float64),
                    waveforms=np.asarray(full_view.windows_mV[blind], dtype=np.float32),
                    led_residual=led_residual, corrected_residual=best_residual,
                    calibration_offset_ps=calibration_offset, model=best_name, mode=mode,
                    k=int(config["reporting"]["top_corrections_k"]), dpi=dpi,
                    window_before_ns=float(best_chosen["window"]["before_ns"]),
                    window_after_ns=float(best_chosen["window"]["after_ns"]),
                )

        normalization_cache.clear()
        mt_feature_cache.clear()

    shutil.rmtree(work_root, ignore_errors=True)
    _write_csv(results_path, rows)
    if nested_rows:
        write_report_csv(nested_path, nested_rows)
    write_summary_results(summary_path, report_rows)
    write_report_csv(output / "report_results.csv", report_rows)
    plot_ctr_vs_voltage(
        plots_root / "validation_ctr_vs_voltage.png", rows=report_rows,
        stage="validation", dpi=dpi, ratio_limit=ratio_limit,
        title="Validation CTR vs voltage",
    )
    plot_ctr_vs_voltage(
        plots_root / "blind_ctr_vs_voltage.png", rows=report_rows,
        stage="blind", dpi=dpi, ratio_limit=ratio_limit,
        title="Blind CTR vs voltage",
    )
    plot_final_bars(plots_root / "blind_ctr_bar_by_voltage.png", rows=report_rows, dpi=dpi)
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
        plots_root / "validation_vs_blind_ctr.png", rows=report_rows,
        selection_stage="validation", dpi=dpi,
    )
    if strategy == "nested":
        plot_ctr_vs_voltage(
            plots_root / "nested_ctr_vs_voltage.png", rows=report_rows,
            stage="nested", dpi=dpi, ratio_limit=ratio_limit,
            title="Nested pipeline CTR vs voltage",
        )
        plot_selection_vs_blind(
            plots_root / "nested_vs_blind_ctr.png", rows=report_rows,
            selection_stage="nested", dpi=dpi,
        )

    manifest = {
        "schema_version": 4,
        "experiment": config["experiment"]["name"],
        "config_hash": config["_config_hash"],
        "config_path": config["_config_path"],
        "prepared_dir": config["preprocessing"]["prepared_dir"],
        "selection_store_dir": config["preprocessing"]["selection_store_dir"],
        "materialized_window_ns": config["preprocessing"]["materialized_window_ns"],
        "row_count": len(rows),
        "codebooks": codebooks,
        "candidate_parameters": candidate_manifest,
        "final_models": final_metadata,
        "fit": config["fit"],
        "validation": config["validation"],
        "protocol": {
            "model_pipeline": "restored unchanged from user-provided working repository",
            "target": "direct antisymmetric LED correction target from working torch_data.py",
            "normalization": "working train-derived shared waveform normalization",
            "selection": "holdout/CV/nested wrapper only; no model implementation replacement",
            "nested": "outer K-fold pipeline evaluation; each outer training fold runs configured inner holdout or CV selection",
            "blind": "single untouched blind partition opened only after final development selection",
            "ctr": "2*sqrt(2*ln(2))*sample standard deviation over all evaluation events",
            "evaluation_rejection": "none after permanent prepared population; pathological models are hidden from figures only",
            "photopeak_cache": "physical/photopeak indices persisted independently from ML windows/models/validation",
            "multithreshold": "working raw native-grid relative threshold implementation; candidates cannot drop events",
        },
        "result_columns": {
            "stage": {"0": "development selection candidate", "1": "blind"},
            "candidate_id": "parameters stored once in candidate_parameters; -1 for fixed LED/CFD",
        },
    }
    atomic_json(manifest_path, manifest)
    logger.info("Study complete | rows=%d | %s", len(rows), output)
    return {
        "output_dir": str(output), "row_count": len(rows),
        "results": str(results_path), "summary_results": str(summary_path),
        "nested_results": str(nested_path) if nested_rows else None,
    }
