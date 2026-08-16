from __future__ import annotations
import argparse
import csv
import json
import math
import sys
import textwrap
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from ml_pipeline.dataset import PreparedDataset, load_prepared_dataset
from ml_pipeline.models import build_model
from ml_pipeline.prepared_data import input_channel_variant_dataset_view, raw_dataset_view
from ml_pipeline.prediction import prediction_window_dataset_view
from ml_pipeline.study import (
    _fit_early_split,
    _random_dev_blind,
    _read_results_csv,
    _seed_for,
    _target_deltas,
    _threshold_crossing_matrix,
)
from ml_pipeline.study_config import CHANNEL_MODES, load_study_config
from ml_pipeline.torch_data import Normalization
from ml_pipeline.training_utils import make_split_loader, predict_loader, resolve_device
from utils.fit import FitResult, fit_delta_times_integer_fs
MODEL_LED = "led"
MODEL_CFD = "cfd"
MODEL_MULTITHRESHOLD = "multithreshold_svr"
LEARNED_MODEL_ORDER = ["linear_svr", "constructive_mlp", "cnn", MODEL_MULTITHRESHOLD]
DISPLAY_MODEL_ORDER = [MODEL_LED, MODEL_CFD, *LEARNED_MODEL_ORDER]
DISPLAY_NAME = {
    MODEL_LED: "LED",
    MODEL_CFD: "CFD",
    "linear_svr": "Linear SVR",
    "constructive_mlp": "Constructive MLP",
    "cnn": "CNN",
    MODEL_MULTITHRESHOLD: "Multithreshold SVR",
}
FWHM_FACTOR = 2.0 * math.sqrt(2.0 * math.log(2.0))
@dataclass
class DistributionFit:
    residual_ps: np.ndarray
    fit: FitResult
    pearson_chi2: float
    pearson_ndof: int
    bootstrap_mean_error_ps: float
    bootstrap_sigma_error_ps: float
    bootstrap_ctr_error_ps: float
    bootstrap_successes: int
    outside_display: int = 0
    @property
    def pearson_chi2_ndof(self) -> float:
        if self.pearson_ndof <= 0:
            return float("nan")
        return float(self.pearson_chi2 / self.pearson_ndof)
def _invert_codebook(codebook: dict[str, Any]) -> dict[int, str]:
    return {int(value): str(key) for key, value in codebook.items()}
def _selected_cv_row(
    rows: list[dict[str, Any]], *, file_id: int, mode_id: int, model_id: int
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if int(row.get("stage", -1)) == 0
        and int(row.get("file_id", -1)) == file_id
        and int(row.get("mode_id", -1)) == mode_id
        and int(row.get("model_id", -1)) == model_id
        and int(row.get("selected", 0)) == 1
    ]
    if len(selected) != 1:
        raise RuntimeError(
            f"Expected exactly one selected CV row for file={file_id}, mode={mode_id}, "
            f"model={model_id}; found {len(selected)}"
        )
    return selected[0]
def _window_map(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(window["id"]): window for window in config["windows_ns"]}
def _format_number(value: Any) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        if abs(value) >= 1000 or (0 < abs(value) < 1e-3):
            return f"{value:.3g}"
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_format_number(v) for v in value) + "]"
    return str(value)
def _candidate_description(
    family: str,
    descriptor: dict[str, Any],
    selected_row: dict[str, Any],
    windows: dict[str, dict[str, Any]],
    *,
    checkpoint_payload: dict[str, Any] | None = None,
) -> str:
    window_id = str(descriptor.get("window", ""))
    window = windows.get(window_id)
    if window is not None:
        window_text = f"window=[-{float(window['before_ns']):g}, {float(window['after_ns']):g}] ns"
    else:
        window_text = f"window={window_id}" if window_id else ""
    cv_text = (
        f"CV CTR={float(selected_row['ctr_ps']):.2f}±"
        f"{float(selected_row.get('ctr_fold_std_ps', float('nan'))):.2f} ps"
    )
    if family == MODEL_MULTITHRESHOLD:
        pieces = [
            "raw multithreshold SVR",
            window_text,
            f"thresholds={_format_number(descriptor.get('thresholds_mV', []))} mV",
            f"kernel={descriptor.get('kernel')}",
            f"C={_format_number(descriptor.get('C'))}",
            f"epsilon={_format_number(descriptor.get('epsilon_ps'))} ps",
        ]
        if str(descriptor.get("kernel")) == "rbf":
            pieces.append(f"gamma={_format_number(descriptor.get('gamma'))}")
        pieces.append(cv_text)
        return " | ".join(piece for piece in pieces if piece)
    overrides = descriptor.get("overrides", {}) or {}
    variant = str(descriptor.get("variant", "raw"))
    subsampling = int(descriptor.get("subsampling", 1))
    pieces = [window_text, f"input={variant}", f"subsampling={subsampling}"]
    if family == "linear_svr":
        c_value = overrides.get("model.C")
        epsilon = overrides.get("model.epsilon_values")
        pieces.insert(0, "Linear SVR")
        if c_value is not None:
            pieces.append(f"C={_format_number(c_value)}")
        if epsilon is not None:
            eps_value = epsilon[0] if isinstance(epsilon, list) and len(epsilon) == 1 else epsilon
            pieces.append(f"epsilon={_format_number(eps_value)} ps")
    elif family == "cnn":
        pieces.insert(0, "CNN")
        context = (checkpoint_payload or {}).get("context", {})
        model_cfg = context.get("model_config", {}) or {}
        for key, label in (
            ("channels", "ch"),
            ("kernel_sizes", "k"),
            ("strides", "stride"),
            ("dilations", "dil"),
            ("adaptive_pool_length", "pool"),
            ("dense_units", "dense"),
        ):
            if key in model_cfg:
                pieces.append(f"{label}={_format_number(model_cfg[key])}")
        if checkpoint_payload is not None and "epoch" in checkpoint_payload:
            pieces.append(f"best_epoch={int(checkpoint_payload['epoch'])}")
        for key, label in (
            ("model.activation", "act"),
            ("model.normalization", "norm"),
            ("optimizer.learning_rate", "lr"),
            ("optimizer.weight_decay", "wd"),
            ("training.early_stop_fraction", "early"),
        ):
            if key in overrides:
                suffix = "%" if key.endswith("early_stop_fraction") else ""
                value = 100.0 * float(overrides[key]) if suffix else overrides[key]
                pieces.append(f"{label}={_format_number(value)}{suffix}")
    elif family == "constructive_mlp":
        pieces.insert(0, "Constructive MLP")
        payload = checkpoint_payload or {}
        context = payload.get("context", {}) or {}
        model_cfg = context.get("model_config", {}) or {}
        units = payload.get("unit_count", model_cfg.get("_trained_units"))
        if units is not None:
            pieces.append(f"units={int(units)}")
        if "epoch" in payload:
            pieces.append(f"best_epoch={int(payload['epoch'])}")
        for key, label in (
            ("activation", "act"),
            ("max_units", "max_units"),
        ):
            if key in model_cfg:
                pieces.append(f"{label}={_format_number(model_cfg[key])}")
        for key, label in (
            ("optimizer.learning_rate", "lr"),
            ("optimizer.weight_decay", "wd"),
            ("training.early_stop_fraction", "early"),
        ):
            if key in overrides:
                suffix = "%" if key.endswith("early_stop_fraction") else ""
                value = 100.0 * float(overrides[key]) if suffix else overrides[key]
                pieces.append(f"{label}={_format_number(value)}{suffix}")
    else:
        pieces.insert(0, family)
    pieces.append(cv_text)
    return " | ".join(piece for piece in pieces if piece)
def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")
def _waveform_residuals(
    config: dict[str, Any],
    dataset: PreparedDataset,
    *,
    mode: str,
    descriptor: dict[str, Any],
    checkpoint_payload: dict[str, Any],
    train_indices: np.ndarray,
    blind_indices: np.ndarray,
    device_name: str,
    batch_size: int,
) -> dict[str, np.ndarray]:
    input_waveforms, target = CHANNEL_MODES[mode]
    variant = str(descriptor["variant"])
    source = input_channel_variant_dataset_view(dataset, input_waveforms, variant)
    windows = _window_map(config)
    window = windows[str(descriptor["window"])]
    view = prediction_window_dataset_view(
        source,
        input_waveforms=input_waveforms,
        target=target,
        before_ns=float(window["before_ns"]),
        after_ns=float(window["after_ns"]),
    )
    context = checkpoint_payload.get("context", {})
    model_type = str(context["model_type"])
    model_cfg = dict(context["model_config"])
    input_length = int(context["input_length"])
    model = build_model(model_type, model_cfg, input_length)
    model.load_state_dict(checkpoint_payload["model_state"])
    normalization = Normalization.from_dict(context["normalization"])
    device = resolve_device(device_name)
    model = model.to(device)
    eval_cfg = {
        "training": {
            "batch_size": int(batch_size),
            "num_workers": 0,
            "pin_memory": bool(device.type == "cuda"),
            "device": device_name,
        },
        "preprocessing": {
            "subsampling_factor": int(descriptor.get("subsampling", context.get("subsampling_factor", 1)))
        },
    }
    def predict(indices: np.ndarray) -> dict[str, np.ndarray]:
        split_view = replace(view, evaluation=np.asarray(indices, dtype=np.int64))
        loader = make_split_loader(
            [split_view],
            "evaluation",
            normalization,
            eval_cfg,
            device,
            shuffle=False,
            subsampling_factor=int(eval_cfg["preprocessing"]["subsampling_factor"]),
        )
        output = predict_loader(model, loader, device)
        return {
            "residual_ps": np.asarray(output["residual_ps"], dtype=np.float64),
            "prediction_ps": np.asarray(output["prediction_ps"], dtype=np.float64),
            "target_ps": np.asarray(output["target_ps"], dtype=np.float64),
        }
    train_output = predict(train_indices)
    blind_output = predict(blind_indices)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "train_residual_ps": train_output["residual_ps"],
        "blind_residual_ps": blind_output["residual_ps"],
        "train_prediction_ps": train_output["prediction_ps"],
        "blind_prediction_ps": blind_output["prediction_ps"],
        "train_target_ps": train_output["target_ps"],
        "blind_target_ps": blind_output["target_ps"],
    }
def _multithreshold_residuals(
    config: dict[str, Any],
    dataset: PreparedDataset,
    *,
    mode: str,
    descriptor: dict[str, Any],
    development: np.ndarray,
    blind: np.ndarray,
) -> dict[str, np.ndarray]:
    raw = raw_dataset_view(dataset)
    input_waveforms, target = CHANNEL_MODES[mode]
    windows = _window_map(config)
    window = windows[str(descriptor["window"])]
    view = prediction_window_dataset_view(
        raw,
        input_waveforms=input_waveforms,
        target=target,
        before_ns=float(window["before_ns"]),
        after_ns=float(window["after_ns"]),
    )
    selected_thresholds = np.asarray(descriptor["thresholds_mV"], dtype=np.float64)
    features = _threshold_crossing_matrix(
        view,
        selected_thresholds,
        chunk_size=int(config["multithreshold"].get("chunk_size", 2048)),
    )
    if np.any(~np.isfinite(features[development])) or np.any(~np.isfinite(features[blind])):
        raise RuntimeError(
            f"Selected multithreshold model for {mode} has missing threshold crossings in train/blind"
        )
    led_ps, _ = _target_deltas(raw, mode, np.arange(dataset.event_id.size, dtype=np.int64))
    target_correction = led_ps - float(dataset.true_tof_ps)
    estimator = make_pipeline(
        StandardScaler(),
        SVR(
            kernel=str(descriptor["kernel"]),
            C=float(descriptor["C"]),
            epsilon=float(descriptor["epsilon_ps"]),
            gamma=descriptor.get("gamma", "scale"),
        ),
    )
    estimator.fit(features[development], target_correction[development])
    def evaluate(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        correction = np.asarray(estimator.predict(features[indices]), dtype=np.float64)
        target = np.asarray(target_correction[indices], dtype=np.float64)
        residual = led_ps[indices] - correction - float(dataset.true_tof_ps)
        return np.asarray(residual, dtype=np.float64), correction, target

    train_residual, train_prediction, train_target = evaluate(development)
    blind_residual, blind_prediction, blind_target = evaluate(blind)
    return {
        "train_residual_ps": train_residual,
        "blind_residual_ps": blind_residual,
        "train_prediction_ps": train_prediction,
        "blind_prediction_ps": blind_prediction,
        "train_target_ps": train_target,
        "blind_target_ps": blind_target,
    }
def _bootstrap_fit_uncertainty(
    values: np.ndarray,
    *,
    label: str,
    fit_config: dict[str, Any],
    replicas: int,
    seed: int,
) -> tuple[float, float, float, int]:
    """Nonparametric event bootstrap of the fitted Gaussian parameters.
    Each replica resamples the *same residual distribution* with replacement and
    repeats the complete Gaussian fit, including the bin-phase scan. This makes
    the quoted uncertainty reflect finite-event fluctuations and fit/binning
    sensitivity rather than the numerical L-BFGS inverse Hessian.
    """
    if replicas <= 1:
        return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.default_rng(int(seed))
    n = int(values.size)
    means: list[float] = []
    sigmas: list[float] = []
    ctrs: list[float] = []
    for replica in range(int(replicas)):
        sample = values[rng.integers(0, n, size=n)]
        sample_fs = np.rint(sample * 1000.0).astype(np.int64)
        boot = fit_delta_times_integer_fs(
            sample_fs,
            method=f"{label} bootstrap",
            parameter=float(replica),
            n_total=n,
            n_selected=n,
            config=fit_config,
        )
        if not boot.success:
            continue
        if not (np.isfinite(boot.mean_ps) and np.isfinite(boot.sigma_ps) and np.isfinite(boot.ctr_ps)):
            continue
        means.append(float(boot.mean_ps))
        sigmas.append(float(boot.sigma_ps))
        ctrs.append(float(boot.ctr_ps))
    successes = len(ctrs)
    minimum_successes = max(20, int(math.ceil(0.5 * replicas)))
    if successes < minimum_successes:
        raise RuntimeError(
            f"{label}: bootstrap Gaussian fit succeeded for only "
            f"{successes}/{replicas} replicas"
        )
    return (
        float(np.std(means, ddof=1)),
        float(np.std(sigmas, ddof=1)),
        float(np.std(ctrs, ddof=1)),
        successes,
    )
def _fit_distribution(
    residual_ps: np.ndarray,
    *,
    label: str,
    fit_config: dict[str, Any],
    bootstrap_replicas: int,
    bootstrap_seed: int,
) -> DistributionFit:
    values = np.asarray(residual_ps, dtype=np.float64).reshape(-1)
    if values.size < 2 or np.any(~np.isfinite(values)):
        raise RuntimeError(f"{label}: residual distribution contains invalid values")
    values_fs = np.rint(values * 1000.0).astype(np.int64)
    fit = fit_delta_times_integer_fs(
        values_fs,
        method=label,
        parameter=0.0,
        n_total=int(values.size),
        n_selected=int(values.size),
        config=fit_config,
    )
    if not fit.success:
        raise RuntimeError(f"{label}: Gaussian fit failed: {fit.message}")
    observed = np.asarray(fit.counts, dtype=np.float64)
    expected = np.asarray(fit.expected, dtype=np.float64)
    mask = np.isfinite(expected) & (expected >= 5.0)
    used_bins = int(np.count_nonzero(mask))
    ndof = used_bins - 3
    if ndof > 0:
        chi2 = float(np.sum((observed[mask] - expected[mask]) ** 2 / expected[mask]))
    else:
        chi2 = float("nan")
        ndof = 0
    mean_err, sigma_err, ctr_err, bootstrap_successes = _bootstrap_fit_uncertainty(
        values,
        label=label,
        fit_config=fit_config,
        replicas=int(bootstrap_replicas),
        seed=int(bootstrap_seed),
    )
    return DistributionFit(
        values,
        fit,
        chi2,
        ndof,
        mean_err,
        sigma_err,
        ctr_err,
        bootstrap_successes,
    )
def _robust_display_range(distributions: dict[str, DistributionFit]) -> tuple[float, float]:
    intervals: list[tuple[float, float]] = []
    all_values: list[np.ndarray] = []
    for result in distributions.values():
        values = result.residual_ps
        all_values.append(values)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        sigma = 1.4826022185 * mad
        if not np.isfinite(sigma) or sigma <= 0.0:
            q16, q84 = np.quantile(values, [0.1586552539, 0.8413447461])
            sigma = 0.5 * float(q84 - q16)
        if np.isfinite(sigma) and sigma > 0.0:
            intervals.append((median - 8.0 * sigma, median + 8.0 * sigma))
    if intervals:
        low = min(value[0] for value in intervals)
        high = max(value[1] for value in intervals)
    else:
        merged = np.concatenate(all_values)
        low, high = np.quantile(merged, [0.005, 0.995])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = -100.0, 100.0
    padding = 0.06 * (high - low)
    return float(low - padding), float(high + padding)
def _rounded_ps_text(value: float, error: float | None = None) -> str:
    value_text = f"{int(round(float(value)))}"
    if error is None or not math.isfinite(float(error)):
        return value_text
    error_value = max(1, int(round(float(error))))
    return f"{value_text} ± {error_value}"
def _summary_text_block(name: str, result: DistributionFit) -> str:
    fit = result.fit
    chi = result.pearson_chi2_ndof
    lines = [
        f"{name}",
        f"CTR = {_rounded_ps_text(fit.ctr_ps, result.bootstrap_ctr_error_ps)} ps",
        f"μ = {_rounded_ps_text(fit.mean_ps, result.bootstrap_mean_error_ps)} ps",
        f"σ = {_rounded_ps_text(fit.sigma_ps, result.bootstrap_sigma_error_ps)} ps",
        f"χ²/ndof = {chi:.2f}" if math.isfinite(chi) else "χ²/ndof = n/a",
    ]
    if result.outside_display:
        lines.append(f"outside = {int(result.outside_display)}")
    return "\n".join(lines)
def _plot_file_mode_model_distribution(
    destination: Path,
    *,
    file_name: str,
    mode: str,
    method: str,
    model_result: DistributionFit,
    led_result: DistributionFit,
    description: str,
    dpi: int,
) -> None:
    """One figure per file/mode/model, comparing the model against the LED baseline."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    display = {MODEL_LED: led_result, method: model_result}
    low, high = _robust_display_range(display)
    for result in display.values():
        result.outside_display = int(
            np.count_nonzero((result.residual_ps < low) | (result.residual_ps > high))
        )
    fig, ax = plt.subplots(figsize=(10.8, 6.4))
    fig.suptitle(
        f"{file_name} | {mode} | {DISPLAY_NAME.get(method, method)}",
        fontsize=14,
        y=0.98,
    )
    wrapped = textwrap.fill(description, width=110)
    fig.text(0.08, 0.93, wrapped, ha='left', va='top', fontsize=8.5)
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.12, top=0.80)
    plot_order = [
        (MODEL_LED, led_result, '#1f77b4'),
        (method, model_result, '#d62728'),
    ]
    for label, result, color in plot_order:
        fit = result.fit
        counts = np.asarray(fit.counts, dtype=np.float64)
        edges = np.asarray(fit.edges_ps, dtype=np.float64)
        width = float(fit.bin_width_ps)
        ax.stairs(
            counts,
            edges,
            linewidth=1.45,
            color=color,
            label=DISPLAY_NAME.get(label, label),
        )
        centers = 0.5 * (edges[:-1] + edges[1:])
        expected_counts = np.asarray(fit.expected, dtype=np.float64)
        visible = (centers >= low) & (centers <= high)
        ax.plot(
            centers[visible],
            expected_counts[visible],
            linestyle='--',
            linewidth=1.4,
            color=color,
        )
    ax.set_xlim(low, high)
    ax.set_xlabel('Residual timing error [ps]')
    ax.set_ylabel('Count')
    ax.grid(True, which='both', alpha=0.25)
    led_box = _summary_text_block(DISPLAY_NAME[MODEL_LED], led_result)
    model_box = _summary_text_block(DISPLAY_NAME.get(method, method), model_result)
    ax.text(
        0.02,
        0.98,
        led_box,
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=9.0,
        linespacing=1.25,
        bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='#1f77b4', alpha=0.95),
        zorder=10,
    )
    ax.text(
        0.98,
        0.98,
        model_box,
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=9.0,
        linespacing=1.25,
        bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='#d62728', alpha=0.95),
        zorder=10,
    )
    fig.savefig(destination, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
def _plot_prediction_vs_target(
    destination: Path,
    *,
    file_name: str,
    mode: str,
    method: str,
    prediction_ps: np.ndarray,
    target_ps: np.ndarray,
    description: str,
    bin_width_ps: float,
    dpi: int,
) -> None:
    """Blind-set predicted correction distribution versus the model's supervision target."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    prediction = np.asarray(prediction_ps, dtype=np.float64).reshape(-1)
    target = np.asarray(target_ps, dtype=np.float64).reshape(-1)
    if prediction.size != target.size or prediction.size == 0:
        raise RuntimeError(
            f"{file_name} | {mode} | {method}: invalid prediction/target sizes "
            f"({prediction.size} vs {target.size})"
        )
    if np.any(~np.isfinite(prediction)) or np.any(~np.isfinite(target)):
        raise RuntimeError(f"{file_name} | {mode} | {method}: non-finite prediction/target")

    pseudo = {
        "prediction": type("_D", (), {"residual_ps": prediction})(),
        "target": type("_D", (), {"residual_ps": target})(),
    }
    low, high = _robust_display_range(pseudo)
    width = float(bin_width_ps)
    start = math.floor(low / width) * width
    stop = math.ceil(high / width) * width
    edges = np.arange(start, stop + 0.5 * width, width, dtype=np.float64)
    if edges.size < 2:
        edges = np.asarray([start, start + width], dtype=np.float64)

    target_counts, _ = np.histogram(target, bins=edges)
    prediction_counts, _ = np.histogram(prediction, bins=edges)
    outside_target = int(np.count_nonzero((target < edges[0]) | (target > edges[-1])))
    outside_prediction = int(
        np.count_nonzero((prediction < edges[0]) | (prediction > edges[-1]))
    )

    fig, ax = plt.subplots(figsize=(10.8, 6.4))
    fig.suptitle(
        f"{file_name} | {mode} | {DISPLAY_NAME.get(method, method)} | blind prediction vs target",
        fontsize=14,
        y=0.98,
    )
    wrapped = textwrap.fill(description, width=110)
    fig.text(0.08, 0.93, wrapped, ha="left", va="top", fontsize=8.5)
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.12, top=0.80)

    ax.stairs(target_counts, edges, linewidth=1.5, color="#1f77b4")
    ax.stairs(prediction_counts, edges, linewidth=1.5, color="#d62728")
    ax.set_xlim(edges[0], edges[-1])
    ax.set_xlabel("Correction [ps]")
    ax.set_ylabel("Count")
    ax.grid(True, which="both", alpha=0.25)

    def stats_box(name: str, values: np.ndarray, outside: int) -> str:
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if values.size > 1 else float("nan")
        lines = [
            name,
            f"N = {values.size}",
            f"mean = {int(round(mean))} ps",
            f"std = {int(round(std))} ps" if math.isfinite(std) else "std = n/a",
        ]
        if outside:
            lines.append(f"outside = {outside}")
        return "\n".join(lines)

    ax.text(
        0.02,
        0.98,
        stats_box("Target", target, outside_target),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.0,
        linespacing=1.25,
        bbox=dict(
            boxstyle="round,pad=0.35", facecolor="white", edgecolor="#1f77b4", alpha=0.95
        ),
        zorder=10,
    )
    ax.text(
        0.98,
        0.98,
        stats_box("Prediction", prediction, outside_prediction),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.0,
        linespacing=1.25,
        bbox=dict(
            boxstyle="round,pad=0.35", facecolor="white", edgecolor="#d62728", alpha=0.95
        ),
        zorder=10,
    )
    fig.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _write_fit_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
def _read_fit_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))
def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected true/false")
def _cached_fit_key(
    *, file_name: str, mode: str, split: str, method: str
) -> tuple[str, str, str, str]:
    return (str(file_name), str(mode), str(split), str(method))
def _cached_fit_map(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    result: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = _cached_fit_key(
            file_name=str(row.get("file", "")),
            mode=str(row.get("mode", "")),
            split=str(row.get("split", "")),
            method=str(row.get("method", "")),
        )
        result[key] = row
    return result
def _reuse_bootstrap_uncertainty(
    result: DistributionFit,
    cached: dict[str, Any],
    *,
    label: str,
) -> None:
    required = ["mean_error_ps", "sigma_error_ps", "ctr_error_ps", "bootstrap_successes"]
    missing = [key for key in required if key not in cached or cached[key] in {None, ""}]
    if missing:
        raise RuntimeError(
            f"{label}: cached fit_results.csv has no bootstrap fields {missing}. "
            "Run once with --rebuild-summary true."
        )
    result.bootstrap_mean_error_ps = float(cached["mean_error_ps"])
    result.bootstrap_sigma_error_ps = float(cached["sigma_error_ps"])
    result.bootstrap_ctr_error_ps = float(cached["ctr_error_ps"])
    result.bootstrap_successes = int(float(cached["bootstrap_successes"]))
def _plot_ctr_vs_voltage(
    destination: Path,
    *,
    mode: str,
    fit_rows: list[dict[str, Any]],
    methods: list[str],
    dpi: int,
) -> None:
    subset = [row for row in fit_rows if row["mode"] == mode and row["split"] == "blind"]
    voltages = sorted({float(row["voltage_V"]) for row in subset})
    if not voltages:
        return
    available = [method for method in methods if any(row["method"] == method for row in subset)]
    x = np.arange(len(voltages), dtype=np.float64)
    width = 0.82 / max(len(available), 1)
    fig, ax = plt.subplots(figsize=(max(9.0, 1.45 * len(voltages) + 5.0), 6.4))
    for index, method in enumerate(available):
        values = []
        errors = []
        for voltage in voltages:
            match = [
                row for row in subset
                if row["method"] == method and float(row["voltage_V"]) == voltage
            ]
            values.append(float(match[0]["ctr_ps"]) if match else np.nan)
            errors.append(float(match[0]["ctr_error_ps"]) if match else np.nan)
        offset = (index - 0.5 * (len(available) - 1)) * width
        ax.bar(x + offset, values, width=width, yerr=errors, capsize=2.5, label=DISPLAY_NAME.get(method, method))
    ax.set_xticks(x, [f"{voltage:g}" for voltage in voltages])
    ax.set_xlabel("Bias voltage [V]")
    ax.set_ylabel("Gaussian-fit CTR [ps]")
    ax.set_title(f"{mode} | blind Gaussian-fit CTR vs voltage")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
def _plot_train_vs_blind(
    destination: Path,
    *,
    mode: str,
    fit_rows: list[dict[str, Any]],
    methods: list[str],
    dpi: int,
) -> None:
    learned = [method for method in methods if method in LEARNED_MODEL_ORDER]
    voltages = sorted({
        float(row["voltage_V"])
        for row in fit_rows
        if row["mode"] == mode and row["method"] in learned
    })
    if not voltages:
        return
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    for method in learned:
        train_values: list[float] = []
        blind_values: list[float] = []
        train_errors: list[float] = []
        blind_errors: list[float] = []
        method_voltages: list[float] = []
        for voltage in voltages:
            train = [
                row for row in fit_rows
                if row["mode"] == mode and row["method"] == method
                and row["split"] == "train" and float(row["voltage_V"]) == voltage
            ]
            blind = [
                row for row in fit_rows
                if row["mode"] == mode and row["method"] == method
                and row["split"] == "blind" and float(row["voltage_V"]) == voltage
            ]
            if not train or not blind:
                continue
            method_voltages.append(voltage)
            train_values.append(float(train[0]["ctr_ps"]))
            blind_values.append(float(blind[0]["ctr_ps"]))
            train_errors.append(float(train[0]["ctr_error_ps"]))
            blind_errors.append(float(blind[0]["ctr_error_ps"]))
        if not method_voltages:
            continue
        blind_line = ax.errorbar(
            method_voltages,
            blind_values,
            yerr=blind_errors,
            marker="o",
            linewidth=1.5,
            capsize=2.5,
            label=f"{DISPLAY_NAME.get(method, method)} blind",
        )
        color = blind_line.lines[0].get_color()
        ax.errorbar(
            method_voltages,
            train_values,
            yerr=train_errors,
            marker="s",
            linestyle="--",
            linewidth=1.25,
            capsize=2.5,
            color=color,
            label=f"{DISPLAY_NAME.get(method, method)} train",
        )
    ax.set_xlabel("Bias voltage [V]")
    ax.set_ylabel("Gaussian-fit CTR [ps]")
    ax.set_title(f"{mode} | final-model train vs blind CTR")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
def _fit_row(
    *,
    file_name: str,
    voltage: float,
    mode: str,
    split: str,
    method: str,
    result: DistributionFit,
    candidate_id: int,
    window_id: str,
    description: str,
    cv_ctr_ps: float = float("nan"),
    cv_ctr_fold_std_ps: float = float("nan"),
) -> dict[str, Any]:
    fit = result.fit
    return {
        "file": file_name,
        "voltage_V": voltage,
        "mode": mode,
        "split": split,
        "method": method,
        "candidate_id": candidate_id,
        "window": window_id,
        "n": int(fit.n_fit),
        "mean_ps": float(fit.mean_ps),
        "mean_error_ps": float(result.bootstrap_mean_error_ps),
        "sigma_ps": float(fit.sigma_ps),
        "sigma_error_ps": float(result.bootstrap_sigma_error_ps),
        "ctr_ps": float(fit.ctr_ps),
        "ctr_error_ps": float(result.bootstrap_ctr_error_ps),
        "bootstrap_successes": int(result.bootstrap_successes),
        "pearson_chi2": float(result.pearson_chi2),
        "pearson_ndof": int(result.pearson_ndof),
        "pearson_chi2_ndof": float(result.pearson_chi2_ndof),
        "poisson_deviance": float(fit.chi2),
        "poisson_ndof": int(fit.ndof),
        "bin_width_ps": float(fit.bin_width_ps),
        "bin_phase_ps": float(fit.bin_phase_ps),
        "cv_ctr_ps": cv_ctr_ps,
        "cv_ctr_fold_std_ps": cv_ctr_fold_std_ps,
        "description": description,
    }
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replay final CV-selected models, Gaussian-fit train/blind residuals, and "
            "produce one readable distribution figure per file/mode/model (with LED baseline comparison) plus voltage summaries."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <experiment output>/gaussian_postfit",
    )
    parser.add_argument("--bin-width-ps", type=float, default=10.0)
    parser.add_argument("--bin-phase-count", type=int, default=10)
    parser.add_argument("--min-events", type=int, default=100)
    parser.add_argument(
        "--bootstrap-replicas",
        type=int,
        default=300,
        help="Bootstrap replicas used only when --rebuild-summary true (default: 300).",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=20260815,
        help="Base seed for deterministic bootstrap uncertainty.",
    )
    parser.add_argument(
        "--rebuild-summary",
        type=_parse_bool,
        default=False,
        metavar="{true,false}",
        help=(
            "false (default): reuse bootstrap uncertainties already stored in "
            "gaussian_postfit/fit_results.csv and regenerate plots without bootstrap; "
            "true: recompute all Gaussian-fit bootstrap uncertainties and overwrite the cache."
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    config = load_study_config(args.config, PROJECT)
    experiment_dir = Path(config["experiment"]["output_dir"])
    manifest_path = experiment_dir / "manifest.json"
    results_path = experiment_dir / "results.csv"
    if not manifest_path.is_file() or not results_path.is_file():
        raise FileNotFoundError(
            f"Completed experiment metadata not found in {experiment_dir}. "
            "Run ml_experiment.py successfully first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("status")) != "complete":
        raise RuntimeError(
            f"Experiment {experiment_dir} is not complete (status={manifest.get('status')!r}). "
            "This post-fit script intentionally runs only after successful completion."
        )
    if manifest.get("config_hash") != config.get("_config_hash"):
        raise RuntimeError(
            "The supplied experiment config does not match the completed run manifest."
        )
    output = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else experiment_dir / "gaussian_postfit"
    )
    output.mkdir(parents=True, exist_ok=True)
    distribution_root = output / "distributions"
    prediction_target_root = output / "prediction_vs_target"
    summary_root = output / "summary"
    fit_results_path = output / "fit_results.csv"
    summary_root.mkdir(parents=True, exist_ok=True)
    # The distribution directory is generated output owned by this script. Clear the old
    # per-file/multi-model layout so the new mode/model layout is not mixed with stale PNGs.
    if distribution_root.exists():
        shutil.rmtree(distribution_root)
    distribution_root.mkdir(parents=True, exist_ok=True)
    if prediction_target_root.exists():
        shutil.rmtree(prediction_target_root)
    prediction_target_root.mkdir(parents=True, exist_ok=True)
    fit_config = {
        "min_events": int(args.min_events),
        "histogram_bin_ps": float(args.bin_width_ps),
        "bin_phase_count": int(args.bin_phase_count),
    }
    if fit_config["histogram_bin_ps"] <= 0.0 or fit_config["bin_phase_count"] <= 0:
        raise ValueError("Gaussian fit bin width and phase count must be positive")
    if bool(args.rebuild_summary) and int(args.bootstrap_replicas) < 20:
        raise ValueError("--bootstrap-replicas must be at least 20 when rebuilding")
    cached_fit_rows: list[dict[str, Any]] = []
    cached_by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    if not bool(args.rebuild_summary):
        if not fit_results_path.is_file():
            raise FileNotFoundError(
                f"Bootstrap cache not found: {fit_results_path}. "
                "Run once with --rebuild-summary true."
            )
        cached_fit_rows = _read_fit_csv(fit_results_path)
        cached_by_key = _cached_fit_map(cached_fit_rows)
        print(f"Reusing cached bootstrap uncertainties from {fit_results_path}")
    rows = _read_results_csv(results_path)
    codebooks = manifest["codebooks"]
    files = _invert_codebook(codebooks["file"])
    modes = _invert_codebook(codebooks["mode"])
    models = _invert_codebook(codebooks["model"])
    model_to_id = {name: model_id for model_id, name in models.items()}
    candidate_parameters = manifest.get("candidate_parameters", {})
    final_models = manifest.get("final_models", {})
    windows = _window_map(config)
    dpi = int(config.get("reporting", {}).get("dpi", 180))
    base_seed = int(config["cross_validation"]["seed"])
    all_fit_rows: list[dict[str, Any]] = []
    def fit_distribution_cached(
        residual_ps: np.ndarray,
        *,
        file_name: str,
        mode: str,
        split: str,
        method: str,
        label: str,
        seed: int,
    ) -> DistributionFit:
        replicas = int(args.bootstrap_replicas) if bool(args.rebuild_summary) else 0
        result = _fit_distribution(
            residual_ps,
            label=label,
            fit_config=fit_config,
            bootstrap_replicas=replicas,
            bootstrap_seed=int(seed),
        )
        if not bool(args.rebuild_summary):
            key = _cached_fit_key(
                file_name=file_name, mode=mode, split=split, method=method
            )
            cached = cached_by_key.get(key)
            if cached is None:
                raise RuntimeError(
                    f"{label}: no cached post-fit row exists. "
                    "Run with --rebuild-summary true to rebuild the bootstrap cache."
                )
            _reuse_bootstrap_uncertainty(result, cached, label=label)
        return result
    for file_id in sorted(files):
        file_name = files[file_id]
        prepared_path = Path(config["preprocessing"]["prepared_dir"]) / Path(file_name).stem
        dataset = load_prepared_dataset(prepared_path)
        expected_fingerprint = str(manifest.get("prepared_fingerprints", {}).get(str(file_id), ""))
        current_fingerprint = str(
            dataset.manifest.get("request_fingerprint", dataset.manifest.get("fingerprint", ""))
        )
        if expected_fingerprint and current_fingerprint != expected_fingerprint:
            raise RuntimeError(
                f"Prepared dataset fingerprint changed for {file_name}; do not post-fit a "
                "different dataset with the saved final models."
            )
        development, blind = _random_dev_blind(
            int(dataset.event_id.size),
            blind_fraction=float(config["cross_validation"]["blind_fraction"]),
            seed=_seed_for(base_seed, file_id, "devblind"),
        )
        voltage_rows = [row for row in rows if int(row.get("file_id", -1)) == file_id]
        voltage = float(voltage_rows[0]["voltage_V"]) if voltage_rows else float("nan")
        print(f"Post-fit file {file_id + 1}/{len(files)}: {file_name}")
        for mode_id in sorted(modes):
            mode = modes[mode_id]
            if mode not in config["channel_modes"]:
                continue
            print(f"  mode={mode}")
            blind_result_map: dict[str, DistributionFit] = {}
            description_map: dict[str, str] = {}
            prediction_target_map: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            led_train, cfd_train = _target_deltas(dataset, mode, development)
            led_blind, cfd_blind = _target_deltas(dataset, mode, blind)
            standard_residuals = {
                MODEL_LED: (
                    led_train - float(dataset.true_tof_ps),
                    led_blind - float(dataset.true_tof_ps),
                )
            }
            if mode == "energy_to_energy":
                standard_residuals[MODEL_CFD] = (
                    cfd_train - float(dataset.true_tof_ps),
                    cfd_blind - float(dataset.true_tof_ps),
                )
            for method, (train_residual, blind_residual) in standard_residuals.items():
                train_fit = fit_distribution_cached(
                    train_residual,
                    file_name=file_name, mode=mode, split="train", method=method,
                    label=f"{file_name} {mode} {method} train",
                    seed=_seed_for(args.bootstrap_seed, file_id, mode_id, method, "train"),
                )
                blind_fit = fit_distribution_cached(
                    blind_residual,
                    file_name=file_name, mode=mode, split="blind", method=method,
                    label=f"{file_name} {mode} {method} blind",
                    seed=_seed_for(args.bootstrap_seed, file_id, mode_id, method, "blind"),
                )
                if method == MODEL_LED:
                    description = "LED baseline"
                else:
                    energy_cfg = {
                        **config["preprocessing"]["common"],
                        **config["preprocessing"].get("energy", {}),
                    }
                    description = f"CFD fraction={float(energy_cfg['cfd_fraction']):g}"
                blind_result_map[method] = blind_fit
                description_map[method] = description
                all_fit_rows.extend([
                    _fit_row(
                        file_name=file_name, voltage=voltage, mode=mode, split="train",
                        method=method, result=train_fit, candidate_id=-1, window_id="",
                        description=description,
                    ),
                    _fit_row(
                        file_name=file_name, voltage=voltage, mode=mode, split="blind",
                        method=method, result=blind_fit, candidate_id=-1, window_id="",
                        description=description,
                    ),
                ])
            for family in ["linear_svr", "constructive_mlp", "cnn"]:
                if family not in model_to_id:
                    continue
                model_id = int(model_to_id[family])
                selected_row = _selected_cv_row(
                    rows, file_id=file_id, mode_id=mode_id, model_id=model_id
                )
                candidate_id = int(selected_row["candidate_id"])
                descriptor = dict(candidate_parameters[str(candidate_id)])
                checkpoint_meta = final_models.get(f"{file_id}:{mode_id}:{model_id}")
                if not isinstance(checkpoint_meta, dict) or "checkpoint" not in checkpoint_meta:
                    raise RuntimeError(
                        f"Final checkpoint metadata missing for {file_name} | {mode} | {family}"
                    )
                checkpoint_path = experiment_dir / str(checkpoint_meta["checkpoint"])
                if not checkpoint_path.is_file():
                    raise FileNotFoundError(f"Final checkpoint not found: {checkpoint_path}")
                payload = _load_checkpoint(checkpoint_path)
                context = payload.get("context", {}) or {}
                model_type = str(context.get("model_type", ""))
                if model_type == "linear_svr":
                    train_indices = development
                else:
                    overrides = descriptor.get("overrides", {}) or {}
                    early_fraction = float(
                        overrides.get(
                            "training.early_stop_fraction",
                            config["cross_validation"]["early_stop_fraction"],
                        )
                    )
                    train_indices, _ = _fit_early_split(
                        development,
                        fraction=early_fraction,
                        seed=_seed_for(base_seed, file_id, "early", "final"),
                    )
                replay = _waveform_residuals(
                    config,
                    dataset,
                    mode=mode,
                    descriptor=descriptor,
                    checkpoint_payload=payload,
                    train_indices=train_indices,
                    blind_indices=blind,
                    device_name=str(args.device),
                    batch_size=int(args.batch_size),
                )
                train_residual = replay["train_residual_ps"]
                blind_residual = replay["blind_residual_ps"]
                prediction_target_map[family] = (
                    replay["blind_prediction_ps"], replay["blind_target_ps"]
                )
                train_fit = fit_distribution_cached(
                    train_residual,
                    file_name=file_name, mode=mode, split="train", method=family,
                    label=f"{file_name} {mode} {family} train",
                    seed=_seed_for(args.bootstrap_seed, file_id, mode_id, family, "train"),
                )
                blind_fit = fit_distribution_cached(
                    blind_residual,
                    file_name=file_name, mode=mode, split="blind", method=family,
                    label=f"{file_name} {mode} {family} blind",
                    seed=_seed_for(args.bootstrap_seed, file_id, mode_id, family, "blind"),
                )
                description = _candidate_description(
                    family, descriptor, selected_row, windows, checkpoint_payload=payload
                )
                blind_result_map[family] = blind_fit
                description_map[family] = description
                all_fit_rows.extend([
                    _fit_row(
                        file_name=file_name, voltage=voltage, mode=mode, split="train",
                        method=family, result=train_fit, candidate_id=candidate_id,
                        window_id=str(descriptor.get("window", "")), description=description,
                        cv_ctr_ps=float(selected_row["ctr_ps"]),
                        cv_ctr_fold_std_ps=float(selected_row.get("ctr_fold_std_ps", float("nan"))),
                    ),
                    _fit_row(
                        file_name=file_name, voltage=voltage, mode=mode, split="blind",
                        method=family, result=blind_fit, candidate_id=candidate_id,
                        window_id=str(descriptor.get("window", "")), description=description,
                        cv_ctr_ps=float(selected_row["ctr_ps"]),
                        cv_ctr_fold_std_ps=float(selected_row.get("ctr_fold_std_ps", float("nan"))),
                    ),
                ])
            if MODEL_MULTITHRESHOLD in model_to_id:
                mt_model_id = int(model_to_id[MODEL_MULTITHRESHOLD])
                try:
                    selected_row = _selected_cv_row(
                        rows, file_id=file_id, mode_id=mode_id, model_id=mt_model_id
                    )
                except RuntimeError:
                    selected_row = None
                if selected_row is not None:
                    candidate_id = int(selected_row["candidate_id"])
                    descriptor = dict(candidate_parameters[str(candidate_id)])
                    replay = _multithreshold_residuals(
                        config,
                        dataset,
                        mode=mode,
                        descriptor=descriptor,
                        development=development,
                        blind=blind,
                    )
                    train_residual = replay["train_residual_ps"]
                    blind_residual = replay["blind_residual_ps"]
                    prediction_target_map[MODEL_MULTITHRESHOLD] = (
                        replay["blind_prediction_ps"], replay["blind_target_ps"]
                    )
                    train_fit = fit_distribution_cached(
                        train_residual,
                        file_name=file_name, mode=mode, split="train", method=MODEL_MULTITHRESHOLD,
                        label=f"{file_name} {mode} {MODEL_MULTITHRESHOLD} train",
                        seed=_seed_for(
                            args.bootstrap_seed, file_id, mode_id, MODEL_MULTITHRESHOLD, "train"
                        ),
                    )
                    blind_fit = fit_distribution_cached(
                        blind_residual,
                        file_name=file_name, mode=mode, split="blind", method=MODEL_MULTITHRESHOLD,
                        label=f"{file_name} {mode} {MODEL_MULTITHRESHOLD} blind",
                        seed=_seed_for(
                            args.bootstrap_seed, file_id, mode_id, MODEL_MULTITHRESHOLD, "blind"
                        ),
                    )
                    description = _candidate_description(
                        MODEL_MULTITHRESHOLD, descriptor, selected_row, windows
                    )
                    blind_result_map[MODEL_MULTITHRESHOLD] = blind_fit
                    description_map[MODEL_MULTITHRESHOLD] = description
                    all_fit_rows.extend([
                        _fit_row(
                            file_name=file_name, voltage=voltage, mode=mode, split="train",
                            method=MODEL_MULTITHRESHOLD, result=train_fit,
                            candidate_id=candidate_id,
                            window_id=str(descriptor.get("window", "")), description=description,
                            cv_ctr_ps=float(selected_row["ctr_ps"]),
                            cv_ctr_fold_std_ps=float(selected_row.get("ctr_fold_std_ps", float("nan"))),
                        ),
                        _fit_row(
                            file_name=file_name, voltage=voltage, mode=mode, split="blind",
                            method=MODEL_MULTITHRESHOLD, result=blind_fit,
                            candidate_id=candidate_id,
                            window_id=str(descriptor.get("window", "")), description=description,
                            cv_ctr_ps=float(selected_row["ctr_ps"]),
                            cv_ctr_fold_std_ps=float(selected_row.get("ctr_fold_std_ps", float("nan"))),
                        ),
                    ])
            if MODEL_LED not in blind_result_map:
                raise RuntimeError(f"Missing blind LED result for {file_name} | {mode}")
            for method, model_result in blind_result_map.items():
                if method == MODEL_LED:
                    continue
                _plot_file_mode_model_distribution(
                    distribution_root / mode / Path(file_name).stem / f"{method}.png",
                    file_name=file_name,
                    mode=mode,
                    method=method,
                    model_result=model_result,
                    led_result=blind_result_map[MODEL_LED],
                    description=description_map.get(method, ""),
                    dpi=dpi,
                )

            for method, (prediction_ps, target_ps) in prediction_target_map.items():
                _plot_prediction_vs_target(
                    prediction_target_root / mode / Path(file_name).stem / f"{method}.png",
                    file_name=file_name,
                    mode=mode,
                    method=method,
                    prediction_ps=prediction_ps,
                    target_ps=target_ps,
                    description=description_map.get(method, ""),
                    bin_width_ps=float(args.bin_width_ps),
                    dpi=dpi,
                )
        del dataset
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if bool(args.rebuild_summary):
        _write_fit_csv(fit_results_path, all_fit_rows)
        summary_rows = all_fit_rows
    else:
        # Preserve the existing bootstrap cache byte-for-byte. The replay above was only
        # needed to redraw Gaussian distributions with the new layout.
        summary_rows = cached_fit_rows
    for mode in config["channel_modes"]:
        methods = [MODEL_LED]
        if mode == "energy_to_energy":
            methods.append(MODEL_CFD)
        methods.extend([family for family in LEARNED_MODEL_ORDER if family in model_to_id])
        _plot_ctr_vs_voltage(
            summary_root / f"ctr_vs_voltage_{mode}.png",
            mode=mode,
            fit_rows=summary_rows,
            methods=methods,
            dpi=dpi,
        )
        _plot_train_vs_blind(
            summary_root / f"train_vs_blind_{mode}.png",
            mode=mode,
            fit_rows=summary_rows,
            methods=methods,
            dpi=dpi,
        )
    metadata = {
        "experiment_output": str(experiment_dir),
        "fit_results": str(fit_results_path),
        "rebuild_summary": bool(args.rebuild_summary),
        "bootstrap": {
            "recomputed": bool(args.rebuild_summary),
            "replicas": int(args.bootstrap_replicas) if bool(args.rebuild_summary) else "reused_from_fit_results.csv",
            "seed": int(args.bootstrap_seed),
        },
        "distribution_layout": (
            "one figure per mode/model; one subplot per voltage; no model overlays"
        ),
        "fit": {
            "histogram_bin_ps": float(args.bin_width_ps),
            "bin_phase_count": int(args.bin_phase_count),
            "min_events": int(args.min_events),
            "parameter_estimator": "binned Gaussian Poisson likelihood; best bin phase by reduced Poisson deviance",
            "goodness_of_fit": "Pearson chi-square on bins with fitted expected count >= 5; ndof = used_bins - 3",
            "event_rejection": "none in Gaussian fit; plot x-range is robustly limited only for display",
        },
        "train_definition": (
            "actual final weight-update subset for CNN/constructive MLP; full development set for "
            "Linear SVR and multithreshold SVR; development population for fixed LED/CFD baselines"
        ),
        "timing_mode_cfd": "excluded from energy_to_timing and timing_to_timing",
        "prediction_vs_target": (
            "blind-set model output versus the exact supervision target used by that model; "
            "waveform models use target_ps from the correction loader, multithreshold uses raw LED-TOF target"
        ),
    }
    (output / "postfit_manifest.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Gaussian post-fit complete: {output}")
if __name__ == "__main__":
    main()
