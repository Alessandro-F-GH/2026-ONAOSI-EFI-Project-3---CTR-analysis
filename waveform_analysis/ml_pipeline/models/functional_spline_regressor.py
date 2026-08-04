from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import BSpline
from torch import nn

from utils.plots import plot_best_fit

from ..common import atomic_json, write_csv_rows
from ..training_context import TrainingContext
from ..training_utils import (
    checkpoint_context,
    evaluate_model,
    make_split_loader,
    resolve_device,
)

_ALLOWED_AMPLITUDE_NORMALIZATION = {"none", "max_abs"}


def validate_config(config: dict[str, Any]) -> None:
    n_basis = int(config.get("n_basis", 16))
    degree = int(config.get("spline_degree", 3))
    if degree < 0:
        raise ValueError("spline_degree must be non-negative")
    if n_basis <= degree:
        raise ValueError("n_basis must be greater than spline_degree")
    normalization = str(config.get("amplitude_normalization", "none"))
    if normalization not in _ALLOWED_AMPLITUDE_NORMALIZATION:
        raise ValueError(
            "amplitude_normalization must be one of "
            f"{sorted(_ALLOWED_AMPLITUDE_NORMALIZATION)}"
        )
    if float(config.get("amplitude_epsilon", 1e-6)) <= 0.0:
        raise ValueError("amplitude_epsilon must be positive")


def validate_training_config(config: dict[str, Any]) -> None:
    regularization = config.get("regularization")
    if not isinstance(regularization, dict):
        raise ValueError("Functional spline training requires a regularization object")
    penalties = regularization.get("smoothness_penalties")
    if not isinstance(penalties, list) or not penalties:
        raise ValueError("regularization.smoothness_penalties must be a non-empty list")
    if any(float(value) <= 0.0 for value in penalties):
        raise ValueError("regularization.smoothness_penalties must be strictly positive")
    if float(regularization.get("ridge_penalty", 1e-10)) < 0.0:
        raise ValueError("regularization.ridge_penalty must be non-negative")
    training = config["training"]
    for name in ("batch_size", "normalization_chunk_size", "feature_chunk_size"):
        if int(training.get(name, 0)) <= 0:
            raise ValueError(f"training.{name} must be positive")


def build_bspline_basis(input_length: int, n_basis: int, degree: int) -> np.ndarray:
    """Return an open-uniform B-spline design matrix with shape [time, basis]."""
    input_length = int(input_length)
    n_basis = int(n_basis)
    degree = int(degree)
    if input_length <= 1:
        raise ValueError("input_length must be greater than one")
    if n_basis <= degree:
        raise ValueError("n_basis must be greater than degree")

    internal_count = n_basis - degree - 1
    internal = (
        np.linspace(0.0, 1.0, internal_count + 2, dtype=np.float64)[1:-1]
        if internal_count > 0
        else np.empty(0, dtype=np.float64)
    )
    knots = np.concatenate(
        [
            np.zeros(degree + 1, dtype=np.float64),
            internal,
            np.ones(degree + 1, dtype=np.float64),
        ]
    )
    time = np.linspace(0.0, 1.0, input_length, dtype=np.float64)
    basis = BSpline.design_matrix(time, knots, degree, extrapolate=False).toarray()
    if basis.shape != (input_length, n_basis):
        raise RuntimeError(
            f"Unexpected B-spline basis shape {basis.shape}; "
            f"expected {(input_length, n_basis)}"
        )
    return np.asarray(basis, dtype=np.float32)


class FunctionalSplineRegressor(nn.Module):
    """Shared functional linear model with correction ``g(s1) - g(s2)``."""

    def __init__(self, config: dict[str, Any], input_length: int) -> None:
        super().__init__()
        validate_config(config)
        self.input_length = int(input_length)
        self.n_basis = int(config.get("n_basis", 16))
        self.spline_degree = int(config.get("spline_degree", 3))
        self.amplitude_normalization = str(config.get("amplitude_normalization", "none"))
        self.amplitude_epsilon = float(config.get("amplitude_epsilon", 1e-6))

        basis = build_bspline_basis(self.input_length, self.n_basis, self.spline_degree)
        self.register_buffer("basis_matrix", torch.from_numpy(basis))
        self.register_buffer("feature_mean", torch.zeros(self.n_basis))
        self.register_buffer("feature_std", torch.ones(self.n_basis))
        self.register_buffer("coefficients", torch.zeros(self.n_basis))

    def _prepare_waveforms(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 2 or waveform.shape[1] != self.input_length:
            raise ValueError(f"Expected [batch, {self.input_length}] waveforms")
        if self.amplitude_normalization == "max_abs":
            scale = waveform.abs().amax(dim=1, keepdim=True).clamp_min(self.amplitude_epsilon)
            waveform = waveform / scale
        return waveform

    def project(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = self._prepare_waveforms(waveform)
        features = waveform @ self.basis_matrix / float(self.input_length)
        return (features - self.feature_mean) / self.feature_std

    def single_channel(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.project(waveform) @ self.coefficients

    def forward(self, waveform_pair: torch.Tensor) -> torch.Tensor:
        if waveform_pair.ndim != 3 or waveform_pair.shape[1] != 2:
            raise ValueError("Expected waveform pairs with shape [batch, 2, length]")
        if waveform_pair.shape[2] != self.input_length:
            raise ValueError(f"Expected waveform length {self.input_length}")
        batch = waveform_pair.shape[0]
        values = self.single_channel(
            waveform_pair.reshape(batch * 2, self.input_length)
        ).reshape(batch, 2)
        return values[:, 0] - values[:, 1]

    def set_fit_parameters(
        self,
        *,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
        coefficients: torch.Tensor,
    ) -> None:
        expected = (self.n_basis,)
        if tuple(feature_mean.shape) != expected:
            raise ValueError(f"feature_mean must have shape {expected}")
        if tuple(feature_std.shape) != expected:
            raise ValueError(f"feature_std must have shape {expected}")
        if tuple(coefficients.shape) != expected:
            raise ValueError(f"coefficients must have shape {expected}")
        if torch.any(feature_std <= 0):
            raise ValueError("feature_std must be strictly positive")
        with torch.no_grad():
            self.feature_mean.copy_(feature_mean.to(self.feature_mean))
            self.feature_std.copy_(feature_std.to(self.feature_std))
            self.coefficients.copy_(coefficients.to(self.coefficients))

    def effective_time_coefficients(self) -> torch.Tensor:
        scaled = self.coefficients / self.feature_std
        return (self.basis_matrix @ scaled) / float(self.input_length)


def build(config: dict[str, Any], input_length: int) -> nn.Module:
    return FunctionalSplineRegressor(config, input_length)


@dataclass(frozen=True)
class FeaturePairs:
    channel_1: np.ndarray
    channel_2: np.ndarray
    target_ps: np.ndarray
    mean_amplitude_mV: np.ndarray


def _project_numpy(waveforms: np.ndarray, model: FunctionalSplineRegressor) -> np.ndarray:
    values = np.asarray(waveforms, dtype=np.float64)
    if model.amplitude_normalization == "max_abs":
        scale = np.max(np.abs(values), axis=2, keepdims=True)
        values = values / np.maximum(scale, model.amplitude_epsilon)
    basis = np.asarray(model.basis_matrix.detach().cpu(), dtype=np.float64)
    return np.einsum("ncl,lk->nck", values, basis, optimize=True) / float(model.input_length)


def _extract_features(
    context: TrainingContext,
    split_name: str,
    model: FunctionalSplineRegressor,
) -> FeaturePairs:
    channel_1: list[np.ndarray] = []
    channel_2: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    mean_amplitudes: list[np.ndarray] = []
    chunk_size = int(context.config["training"]["feature_chunk_size"])

    for dataset in context.datasets:
        indices = np.asarray(getattr(dataset, split_name), dtype=np.int64)
        for start in range(0, indices.size, chunk_size):
            selected = indices[start : start + chunk_size]
            pair = np.asarray(dataset.windows_mV[selected], dtype=np.float64)
            pair = (pair - context.normalization.mean_mV) / context.normalization.std_mV
            projected = _project_numpy(pair, model)
            led_delta_ps = (
                np.asarray(dataset.led_time_fs[selected, 0], dtype=np.float64)
                - np.asarray(dataset.led_time_fs[selected, 1], dtype=np.float64)
            ) / 1000.0
            channel_1.append(projected[:, 0, :])
            channel_2.append(projected[:, 1, :])
            targets.append(led_delta_ps - dataset.true_tof_ps)
            mean_amplitudes.append(
                np.mean(np.asarray(dataset.amplitude_mV[selected], dtype=np.float64), axis=1)
            )
    if not targets:
        raise RuntimeError(f"No events found for split {split_name!r}")
    return FeaturePairs(
        channel_1=np.concatenate(channel_1, axis=0),
        channel_2=np.concatenate(channel_2, axis=0),
        target_ps=np.concatenate(targets, axis=0),
        mean_amplitude_mV=np.concatenate(mean_amplitudes, axis=0),
    )


def _fit_feature_scaling(train_features: FeaturePairs) -> tuple[np.ndarray, np.ndarray]:
    all_features = np.concatenate([train_features.channel_1, train_features.channel_2], axis=0)
    mean = np.mean(all_features, axis=0)
    std = np.maximum(np.std(all_features, axis=0), 1e-8)
    return mean, std


def _difference_design(values: FeaturePairs, feature_std: np.ndarray) -> np.ndarray:
    return (values.channel_1 - values.channel_2) / feature_std


def _second_difference_penalty(n_basis: int) -> np.ndarray:
    if n_basis < 3:
        return np.zeros((n_basis, n_basis), dtype=np.float64)
    second_difference = np.diff(np.eye(n_basis, dtype=np.float64), n=2, axis=0)
    return second_difference.T @ second_difference


def _solve_coefficients(
    gram: np.ndarray,
    rhs: np.ndarray,
    penalty: np.ndarray,
    smoothness: float,
    ridge: float,
) -> np.ndarray:
    system = gram + smoothness * penalty + ridge * np.eye(gram.shape[0], dtype=np.float64)
    try:
        return np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(system, rhs, rcond=None)[0]


def _rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    residual = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return float(np.sqrt(np.mean(residual * residual)))


def _save_regularization_plot(rows: list[dict[str, Any]], path: Any, dpi: int) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 5.5))
    penalties = [float(row["smoothness_penalty"]) for row in rows]
    axis.semilogx(penalties, [float(row["train_rmse_ps"]) for row in rows], marker="o", label="Train")
    axis.semilogx(
        penalties,
        [float(row["validation_rmse_ps"]) for row in rows],
        marker="o",
        label="Validation",
    )
    axis.set_xlabel("Smoothness penalty")
    axis.set_ylabel("Pairwise RMSE [ps]")
    axis.grid(True, alpha=0.22)
    axis.legend()
    figure.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)


def _save_coefficient_plot(relative_time_ps: np.ndarray, coefficient: np.ndarray, path: Any, dpi: int) -> None:
    figure, axis = plt.subplots(figsize=(9.0, 5.5))
    axis.plot(relative_time_ps, coefficient)
    axis.axhline(0.0, linewidth=0.8)
    axis.set_xlabel("Time relative to trigger [ps]")
    axis.set_ylabel("Timing coefficient [ps / normalized sample]")
    axis.grid(True, alpha=0.22)
    figure.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)


def _save_prediction_plot(target: np.ndarray, prediction: np.ndarray, path: Any, dpi: int) -> None:
    figure, axis = plt.subplots(figsize=(6.5, 6.0))
    axis.scatter(target, prediction, s=9, alpha=0.35)
    finite = np.concatenate([target[np.isfinite(target)], prediction[np.isfinite(prediction)]])
    if finite.size:
        low, high = float(np.min(finite)), float(np.max(finite))
        axis.plot([low, high], [low, high], linewidth=1.0, linestyle="--")
    axis.set_xlabel("Target LED error [ps]")
    axis.set_ylabel("Predicted correction [ps]")
    axis.grid(True, alpha=0.22)
    figure.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)


def _save_residual_amplitude_plot(amplitude: np.ndarray, residual: np.ndarray, path: Any, dpi: int) -> None:
    figure, axis = plt.subplots(figsize=(8.0, 5.5))
    axis.scatter(amplitude, residual, s=9, alpha=0.35)
    axis.axhline(0.0, linewidth=0.8)
    axis.set_xlabel("Mean pulse amplitude [mV]")
    axis.set_ylabel("Corrected timing residual [ps]")
    axis.grid(True, alpha=0.22)
    figure.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)


def train(context: TrainingContext) -> dict[str, Any]:
    config = context.config
    reference_time = np.asarray(context.datasets[0].relative_time_ps, dtype=np.float64)
    for dataset in context.datasets[1:]:
        current = np.asarray(dataset.relative_time_ps, dtype=np.float64)
        if current.shape != reference_time.shape or not np.allclose(
            current, reference_time, rtol=0.0, atol=1e-6
        ):
            raise ValueError("Training datasets use different waveform time axes")

    device = resolve_device(config["training"].get("device", "auto"))
    model = build(context.model_config, context.input_length)
    if not isinstance(model, FunctionalSplineRegressor):
        raise TypeError("Functional spline builder returned an incompatible model")

    context.logger.info("Extracting B-spline waveform features")
    train_features = _extract_features(context, "train", model)
    validation_features = _extract_features(context, "validation", model)
    feature_mean, feature_std = _fit_feature_scaling(train_features)
    x_train = _difference_design(train_features, feature_std)
    x_validation = _difference_design(validation_features, feature_std)
    gram = x_train.T @ x_train
    rhs = x_train.T @ train_features.target_ps
    penalty = _second_difference_penalty(model.n_basis)

    smoothness_penalties = sorted(
        {float(value) for value in config["regularization"]["smoothness_penalties"]}
    )
    ridge = float(config["regularization"].get("ridge_penalty", 1e-10))
    rows: list[dict[str, Any]] = []
    best_coefficients: np.ndarray | None = None
    best_penalty = math.nan
    best_validation_rmse = math.inf

    for smoothness in smoothness_penalties:
        coefficients = _solve_coefficients(gram, rhs, penalty, smoothness, ridge)
        train_rmse = _rmse(x_train @ coefficients, train_features.target_ps)
        validation_rmse = _rmse(x_validation @ coefficients, validation_features.target_ps)
        if validation_rmse < best_validation_rmse:
            best_validation_rmse = validation_rmse
            best_penalty = smoothness
            best_coefficients = coefficients.copy()
        rows.append(
            {
                "smoothness_penalty": smoothness,
                "ridge_penalty": ridge,
                "train_rmse_ps": train_rmse,
                "validation_rmse_ps": validation_rmse,
                "selected_best": False,
            }
        )
        context.logger.info(
            "Smoothness %.6g | train RMSE %.3f ps | validation RMSE %.3f ps",
            smoothness,
            train_rmse,
            validation_rmse,
        )

    if best_coefficients is None:
        raise RuntimeError("No valid functional spline solution was produced")
    for row in rows:
        row["selected_best"] = float(row["smoothness_penalty"]) == float(best_penalty)
    write_csv_rows(context.output_dir / "regularization_scan.csv", rows)

    model.set_fit_parameters(
        feature_mean=torch.from_numpy(feature_mean.astype(np.float32)),
        feature_std=torch.from_numpy(feature_std.astype(np.float32)),
        coefficients=torch.from_numpy(best_coefficients.astype(np.float32)),
    )
    model = model.to(device)
    train_loader = make_split_loader(
        context.datasets, "train", context.normalization, config, device, shuffle=False
    )
    validation_loader = make_split_loader(
        context.datasets, "validation", context.normalization, config, device, shuffle=False
    )
    train_metrics, train_fit, train_prediction = evaluate_model(
        model, train_loader, device, config["fit"], "Functional spline train residual"
    )
    validation_metrics, validation_fit, validation_prediction = evaluate_model(
        model,
        validation_loader,
        device,
        config["fit"],
        "Functional spline validation residual",
    )

    final_model_config = copy.deepcopy(context.model_config)
    final_model_config["selected_smoothness_penalty"] = float(best_penalty)
    metadata = checkpoint_context(
        context,
        model_config=final_model_config,
        training_strategy=(
            "closed-form functional linear regression with B-spline projection "
            "and validation-selected second-difference regularization"
        ),
    )
    payload = {"model_state": model.state_dict(), "epoch": 1, "context": metadata}
    best_path = context.checkpoint_dir / "best.pt"
    last_path = context.checkpoint_dir / "last.pt"
    torch.save(payload, best_path)
    torch.save(payload, last_path)

    dpi = int(config["plotting"].get("dpi", 180))
    _save_regularization_plot(rows, context.plot_dir / "regularization_scan.png", dpi)
    effective = np.asarray(model.effective_time_coefficients().detach().cpu(), dtype=np.float64)
    _save_coefficient_plot(
        reference_time,
        effective,
        context.plot_dir / "coefficient_function.png",
        dpi,
    )
    _save_prediction_plot(
        np.asarray(validation_prediction["target_ps"], dtype=np.float64),
        np.asarray(validation_prediction["prediction_ps"], dtype=np.float64),
        context.plot_dir / "validation_predicted_vs_target.png",
        dpi,
    )
    _save_residual_amplitude_plot(
        validation_features.mean_amplitude_mV,
        np.asarray(validation_prediction["residual_ps"], dtype=np.float64),
        context.plot_dir / "validation_residual_vs_amplitude.png",
        dpi,
    )
    plot_best_fit(train_fit, context.plot_dir / "best_train_gaussian_fit.png", dpi=dpi)
    plot_best_fit(
        validation_fit,
        context.plot_dir / "best_validation_gaussian_fit.png",
        dpi=dpi,
    )

    write_csv_rows(
        context.output_dir / "basis_coefficients.csv",
        [
            {
                "basis_index": index,
                "coefficient": float(best_coefficients[index]),
                "training_feature_mean": float(feature_mean[index]),
                "training_feature_std": float(feature_std[index]),
            }
            for index in range(model.n_basis)
        ],
    )
    write_csv_rows(
        context.output_dir / "coefficient_function.csv",
        [
            {
                "sample_index": index,
                "relative_time_ps": float(reference_time[index]),
                "effective_coefficient_ps_per_normalized_sample": float(effective[index]),
            }
            for index in range(context.input_length)
        ],
    )

    summary = {
        "model_type": context.model_type,
        "model_name": context.model_name,
        "best_epoch": 1,
        "best_validation_rmse_ps": float(validation_metrics["rmse_ps"]),
        "best_validation_ctr_ps": float(validation_metrics["ctr_ps"]),
        "best_validation_bias_ps": float(validation_metrics["bias_ps"]),
        "best_checkpoint": str(best_path.resolve()),
        "last_checkpoint": str(last_path.resolve()),
        "train_dir": str(context.output_dir.resolve()),
        "input_length": int(context.input_length),
        "n_basis": int(model.n_basis),
        "spline_degree": int(model.spline_degree),
        "selected_smoothness_penalty": float(best_penalty),
        "ridge_penalty": ridge,
        "normalization": context.normalization.as_dict(),
        "training_datasets": [str(dataset.directory) for dataset in context.datasets],
        "optimizer": "closed-form linear solve",
        "regularization_scan_rows": len(rows),
        "final_train_rmse_ps": float(train_metrics["rmse_ps"]),
    }
    atomic_json(context.output_dir / "training_summary.json", summary)
    return summary
