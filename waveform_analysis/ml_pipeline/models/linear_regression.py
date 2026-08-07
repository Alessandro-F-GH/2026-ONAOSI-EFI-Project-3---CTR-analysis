from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn

from ..common import atomic_json
from ..input_transform import component_subsampling_indices
from ..training_context import TrainingContext
from ..torch_data import factored_correction_target_ps
from ..training_utils import (
    checkpoint_context,
    evaluate_model,
    make_split_loader,
    resolve_device,
)
from .spec import ModelSpec


_REGULARIZATION_ALIASES = {
    "none": "none",
    "ols": "none",
    "linear": "none",
    "linear_regression": "none",
    "ridge": "ridge",
    "l2": "ridge",
    "lasso": "lasso",
    "l1": "lasso",
}

_LOSS_ALIASES = {
    "variance": "variance",
    "rmse": "rmse",
    "variance_bias": "variance_bias",
    "variance_plus_bias": "variance_bias",
    "variance+bias": "variance_bias",
}


def _regularization(config: dict[str, Any]) -> str:
    raw = str(config.get("regularization", "none")).strip().lower()
    try:
        return _REGULARIZATION_ALIASES[raw]
    except KeyError as exc:
        raise ValueError(
            "regularization must be one of ['none', 'ridge', 'lasso']"
        ) from exc


def _selection_loss(config: dict[str, Any]) -> str:
    raw = str(config.get("loss", {}).get("type", "rmse")).strip().lower()
    try:
        return _LOSS_ALIASES[raw]
    except KeyError as exc:
        raise ValueError(
            "model.loss.type must be one of "
            "['variance', 'rmse', 'variance_bias']"
        ) from exc


def validate_config(config: dict[str, Any]) -> None:
    regularization = _regularization(config)
    alpha = float(config.get("alpha", 1.0))
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be finite and non-negative")
    if regularization in {"ridge", "lasso"} and alpha <= 0.0:
        raise ValueError(f"alpha must be positive for {regularization} regularization")
    if float(config.get("tolerance", 1.0e-4)) <= 0.0:
        raise ValueError("tolerance must be positive")
    if int(config.get("max_iterations", 10000)) <= 0:
        raise ValueError("max_iterations must be positive")
    if str(config.get("lasso_selection", "cyclic")) not in {"cyclic", "random"}:
        raise ValueError("lasso_selection must be 'cyclic' or 'random'")
    ridge_solver = str(config.get("ridge_solver", "auto"))
    if ridge_solver not in {
        "auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs"
    }:
        raise ValueError("Unsupported ridge_solver")
    loss = config.get("loss", {})
    if not isinstance(loss, dict):
        raise ValueError("loss must be an object")
    _selection_loss(config)
    if float(loss.get("bias_weight", 0.0)) < 0.0:
        raise ValueError("loss.bias_weight must be non-negative")
    normalization = str(loss.get("bias_normalization", "none"))
    if normalization not in {"none", "target_std"}:
        raise ValueError("loss.bias_normalization must be 'none' or 'target_std'")
    if float(loss.get("minimum_scale", 1.0e-8)) <= 0.0:
        raise ValueError("loss.minimum_scale must be positive")


def validate_training_config(config: dict[str, Any]) -> None:
    training = config["training"]
    for name in ("batch_size", "normalization_chunk_size"):
        if int(training.get(name, 0)) <= 0:
            raise ValueError(f"training.{name} must be positive")
    if bool(training.get("random_pair_swap", False)):
        raise ValueError(
            "training.random_pair_swap is unnecessary for linear_regression: "
            "swapping the detector pair negates both the feature difference and target"
        )
    baseline_guard_metric = training.get("baseline_guard_metric")
    if baseline_guard_metric not in (None, "validation_rmse", "validation_ctr"):
        raise ValueError(
            "training.baseline_guard_metric must be null, "
            "'validation_rmse', or 'validation_ctr'"
        )


class LinearPairRegressor(nn.Module):
    """Shared linear branch for an ordered detector pair.

    For ``g(s)=w^T s`` the learned pair correction is
    ``g(s1)-g(s2)=w^T(s1-s2)``.  The scalar pair bias is reserved for the
    mandatory training-residual mean calibration, preserving the shared-branch
    interpretation while allowing a non-zero global correction offset.
    """

    def __init__(self, input_length: int) -> None:
        super().__init__()
        self.input_length = int(input_length)
        self.weight = nn.Parameter(
            torch.zeros(self.input_length, dtype=torch.float32),
            requires_grad=False,
        )
        self.pair_output_bias_ps = nn.Parameter(
            torch.zeros((), dtype=torch.float32),
            requires_grad=False,
        )

    def forward(self, waveform_pair: torch.Tensor) -> torch.Tensor:
        if waveform_pair.ndim != 3 or waveform_pair.shape[1] != 2:
            raise ValueError("Expected waveform pairs with shape [batch, 2, length]")
        if waveform_pair.shape[2] != self.input_length:
            raise ValueError(f"Expected waveform length {self.input_length}")
        difference = waveform_pair[:, 0, :] - waveform_pair[:, 1, :]
        return difference @ self.weight + self.pair_output_bias_ps


def build(config: dict[str, Any], input_length: int) -> nn.Module:
    del config
    return LinearPairRegressor(input_length)


class _ZeroCorrectionModel(nn.Module):
    apply_window_anchor_shift = False

    def forward(self, waveform_pair: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            waveform_pair.shape[0],
            dtype=waveform_pair.dtype,
            device=waveform_pair.device,
        )


def _split_matrix(
    context: TrainingContext,
    split_name: str,
    *,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    counts = [
        int(np.asarray(getattr(dataset, split_name)).size)
        for dataset in context.datasets
    ]
    total = int(sum(counts))
    if total == 0:
        raise ValueError(
            f"Cannot train/evaluate linear_regression on empty {split_name} split"
        )

    features = np.empty((total, context.input_length), dtype=np.float64)
    target = np.empty(total, dtype=np.float64)
    cursor = 0
    scale = np.asarray(context.normalization.std_mV, dtype=np.float64)
    if scale.ndim not in {0, 1}:
        raise ValueError("Invalid waveform normalization standard deviation shape")
    if scale.ndim == 1 and int(scale.size) != int(context.input_length):
        raise ValueError(
            "Feature normalization length does not match linear_regression input: "
            f"{scale.size} != {context.input_length}"
        )
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("Invalid waveform normalization standard deviation")

    for dataset in context.datasets:
        indices = np.asarray(getattr(dataset, split_name), dtype=np.int64)
        for start in range(0, indices.size, chunk_size):
            selected = indices[start : start + chunk_size]
            pair = np.asarray(dataset.windows_mV[selected], dtype=np.float64)
            component_lengths = dataset.manifest.get("input_component_lengths")
            lengths = (
                [int(value) for value in component_lengths]
                if isinstance(component_lengths, list)
                else [int(dataset.input_length)]
            )
            source_indices = component_subsampling_indices(
                lengths, context.subsampling_factor
            )
            pair = pair[..., source_indices]
            size = int(selected.size)
            # The shared normalization mean cancels in the detector difference.
            features[cursor : cursor + size] = (
                pair[:, 0, :] - pair[:, 1, :]
            ) / scale
            target[cursor : cursor + size] = factored_correction_target_ps(
                dataset, selected
            )
            cursor += size
    return features, target


def _residual_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = np.asarray(target, dtype=np.float64) - np.asarray(
        prediction, dtype=np.float64
    )
    bias = float(np.mean(residual))
    variance = float(np.mean((residual - bias) ** 2))
    rmse = float(np.sqrt(np.mean(residual**2)))
    return {"bias_ps": bias, "variance_ps2": variance, "rmse_ps": rmse}


def _selection_value(
    metrics: dict[str, float],
    *,
    loss_type: str,
    bias_weight: float,
    bias_scale_ps: float = 1.0,
) -> float:
    if loss_type == "variance":
        return float(metrics["variance_ps2"])
    if loss_type == "rmse":
        return float(metrics["rmse_ps"])
    if loss_type == "variance_bias":
        scale = max(float(bias_scale_ps), np.finfo(np.float64).eps)
        return float(
            metrics["variance_ps2"]
            + bias_weight * (metrics["bias_ps"] / scale) ** 2
        )
    raise ValueError(f"Unsupported linear-regression selection loss: {loss_type}")


def _calibrated_bias(
    coefficient: np.ndarray,
    features: np.ndarray,
    target: np.ndarray,
) -> float:
    raw_prediction = features @ np.asarray(coefficient, dtype=np.float64)
    return float(np.mean(np.asarray(target, dtype=np.float64) - raw_prediction))


def _assign_model(
    model: LinearPairRegressor,
    coefficient: np.ndarray,
    pair_bias_ps: float,
) -> None:
    coefficient = np.asarray(coefficient, dtype=np.float32).reshape(-1)
    if coefficient.shape != (model.input_length,):
        raise ValueError(
            f"Linear-regression coefficient shape {coefficient.shape} does not match "
            f"input length {model.input_length}"
        )
    with torch.no_grad():
        model.weight.copy_(torch.from_numpy(coefficient).to(model.weight.device))
        model.pair_output_bias_ps.fill_(float(pair_bias_ps))


def _make_estimator(model_config: dict[str, Any], random_state: int):
    try:
        from sklearn.linear_model import Lasso, LinearRegression, Ridge
    except ImportError as exc:
        raise RuntimeError(
            "linear_regression requires scikit-learn. Install it with "
            "'python -m pip install scikit-learn'."
        ) from exc

    regularization = _regularization(model_config)
    alpha = float(model_config.get("alpha", 1.0))
    if regularization == "none":
        return LinearRegression(fit_intercept=False), regularization, 0.0
    if regularization == "ridge":
        return (
            Ridge(
                alpha=alpha,
                fit_intercept=False,
                solver=str(model_config.get("ridge_solver", "auto")),
                tol=float(model_config.get("tolerance", 1.0e-4)),
                max_iter=int(model_config.get("max_iterations", 10000)),
                random_state=random_state,
            ),
            regularization,
            alpha,
        )
    return (
        Lasso(
            alpha=alpha,
            fit_intercept=False,
            tol=float(model_config.get("tolerance", 1.0e-4)),
            max_iter=int(model_config.get("max_iterations", 10000)),
            selection=str(model_config.get("lasso_selection", "cyclic")),
            random_state=random_state,
        ),
        regularization,
        alpha,
    )


def train(context: TrainingContext) -> dict[str, Any]:
    config = context.config
    model_config = context.model_config
    artifacts = dict(config.get("artifacts", {}))
    save_last_checkpoint = bool(artifacts.get("save_last_checkpoint", True))
    save_summary = bool(artifacts.get("save_summary", True))

    device = resolve_device(config["training"].get("device", "auto"))
    train_loader = make_split_loader(
        context.datasets,
        "train",
        context.normalization,
        config,
        device,
        shuffle=False,
        subsampling_factor=context.subsampling_factor,
    )
    validation_loader = make_split_loader(
        context.datasets,
        "validation",
        context.normalization,
        config,
        device,
        shuffle=False,
        subsampling_factor=context.subsampling_factor,
    )
    zero_model = _ZeroCorrectionModel().to(device)
    baseline_train_metrics, _baseline_train_fit, _ = evaluate_model(
        zero_model,
        train_loader,
        device,
        config["fit"],
        "Uncorrected train LED",
    )
    baseline_validation_metrics, _baseline_validation_fit, _ = evaluate_model(
        zero_model,
        validation_loader,
        device,
        config["fit"],
        "Uncorrected validation LED",
    )

    chunk_size = int(
        config["training"].get(
            "linear_materialization_chunk_size",
            config["training"].get("normalization_chunk_size", 2048),
        )
    )
    if chunk_size <= 0:
        raise ValueError("training.linear_materialization_chunk_size must be positive")
    train_features, train_target = _split_matrix(
        context, "train", chunk_size=chunk_size
    )
    validation_features, validation_target = _split_matrix(
        context, "validation", chunk_size=chunk_size
    )

    random_state = int(
        config["training"].get(
            "initialization_seed",
            config["training"].get("seed", 12345),
        )
    )
    estimator, regularization, alpha = _make_estimator(model_config, random_state)
    estimator.fit(train_features, train_target)
    coefficient = np.asarray(estimator.coef_, dtype=np.float64).reshape(-1)
    if coefficient.shape != (context.input_length,):
        raise RuntimeError(
            f"Estimator returned coefficient shape {coefficient.shape}; "
            f"expected {(context.input_length,)}"
        )

    pair_bias_ps = _calibrated_bias(coefficient, train_features, train_target)
    model = build(model_config, context.input_length).to(device)
    assert isinstance(model, LinearPairRegressor)
    _assign_model(model, coefficient, pair_bias_ps)

    train_metrics, _train_fit, _ = evaluate_model(
        model,
        train_loader,
        device,
        config["fit"],
        "Final calibrated linear-regression train residual",
    )
    validation_metrics, _validation_fit, _ = evaluate_model(
        model,
        validation_loader,
        device,
        config["fit"],
        "Final calibrated linear-regression validation residual",
    )

    baseline_guard_metric = config["training"].get("baseline_guard_metric")
    baseline_guard_applied = False
    if baseline_guard_metric is not None:
        key = {
            "validation_rmse": "rmse_ps",
            "validation_ctr": "ctr_ps",
        }[str(baseline_guard_metric)]
        if float(validation_metrics[key]) > float(baseline_validation_metrics[key]):
            context.logger.warning(
                "Linear regression is worse than uncorrected LED on %s; "
                "selecting calibrated constant correction",
                baseline_guard_metric,
            )
            coefficient = np.zeros(context.input_length, dtype=np.float64)
            pair_bias_ps = _calibrated_bias(
                coefficient, train_features, train_target
            )
            _assign_model(model, coefficient, pair_bias_ps)
            baseline_guard_applied = True
            train_metrics, _train_fit, _ = evaluate_model(
                model,
                train_loader,
                device,
                config["fit"],
                "Selected calibrated baseline train residual",
            )
            validation_metrics, _validation_fit, _ = evaluate_model(
                model,
                validation_loader,
                device,
                config["fit"],
                "Selected calibrated baseline validation residual",
            )

    loss_type = _selection_loss(model_config)
    loss_config = model_config.get("loss", {})
    bias_weight = float(loss_config.get("bias_weight", 0.0))
    bias_normalization = str(loss_config.get("bias_normalization", "none"))
    minimum_scale = float(loss_config.get("minimum_scale", 1.0e-8))
    bias_scale_ps = (
        max(float(np.std(train_target, ddof=0)), minimum_scale)
        if bias_normalization == "target_std"
        else 1.0
    )
    validation_residual_metrics = {
        "variance_ps2": max(
            float(validation_metrics["rmse_ps"]) ** 2
            - float(validation_metrics["bias_ps"]) ** 2,
            0.0,
        ),
        "rmse_ps": float(validation_metrics["rmse_ps"]),
        "bias_ps": float(validation_metrics["bias_ps"]),
    }
    selection_value = _selection_value(
        validation_residual_metrics,
        loss_type=loss_type,
        bias_weight=bias_weight,
        bias_scale_ps=bias_scale_ps,
    )

    checkpoint_metadata = checkpoint_context(
        context,
        training_strategy=(
            "scikit-learn linear least squares on normalized detector-pair "
            f"differences with {regularization} regularization"
        ),
    )
    checkpoint_metadata["linear_regression"] = {
        "regularization": regularization,
        "alpha": float(alpha),
        "selection_loss_type": loss_type,
        "bias_weight": bias_weight,
        "bias_normalization": bias_normalization,
        "bias_scale_ps": bias_scale_ps,
        "baseline_guard_selected": baseline_guard_applied,
        "nonzero_coefficient_count": int(np.count_nonzero(coefficient)),
        "coefficient_l1_norm": float(np.linalg.norm(coefficient, ord=1)),
        "coefficient_l2_norm": float(np.linalg.norm(coefficient, ord=2)),
    }
    checkpoint_metadata["final_bias_calibration"] = {
        "enforced": True,
        "reference_split": "train",
        "quantity_zeroed": "arithmetic mean of corrected residual",
        "mode": "residual_mean",
        "pair_output_bias_ps": float(pair_bias_ps),
        "final_train_bias_ps": float(train_metrics["bias_ps"]),
    }

    best_path = context.checkpoint_dir / "best.pt"
    last_path = context.checkpoint_dir / "last.pt"
    payload = {
        "model_state": model.state_dict(),
        "epoch": 0,
        "context": checkpoint_metadata,
    }
    torch.save(payload, best_path)
    if save_last_checkpoint:
        torch.save(payload, last_path)

    weight_path = context.output_dir / "linear_regression_weight.npy"
    np.save(weight_path, model.weight.detach().cpu().numpy())
    atomic_json(
        context.output_dir / "linear_regression_selection.json",
        {
            "regularization": regularization,
            "alpha": float(alpha),
            "selection_loss_type": loss_type,
            "selection_value": float(selection_value),
            "pair_output_bias_ps": float(pair_bias_ps),
            "baseline_guard_applied": baseline_guard_applied,
            "nonzero_coefficient_count": int(np.count_nonzero(coefficient)),
            "coefficient_l1_norm": float(np.linalg.norm(coefficient, ord=1)),
            "coefficient_l2_norm": float(np.linalg.norm(coefficient, ord=2)),
        },
    )

    summary = {
        "model_type": context.model_type,
        "model_name": context.model_name,
        "best_epoch": 0,
        "best_selection_metric": loss_type,
        "best_selection_value": float(selection_value),
        "best_validation_rmse_ps": float(validation_metrics["rmse_ps"]),
        "best_validation_ctr_ps": float(validation_metrics["ctr_ps"]),
        "best_validation_bias_ps": float(validation_metrics["bias_ps"]),
        "best_validation_variance_ps2": float(
            validation_residual_metrics["variance_ps2"]
        ),
        "uncorrected_led_validation_rmse_ps": float(
            baseline_validation_metrics["rmse_ps"]
        ),
        "uncorrected_led_validation_ctr_ps": float(
            baseline_validation_metrics["ctr_ps"]
        ),
        "uncorrected_led_validation_bias_ps": float(
            baseline_validation_metrics["bias_ps"]
        ),
        "baseline_guard_metric": baseline_guard_metric,
        "baseline_guard_applied": baseline_guard_applied,
        "best_checkpoint": str(best_path.resolve()),
        "last_checkpoint": str(last_path.resolve()) if last_path.is_file() else "",
        "train_dir": str(context.output_dir.resolve()),
        "input_length": int(context.input_length),
        "input_transform": context.input_transform,
        "subsampling_factor": int(context.subsampling_factor),
        "input_waveform_source": context.input_waveform_source,
        "prediction_target": context.prediction_target,
        "input_cache_paths": [str(path) for path in context.input_cache_dirs],
        "pair_output_bias_ps": float(pair_bias_ps),
        "final_bias_calibration": {
            "enforced": True,
            "reference_split": "train",
            "mode": "residual_mean",
            "final_train_bias_ps": float(train_metrics["bias_ps"]),
        },
        "normalization": context.normalization.as_dict(),
        "training_datasets": [str(dataset.directory) for dataset in context.datasets],
        "optimizer": f"scikit-learn {type(estimator).__name__}",
        "regularization": regularization,
        "alpha": float(alpha),
        "coefficient_path": str(weight_path.resolve()),
        "coefficient_l1_norm": float(np.linalg.norm(coefficient, ord=1)),
        "coefficient_l2_norm": float(np.linalg.norm(coefficient, ord=2)),
        "nonzero_coefficient_count": int(np.count_nonzero(coefficient)),
        "final_train_rmse_ps": float(train_metrics["rmse_ps"]),
        "final_train_bias_ps": float(train_metrics["bias_ps"]),
        "data_view": dict(context.data_view),
        "data_seed": int(
            config["training"].get(
                "data_seed", config["training"].get("seed", 12345)
            )
        ),
        "model_parameter_count": int(context.input_length + 1),
    }
    if save_summary:
        atomic_json(context.output_dir / "training_summary.json", summary)
    return summary


MODEL_SPEC = ModelSpec(
    name="linear_regression",
    builder=build,
    validator=validate_config,
    training_validator=validate_training_config,
    trainer=train,
    complexity_counter=lambda _config, input_length: int(input_length) + 1,
)
