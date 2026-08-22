from __future__ import annotations

import sys

import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import torch
from torch import nn

from utils_fit.plotting import plot_gaussian_fit
from ..common import atomic_json, write_csv_rows
from ..input_transform import component_subsampling_indices
from ..training_context import TrainingContext
from ..torch_data import factored_correction_target_ps
from ..training_utils import (
    checkpoint_context,
    evaluate_model,
    evaluate_model_with_optional_fit,
    ctr_log_text,
    make_split_loader,
    resolve_device,
)
from .spec import ModelSpec


_LOSS_ALIASES = {
    "variance": "variance",
    "rmse": "rmse",
    "variance_bias": "variance_bias",
    "variance_plus_bias": "variance_bias",
    "variance+bias": "variance_bias",
}


def _normalized_selection_loss(config: dict[str, Any]) -> str:
    raw = str(config.get("loss", {}).get("type", "variance")).strip().lower()
    try:
        return _LOSS_ALIASES[raw]
    except KeyError as exc:
        raise ValueError(
            "model.loss.type must be one of "
            "['variance', 'rmse', 'variance_bias']"
        ) from exc


def validate_config(config: dict[str, Any]) -> None:
    if float(config.get("C", 1.0)) <= 0.0:
        raise ValueError("C must be positive")
    epsilon_values = config.get("epsilon_values")
    if not isinstance(epsilon_values, list) or not epsilon_values:
        raise ValueError("epsilon_values must be a non-empty list")
    values = [float(value) for value in epsilon_values]
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("epsilon_values must contain finite non-negative values")
    if len(set(values)) != len(values):
        raise ValueError("epsilon_values must not contain duplicates")
    if str(config.get("svm_loss", "epsilon_insensitive")) not in (
        "epsilon_insensitive",
        "squared_epsilon_insensitive",
    ):
        raise ValueError(
            "svm_loss must be 'epsilon_insensitive' or "
            "'squared_epsilon_insensitive'"
        )
    if float(config.get("tolerance", 1.0e-4)) <= 0.0:
        raise ValueError("tolerance must be positive")
    if int(config.get("max_iterations", 10000)) <= 0:
        raise ValueError("max_iterations must be positive")
    dual = config.get("dual", "auto")
    if dual not in (True, False, "auto"):
        raise ValueError("dual must be true, false, or 'auto'")
    loss = config.get("loss", {})
    if not isinstance(loss, dict):
        raise ValueError("loss must be an object")
    _normalized_selection_loss(config)
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
            "training.random_pair_swap is unnecessary for linear_svr: swapping "
            "the detector pair simply negates both the feature difference and target"
        )
    baseline_guard_metric = training.get("baseline_guard_metric")
    if baseline_guard_metric not in (None, "validation_rmse", "validation_ctr"):
        raise ValueError(
            "training.baseline_guard_metric must be null, "
            "'validation_rmse', or 'validation_ctr'"
        )


class LinearPairSVR(nn.Module):
    """Linear shared-branch pair model.

    For a single-channel score ``g(s) = w^T s``, the pair correction is
    exactly ``g(s1) - g(s2) = w^T (s1 - s2)``.
    """

    def __init__(self, input_length: int) -> None:
        super().__init__()
        self.input_length = int(input_length)
        self.weight = nn.Parameter(
            torch.zeros(self.input_length, dtype=torch.float32),
            requires_grad=False,
        )

    def forward(self, waveform_pair: torch.Tensor) -> torch.Tensor:
        if waveform_pair.ndim != 3 or waveform_pair.shape[1] != 2:
            raise ValueError("Expected waveform pairs with shape [batch, 2, length]")
        if waveform_pair.shape[2] != self.input_length:
            raise ValueError(f"Expected waveform length {self.input_length}")
        difference = waveform_pair[:, 0, :] - waveform_pair[:, 1, :]
        return difference @ self.weight


def build(config: dict[str, Any], input_length: int) -> nn.Module:
    del config
    return LinearPairSVR(input_length)


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
    counts = [int(np.asarray(getattr(dataset, split_name)).size) for dataset in context.datasets]
    total = int(sum(counts))
    if total == 0:
        raise ValueError(f"Cannot train/evaluate linear_svr on empty {split_name} split")

    features = np.empty((total, context.input_length), dtype=np.float64)
    target = np.empty(total, dtype=np.float64)
    cursor = 0
    scale = np.asarray(context.normalization.std_mV, dtype=np.float64)
    if scale.ndim not in {0, 1}:
        raise ValueError("Invalid waveform normalization standard deviation shape")
    if scale.ndim == 1 and int(scale.size) != int(context.input_length):
        raise ValueError(
            "Feature normalization length does not match linear_svr input: "
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
            # The common normalization mean cancels in the detector difference:
            # (s1 - mean)/std - (s2 - mean)/std = (s1 - s2)/std.
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
    return {
        "bias_ps": bias,
        "variance_ps2": variance,
        "rmse_ps": rmse,
    }


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
        # Bias is squared so both terms have units of ps^2 and positive/negative
        # validation bias cannot cancel the objective.
        scale = max(float(bias_scale_ps), np.finfo(np.float64).eps)
        return float(
            metrics["variance_ps2"]
            + bias_weight * (metrics["bias_ps"] / scale) ** 2
        )
    raise ValueError(f"Unsupported linear_svr selection loss: {loss_type}")


def _assign_model(
    model: LinearPairSVR,
    coefficient: np.ndarray,
) -> None:
    coefficient = np.asarray(coefficient, dtype=np.float32).reshape(-1)
    if coefficient.shape != (model.input_length,):
        raise ValueError(
            f"LinearSVR coefficient shape {coefficient.shape} does not match "
            f"input length {model.input_length}"
        )
    with torch.no_grad():
        model.weight.copy_(torch.from_numpy(coefficient).to(model.weight.device))


def _plot_epsilon_scan(rows: list[dict[str, Any]], path: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    epsilon = np.asarray([float(row["epsilon_ps"]) for row in rows], dtype=np.float64)
    objective = np.asarray(
        [float(row["validation_selection_loss"]) for row in rows],
        dtype=np.float64,
    )
    order = np.argsort(epsilon)
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    axis.plot(epsilon[order], objective[order], marker="o")
    axis.set_xlabel("SVR epsilon [ps]")
    axis.set_ylabel("Validation selection loss")
    axis.set_title("Linear SVR epsilon scan")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def train(context: TrainingContext) -> dict[str, Any]:
    try:
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.svm import LinearSVR
    except ImportError as exc:
        raise RuntimeError(
            "linear_svr requires scikit-learn. Install it with "
            "'python -m pip install scikit-learn'."
        ) from exc

    config = context.config
    model_config = context.model_config
    artifacts = dict(config.get("artifacts", {}))
    save_history = bool(artifacts.get("save_history", True))
    save_plots = bool(artifacts.get("save_plots", True))
    save_best_checkpoint = bool(artifacts.get("save_best_checkpoint", True))
    save_last_checkpoint = bool(artifacts.get("save_last_checkpoint", True))
    save_summary = bool(artifacts.get("save_summary", True))
    save_model_artifacts = bool(artifacts.get("save_model_artifacts", True))
    perform_internal_fit = bool(artifacts.get("perform_internal_gaussian_fit", True))

    # LinearSVR/liblinear training is CPU-only.  The configured device is still
    # used for the common final evaluation path and checkpoint compatibility.
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
    baseline_train_metrics, baseline_train_fit, _ = evaluate_model_with_optional_fit(
        zero_model,
        train_loader,
        device,
        config["fit"],
        "Uncorrected train LED",
        perform_fit=perform_internal_fit,
    )
    baseline_validation_metrics, baseline_validation_fit, _ = evaluate_model_with_optional_fit(
        zero_model,
        validation_loader,
        device,
        config["fit"],
        "Uncorrected validation LED",
        perform_fit=perform_internal_fit,
    )
    context.logger.debug(
        "LED baseline | train RMSE %.1f ps %s | val RMSE %.1f ps %s bias %.1f ps",
        baseline_train_metrics["rmse_ps"],
        ctr_log_text(baseline_train_metrics),
        baseline_validation_metrics["rmse_ps"],
        ctr_log_text(baseline_validation_metrics),
        baseline_validation_metrics["bias_ps"],
    )

    chunk_size = int(
        config["training"].get(
            "svr_materialization_chunk_size",
            config["training"].get("normalization_chunk_size", 2048),
        )
    )
    if chunk_size <= 0:
        raise ValueError("training.svr_materialization_chunk_size must be positive")
    train_features, train_target = _split_matrix(
        context, "train", chunk_size=chunk_size
    )
    validation_features, validation_target = _split_matrix(
        context, "validation", chunk_size=chunk_size
    )
    context.logger.debug(
        "Linear SVR matrices | train %s | validation %s | dtype float64",
        tuple(train_features.shape),
        tuple(validation_features.shape),
    )

    epsilon_values = [float(value) for value in model_config["epsilon_values"]]
    loss_type = _normalized_selection_loss(model_config)
    loss_config = model_config.get("loss", {})
    bias_weight = float(loss_config.get("bias_weight", 0.0))
    bias_normalization = str(loss_config.get("bias_normalization", "none"))
    minimum_scale = float(loss_config.get("minimum_scale", 1.0e-8))
    bias_scale_ps = (
        max(float(np.std(train_target, ddof=0)), minimum_scale)
        if bias_normalization == "target_std"
        else 1.0
    )
    random_state = int(
        config["training"].get(
            "initialization_seed",
            config["training"].get("seed", 12345),
        )
    )
    common_parameters = {
        "C": float(model_config.get("C", 1.0)),
        "loss": str(model_config.get("svm_loss", "epsilon_insensitive")),
        "fit_intercept": False,
        "tol": float(model_config.get("tolerance", 1.0e-4)),
        "dual": model_config.get("dual", "auto"),
        "max_iter": int(model_config.get("max_iterations", 10000)),
        "random_state": random_state,
        "verbose": 0,
    }

    rows: list[dict[str, Any]] = []
    best_value = math.inf
    best_rmse = math.inf
    best_epsilon = float("nan")
    best_coefficient: np.ndarray | None = None
    last_coefficient: np.ndarray | None = None
    last_epsilon = float("nan")

    for scan_index, epsilon in enumerate(epsilon_values, start=1):
        estimator = LinearSVR(epsilon=epsilon, **common_parameters)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            estimator.fit(train_features, train_target)
        converged = not any(
            issubclass(item.category, ConvergenceWarning) for item in caught
        )
        coefficient = np.asarray(estimator.coef_, dtype=np.float64).reshape(-1)
        train_prediction = train_features @ coefficient
        validation_prediction = validation_features @ coefficient
        train_metrics = _residual_metrics(train_target, train_prediction)
        validation_metrics = _residual_metrics(
            validation_target, validation_prediction
        )
        objective = _selection_value(
            validation_metrics,
            loss_type=loss_type,
            bias_weight=bias_weight,
            bias_scale_ps=bias_scale_ps,
        )
        row = {
            "scan_index": scan_index,
            "epsilon_ps": epsilon,
            "C": common_parameters["C"],
            "svm_loss": common_parameters["loss"],
            "converged": converged,
            "iterations": int(np.max(np.atleast_1d(estimator.n_iter_))),
            "train_rmse_ps": train_metrics["rmse_ps"],
            "train_variance_ps2": train_metrics["variance_ps2"],
            "train_bias_ps": train_metrics["bias_ps"],
            "validation_rmse_ps": validation_metrics["rmse_ps"],
            "validation_variance_ps2": validation_metrics["variance_ps2"],
            "validation_bias_ps": validation_metrics["bias_ps"],
            "selection_loss_type": loss_type,
            "bias_weight": bias_weight,
            "validation_selection_loss": objective,
            "selected_best": False,
        }
        better = (
            objective < best_value
            or (
                math.isclose(objective, best_value, rel_tol=0.0, abs_tol=1.0e-12)
                and validation_metrics["rmse_ps"] < best_rmse
            )
        )
        if better:
            for previous in rows:
                previous["selected_best"] = False
            best_value = objective
            best_rmse = validation_metrics["rmse_ps"]
            best_epsilon = epsilon
            best_coefficient = coefficient.copy()
            row["selected_best"] = True
        rows.append(row)
        last_coefficient = coefficient.copy()
        last_epsilon = epsilon
        context.logger.debug(
            "SVR epsilon %g ps | val variance %.6f ps^2 | val RMSE %.3f ps | "
            "val bias %.3f ps | objective %.6f | converged=%s",
            epsilon,
            validation_metrics["variance_ps2"],
            validation_metrics["rmse_ps"],
            validation_metrics["bias_ps"],
            objective,
            converged,
        )

    if best_coefficient is None:
        raise RuntimeError("Linear SVR epsilon scan produced no candidate model")

    model = build(model_config, context.input_length).to(device)
    assert isinstance(model, LinearPairSVR)
    _assign_model(model, best_coefficient)
    checkpoint_metadata = checkpoint_context(
        context,
        training_strategy=(
            "scikit-learn LinearSVR epsilon-insensitive optimization on normalized "
            "detector-pair differences; epsilon selected on validation residual loss"
        ),
    )
    checkpoint_metadata["linear_svr"] = {
        "selected_epsilon_ps": best_epsilon,
        "epsilon_values_ps": epsilon_values,
        "selection_loss_type": loss_type,
        "bias_weight": bias_weight,
        "bias_normalization": bias_normalization,
        "bias_scale_ps": bias_scale_ps,
        **common_parameters,
    }

    # No additive calibration is permitted: the selected model remains exactly
    # c(s1,s2)=g(s1)-g(s2).
    _assign_model(model, best_coefficient)
    train_metrics, train_fit, _ = evaluate_model_with_optional_fit(
        model,
        train_loader,
        device,
        config["fit"],
        "Final linear SVR train residual",
        perform_fit=perform_internal_fit,
    )
    validation_metrics, validation_fit, _ = evaluate_model_with_optional_fit(
        model,
        validation_loader,
        device,
        config["fit"],
        "Final linear SVR validation residual",
        perform_fit=perform_internal_fit,
    )

    baseline_guard_metric = config["training"].get("baseline_guard_metric")
    baseline_guard_applied = False
    if baseline_guard_metric is not None:
        key = {
            "validation_rmse": "rmse_ps",
            "validation_ctr": "ctr_ps",
        }[str(baseline_guard_metric)]
        corrected_value = float(validation_metrics[key])
        baseline_value = float(baseline_validation_metrics[key])
        if corrected_value > baseline_value:
            context.logger.warning(
                "Linear SVR is worse than the uncorrected baseline on %s "
                "(%.3f > %.3f); selecting zero correction",
                baseline_guard_metric,
                corrected_value,
                baseline_value,
            )
            best_coefficient = np.zeros(context.input_length, dtype=np.float64)
            _assign_model(model, best_coefficient)
            baseline_guard_applied = True
            best_epsilon = float("nan")
            checkpoint_metadata["linear_svr"]["selected_epsilon_ps"] = None
            checkpoint_metadata["linear_svr"]["baseline_guard_selected"] = True
            train_metrics, train_fit, _ = evaluate_model(
                model,
                train_loader,
                device,
                config["fit"],
                "Selected zero-correction train residual",
            )
            validation_metrics, validation_fit, _ = evaluate_model(
                model,
                validation_loader,
                device,
                config["fit"],
                "Selected zero-correction validation residual",
            )

    final_validation_residual_metrics = {
        "variance_ps2": max(
            float(validation_metrics["rmse_ps"]) ** 2
            - float(validation_metrics["bias_ps"]) ** 2,
            0.0,
        ),
        "rmse_ps": float(validation_metrics["rmse_ps"]),
        "bias_ps": float(validation_metrics["bias_ps"]),
    }
    best_value = _selection_value(
        final_validation_residual_metrics,
        loss_type=loss_type,
        bias_weight=bias_weight,
        bias_scale_ps=bias_scale_ps,
    )


    best_path = context.checkpoint_dir / "best.pt"
    last_path = context.checkpoint_dir / "last.pt"
    if save_best_checkpoint:
        torch.save(
            {
                "model_state": model.state_dict(),
                "epoch": 0,
                "context": checkpoint_metadata,
            },
            best_path,
        )
    if save_last_checkpoint and last_coefficient is not None:
        last_model = build(model_config, context.input_length)
        assert isinstance(last_model, LinearPairSVR)
        _assign_model(last_model, last_coefficient)
        last_context = dict(checkpoint_metadata)
        last_context["linear_svr"] = dict(checkpoint_metadata["linear_svr"])
        last_context["linear_svr"]["selected_epsilon_ps"] = last_epsilon
        torch.save(
            {
                "model_state": last_model.state_dict(),
                "epoch": 0,
                "context": last_context,
            },
            last_path,
        )

    if save_history:
        write_csv_rows(context.output_dir / "epsilon_scan.csv", rows)
    if save_model_artifacts:
        np.save(
            context.output_dir / "linear_svr_weight.npy",
            model.weight.detach().cpu().numpy(),
        )
        atomic_json(
            context.output_dir / "linear_svr_selection.json",
            {
                "selected_epsilon_ps": None
                if not math.isfinite(best_epsilon)
                else best_epsilon,
                "selection_loss_type": loss_type,
                "bias_weight": bias_weight,
                "selection_value": best_value,
                "baseline_guard_applied": baseline_guard_applied,
            },
        )

    if save_plots:
        context.plot_dir.mkdir(parents=True, exist_ok=True)
        dpi = int(config.get("plotting", {}).get("dpi", 180))
        _plot_epsilon_scan(rows, context.plot_dir / "epsilon_scan.png", dpi)
        plot_gaussian_fit(
            train_fit,
            context.plot_dir / "best_train_gaussian_fit.png",
            dpi=dpi,
        )
        plot_gaussian_fit(
            validation_fit,
            context.plot_dir / "best_validation_gaussian_fit.png",
            dpi=dpi,
        )

    summary = {
        "model_type": context.model_type,
        "model_name": context.model_name,
        "best_epoch": 0,
        "best_selection_metric": loss_type,
        "best_selection_value": float(best_value),
        "selected_epsilon_ps": None
        if not math.isfinite(best_epsilon)
        else float(best_epsilon),
        "epsilon_values_ps": epsilon_values,
        "best_validation_rmse_ps": float(validation_metrics["rmse_ps"]),
        "best_validation_ctr_ps": float(validation_metrics["ctr_ps"]),
        "best_validation_bias_ps": float(validation_metrics["bias_ps"]),
        "best_validation_variance_ps2": float(
            final_validation_residual_metrics["variance_ps2"]
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
        "best_checkpoint": str(best_path.resolve()) if best_path.is_file() else "",
        "last_checkpoint": str(last_path.resolve()) if last_path.is_file() else "",
        "train_dir": str(context.output_dir.resolve()),
        "input_length": int(context.input_length),
        "input_transform": context.input_transform,
        "subsampling_factor": int(context.subsampling_factor),
        "input_waveform_source": context.input_waveform_source,
        "prediction_target": context.prediction_target,
        "input_cache_paths": [str(path) for path in context.input_cache_dirs],
        "normalization": context.normalization.as_dict(),
        "training_datasets": [str(dataset.directory) for dataset in context.datasets],
        "optimizer": "scikit-learn LinearSVR (liblinear)",
        "history_rows": len(rows),
        "final_train_rmse_ps": float(train_metrics["rmse_ps"]),
        "final_train_bias_ps": float(train_metrics["bias_ps"]),
        "data_view": dict(context.data_view),
        "data_seed": int(
            config["training"].get("data_seed", config["training"].get("seed", 12345))
        ),
        "model_parameter_count": int(context.input_length),
    }
    if save_summary:
        atomic_json(context.output_dir / "training_summary.json", summary)
    summary["_trained_model"] = model
    summary["_checkpoint_context"] = checkpoint_metadata
    return summary


MODEL_SPEC = ModelSpec(
    name="linear_svr",
    builder=build,
    validator=validate_config,
    training_validator=validate_training_config,
    trainer=train,
    complexity_counter=lambda _config, input_length: int(input_length) + 1,
)
