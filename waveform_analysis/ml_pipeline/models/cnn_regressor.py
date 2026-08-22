from __future__ import annotations

from pathlib import Path

import sys

import copy

import hashlib
import math
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from torch import nn

from utils_fit.plotting import plot_gaussian_fit
from ..common import atomic_json, write_csv_rows
from ..losses import mse_residual_loss, var_bias_loss, var_bias_value_from_metrics
from ..plots import plot_training_history
from ..training_context import TrainingContext
from ..torch_data import factored_correction_target_ps
from .spec import ModelSpec
from ..training_utils import (
    checkpoint_context,
    evaluate_model,
    evaluate_model_with_optional_fit,
    ctr_log_text,
    fit_schedule_for_epoch,
    make_split_loader,
    predict_loader,
    randomly_swap_paired_batch,
    resolve_device,
    validate_fit_schedule,
)




def model_state_hash(model_or_state: nn.Module | dict[str, torch.Tensor]) -> str:
    """Stable hash of a model state, including parameter names and tensor bytes."""
    state = model_or_state.state_dict() if isinstance(model_or_state, nn.Module) else model_or_state
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()

def _activation(name: str) -> nn.Module:
    key = str(name).lower()
    choices: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
        "identity": nn.Identity,
    }
    if key not in choices:
        raise ValueError(f"Unsupported activation: {name}")
    return choices[key]()


def _as_int_list(config: dict[str, Any], name: str) -> list[int]:
    value = config.get(name)
    if not isinstance(value, list) or not value or any(int(item) <= 0 for item in value):
        raise ValueError(f"{name} must be a non-empty list of positive integers")
    return [int(item) for item in value]


def validate_config(config: dict[str, Any]) -> None:
    channels = _as_int_list(config, "channels")
    kernel_sizes = _as_int_list(config, "kernel_sizes")
    strides = _as_int_list(config, "strides")
    if not (len(channels) == len(kernel_sizes) == len(strides)):
        raise ValueError("channels, kernel_sizes, and strides must have the same length")
    if any(kernel % 2 == 0 for kernel in kernel_sizes):
        raise ValueError("kernel_sizes must be odd so temporal alignment is preserved")
    dilations = config.get("dilations", [1] * len(channels))
    if not isinstance(dilations, list) or len(dilations) != len(channels):
        raise ValueError("dilations must be a list with one value per convolution block")
    if any(int(value) <= 0 for value in dilations):
        raise ValueError("dilations must contain positive integers")
    dense_units = config.get("dense_units", [])
    if not isinstance(dense_units, list) or any(int(value) <= 0 for value in dense_units):
        raise ValueError("dense_units must be a list of positive integers")
    pool_length = config.get("adaptive_pool_length")
    if pool_length is not None and int(pool_length) <= 0:
        raise ValueError("adaptive_pool_length must be positive or null")
    if config.get("activation", "silu") not in ("relu", "gelu", "silu", "tanh", "identity"):
        raise ValueError("Unsupported activation")
    normalization = str(config.get("normalization", "none")).lower()
    if normalization not in ("none", "batch", "group"):
        raise ValueError("normalization must be one of ['none', 'batch', 'group']")
    groups = int(config.get("group_norm_groups", 1))
    if groups <= 0:
        raise ValueError("group_norm_groups must be positive")
    if normalization == "group" and any(channel % groups != 0 for channel in channels):
        raise ValueError("Every channel count must be divisible by group_norm_groups")
    for name in ("conv_dropout", "dense_dropout"):
        value = float(config.get(name, 0.0))
        if not 0.0 <= value < 1.0:
            raise ValueError(f"{name} must lie in [0, 1)")
    bound = config.get("max_abs_single_channel_output_ps")
    if bound is not None and float(bound) <= 0.0:
        raise ValueError("max_abs_single_channel_output_ps must be positive when provided")

def validate_training_config(config: dict[str, Any]) -> None:
    optimizer = config.get("optimizer")
    if not isinstance(optimizer, dict):
        raise ValueError("CNN training requires an optimizer object")
    if float(optimizer.get("learning_rate", 0.0)) <= 0.0:
        raise ValueError("optimizer.learning_rate must be positive")
    if float(optimizer.get("weight_decay", 0.0)) < 0.0:
        raise ValueError("optimizer.weight_decay must be non-negative")
    training = config["training"]
    validate_fit_schedule(training)
    for name in ("epochs", "batch_size", "normalization_chunk_size"):
        if int(training.get(name, 0)) <= 0:
            raise ValueError(f"training.{name} must be positive")
    random_pair_swap = training.get("random_pair_swap", False)
    if not isinstance(random_pair_swap, bool):
        raise ValueError("training.random_pair_swap must be a boolean")
    baseline_guard_metric = training.get("baseline_guard_metric")
    if baseline_guard_metric not in (None, "validation_rmse", "validation_ctr"):
        raise ValueError(
            "training.baseline_guard_metric must be null, "
            "'validation_rmse', or 'validation_ctr'"
        )
    loss = config.get("model", {}).get("loss", {})
    if not isinstance(loss, dict):
        raise ValueError("model.loss must be an object when provided")
    loss_type = str(loss.get("type", "mse"))
    if loss_type not in ("mse", "var_bias"):
        raise ValueError("model.loss.type must be one of ['mse', 'var_bias']")
    if float(loss.get("bias_weight", 0.0)) < 0.0:
        raise ValueError("model.loss.bias_weight must be non-negative")


class _ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        stride: int,
        dilation: int,
        activation: str,
        normalization: str,
        group_norm_groups: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        layers: list[nn.Module] = [
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=normalization == "none",
            )
        ]
        if normalization == "batch":
            layers.append(nn.BatchNorm1d(out_channels))
        elif normalization == "group":
            layers.append(nn.GroupNorm(group_norm_groups, out_channels))
        layers.append(_activation(activation))
        if dropout > 0.0:
            layers.append(nn.Dropout1d(dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class SingleChannelCNN(nn.Module):
    """Small strided 1-D CNN for long, strongly autocorrelated waveforms.

    Strided convolutions reduce the temporal dimension before the dense head, so
    parameter count and memory do not grow linearly with the original waveform
    length. Adaptive pooling gives a fixed-size representation for any compatible
    window length.
    """

    def __init__(self, config: dict[str, Any], input_length: int) -> None:
        super().__init__()
        channels = [int(value) for value in config["channels"]]
        kernel_sizes = [int(value) for value in config["kernel_sizes"]]
        strides = [int(value) for value in config["strides"]]
        dilations = [int(value) for value in config.get("dilations", [1] * len(channels))]
        activation = str(config.get("activation", "silu"))
        normalization = str(config.get("normalization", "none")).lower()
        group_norm_groups = int(config.get("group_norm_groups", 1))
        conv_dropout = float(config.get("conv_dropout", 0.0))
        dense_dropout = float(config.get("dense_dropout", 0.0))
        pool_value = config.get("adaptive_pool_length")
        pool_length = None if pool_value is None else int(pool_value)
        dense_units = [int(value) for value in config.get("dense_units", [])]

        blocks: list[nn.Module] = []
        in_channels = 1
        for out_channels, kernel_size, stride, dilation in zip(
            channels, kernel_sizes, strides, dilations
        ):
            blocks.append(
                _ConvBlock(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    activation=activation,
                    normalization=normalization,
                    group_norm_groups=group_norm_groups,
                    dropout=conv_dropout,
                )
            )
            in_channels = out_channels
        self.features = nn.Sequential(*blocks)
        feature_length = int(input_length)
        for stride in strides:
            feature_length = (feature_length + stride - 1) // stride
        if pool_length is None:
            self.pool = nn.Identity()
            pooled_length = feature_length
        else:
            self.pool = nn.AdaptiveAvgPool1d(pool_length)
            pooled_length = pool_length

        head: list[nn.Module] = []
        in_features = channels[-1] * pooled_length
        for out_features in dense_units:
            head.append(nn.Linear(in_features, out_features))
            head.append(_activation(activation))
            if dense_dropout > 0.0:
                head.append(nn.Dropout(dense_dropout))
            in_features = out_features
        head.append(nn.Linear(in_features, 1))
        self.head = nn.Sequential(*head)
        self.input_length = int(input_length)
        self.total_stride = math.prod(strides)
        self.encoded_length = int(pooled_length)
        bound = config.get("max_abs_single_channel_output_ps")
        self.output_bound_ps = None if bound is None else float(bound)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 3 and waveform.shape[1] == 1:
            waveform = waveform[:, 0, :]
        if waveform.ndim != 2 or waveform.shape[1] != self.input_length:
            raise ValueError(f"Expected [batch, {self.input_length}] single-channel waveforms")
        encoded = self.features(waveform.unsqueeze(1))
        encoded = self.pool(encoded).flatten(1)
        output = self.head(encoded).squeeze(-1)
        if self.output_bound_ps is not None:
            output = self.output_bound_ps * torch.tanh(output / self.output_bound_ps)
        return output


class AntisymmetricCNNRegressor(nn.Module):
    """One shared CNN score g applied to both detectors: g(s1)-g(s2)."""

    def __init__(self, config: dict[str, Any], input_length: int) -> None:
        super().__init__()
        self.shared = SingleChannelCNN(config, input_length)
        self.input_length = int(input_length)

    def forward(self, waveform_pair: torch.Tensor) -> torch.Tensor:
        if waveform_pair.ndim != 3 or waveform_pair.shape[1] != 2:
            raise ValueError("Expected waveform pairs with shape [batch, 2, length]")
        if waveform_pair.shape[2] != self.input_length:
            raise ValueError(f"Expected waveform length {self.input_length}")
        batch = waveform_pair.shape[0]
        output = self.shared(
            waveform_pair.reshape(batch * 2, self.input_length)
        ).reshape(batch, 2)
        return output[:, 0] - output[:, 1]


def build(config: dict[str, Any], input_length: int) -> nn.Module:
    return AntisymmetricCNNRegressor(config, input_length)


class _ZeroCorrectionModel(nn.Module):
    apply_window_anchor_shift = False

    def forward(self, waveform_pair: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            waveform_pair.shape[0],
            dtype=waveform_pair.dtype,
            device=waveform_pair.device,
        )


def _set_zero_correction(model: AntisymmetricCNNRegressor) -> None:
    final_linear = next(
        layer for layer in reversed(model.shared.head) if isinstance(layer, nn.Linear)
    )
    with torch.no_grad():
        final_linear.weight.zero_()
        if final_linear.bias is not None:
            final_linear.bias.zero_()


def _target_scale_from_datasets(context: TrainingContext, minimum_scale: float) -> float:
    values: list[torch.Tensor] = []
    for dataset in context.datasets:
        indices = dataset.train
        values.append(
            torch.as_tensor(
                factored_correction_target_ps(dataset, indices),
                dtype=torch.float64,
            )
        )
    target = torch.cat(values)
    return max(float(torch.std(target, unbiased=False).item()), minimum_scale)


def train(context: TrainingContext) -> dict[str, Any]:
    config = context.config
    artifacts = dict(config.get("artifacts", {}))
    save_history = bool(artifacts.get("save_history", True))
    save_plots = bool(artifacts.get("save_plots", True))
    save_best_checkpoint = bool(artifacts.get("save_best_checkpoint", True))
    save_last_checkpoint = bool(artifacts.get("save_last_checkpoint", True))
    save_summary = bool(artifacts.get("save_summary", True))
    perform_internal_fit = bool(artifacts.get("perform_internal_gaussian_fit", True))

    device = resolve_device(config["training"].get("device", "auto"))
    train_loader = make_split_loader(
        context.datasets, "train", context.normalization, config, device, shuffle=True,
        subsampling_factor=context.subsampling_factor,
    )
    train_eval_loader = make_split_loader(
        context.datasets, "train", context.normalization, config, device, shuffle=False,
        subsampling_factor=context.subsampling_factor,
    )
    validation_loader = make_split_loader(
        context.datasets, "validation", context.normalization, config, device, shuffle=False,
        subsampling_factor=context.subsampling_factor,
    )
    zero_model = _ZeroCorrectionModel().to(device)
    baseline_train_metrics, _baseline_train_fit, _ = evaluate_model_with_optional_fit(
        zero_model, train_eval_loader, device, config["fit"], "Uncorrected train LED",
        perform_fit=perform_internal_fit,
    )
    baseline_validation_metrics, _baseline_validation_fit, _ = evaluate_model_with_optional_fit(
        zero_model, validation_loader, device, config["fit"], "Uncorrected validation LED",
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

    initialization_seed = int(
        config["training"].get(
            "initialization_seed",
            config["training"].get("seed", 12345),
        )
    )
    torch.manual_seed(initialization_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(initialization_seed)
    model = build(context.model_config, context.input_length).to(device)
    assert isinstance(model, AntisymmetricCNNRegressor)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    context.logger.debug(
        "CNN architecture | input length %d | encoded length %d | total stride %d | parameters %d | dense units %s",
        context.input_length,
        model.shared.encoded_length,
        model.shared.total_stride,
        parameter_count,
        context.model_config.get("dense_units", []),
    )
    initial_state_hash = model_state_hash(model)
    first_parameter = next(model.parameters()).detach().reshape(-1)[0].item()
    context.logger.debug(
        "Actual model initialization seed %d | hash %s | first weight %.9g",
        initialization_seed,
        initial_state_hash[:12],
        float(first_parameter),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["optimizer"]["learning_rate"]),
        weight_decay=float(config["optimizer"].get("weight_decay", 0.0)),
    )
    amp_enabled = bool(config["training"].get("mixed_precision", True)) and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    epochs = int(config["training"]["epochs"])
    patience = int(config["training"].get("early_stopping_patience", epochs))
    min_delta = float(config["training"].get("early_stopping_min_delta_ps", 0.0))
    gradient_clip = config["training"].get("gradient_clip_norm")
    best_value = math.inf
    best_epoch = 0
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    best_path = context.checkpoint_dir / "best.pt"
    last_path = context.checkpoint_dir / "last.pt"
    best_state: dict[str, torch.Tensor] | None = None
    checkpoint_metadata = checkpoint_context(
        context,
        training_strategy="Adam optimization of a shared strided 1-D CNN with antisymmetric pair output",
    )

    loss_config = dict(config.get("model", {}).get("loss", {}))
    loss_type = str(loss_config.get("type", "mse"))
    bias_weight = float(loss_config.get("bias_weight", 0.0)) if loss_type == "var_bias" else 0.0
    bias_norm = str(loss_config.get("bias_normalization", "target_std"))
    minimum_scale = float(loss_config.get("minimum_scale", 1e-8))
    target_scale = (
        _target_scale_from_datasets(context, minimum_scale)
        if bias_norm == "target_std"
        else 1.0
    )
    selection_metric = str(config["training"].get("selection_metric", "validation_rmse"))
    allowed_selection = {
        "validation_rmse",
        "validation_loss",
        "absolute_validation_bias",
        "validation_ctr",
    }
    if selection_metric not in allowed_selection:
        raise ValueError(
            f"Unsupported training.selection_metric {selection_metric!r}; "
            f"available: {sorted(allowed_selection)}"
        )

    random_pair_swap = bool(config["training"].get("random_pair_swap", False))
    pair_swap_generator = torch.Generator(device="cpu")
    pair_swap_generator.manual_seed(
        int(config["training"].get("data_seed", config["training"].get("seed", 12345)))
    )
    if random_pair_swap:
        context.logger.debug(
            "Random ordered-pair swapping enabled for training batches (probability 0.5)"
        )
    context.logger.debug("Training %s with Adam on %s", context.model_name, device)
    for epoch in range(1, epochs + 1):
        model.train()
        for waveforms, target, led_delta, cfd_delta, true_tof, anchor_shift in train_loader:
            if random_pair_swap:
                (
                    waveforms, target, led_delta, cfd_delta, true_tof, anchor_shift
                ) = randomly_swap_paired_batch(
                    waveforms,
                    target,
                    led_delta,
                    cfd_delta,
                    true_tof,
                    anchor_shift,
                    generator=pair_swap_generator,
                    probability=0.5,
                )
            waveforms = waveforms.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                prediction = model(waveforms)
                if loss_type == "var_bias":
                    loss, _penalty = var_bias_loss(
                        prediction,
                        target,
                        bias_weight=bias_weight,
                        target_scale=target_scale,
                    )
                else:
                    loss = mse_residual_loss(prediction, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip))
            scaler.step(optimizer)
            scaler.update()

        fit_train, fit_validation = fit_schedule_for_epoch(
            config["training"], epoch, selection_metric=selection_metric
        )
        train_metrics, _train_fit, train_prediction = evaluate_model_with_optional_fit(
            model,
            train_eval_loader,
            device,
            config["fit"],
            "Train residual",
            perform_fit=fit_train,
        )
        validation_metrics, _validation_fit, validation_prediction = evaluate_model_with_optional_fit(
            model,
            validation_loader,
            device,
            config["fit"],
            "Validation residual",
            perform_fit=fit_validation,
        )
        train_bias = float(train_metrics["bias_ps"])
        validation_bias = float(validation_metrics["bias_ps"])
        if loss_type == "var_bias":
            train_loss, train_variance, _train_bias_penalty = var_bias_value_from_metrics(
                rmse_ps=float(train_metrics["rmse_ps"]),
                bias_ps=train_bias,
                bias_weight=bias_weight,
                target_scale=target_scale,
            )
            validation_loss, validation_variance, validation_bias_penalty = (
                var_bias_value_from_metrics(
                    rmse_ps=float(validation_metrics["rmse_ps"]),
                    bias_ps=validation_bias,
                    bias_weight=bias_weight,
                    target_scale=target_scale,
                )
            )
        else:
            train_loss = float(train_metrics["rmse_ps"] ** 2)
            validation_loss = float(validation_metrics["rmse_ps"] ** 2)
            train_variance = max(train_loss - train_bias**2, 0.0)
            validation_variance = max(validation_loss - validation_bias**2, 0.0)
            validation_bias_penalty = 0.0
        row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_rmse_ps": float(train_metrics["rmse_ps"]),
            "validation_rmse_ps": float(validation_metrics["rmse_ps"]),
            "train_ctr_ps": float(train_metrics["ctr_ps"]),
            "validation_ctr_ps": float(validation_metrics["ctr_ps"]),
            "train_bias_ps": train_bias,
            "validation_bias_ps": validation_bias,
            "train_variance_ps2": float(train_variance),
            "validation_variance_ps2": float(validation_variance),
            "train_loss": float(train_loss),
            "validation_loss": float(validation_loss),
            "train_prediction_mean_ps": float(
                torch.as_tensor(train_prediction["prediction_ps"], dtype=torch.float64).mean().item()
            ),
            "train_prediction_residual_mean_ps": float(
                (
                    torch.as_tensor(train_prediction["prediction_ps"], dtype=torch.float64)
                    - torch.as_tensor(train_prediction["target_ps"], dtype=torch.float64)
                ).mean().item()
            ),
            "validation_prediction_mean_ps": float(
                torch.as_tensor(validation_prediction["prediction_ps"], dtype=torch.float64).mean().item()
            ),
            "bias_penalty": float(validation_bias_penalty),
            "train_fit_performed": bool(train_metrics["fit_performed"]),
            "validation_fit_performed": bool(validation_metrics["fit_performed"]),
            "selected_best": False,
        }
        metric_values = {
            "validation_rmse": row["validation_rmse_ps"],
            "validation_loss": row["validation_loss"],
            "absolute_validation_bias": abs(row["validation_bias_ps"]),
            "validation_ctr": row["validation_ctr_ps"],
        }
        current = float(metric_values[selection_metric])
        if current < best_value - min_delta:
            best_value = current
            best_epoch = epoch
            bad_epochs = 0
            row["selected_best"] = True
            best_state = copy.deepcopy(model.state_dict())
            if save_best_checkpoint:
                torch.save(
                    {"model_state": best_state, "epoch": epoch, "context": checkpoint_metadata},
                    best_path,
                )
        else:
            bad_epochs += 1
        history.append(row)
        if save_last_checkpoint:
            torch.save(
                {"model_state": model.state_dict(), "epoch": epoch, "context": checkpoint_metadata},
                last_path,
            )
        context.logger.debug(
            "Epoch %d/%d | train RMSE %.1f ps | val RMSE %.1f ps | %s | val bias %.1f ps",
            epoch,
            epochs,
            row["train_rmse_ps"],
            row["validation_rmse_ps"],
            ctr_log_text(validation_metrics),
            row["validation_bias_ps"],
        )
        if bad_epochs >= patience:
            context.logger.info("Early stopping after %d epochs without improvement", bad_epochs)
            break

    if best_state is None:
        raise RuntimeError("Training completed without a valid in-memory best state")
    model.load_state_dict(best_state)
    train_metrics, train_fit, _ = evaluate_model_with_optional_fit(
        model, train_eval_loader, device, config["fit"], "Best train residual",
        perform_fit=perform_internal_fit,
    )
    validation_metrics, validation_fit, _ = evaluate_model_with_optional_fit(
        model, validation_loader, device, config["fit"], "Best validation residual",
        perform_fit=perform_internal_fit,
    )

    baseline_guard_metric = config["training"].get("baseline_guard_metric")
    baseline_guard_applied = False
    if baseline_guard_metric is not None:
        baseline_guard_metric = str(baseline_guard_metric)
        metric_keys = {
            "validation_rmse": "rmse_ps",
            "validation_ctr": "ctr_ps",
        }
        if baseline_guard_metric not in metric_keys:
            raise ValueError(
                "training.baseline_guard_metric must be null, "
                "'validation_rmse', or 'validation_ctr'"
            )
        key = metric_keys[baseline_guard_metric]
        corrected_value = float(validation_metrics[key])
        baseline_value = float(baseline_validation_metrics[key])
        if corrected_value > baseline_value:
            context.logger.warning(
                "Learned correction is worse than uncorrected LED on %s "
                "(%.3f > %.3f); selecting zero correction",
                baseline_guard_metric,
                corrected_value,
                baseline_value,
            )
            _set_zero_correction(model)
            best_state = copy.deepcopy(model.state_dict())
            if save_best_checkpoint:
                torch.save(
                    {"model_state": best_state, "epoch": 0, "context": checkpoint_metadata},
                    best_path,
                )
            best_epoch = 0
            best_value = baseline_value
            baseline_guard_applied = True
            train_metrics, train_fit, _ = evaluate_model(
                model, train_eval_loader, device, config["fit"], "Selected uncorrected train LED"
            )
            validation_metrics, validation_fit, _ = evaluate_model(
                model, validation_loader, device, config["fit"], "Selected uncorrected validation LED"
            )

    # Exact antisymmetry is preserved through finalization: there is no additive
    # pair-output calibration after early stopping. Persist the selected state as-is.
    final_train_bias_ps = float(train_metrics["bias_ps"])
    best_state = copy.deepcopy(model.state_dict())
    if save_best_checkpoint:
        torch.save(
            {
                "model_state": best_state,
                "epoch": int(best_epoch),
                "context": checkpoint_metadata,
            },
            best_path,
        )

    if save_history:
        write_csv_rows(context.output_dir / "training_metrics.csv", history)
    if save_plots:
        context.plot_dir.mkdir(parents=True, exist_ok=True)
        dpi = int(config.get("plotting", {}).get("dpi", 180))
        plot_training_history(history, context.plot_dir, dpi)
        plot_gaussian_fit(train_fit, context.plot_dir / "best_train_gaussian_fit.png", dpi=dpi)
        plot_gaussian_fit(
            validation_fit,
            context.plot_dir / "best_validation_gaussian_fit.png",
            dpi=dpi,
        )

    summary = {
        "model_type": context.model_type,
        "model_name": context.model_name,
        "best_epoch": int(best_epoch),
        "best_selection_metric": selection_metric,
        "best_selection_value": float(best_value),
        "best_validation_rmse_ps": float(validation_metrics["rmse_ps"]),
        "best_validation_ctr_ps": float(validation_metrics["ctr_ps"]),
        "best_validation_bias_ps": float(validation_metrics["bias_ps"]),
        "uncorrected_led_validation_rmse_ps": float(baseline_validation_metrics["rmse_ps"]),
        "uncorrected_led_validation_ctr_ps": float(baseline_validation_metrics["ctr_ps"]),
        "uncorrected_led_validation_bias_ps": float(baseline_validation_metrics["bias_ps"]),
        "baseline_guard_metric": baseline_guard_metric,
        "baseline_guard_applied": baseline_guard_applied,
        "best_checkpoint": str(best_path.resolve()) if best_path.is_file() else "",
        "last_checkpoint": str(last_path.resolve()) if last_path.is_file() else "",
        "train_dir": str(context.output_dir.resolve()),
        "input_length": int(context.input_length),
        "input_transform": context.input_transform,
        "input_waveform_source": context.input_waveform_source,
        "prediction_target": context.prediction_target,
        "input_cache_paths": [str(path) for path in context.input_cache_dirs],
        "normalization": context.normalization.as_dict(),
        "training_datasets": [str(dataset.directory) for dataset in context.datasets],
        "optimizer": "Adam",
        "history_rows": len(history),
        "final_train_rmse_ps": float(train_metrics["rmse_ps"]),
        "final_train_bias_ps": float(train_metrics["bias_ps"]),
        "data_view": dict(context.data_view),
        "data_seed": int(config["training"].get("data_seed", config["training"].get("seed", 12345))),
        "initialization_seed": int(initialization_seed),
        "initial_state_hash": initial_state_hash,
        "initial_first_weight": float(first_parameter),
        "random_pair_swap": random_pair_swap,
        "random_pair_swap_probability": 0.5 if random_pair_swap else 0.0,
        "model_parameter_count": int(parameter_count),
        "cnn_architecture": {
            "channels": [int(value) for value in context.model_config["channels"]],
            "kernel_sizes": [int(value) for value in context.model_config["kernel_sizes"]],
            "strides": [int(value) for value in context.model_config["strides"]],
            "dilations": [
                int(value)
                for value in context.model_config.get(
                    "dilations", [1] * len(context.model_config["channels"])
                )
            ],
            "total_stride": int(model.shared.total_stride),
            "encoded_length": int(model.shared.encoded_length),
            "adaptive_pool_length": context.model_config.get("adaptive_pool_length"),
            "dense_units": [
                int(value) for value in context.model_config.get("dense_units", [])
            ],
        },
    }
    if save_summary:
        atomic_json(context.output_dir / "training_summary.json", summary)
    # Private in-process handles are intentionally added only after serialization.
    # CV callers use them to score the untouched fold without writing checkpoints.
    summary["_trained_model"] = model
    summary["_checkpoint_context"] = checkpoint_metadata
    return summary


MODEL_SPEC = ModelSpec(
    name="cnn_regressor",
    builder=build,
    validator=validate_config,
    training_validator=validate_training_config,
    trainer=train,
)
