from __future__ import annotations

import hashlib
import math
from typing import Any

import torch
from torch import nn

from utils.plots import plot_best_fit

from ..common import atomic_json, write_csv_rows
from ..losses import mse_bias_loss, mse_residual_loss
from ..plots import plot_training_history
from ..training_context import TrainingContext
from .spec import ModelSpec
from ..training_utils import (
    checkpoint_context,
    evaluate_model,
    evaluate_model_with_optional_fit,
    fit_schedule_for_epoch,
    make_split_loader,
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


def validate_config(config: dict[str, Any]) -> None:
    hidden_units = config.get("hidden_units", [])
    if not isinstance(hidden_units, list) or any(int(value) <= 0 for value in hidden_units):
        raise ValueError("hidden_units must be a list of positive integers")
    if config.get("activation", "relu") not in ("relu", "gelu", "silu", "tanh", "identity"):
        raise ValueError("Unsupported activation")
    dropout = float(config.get("dropout", 0.0))
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must lie in [0, 1)")
    bound = config.get("max_abs_single_channel_output_ps")
    if bound is not None and float(bound) <= 0.0:
        raise ValueError("max_abs_single_channel_output_ps must be positive when provided")


def validate_training_config(config: dict[str, Any]) -> None:
    optimizer = config.get("optimizer")
    if not isinstance(optimizer, dict):
        raise ValueError("MLP training requires an optimizer object")
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
    if loss_type not in ("mse", "mse_bias"):
        raise ValueError("model.loss.type must be one of ['mse', 'mse_bias']")
    if float(loss.get("bias_weight", 0.0)) < 0.0:
        raise ValueError("model.loss.bias_weight must be non-negative")


class SingleChannelMLP(nn.Module):
    def __init__(self, config: dict[str, Any], input_length: int) -> None:
        super().__init__()
        hidden_units = [int(value) for value in config.get("hidden_units", [])]
        activation = str(config.get("activation", "relu"))
        dropout = float(config.get("dropout", 0.0))
        batch_norm = bool(config.get("batch_norm", False))
        layers: list[nn.Module] = []
        in_features = int(input_length)
        for out_features in hidden_units:
            layers.append(nn.Linear(in_features, out_features, bias=not batch_norm))
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_features))
            layers.append(_activation(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_features = out_features
        layers.append(nn.Linear(in_features, 1))
        self.network = nn.Sequential(*layers)
        self.input_length = int(input_length)
        bound = config.get("max_abs_single_channel_output_ps")
        self.output_bound_ps = None if bound is None else float(bound)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 3 and waveform.shape[1] == 1:
            waveform = waveform[:, 0, :]
        if waveform.ndim != 2 or waveform.shape[1] != self.input_length:
            raise ValueError(f"Expected [batch, {self.input_length}] single-channel waveforms")
        output = self.network(waveform).squeeze(-1)
        if self.output_bound_ps is not None:
            output = self.output_bound_ps * torch.tanh(output / self.output_bound_ps)
        return output


class AntisymmetricMLPRegressor(nn.Module):
    """Shared single-channel MLP with correction ``g(w1) - g(w2)``."""

    def __init__(self, config: dict[str, Any], input_length: int) -> None:
        super().__init__()
        self.shared = SingleChannelMLP(config, input_length)
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
    return AntisymmetricMLPRegressor(config, input_length)


class _ZeroCorrectionModel(nn.Module):
    def forward(self, waveform_pair: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            waveform_pair.shape[0],
            dtype=waveform_pair.dtype,
            device=waveform_pair.device,
        )


def _set_zero_correction(model: AntisymmetricMLPRegressor) -> None:
    final_linear = next(
        layer for layer in reversed(model.shared.network) if isinstance(layer, nn.Linear)
    )
    with torch.no_grad():
        final_linear.weight.zero_()
        if final_linear.bias is not None:
            final_linear.bias.zero_()


def _target_scale_from_datasets(context: TrainingContext, minimum_scale: float) -> float:
    values: list[torch.Tensor] = []
    for dataset in context.datasets:
        indices = dataset.train
        led_delta = (
            torch.as_tensor(dataset.led_time_fs[indices, 0].copy(), dtype=torch.float64)
            - torch.as_tensor(dataset.led_time_fs[indices, 1].copy(), dtype=torch.float64)
        ) / 1000.0
        values.append(led_delta - float(dataset.true_tof_ps))
    target = torch.cat(values)
    return max(float(torch.std(target, unbiased=False).item()), minimum_scale)


def train(context: TrainingContext) -> dict[str, Any]:
    config = context.config
    artifacts = dict(config.get("artifacts", {}))
    save_history = bool(artifacts.get("save_history", True))
    save_plots = bool(artifacts.get("save_plots", True))
    save_last_checkpoint = bool(artifacts.get("save_last_checkpoint", True))
    save_summary = bool(artifacts.get("save_summary", True))

    device = resolve_device(config["training"].get("device", "auto"))
    train_loader = make_split_loader(
        context.datasets, "train", context.normalization, config, device, shuffle=True
    )
    train_eval_loader = make_split_loader(
        context.datasets, "train", context.normalization, config, device, shuffle=False
    )
    validation_loader = make_split_loader(
        context.datasets, "validation", context.normalization, config, device, shuffle=False
    )
    zero_model = _ZeroCorrectionModel().to(device)
    baseline_train_metrics, _baseline_train_fit, _ = evaluate_model(
        zero_model, train_eval_loader, device, config["fit"], "Uncorrected train LED"
    )
    baseline_validation_metrics, _baseline_validation_fit, _ = evaluate_model(
        zero_model, validation_loader, device, config["fit"], "Uncorrected validation LED"
    )
    context.logger.info(
        "Uncorrected LED baseline | train RMSE %.3f ps CTR %.3f ps | "
        "validation RMSE %.3f ps CTR %.3f ps bias %.3f ps",
        baseline_train_metrics["rmse_ps"],
        baseline_train_metrics["ctr_ps"],
        baseline_validation_metrics["rmse_ps"],
        baseline_validation_metrics["ctr_ps"],
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
    initial_state_hash = model_state_hash(model)
    first_parameter = next(model.parameters()).detach().reshape(-1)[0].item()
    context.logger.info(
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
    checkpoint_metadata = checkpoint_context(
        context,
        training_strategy="Adam optimization of antisymmetric pairwise residual objective",
    )

    loss_config = dict(config.get("model", {}).get("loss", {}))
    loss_type = str(loss_config.get("type", "mse"))
    bias_weight = float(loss_config.get("bias_weight", 0.0)) if loss_type == "mse_bias" else 0.0
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
        context.logger.info(
            "Random ordered-pair swapping enabled for training batches (probability 0.5)"
        )

    context.logger.info("Training %s with Adam on %s", context.model_name, device)
    for epoch in range(1, epochs + 1):
        model.train()
        for waveforms, target, led_delta, cfd_delta, true_tof in train_loader:
            if random_pair_swap:
                waveforms, target, led_delta, cfd_delta, true_tof = randomly_swap_paired_batch(
                    waveforms,
                    target,
                    led_delta,
                    cfd_delta,
                    true_tof,
                    generator=pair_swap_generator,
                    probability=0.5,
                )
            waveforms = waveforms.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                prediction = model(waveforms)
                if loss_type == "mse_bias":
                    loss, _penalty = mse_bias_loss(
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
        train_metrics, _train_fit, _ = evaluate_model_with_optional_fit(
            model,
            train_eval_loader,
            device,
            config["fit"],
            "Train residual",
            perform_fit=fit_train,
        )
        validation_metrics, _validation_fit, _ = evaluate_model_with_optional_fit(
            model,
            validation_loader,
            device,
            config["fit"],
            "Validation residual",
            perform_fit=fit_validation,
        )
        validation_bias = float(validation_metrics["bias_ps"])
        validation_loss = float(validation_metrics["rmse_ps"] ** 2)
        if loss_type == "mse_bias":
            validation_loss += bias_weight * (validation_bias / target_scale) ** 2
        row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_rmse_ps": float(train_metrics["rmse_ps"]),
            "validation_rmse_ps": float(validation_metrics["rmse_ps"]),
            "train_ctr_ps": float(train_metrics["ctr_ps"]),
            "validation_ctr_ps": float(validation_metrics["ctr_ps"]),
            "train_bias_ps": float(train_metrics["bias_ps"]),
            "validation_bias_ps": validation_bias,
            "train_loss": float(train_metrics["rmse_ps"] ** 2),
            "validation_loss": validation_loss,
            "bias_penalty": float(
                bias_weight * (validation_bias / target_scale) ** 2
                if loss_type == "mse_bias"
                else 0.0
            ),
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
            torch.save(
                {"model_state": model.state_dict(), "epoch": epoch, "context": checkpoint_metadata},
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
        context.logger.info(
            "Epoch %d/%d | train RMSE %.3f ps | val RMSE %.3f ps | val CTR %s | val bias %.3f ps",
            epoch,
            epochs,
            row["train_rmse_ps"],
            row["validation_rmse_ps"],
            (f"{row['validation_ctr_ps']:.3f} ps" if math.isfinite(row["validation_ctr_ps"]) else "not fitted"),
            row["validation_bias_ps"],
        )
        if bad_epochs >= patience:
            context.logger.info("Early stopping after %d epochs without improvement", bad_epochs)
            break

    if not best_path.is_file():
        raise RuntimeError("Training completed without a valid checkpoint")
    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state"])
    train_metrics, train_fit, _ = evaluate_model(
        model, train_eval_loader, device, config["fit"], "Best train residual"
    )
    validation_metrics, validation_fit, _ = evaluate_model(
        model, validation_loader, device, config["fit"], "Best validation residual"
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
            torch.save(
                {"model_state": model.state_dict(), "epoch": 0, "context": checkpoint_metadata},
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

    if save_history:
        write_csv_rows(context.output_dir / "training_metrics.csv", history)
    if save_plots:
        context.plot_dir.mkdir(parents=True, exist_ok=True)
        dpi = int(config.get("plotting", {}).get("dpi", 180))
        plot_training_history(history, context.plot_dir, dpi)
        plot_best_fit(train_fit, context.plot_dir / "best_train_gaussian_fit.png", dpi=dpi)
        plot_best_fit(
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
        "best_checkpoint": str(best_path.resolve()),
        "last_checkpoint": str(last_path.resolve()) if last_path.is_file() else "",
        "train_dir": str(context.output_dir.resolve()),
        "input_length": int(context.input_length),
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
        "model_parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
    }
    if save_summary:
        atomic_json(context.output_dir / "training_summary.json", summary)
    return summary


MODEL_SPEC = ModelSpec(
    name="mlp_regressor",
    builder=build,
    validator=validate_config,
    training_validator=validate_training_config,
    trainer=train,
)
