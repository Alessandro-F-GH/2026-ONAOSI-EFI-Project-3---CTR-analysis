from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from utils.plots import plot_best_fit

from ..common import atomic_json, write_csv_rows
from ..losses import mse_residual_loss, var_bias_loss, var_bias_value_from_metrics
from ..plots import plot_training_history
from ..training_context import TrainingContext
from ..torch_data import factored_correction_target_ps
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
from .spec import ModelSpec

_TRAINED_UNITS_KEY = "_trained_units"


def _model_state_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def validate_config(config: dict[str, Any]) -> None:
    activation = str(config.get("activation", "silu")).lower()
    if activation not in {"silu", "tanh", "gelu", "relu", "identity"}:
        raise ValueError("constructive_mlp_encoder activation must be silu, tanh, gelu, relu or identity")
    max_units = int(config.get("max_units", 16))
    if max_units <= 0:
        raise ValueError("max_units must be positive")
    trained_units = config.get(_TRAINED_UNITS_KEY)
    if trained_units is not None:
        trained_units = int(trained_units)
        if trained_units <= 0 or trained_units > max_units:
            raise ValueError(f"{_TRAINED_UNITS_KEY} must lie in [1, max_units]")
    unit_bias = config.get("unit_bias", True)
    if not isinstance(unit_bias, bool):
        raise ValueError("unit_bias must be boolean")
    bound = config.get("max_abs_single_channel_output_ps")
    if bound is not None and float(bound) <= 0.0:
        raise ValueError(
            "max_abs_single_channel_output_ps must be positive when provided"
        )


def validate_training_config(config: dict[str, Any]) -> None:
    optimizer = config.get("optimizer")
    if not isinstance(optimizer, dict):
        raise ValueError("Constructive training requires an optimizer object")
    if float(optimizer.get("learning_rate", 0.0)) <= 0.0:
        raise ValueError("optimizer.learning_rate must be positive")
    if float(optimizer.get("weight_decay", 0.0)) < 0.0:
        raise ValueError("optimizer.weight_decay must be non-negative")

    training = config["training"]
    validate_fit_schedule(training)
    for name in ("epochs_per_unit", "batch_size", "normalization_chunk_size"):
        if int(training.get(name, 0)) <= 0:
            raise ValueError(f"training.{name} must be positive")
    patience = int(
        training.get("unit_early_stopping_patience", training["epochs_per_unit"])
    )
    if patience <= 0:
        raise ValueError("training.unit_early_stopping_patience must be positive")
    if float(training.get("unit_early_stopping_min_delta_ps", 0.0)) < 0.0:
        raise ValueError(
            "training.unit_early_stopping_min_delta_ps must be non-negative"
        )
    if float(training.get("min_unit_improvement_ps", 0.0)) < 0.0:
        raise ValueError("training.min_unit_improvement_ps must be non-negative")
    relative = float(training.get("min_relative_unit_improvement", 0.0))
    if relative < 0.0:
        raise ValueError(
            "training.min_relative_unit_improvement must be non-negative"
        )
    random_pair_swap = training.get("random_pair_swap", False)
    if not isinstance(random_pair_swap, bool):
        raise ValueError("training.random_pair_swap must be boolean")

    loss = config.get("model", {}).get("loss", {})
    if not isinstance(loss, dict):
        raise ValueError("model.loss must be an object when provided")
    loss_type = str(loss.get("type", "mse"))
    if loss_type not in {"mse", "var_bias"}:
        raise ValueError("model.loss.type must be 'mse' or 'var_bias'")
    if float(loss.get("bias_weight", 0.0)) < 0.0:
        raise ValueError("model.loss.bias_weight must be non-negative")


class ConstructiveUnit(nn.Module):
    """One scalar nonlinear unit fed by raw input and all frozen units."""

    def __init__(
        self,
        input_length: int,
        previous_units: int,
        *,
        bias: bool,
        activation: str,
    ) -> None:
        super().__init__()
        self.raw_input_length = int(input_length)
        self.previous_units = int(previous_units)
        self.linear = nn.Linear(
            self.raw_input_length + self.previous_units,
            1,
            bias=bias,
        )
        key = str(activation).lower()
        activations: dict[str, nn.Module] = {
            "silu": nn.SiLU(),
            "tanh": nn.Tanh(),
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "identity": nn.Identity(),
        }
        self.activation_name = key
        self.activation = activations[key]

    def forward(
        self,
        waveform: torch.Tensor,
        previous_hidden: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.previous_units == 0:
            values = waveform
        else:
            if previous_hidden is None or previous_hidden.shape[1] != self.previous_units:
                raise ValueError("Incorrect number of frozen hidden-unit inputs")
            values = torch.cat([waveform, previous_hidden], dim=1)
        return self.activation(self.linear(values))


class ConstructiveEncoder(nn.Module):
    def __init__(
        self,
        input_length: int,
        unit_count: int,
        *,
        unit_bias: bool,
        activation: str,
    ) -> None:
        super().__init__()
        self.input_length = int(input_length)
        self.unit_bias = bool(unit_bias)
        self.activation_name = str(activation).lower()
        self.units = nn.ModuleList(
            [
                ConstructiveUnit(
                    self.input_length,
                    previous_units=index,
                    bias=self.unit_bias,
                    activation=self.activation_name,
                )
                for index in range(int(unit_count))
            ]
        )

    @property
    def unit_count(self) -> int:
        return len(self.units)

    def add_unit(self, *, seed: int, device: torch.device) -> None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        unit = ConstructiveUnit(
            self.input_length,
            previous_units=self.unit_count,
            bias=self.unit_bias,
            activation=self.activation_name,
        ).to(device)
        self.units.append(unit)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 3 and waveform.shape[1] == 1:
            waveform = waveform[:, 0, :]
        if waveform.ndim != 2 or int(waveform.shape[1]) != self.input_length:
            raise ValueError(
                f"Expected single-channel waveforms with shape [batch, {self.input_length}]"
            )
        hidden: list[torch.Tensor] = []
        for unit in self.units:
            previous = torch.cat(hidden, dim=1) if hidden else None
            hidden.append(unit(waveform, previous))
        if not hidden:
            return waveform.new_empty((waveform.shape[0], 0))
        return torch.cat(hidden, dim=1)



class AntisymmetricConstructiveMLPEncoder(nn.Module):
    """Shared constructive encoder with prediction g(s1)-g(s2)."""

    def __init__(self, config: dict[str, Any], input_length: int) -> None:
        super().__init__()
        self.input_length = int(input_length)
        self.unit_bias = bool(config.get("unit_bias", True))
        self.activation_name = str(config.get("activation", "silu")).lower()
        unit_count = int(config.get(_TRAINED_UNITS_KEY, config.get("initial_units", 1)))
        self.encoder = ConstructiveEncoder(
            self.input_length,
            unit_count,
            unit_bias=self.unit_bias,
            activation=self.activation_name,
        )
        self.output_weights = nn.ParameterList(
            [nn.Parameter(torch.tensor(1.0, dtype=torch.float32)) for _ in range(unit_count)]
        )
        bound = config.get("max_abs_single_channel_output_ps")
        self.output_bound_ps = None if bound is None else float(bound)

    @property
    def unit_count(self) -> int:
        return self.encoder.unit_count

    def add_unit(self, *, seed: int, device: torch.device) -> None:
        self.encoder.add_unit(seed=seed, device=device)
        # Start every added branch as an exact zero residual so the previously
        # accepted predictor is unchanged, while keeping non-zero gradients for
        # the new unit's input weights.
        with torch.no_grad():
            self.encoder.units[-1].linear.weight.zero_()
            if self.encoder.units[-1].linear.bias is not None:
                self.encoder.units[-1].linear.bias.zero_()
        coefficient = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device=device))
        self.output_weights.append(coefficient)

    def freeze_all(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def train_only_last_unit(self) -> None:
        if self.unit_count <= 0:
            raise RuntimeError("Cannot train an empty constructive model")
        self.freeze_all()
        for parameter in self.encoder.units[-1].parameters():
            parameter.requires_grad_(True)
        self.output_weights[-1].requires_grad_(True)

    def encode_single(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.encoder(waveform)

    def encode_pair(self, waveform_pair: torch.Tensor) -> torch.Tensor:
        if waveform_pair.ndim != 3 or int(waveform_pair.shape[1]) != 2:
            raise ValueError("Expected waveform pairs with shape [batch, 2, length]")
        if int(waveform_pair.shape[2]) != self.input_length:
            raise ValueError(f"Expected waveform length {self.input_length}")
        batch = int(waveform_pair.shape[0])
        hidden = self.encode_single(
            waveform_pair.reshape(batch * 2, self.input_length)
        )
        return hidden.reshape(batch, 2, self.unit_count)

    def single_channel_prediction(self, waveform: torch.Tensor) -> torch.Tensor:
        hidden = self.encode_single(waveform)
        if self.unit_count == 0:
            output = waveform.new_zeros(waveform.shape[0])
        else:
            coefficients = torch.stack(list(self.output_weights))
            output = hidden @ coefficients
        if self.output_bound_ps is not None:
            output = self.output_bound_ps * torch.tanh(output / self.output_bound_ps)
        return output

    def forward(self, waveform_pair: torch.Tensor) -> torch.Tensor:
        hidden = self.encode_pair(waveform_pair)
        coefficients = torch.stack(list(self.output_weights))
        single = hidden @ coefficients
        if self.output_bound_ps is not None:
            single = self.output_bound_ps * torch.tanh(single / self.output_bound_ps)
        return single[:, 0] - single[:, 1]



def build(config: dict[str, Any], input_length: int) -> nn.Module:
    return AntisymmetricConstructiveMLPEncoder(config, input_length)


def count_complexity(config: dict[str, Any], input_length: int) -> int:
    units = int(config.get(_TRAINED_UNITS_KEY, config.get("max_units", 16)))
    bias_count = 1 if bool(config.get("unit_bias", True)) else 0
    encoder_parameters = sum(
        int(input_length) + previous + bias_count for previous in range(units)
    )
    return int(encoder_parameters + units)


def _target_scale_from_datasets(context: TrainingContext, minimum_scale: float) -> float:
    values: list[np.ndarray] = []
    for dataset in context.datasets:
        indices = np.asarray(dataset.train, dtype=np.int64)
        led_delta = (
            np.asarray(dataset.led_time_fs[indices, 0], dtype=np.float64)
            - np.asarray(dataset.led_time_fs[indices, 1], dtype=np.float64)
        ) / 1000.0
        values.append(led_delta - float(dataset.true_tof_ps))
    target = np.concatenate(values)
    return max(float(np.std(target)), minimum_scale)


def _checkpoint_model_config(
    base_config: dict[str, Any], unit_count: int
) -> dict[str, Any]:
    value = copy.deepcopy(base_config)
    value[_TRAINED_UNITS_KEY] = int(unit_count)
    return value


def _selection_value(metrics: dict[str, Any], loss_value: float, name: str) -> float:
    values = {
        "validation_rmse": float(metrics["rmse_ps"]),
        "validation_loss": float(loss_value),
        "absolute_validation_bias": abs(float(metrics["bias_ps"])),
        "validation_ctr": float(metrics["ctr_ps"]),
    }
    if name not in values:
        raise ValueError(
            f"Unsupported training.selection_metric {name!r}; available: {sorted(values)}"
        )
    return values[name]


def _save_encoder_artifacts(
    model: AntisymmetricConstructiveMLPEncoder,
    output_dir: Path,
) -> dict[str, str]:
    """Persist compact structural information for the selected nonlinear model."""
    output_weights = np.asarray(
        [float(value.detach().cpu().item()) for value in model.output_weights],
        dtype=np.float64,
    )
    np.save(output_dir / "encoder_output_weights.npy", output_weights)
    rows = []
    for index, unit in enumerate(model.encoder.units):
        direct = unit.linear.weight.detach().cpu().numpy().reshape(-1).astype(np.float64)
        rows.append(
            {
                "unit_index": index,
                "raw_input_weight_l2": float(np.linalg.norm(direct[: model.input_length])),
                "previous_unit_weight_l2": float(np.linalg.norm(direct[model.input_length :])),
                "output_weight": float(output_weights[index]),
                "absolute_output_weight": float(abs(output_weights[index])),
                "activation": model.activation_name,
            }
        )
    write_csv_rows(output_dir / "encoder_units.csv", rows)
    return {
        "encoder_output_weights": str((output_dir / "encoder_output_weights.npy").resolve()),
        "encoder_units": str((output_dir / "encoder_units.csv").resolve()),
    }


def _plot_growth(rows: list[dict[str, Any]], destination: Path, dpi: int) -> None:
    if not rows:
        return
    units = [int(row["unit_count"]) for row in rows]
    train_rmse = [float(row["train_rmse_ps"]) for row in rows]
    validation_rmse = [float(row["validation_rmse_ps"]) for row in rows]
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    axis.plot(units, train_rmse, marker="o", label="Train RMSE")
    axis.plot(units, validation_rmse, marker="o", label="Validation RMSE")
    axis.set_xlabel("Accepted constructive units")
    axis.set_ylabel("RMSE [ps]")
    axis.set_title("Constructive encoder growth")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def train(context: TrainingContext) -> dict[str, Any]:
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
        context.datasets,
        "validation",
        context.normalization,
        config,
        device,
        shuffle=False,
        subsampling_factor=context.subsampling_factor,
    )

    max_units = int(model_config.get("max_units", 16))
    epochs_per_unit = int(config["training"]["epochs_per_unit"])
    patience = int(
        config["training"].get("unit_early_stopping_patience", epochs_per_unit)
    )
    stage_min_delta = float(
        config["training"].get("unit_early_stopping_min_delta_ps", 0.0)
    )
    minimum_improvement = float(
        config["training"].get("min_unit_improvement_ps", 0.1)
    )
    minimum_relative = float(
        config["training"].get("min_relative_unit_improvement", 0.0)
    )
    selection_metric = str(
        config["training"].get("selection_metric", "validation_rmse")
    )
    gradient_clip = config["training"].get("gradient_clip_norm")
    initialization_seed = int(
        config["training"].get(
            "initialization_seed", config["training"].get("seed", 12345)
        )
    )
    random_pair_swap = bool(config["training"].get("random_pair_swap", False))
    pair_swap_generator = torch.Generator(device="cpu")
    pair_swap_generator.manual_seed(
        int(config["training"].get("data_seed", config["training"].get("seed", 12345)))
    )

    loss_config = dict(config.get("model", {}).get("loss", {}))
    loss_type = str(loss_config.get("type", "mse"))
    bias_weight = (
        float(loss_config.get("bias_weight", 0.0))
        if loss_type == "var_bias"
        else 0.0
    )
    bias_norm = str(loss_config.get("bias_normalization", "target_std"))
    minimum_scale = float(loss_config.get("minimum_scale", 1e-8))
    target_scale = (
        _target_scale_from_datasets(context, minimum_scale)
        if bias_norm == "target_std"
        else 1.0
    )

    torch.manual_seed(initialization_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(initialization_seed)
    model = AntisymmetricConstructiveMLPEncoder(
        {**model_config, _TRAINED_UNITS_KEY: 1}, context.input_length
    ).to(device)
    model.output_weights[0].data.fill_(1.0)
    initial_state_hash = _model_state_hash(model)
    initial_first_weight = float(
        next(model.parameters()).detach().reshape(-1)[0].cpu().item()
    )

    best_path = context.checkpoint_dir / "best.pt"
    last_path = context.checkpoint_dir / "last.pt"
    history: list[dict[str, Any]] = []
    growth_rows: list[dict[str, Any]] = []
    accepted_state: dict[str, torch.Tensor] | None = None
    accepted_metric = math.inf
    accepted_rmse = math.inf
    accepted_units = 0
    accepted_global_epoch = 0
    global_epoch = 0
    stop_reason = "maximum_units_reached"

    amp_enabled = (
        bool(config["training"].get("mixed_precision", False))
        and device.type == "cuda"
    )

    context.logger.info(
        "Constructive nonlinear encoder | max units %d | epochs/unit %d | "
        "minimum validation RMSE improvement %.6g ps",
        max_units,
        epochs_per_unit,
        minimum_improvement,
    )

    for unit_index in range(max_units):
        if unit_index > 0:
            model.add_unit(
                seed=initialization_seed + unit_index,
                device=device,
            )
        model.train_only_last_unit()
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.Adam(
            trainable,
            lr=float(config["optimizer"]["learning_rate"]),
            weight_decay=float(config["optimizer"].get("weight_decay", 0.0)),
        )
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        except AttributeError:  # pragma: no cover
            scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

        stage_best_metric = math.inf
        stage_best_rmse = math.inf
        stage_best_state: dict[str, torch.Tensor] | None = None
        stage_best_epoch = 0
        bad_epochs = 0

        context.logger.debug(
            "Training constructive unit %d/%d with %d trainable parameters",
            unit_index + 1,
            max_units,
            sum(parameter.numel() for parameter in trainable),
        )
        for unit_epoch in range(1, epochs_per_unit + 1):
            global_epoch += 1
            model.train()
            batch_losses: list[float] = []
            for waveforms, target, led_delta, cfd_delta, true_tof, anchor_shift in train_loader:
                if random_pair_swap:
                    (
                        waveforms,
                        target,
                        led_delta,
                        cfd_delta,
                        true_tof,
                        anchor_shift,
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
                    torch.nn.utils.clip_grad_norm_(trainable, float(gradient_clip))
                scaler.step(optimizer)
                scaler.update()
                batch_losses.append(float(loss.detach().cpu().item()))

            fit_train, fit_validation = fit_schedule_for_epoch(
                config["training"], global_epoch, selection_metric=selection_metric
            )
            train_metrics, _train_fit, _ = evaluate_model_with_optional_fit(
                model,
                train_eval_loader,
                device,
                config["fit"],
                "Constructive train residual",
                perform_fit=fit_train,
            )
            validation_metrics, _validation_fit, _ = evaluate_model_with_optional_fit(
                model,
                validation_loader,
                device,
                config["fit"],
                "Constructive validation residual",
                perform_fit=fit_validation,
            )
            validation_bias = float(validation_metrics["bias_ps"])
            if loss_type == "var_bias":
                validation_loss, validation_variance, validation_bias_penalty = (
                    var_bias_value_from_metrics(
                        rmse_ps=float(validation_metrics["rmse_ps"]),
                        bias_ps=validation_bias,
                        bias_weight=bias_weight,
                        target_scale=target_scale,
                    )
                )
            else:
                validation_loss = float(validation_metrics["rmse_ps"] ** 2)
                validation_variance = max(
                    validation_loss - validation_bias**2, 0.0
                )
                validation_bias_penalty = 0.0
            current = _selection_value(
                validation_metrics, validation_loss, selection_metric
            )
            selected = False
            if math.isfinite(current) and current < stage_best_metric - stage_min_delta:
                stage_best_metric = current
                stage_best_rmse = float(validation_metrics["rmse_ps"])
                stage_best_state = copy.deepcopy(model.state_dict())
                stage_best_epoch = unit_epoch
                bad_epochs = 0
                selected = True
            else:
                bad_epochs += 1
            row = {
                "epoch": global_epoch,
                "unit_count": unit_index + 1,
                "unit_epoch": unit_epoch,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train_rmse_ps": float(train_metrics["rmse_ps"]),
                "validation_rmse_ps": float(validation_metrics["rmse_ps"]),
                "train_ctr_ps": float(train_metrics["ctr_ps"]),
                "validation_ctr_ps": float(validation_metrics["ctr_ps"]),
                "train_bias_ps": float(train_metrics["bias_ps"]),
                "validation_bias_ps": validation_bias,
                "validation_variance_ps2": float(validation_variance),
                "train_loss": (
                    float(np.mean(batch_losses)) if batch_losses else math.nan
                ),
                "validation_loss": float(validation_loss),
                "bias_penalty": float(validation_bias_penalty),
                "train_fit_performed": bool(train_metrics["fit_performed"]),
                "validation_fit_performed": bool(validation_metrics["fit_performed"]),
                "selected_best": selected,
            }
            history.append(row)
            context.logger.debug(
                "Unit %d | epoch %d/%d | train RMSE %.3f ps | val RMSE %.3f ps | "
                "val CTR %.3f ps | val bias %.3f ps",
                unit_index + 1,
                unit_epoch,
                epochs_per_unit,
                row["train_rmse_ps"],
                row["validation_rmse_ps"],
                row["validation_ctr_ps"],
                row["validation_bias_ps"],
            )
            if bad_epochs >= patience:
                context.logger.debug(
                    "Unit %d early stopping after %d epochs without improvement",
                    unit_index + 1,
                    bad_epochs,
                )
                break

        if stage_best_state is None:
            raise RuntimeError(
                f"Constructive unit {unit_index + 1} produced no finite validation metric"
            )
        model.load_state_dict(stage_best_state)
        train_metrics, train_fit, _ = evaluate_model_with_optional_fit(
            model,
            train_eval_loader,
            device,
            config["fit"],
            f"Constructive {unit_index + 1}-unit train residual",
            perform_fit=perform_internal_fit,
        )
        validation_metrics, validation_fit, _ = evaluate_model_with_optional_fit(
            model,
            validation_loader,
            device,
            config["fit"],
            f"Constructive {unit_index + 1}-unit validation residual",
            perform_fit=perform_internal_fit,
        )

        improvement = (
            math.inf if accepted_units == 0 else accepted_rmse - stage_best_rmse
        )
        required_improvement = max(
            minimum_improvement,
            0.0 if accepted_units == 0 else accepted_rmse * minimum_relative,
        )
        accepted = accepted_units == 0 or improvement >= required_improvement
        growth_row = {
            "unit_count": unit_index + 1,
            "accepted": bool(accepted),
            "best_unit_epoch": int(stage_best_epoch),
            "train_rmse_ps": float(train_metrics["rmse_ps"]),
            "validation_rmse_ps": float(validation_metrics["rmse_ps"]),
            "validation_ctr_ps": float(validation_metrics["ctr_ps"]),
            "validation_bias_ps": float(validation_metrics["bias_ps"]),
            "selection_metric": selection_metric,
            "selection_value": float(stage_best_metric),
            "rmse_improvement_ps": (
                "" if accepted_units == 0 else float(improvement)
            ),
            "required_rmse_improvement_ps": (
                "" if accepted_units == 0 else float(required_improvement)
            ),
        }
        growth_rows.append(growth_row)

        if not accepted:
            stop_reason = "marginal_validation_improvement"
            context.logger.info(
                "Rejected constructive unit %d: validation RMSE improvement %.6g ps "
                "is below required %.6g ps",
                unit_index + 1,
                improvement,
                required_improvement,
            )
            if accepted_state is None:
                raise RuntimeError("Internal error: no accepted constructive state")
            accepted_config = _checkpoint_model_config(
                model_config, accepted_units
            )
            model = build(accepted_config, context.input_length).to(device)
            assert isinstance(model, AntisymmetricConstructiveMLPEncoder)
            model.load_state_dict(accepted_state)
            break

        accepted_units = unit_index + 1
        accepted_metric = stage_best_metric
        accepted_rmse = stage_best_rmse
        accepted_global_epoch = global_epoch - (unit_epoch - stage_best_epoch)
        accepted_state = copy.deepcopy(model.state_dict())
        model.freeze_all()
        accepted_config = _checkpoint_model_config(model_config, accepted_units)
        metadata = checkpoint_context(
            context,
            model_config=accepted_config,
            training_strategy=(
                "Greedy constructive nonlinear encoder: train one new unit on the full "
                "waveform plus all frozen hidden units, freeze it, and stop when early-stop "
                "validation improvement is marginal"
            ),
        )
        payload = {
            "model_state": accepted_state,
            "epoch": int(accepted_global_epoch),
            "unit_count": int(accepted_units),
            "context": metadata,
        }
        if save_best_checkpoint:
            torch.save(payload, best_path)
        if save_last_checkpoint:
            torch.save(payload, last_path)
        context.logger.info(
            "Accepted constructive unit %d | validation RMSE %.3f ps%s",
            accepted_units,
            accepted_rmse,
            ""
            if accepted_units == 1
            else f" | improvement {improvement:.3f} ps",
        )

    if accepted_state is None:
        raise RuntimeError("Constructive training completed without an accepted unit")
    accepted_config = _checkpoint_model_config(model_config, accepted_units)
    model = build(accepted_config, context.input_length).to(device)
    assert isinstance(model, AntisymmetricConstructiveMLPEncoder)
    model.load_state_dict(accepted_state)
    model.freeze_all()
    train_metrics, train_fit, _ = evaluate_model_with_optional_fit(
        model, train_eval_loader, device, config["fit"], "Best constructive train residual",
        perform_fit=perform_internal_fit,
    )
    validation_metrics, validation_fit, _ = evaluate_model_with_optional_fit(
        model,
        validation_loader,
        device,
        config["fit"],
        "Best constructive validation residual",
        perform_fit=perform_internal_fit,
    )

    if save_history:
        write_csv_rows(context.output_dir / "training_metrics.csv", history)
    if save_model_artifacts:
        write_csv_rows(context.output_dir / "constructive_growth.csv", growth_rows)
    dpi = int(config.get("plotting", {}).get("dpi", 180))
    if save_plots:
        context.plot_dir.mkdir(parents=True, exist_ok=True)
        plot_training_history(history, context.plot_dir, dpi)
        _plot_growth(
            [row for row in growth_rows if bool(row["accepted"])],
            context.plot_dir / "constructive_growth.png",
            dpi,
        )
        plot_best_fit(
            train_fit,
            context.plot_dir / "best_train_gaussian_fit.png",
            dpi=dpi,
        )
        plot_best_fit(
            validation_fit,
            context.plot_dir / "best_validation_gaussian_fit.png",
            dpi=dpi,
        )
    encoder_artifacts = (
        _save_encoder_artifacts(model, context.output_dir) if save_model_artifacts else {}
    )

    summary = {
        "model_type": context.model_type,
        "model_name": context.model_name,
        "best_epoch": int(accepted_global_epoch),
        "best_selection_metric": selection_metric,
        "best_selection_value": float(accepted_metric),
        "best_validation_rmse_ps": float(validation_metrics["rmse_ps"]),
        "best_validation_ctr_ps": float(validation_metrics["ctr_ps"]),
        "best_validation_bias_ps": float(validation_metrics["bias_ps"]),
        "best_checkpoint": str(best_path.resolve()) if best_path.is_file() else "",
        "last_checkpoint": str(last_path.resolve()) if last_path.is_file() else "",
        "train_dir": str(context.output_dir.resolve()),
        "input_length": int(context.input_length),
        "input_transform": context.input_transform,
        "input_cache_paths": [str(path) for path in context.input_cache_dirs],
        "encoded_dimension": int(accepted_units),
        "accepted_units": int(accepted_units),
        "attempted_units": int(len(growth_rows)),
        "maximum_units": int(max_units),
        "stop_reason": stop_reason,
        "minimum_unit_improvement_ps": float(minimum_improvement),
        "minimum_relative_unit_improvement": float(minimum_relative),
        "activation": model.activation_name,
        "normalization": context.normalization.as_dict(),
        "training_datasets": [str(dataset.directory) for dataset in context.datasets],
        "optimizer": "Adam",
        "history_rows": len(history),
        "final_train_rmse_ps": float(train_metrics["rmse_ps"]),
        "final_train_bias_ps": float(train_metrics["bias_ps"]),
        "data_view": dict(context.data_view),
        "data_seed": int(
            config["training"].get("data_seed", config["training"].get("seed", 12345))
        ),
        "initialization_seed": int(initialization_seed),
        "initial_state_hash": initial_state_hash,
        "initial_first_weight": float(initial_first_weight),
        "random_pair_swap": random_pair_swap,
        "random_pair_swap_probability": 0.5 if random_pair_swap else 0.0,
        "model_parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "objective": (
            "Residual variance plus normalized squared-bias penalty"
            if loss_type == "var_bias"
            else "Direct MSE of g(s1)-g(s2) against LED_delta-true_TOF"
        ),
        "architecture_note": (
            "Each accepted unit receives the full normalized waveform plus all previously "
            "accepted frozen units; only the new unit and its output coefficient are trained."
        ),
        "artifacts": {
            **encoder_artifacts,
            **(
                {"constructive_growth": str((context.output_dir / "constructive_growth.csv").resolve())}
                if save_model_artifacts
                else {}
            ),
        },
    }
    if save_summary:
        atomic_json(context.output_dir / "training_summary.json", summary)
    summary["_trained_model"] = model
    summary["_checkpoint_context"] = checkpoint_context(
        context,
        model_config=accepted_config,
        training_strategy=(
            "Greedy constructive nonlinear encoder: train one new unit on the full waveform "
            "plus all frozen hidden units, then freeze accepted units"
        ),
    )
    return summary


MODEL_SPEC = ModelSpec(
    name="constructive_mlp_encoder",
    builder=build,
    validator=validate_config,
    training_validator=validate_training_config,
    trainer=train,
    complexity_counter=count_complexity,
)
