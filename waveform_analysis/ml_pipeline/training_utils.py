from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from utils.fit import FitResult

from .dataset import PreparedDataset
from .metrics import distribution_metrics, fit_times_ps
from .torch_data import CorrectionDataset, Normalization
from .training_context import TrainingContext


def resolve_device(requested: str) -> torch.device:
    value = str(requested).lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def randomly_swap_paired_batch(
    waveforms: torch.Tensor,
    target: torch.Tensor,
    led_delta: torch.Tensor,
    cfd_delta: torch.Tensor,
    true_tof: torch.Tensor,
    *,
    generator: torch.Generator,
    probability: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Randomly reverse ordered detector pairs while preserving supervision.

    Swapping ``(s1, s2)`` to ``(s2, s1)`` reverses every ordered time
    difference. Therefore the LED/CFD differences, true TOF and correction
    target must all change sign for the selected events.
    """

    if waveforms.ndim != 3 or waveforms.shape[1] != 2:
        raise ValueError("Expected waveform pairs with shape [batch, 2, length]")
    batch_size = int(waveforms.shape[0])
    for name, values in (
        ("target", target),
        ("led_delta", led_delta),
        ("cfd_delta", cfd_delta),
        ("true_tof", true_tof),
    ):
        if values.ndim != 1 or int(values.shape[0]) != batch_size:
            raise ValueError(f"{name} must have shape [batch]")

    probability = float(probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("Pair-swap probability must lie in [0, 1]")
    if batch_size == 0 or probability == 0.0:
        return waveforms, target, led_delta, cfd_delta, true_tof

    swap_mask = torch.rand(batch_size, generator=generator) < probability
    if not bool(torch.any(swap_mask)):
        return waveforms, target, led_delta, cfd_delta, true_tof

    waveforms = waveforms.clone()
    target = target.clone()
    led_delta = led_delta.clone()
    cfd_delta = cfd_delta.clone()
    true_tof = true_tof.clone()

    waveforms[swap_mask] = waveforms[swap_mask][:, [1, 0], :]
    target[swap_mask] = -target[swap_mask]
    led_delta[swap_mask] = -led_delta[swap_mask]
    cfd_delta[swap_mask] = -cfd_delta[swap_mask]
    true_tof[swap_mask] = -true_tof[swap_mask]
    return waveforms, target, led_delta, cfd_delta, true_tof


def make_split_loader(
    datasets: list[PreparedDataset],
    split_name: str,
    normalization: Normalization,
    config: dict[str, Any],
    device: torch.device,
    *,
    shuffle: bool,
) -> DataLoader:
    views = []
    for dataset in datasets:
        indices = np.asarray(getattr(dataset, split_name), dtype=np.int64)
        views.append(CorrectionDataset(dataset, indices, normalization))
    combined = ConcatDataset(views)

    training = config["training"]
    workers = int(training.get("num_workers", 0))
    kwargs: dict[str, Any] = {
        "batch_size": int(training["batch_size"]),
        "shuffle": bool(shuffle),
        "drop_last": False,
        "num_workers": workers,
        "pin_memory": bool(training.get("pin_memory", device.type == "cuda")),
    }
    if workers > 0:
        kwargs["persistent_workers"] = bool(training.get("persistent_workers", False))
        kwargs["prefetch_factor"] = int(training.get("prefetch_factor", 2))
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(
            int(training.get("data_seed", training.get("seed", 12345)))
        )
        kwargs["generator"] = generator
    return DataLoader(combined, **kwargs)


def predict_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray | float]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    led: list[np.ndarray] = []
    cfd: list[np.ndarray] = []
    true_tof: list[np.ndarray] = []
    with torch.no_grad():
        for waveforms, target, led_delta, cfd_delta, tof in loader:
            prediction = model(waveforms.to(device, non_blocking=True))
            predictions.append(prediction.detach().cpu().numpy().astype(np.float64))
            targets.append(target.numpy().astype(np.float64))
            led.append(led_delta.numpy().astype(np.float64))
            cfd.append(cfd_delta.numpy().astype(np.float64))
            true_tof.append(tof.numpy().astype(np.float64))
    if not predictions:
        raise RuntimeError("Cannot evaluate an empty data loader")

    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    led_delta = np.concatenate(led)
    cfd_delta = np.concatenate(cfd)
    true = np.concatenate(true_tof)
    corrected = led_delta - prediction
    residual = corrected - true
    return {
        "prediction_ps": prediction,
        "target_ps": target,
        "led_ps": led_delta,
        "cfd_ps": cfd_delta,
        "true_tof_ps": true,
        "corrected_ps": corrected,
        "residual_ps": residual,
        "rmse_ps": float(np.sqrt(np.mean(residual * residual))),
    }


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    fit_config: dict[str, Any],
    label: str,
) -> tuple[dict[str, Any], FitResult | None, dict[str, np.ndarray | float]]:
    return evaluate_model_with_optional_fit(
        model, loader, device, fit_config, label, perform_fit=True
    )


def evaluate_model_with_optional_fit(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    fit_config: dict[str, Any],
    label: str,
    *,
    perform_fit: bool,
) -> tuple[dict[str, Any], FitResult | None, dict[str, np.ndarray | float]]:
    prediction = predict_loader(model, loader, device)
    residuals = np.asarray(prediction["residual_ps"], dtype=np.float64)
    arithmetic_bias = float(np.mean(residuals))
    fit: FitResult | None = None
    ctr_ps = float("nan")
    gaussian_bias_ps = float("nan")
    if perform_fit:
        fit = fit_times_ps(residuals, label, fit_config)
        distribution = distribution_metrics(residuals, true_value_ps=0.0, fit=fit)
        ctr_ps = float(distribution["ctr_ps"])
        gaussian_bias_ps = float(distribution["gaussian_bias_ps"])
    row = {
        "rmse_ps": float(prediction["rmse_ps"]),
        "ctr_ps": ctr_ps,
        "bias_ps": arithmetic_bias,
        "arithmetic_bias_ps": arithmetic_bias,
        "gaussian_bias_ps": gaussian_bias_ps,
        "fit_performed": bool(perform_fit),
    }
    return row, fit, prediction


def fit_schedule_for_epoch(
    training: dict[str, Any],
    epoch: int,
    *,
    selection_metric: str,
) -> tuple[bool, bool]:
    interval = int(training.get("fit_interval_epochs", 1))
    if interval < 0:
        raise ValueError("training.fit_interval_epochs must be non-negative")
    scheduled = interval > 0 and int(epoch) % interval == 0
    fit_train = bool(training.get("fit_train_during_training", True)) and scheduled
    fit_validation = (
        bool(training.get("fit_validation_during_training", True)) and scheduled
    )
    if selection_metric == "validation_ctr":
        fit_validation = True
    return fit_train, fit_validation


def validate_fit_schedule(training: dict[str, Any]) -> None:
    interval = int(training.get("fit_interval_epochs", 1))
    if interval < 0:
        raise ValueError("training.fit_interval_epochs must be non-negative")
    for name in ("fit_train_during_training", "fit_validation_during_training"):
        value = training.get(name, True)
        if not isinstance(value, bool):
            raise ValueError(f"training.{name} must be boolean")


def checkpoint_context(
    context: TrainingContext,
    *,
    model_config: dict[str, Any] | None = None,
    training_strategy: str,
) -> dict[str, Any]:
    contract_fields = (
        "led_timestamp_source",
        "cfd_timestamp_source",
        "ml_window_alignment_source",
        "timing_channel_waveforms_saved",
    )
    dataset_contract = {
        field: context.datasets[0].manifest.get(field) for field in contract_fields
    }
    return {
        "model_type": context.model_type,
        "model_name": context.model_name,
        "model_config": dict(context.model_config if model_config is None else model_config),
        "input_length": int(context.input_length),
        "normalization": context.normalization.as_dict(),
        "training_dataset_fingerprints": [
            dataset.manifest["fingerprint"] for dataset in context.datasets
        ],
        "training_dataset_paths": [str(dataset.directory) for dataset in context.datasets],
        "target_definition": "LED time difference minus known true TOF",
        "training_strategy": training_strategy,
        "data_view": dict(context.data_view),
        "relative_time_ps_start": float(context.datasets[0].relative_time_ps[0]),
        "relative_time_ps_stop": float(context.datasets[0].relative_time_ps[-1]),
        "dataset_contract": dataset_contract,
    }
