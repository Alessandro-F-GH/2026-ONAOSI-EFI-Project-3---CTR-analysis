from __future__ import annotations

from dataclasses import dataclass
import csv
from datetime import datetime, timedelta
import json
import logging
import math
import time
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .cnn_cache import load_cnn_dataset_cache
from .cnn_models import build_model

LOGGER = logging.getLogger(__name__)


def _format_duration(seconds: float) -> str:
    """Format a non-negative duration as HH:MM:SS."""
    if not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _parameter_count(model: nn.Module) -> int:
    target = model.module if isinstance(model, nn.DataParallel) else model
    return sum(parameter.numel() for parameter in target.parameters())


class ArrayRegressionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> None:
        self.x = np.asarray(x, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        if self.x.ndim != 3 or self.x.shape[1] != 2:
            raise ValueError("waveform array must have shape [events, 2, samples]")
        if self.y.shape != (self.x.shape[0],):
            raise ValueError("target shape does not match waveform count")

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        values = (self.x[index] - self.mean) / self.std
        return torch.from_numpy(values), torch.tensor(self.y[index], dtype=torch.float32)


@dataclass(frozen=True)
class TrainingResult:
    model_type: str
    seed: int
    checkpoint: str
    best_epoch: int
    best_validation_loss: float
    device: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "seed": self.seed,
            "checkpoint": self.checkpoint,
            "best_epoch": self.best_epoch,
            "best_validation_loss": self.best_validation_loss,
            "device": self.device,
        }


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def resolve_device(requested: str) -> torch.device:
    value = str(requested).lower()
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if value.startswith("cuda") and not torch.cuda.is_available():
        LOGGER.warning("CUDA device %s requested but CUDA is unavailable; using CPU", requested)
        return torch.device("cpu")
    return torch.device(requested)


def channel_normalization(
    x: np.ndarray,
    normalization_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    kind = str(normalization_config.get("type", "train_channel_zscore"))
    epsilon = float(normalization_config.get("epsilon", 1e-6))
    if kind == "none":
        return (
            np.zeros((2, 1), dtype=np.float32),
            np.ones((2, 1), dtype=np.float32),
        )
    values = np.asarray(x, dtype=np.float64)
    mean = np.mean(values, axis=(0, 2), keepdims=False).reshape(2, 1)
    std = np.std(values, axis=(0, 2), keepdims=False).reshape(2, 1)
    std = np.maximum(std, epsilon)
    return mean.astype(np.float32), std.astype(np.float32)


def _loss_function(config: dict[str, Any]) -> nn.Module:
    loss_name = str(config.get("loss", "mse"))
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "huber":
        return nn.HuberLoss(delta=float(config.get("huber_delta_ps", 20.0)))
    raise ValueError(f"unsupported loss: {loss_name}")


def _unwrap_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    if isinstance(model, nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    mixed_precision: bool,
    gradient_clip_norm: float | None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_events = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(
                device_type=device.type,
                enabled=mixed_precision and device.type == "cuda",
            ):
                predictions = model(inputs)
                loss = loss_function(predictions, targets)
            if training:
                scaler.scale(loss).backward()
                if gradient_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
        batch_size = inputs.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_events += batch_size
    return total_loss / max(total_events, 1)


def train_model_run(task: dict[str, Any]) -> dict[str, Any]:
    """Train one direct or correction CNN realization.

    This function is deliberately top-level and JSON/pickle-friendly so the
    experiment runner can execute independent seeds concurrently with a process
    pool on Windows, Linux, or macOS.
    """
    model_type = str(task["model_type"])
    seed = int(task["seed"])
    dataset_path = Path(task["dataset_path"])
    run_dir = Path(task["run_dir"])
    model_config = task["model_config"]
    preprocessing_config = task["preprocessing_config"]
    device = resolve_device(str(task.get("device", "auto")))
    deterministic = bool(model_config["parallel"].get("deterministic", True))
    seed_everything(seed, deterministic)
    torch.set_num_threads(max(1, int(model_config["parallel"].get("cpu_threads_per_run", 1))))
    run_dir.mkdir(parents=True, exist_ok=True)

    cache = load_cnn_dataset_cache(dataset_path)
    if model_type == "direct":
        train_x = cache["direct_train_x"]
        train_y = cache["direct_train_y"]
        validation_x = cache["direct_validation_x"]
        validation_y = cache["direct_validation_y"]
        correction_center_ps = 0.0
    elif model_type == "correction":
        train_x = cache["correction_train_x"]
        train_led = cache["correction_train_led_delta_ps"].astype(np.float32)
        correction_center_ps = float(np.mean(train_led))
        train_y = train_led - correction_center_ps
        validation_x = cache["correction_validation_x"]
        validation_y = (
            cache["correction_validation_led_delta_ps"].astype(np.float32)
            - correction_center_ps
        )
    else:
        raise ValueError(f"unknown model type: {model_type}")

    mean, std = channel_normalization(train_x, preprocessing_config["normalization"])
    training_config = model_config["training"]
    train_dataset = ArrayRegressionDataset(train_x, train_y, mean=mean, std=std)
    validation_dataset = ArrayRegressionDataset(
        validation_x,
        validation_y,
        mean=mean,
        std=std,
    )
    batch_size = int(training_config["batch_size"])
    num_workers = int(training_config.get("num_workers", 0))
    pin_memory = bool(training_config.get("pin_memory", True)) and device.type == "cuda"
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    model = build_model(model_type, model_config).to(device)
    if (
        bool(model_config["parallel"].get("use_data_parallel", True))
        and device.type == "cuda"
        and torch.cuda.device_count() > 1
    ):
        model = nn.DataParallel(model)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(training_config.get("lr_factor", 0.5)),
        patience=int(training_config.get("lr_patience", 7)),
        min_lr=float(training_config.get("min_learning_rate", 1e-6)),
    )
    loss_function = _loss_function(training_config)
    mixed_precision = bool(training_config.get("mixed_precision", True))
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=mixed_precision and device.type == "cuda",
    )
    gradient_clip_value = training_config.get("gradient_clip_norm")
    gradient_clip = None if gradient_clip_value is None else float(gradient_clip_value)
    epochs = int(training_config["epochs"])
    patience = int(training_config["early_stopping_patience"])
    best_loss = math.inf
    best_epoch = -1
    epochs_without_improvement = 0
    history: list[dict[str, float | int | str]] = []
    checkpoint_path = run_dir / "best_model.pt"
    run_number = int(task.get("run_number", 1))
    total_runs = int(task.get("total_runs", 1))
    parallel_runs = max(1, int(task.get("parallel_runs", 1)))
    log_every = max(1, int(training_config.get("epoch_log_every", 1)))
    smoothing_epochs = max(1, int(training_config.get("eta_smoothing_epochs", 5)))
    run_start = time.perf_counter()
    epoch_durations: list[float] = []
    LOGGER.info(
        "Starting run %d/%d | %s seed=%d | device=%s | parameters=%d | "
        "train_events=%d | validation_events=%d | max_epochs=%d | batch=%d",
        run_number,
        total_runs,
        model_type,
        seed,
        device,
        _parameter_count(model),
        len(train_dataset),
        len(validation_dataset),
        epochs,
        batch_size,
    )

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        if epoch == 1:
            LOGGER.info(
                "Run %d/%d | %s seed=%d | epoch 1/%d started",
                run_number,
                total_runs,
                model_type,
                seed,
                epochs,
            )
        train_loss = _run_epoch(
            model,
            train_loader,
            loss_function,
            device,
            optimizer=optimizer,
            scaler=scaler,
            mixed_precision=mixed_precision,
            gradient_clip_norm=gradient_clip,
        )
        validation_loss = _run_epoch(
            model,
            validation_loader,
            loss_function,
            device,
            optimizer=None,
            scaler=scaler,
            mixed_precision=mixed_precision,
            gradient_clip_norm=None,
        )
        scheduler.step(validation_loss)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        epoch_seconds = time.perf_counter() - epoch_start
        elapsed_seconds = time.perf_counter() - run_start
        epoch_durations.append(epoch_seconds)
        recent_average = float(np.mean(epoch_durations[-smoothing_epochs:]))
        overall_average = float(np.mean(epoch_durations))
        conservative_epoch_seconds = max(recent_average, overall_average)
        remaining_epochs = max(0, epochs - epoch)
        eta_max_seconds = conservative_epoch_seconds * remaining_epochs
        max_total_seconds = elapsed_seconds + eta_max_seconds
        maximum_finish = datetime.now() + timedelta(seconds=eta_max_seconds)
        experiment_eta_max_seconds: float | None = None
        if parallel_runs == 1:
            remaining_runs = max(0, total_runs - run_number)
            experiment_eta_max_seconds = (
                eta_max_seconds + remaining_runs * conservative_epoch_seconds * epochs
            )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": learning_rate,
                "epoch_seconds": epoch_seconds,
                "elapsed_seconds": elapsed_seconds,
                "eta_max_seconds": eta_max_seconds,
                "estimated_max_total_seconds": max_total_seconds,
                "estimated_max_finish": maximum_finish.isoformat(timespec="seconds"),
            }
        )
        if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
            message = (
                "Run %d/%d | %s seed=%d | epoch %d/%d (%.1f%%) | "
                "train=%.4f val=%.4f lr=%.3g | epoch_time=%s | elapsed=%s | "
                "eta_max=%s | max_finish=%s"
            )
            arguments: list[Any] = [
                run_number,
                total_runs,
                model_type,
                seed,
                epoch,
                epochs,
                100.0 * epoch / epochs,
                train_loss,
                validation_loss,
                learning_rate,
                _format_duration(epoch_seconds),
                _format_duration(elapsed_seconds),
                _format_duration(eta_max_seconds),
                maximum_finish.strftime("%Y-%m-%d %H:%M:%S"),
            ]
            if experiment_eta_max_seconds is not None:
                message += " | experiment_eta_max~%s"
                arguments.append(_format_duration(experiment_eta_max_seconds))
            LOGGER.info(message, *arguments)
        if validation_loss < best_loss - float(training_config.get("minimum_improvement", 1e-6)):
            best_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_type": model_type,
                    "seed": seed,
                    "state_dict": _unwrap_state_dict(model),
                    "normalization_mean": mean,
                    "normalization_std": std,
                    "correction_center_ps": correction_center_ps,
                    "model_config": model_config,
                    "preprocessing_config": preprocessing_config,
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_loss,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            LOGGER.info(
                "%s seed=%d early stopping at epoch %d (best=%d)",
                model_type,
                seed,
                epoch,
                best_epoch,
            )
            break

    with (run_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    summary = TrainingResult(
        model_type=model_type,
        seed=seed,
        checkpoint=str(checkpoint_path),
        best_epoch=best_epoch,
        best_validation_loss=float(best_loss),
        device=str(device),
    )
    with (run_dir / "training_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary.as_dict(), stream, indent=2)
    return summary.as_dict()


def load_trained_model(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    target_device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=target_device, weights_only=False)
    model = build_model(checkpoint["model_type"], checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(target_device)
    model.eval()
    return model, checkpoint


def predict_array(
    model: nn.Module,
    x: np.ndarray,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    device: str | torch.device,
    batch_size: int,
    num_workers: int = 0,
) -> np.ndarray:
    target_device = torch.device(device)
    dummy_y = np.zeros(x.shape[0], dtype=np.float32)
    dataset = ArrayRegressionDataset(x, dummy_y, mean=mean, std=std)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=target_device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, _ in loader:
            values = model(inputs.to(target_device, non_blocking=True))
            predictions.append(values.detach().cpu().numpy())
    return np.concatenate(predictions).astype(np.float64)
