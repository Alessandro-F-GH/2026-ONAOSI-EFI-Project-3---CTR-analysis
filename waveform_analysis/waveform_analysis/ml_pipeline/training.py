from __future__ import annotations

import csv
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from utils.fit import fit_delta_times_integer_fs

from .common import atomic_json, canonical_hash, json_safe, set_global_seed
from .data import EnergyCache, SplitData
from .model import AntisymmetricCorrectionCNN
from .plots import plot_training_history
from .torch_data import (
    CorrectionDataset,
    EpochRandomSampler,
    Normalization,
    compute_normalization,
)


def _resolve_device(requested: str) -> torch.device:
    requested = str(requested).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _make_optimizer(model: nn.Module, config: dict[str, Any]) -> Optimizer:
    name = str(config["optimizer"].get("name", "adamw")).lower()
    kwargs = {
        "lr": float(config["optimizer"]["learning_rate"]),
        "weight_decay": float(config["optimizer"].get("weight_decay", 0.0)),
    }
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    raise ValueError(f"Unsupported optimizer: {name}")


def _make_scheduler(optimizer: Optimizer, config: dict[str, Any]) -> Any:
    scheduler_config = config["scheduler"]
    name = str(scheduler_config.get("name", "reduce_on_plateau")).lower()
    if name == "none":
        return None
    if name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(scheduler_config.get("factor", 0.5)),
            patience=int(scheduler_config.get("patience", 5)),
            min_lr=float(scheduler_config.get("minimum_learning_rate", 1e-7)),
        )
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config["training"]["epochs"]),
            eta_min=float(scheduler_config.get("minimum_learning_rate", 1e-7)),
        )
    raise ValueError(f"Unsupported scheduler: {name}")


def _loader_kwargs(pipeline_config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    parallel = pipeline_config["parallelization"]
    workers = int(parallel.get("training_num_workers", 0))
    kwargs: dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": bool(parallel.get("pin_memory", device.type == "cuda")),
    }
    if workers > 0:
        kwargs["persistent_workers"] = bool(parallel.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(parallel.get("prefetch_factor", 2))
    return kwargs


def _make_loaders(
    cache: EnergyCache,
    splits: SplitData,
    normalization: Normalization,
    led_center_ps: float,
    pipeline_config: dict[str, Any],
    cnn_config: dict[str, Any],
    device: torch.device,
) -> tuple[CorrectionDataset, CorrectionDataset, DataLoader, DataLoader, EpochRandomSampler]:
    training_dataset = CorrectionDataset(
        cache, splits.train, normalization, led_center_ps
    )
    validation_dataset = CorrectionDataset(
        cache, splits.validation, normalization, led_center_ps
    )
    sampler = EpochRandomSampler(
        len(training_dataset), int(cnn_config["training"].get("seed", 12345))
    )
    batch_size = int(cnn_config["training"]["batch_size"])
    common = _loader_kwargs(pipeline_config, device)
    training_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=False,
        **common,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )
    return (
        training_dataset,
        validation_dataset,
        training_loader,
        validation_loader,
        sampler,
    )


def _led_deltas_ps(cache: EnergyCache, indices: np.ndarray) -> np.ndarray:
    times = np.asarray(cache.led_time_fs[indices], dtype=np.int64)
    return (times[:, 0] - times[:, 1]).astype(np.float64) / 1000.0


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    next_batch: int,
    accumulator: dict[str, float],
    history: list[dict[str, Any]],
    best_value: float,
    bad_epochs: int,
    context: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": None if scheduler is None else scheduler.state_dict(),
        "scaler_state": None if scaler is None else scaler.state_dict(),
        "epoch": int(epoch),
        "next_batch": int(next_batch),
        "train_accumulator": accumulator,
        "history": history,
        "best_value": float(best_value),
        "bad_epochs": int(bad_epochs),
        "context": context,
        "rng_state": _capture_rng_state(),
    }


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)


def _write_history(path: Path, history: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if not history:
        temporary.write_text("", encoding="utf-8")
    else:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows([json_safe(row) for row in history])
    os.replace(temporary, path)


@torch.no_grad()
def predict_loader(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, np.ndarray | float]:
    model.eval()
    corrections: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    led_values: list[np.ndarray] = []
    loss_sum = 0.0
    count = 0
    for waveforms, target, led_delta in loader:
        waveforms = waveforms.to(device, non_blocking=True)
        target_device = target.to(device, non_blocking=True)
        prediction = model(waveforms)
        loss = torch.mean((prediction - target_device) ** 2)
        batch_count = int(target.shape[0])
        loss_sum += float(loss.item()) * batch_count
        count += batch_count
        corrections.append(prediction.detach().cpu().numpy().astype(np.float64))
        targets.append(target.numpy().astype(np.float64))
        led_values.append(led_delta.numpy().astype(np.float64))
    if count == 0:
        raise RuntimeError("Cannot evaluate an empty data loader")
    correction = np.concatenate(corrections)
    target_array = np.concatenate(targets)
    led = np.concatenate(led_values)
    corrected = led - correction
    return {
        "loss": loss_sum / count,
        "correction_ps": correction,
        "target_ps": target_array,
        "led_ps": led,
        "corrected_ps": corrected,
        "corrected_std_ps": float(np.std(corrected, ddof=1)) if count > 1 else np.nan,
    }


def _fit_ctr(values_ps: np.ndarray, method: str, fit_config: dict[str, Any]) -> float:
    values_fs = np.rint(np.asarray(values_ps, dtype=np.float64) * 1000.0).astype(np.int64)
    fit = fit_delta_times_integer_fs(
        values_fs,
        method=method,
        parameter=0.0,
        n_total=int(values_fs.size),
        n_selected=int(values_fs.size),
        config=fit_config,
    )
    return float(fit.ctr_ps) if fit.success else np.nan


def train_cnn(
    cache: EnergyCache,
    splits: SplitData,
    pipeline_config: dict[str, Any],
    cnn_config: dict[str, Any],
    *,
    resume: bool,
    restart: bool,
    logger: Any,
) -> dict[str, Any]:
    checkpoint_dir = Path(pipeline_config["paths"]["checkpoint_dir"])
    plot_dir = Path(pipeline_config["paths"]["plot_dir"]) / "training"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if restart:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(plot_dir, ignore_errors=True)
    last_path = checkpoint_dir / "last.pt"
    interrupted_path = checkpoint_dir / "interrupted.pt"
    best_path = checkpoint_dir / "best_validation.pt"
    if not resume and not restart and (last_path.exists() or best_path.exists()):
        raise RuntimeError(
            "Training checkpoints already exist. Use --resume to continue or --restart to start over."
        )

    seed = int(cnn_config["training"].get("seed", 12345))
    set_global_seed(seed)
    parallel = pipeline_config["parallelization"]
    torch_threads = int(parallel.get("torch_num_threads", 0))
    if torch_threads > 0:
        torch.set_num_threads(torch_threads)
    interop_threads = int(parallel.get("torch_num_interop_threads", 0))
    if interop_threads > 0:
        try:
            torch.set_num_interop_threads(interop_threads)
        except RuntimeError:
            logger.warning("torch inter-op thread count could not be changed in this process")

    device = _resolve_device(cnn_config["training"].get("device", "auto"))
    logger.info("Training device: %s", device)

    train_led = _led_deltas_ps(cache, splits.train)
    led_center_ps = float(np.mean(train_led))
    normalization = compute_normalization(
        cache.windows_mV,
        splits.train,
        chunk_size=int(pipeline_config["cache"].get("normalization_chunk_size", 2048)),
    )
    context = {
        "pipeline_config_hash": pipeline_config["_config_hash"],
        "cnn_config_hash": cnn_config["_config_hash"],
        "dataset_fingerprint": cache.manifest["fingerprint"],
        "split_fingerprint": splits.manifest["fingerprint"],
        "training_context_fingerprint": canonical_hash(
            {
                "pipeline": pipeline_config["_config_hash"],
                "cnn": cnn_config["_config_hash"],
                "dataset": cache.manifest["fingerprint"],
                "split": splits.manifest["fingerprint"],
            }
        ),
        "led_center_ps": led_center_ps,
        "normalization": normalization.as_dict(),
        "model": "AntisymmetricCorrectionCNN",
        "estimator": "y_theta(s1,s2)=g_theta(s1)-g_theta(s2)",
        "input_channels": cache.manifest["energy_channels_one_based"],
        "test_used_during_training": False,
    }
    atomic_json(checkpoint_dir / "training_context.json", context)

    (
        training_dataset,
        validation_dataset,
        training_loader,
        validation_loader,
        sampler,
    ) = _make_loaders(
        cache,
        splits,
        normalization,
        led_center_ps,
        pipeline_config,
        cnn_config,
        device,
    )
    # A non-shuffled loader is used only for epoch-end train metrics.
    train_metric_loader = DataLoader(
        training_dataset,
        batch_size=int(cnn_config["training"]["batch_size"]),
        shuffle=False,
        drop_last=False,
        **_loader_kwargs(pipeline_config, device),
    )

    model = AntisymmetricCorrectionCNN(cnn_config).to(device)
    optimizer = _make_optimizer(model, cnn_config)
    scheduler = _make_scheduler(optimizer, cnn_config)
    amp_enabled = bool(cnn_config["training"].get("mixed_precision", True)) and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except AttributeError:  # PyTorch 2.2 compatibility
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    start_epoch = 0
    start_batch = 0
    accumulator = {"loss_sum": 0.0, "count": 0.0}
    history: list[dict[str, Any]] = []
    best_value = math.inf
    bad_epochs = 0

    if resume:
        candidates = [path for path in (last_path, interrupted_path) if path.is_file()]
        if not candidates:
            raise FileNotFoundError("No resumable checkpoint was found")
        resume_path = max(candidates, key=lambda item: item.stat().st_mtime_ns)
        checkpoint = _load_checkpoint(resume_path, device)
        if checkpoint["context"]["training_context_fingerprint"] != context[
            "training_context_fingerprint"
        ]:
            raise RuntimeError(
                "Checkpoint does not match current data, split, preprocessing, or CNN configuration"
            )
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scheduler is not None and checkpoint["scheduler_state"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        if checkpoint.get("scaler_state") is not None:
            scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint["epoch"])
        start_batch = int(checkpoint["next_batch"])
        accumulator = {
            "loss_sum": float(checkpoint["train_accumulator"]["loss_sum"]),
            "count": float(checkpoint["train_accumulator"]["count"]),
        }
        history = list(checkpoint["history"])
        best_value = float(checkpoint["best_value"])
        bad_epochs = int(checkpoint["bad_epochs"])
        _restore_rng_state(checkpoint["rng_state"])
        logger.info(
            "Resuming from %s at epoch %d, next batch %d",
            resume_path,
            start_epoch + 1,
            start_batch,
        )

    epochs = int(cnn_config["training"]["epochs"])
    log_every = max(1, int(cnn_config["training"].get("log_every_batches", 20)))
    checkpoint_every = max(
        0, int(cnn_config["checkpointing"].get("every_batches", 0))
    )
    gradient_clip = cnn_config["training"].get("gradient_clip_norm")
    fit_every = max(1, int(cnn_config["training"].get("fit_metrics_every_epochs", 1)))
    monitor = str(cnn_config["training"].get("monitor", "validation_ctr_ps"))
    patience = int(cnn_config["training"].get("early_stopping_patience", 15))
    min_delta = float(cnn_config["training"].get("early_stopping_min_delta", 0.0))

    current_epoch = start_epoch
    current_next_batch = start_batch
    started = time.time()
    try:
        for epoch in range(start_epoch, epochs):
            current_epoch = epoch
            sampler.set_epoch(epoch)
            model.train()
            epoch_start_batch = start_batch if epoch == start_epoch else 0
            if epoch != start_epoch:
                accumulator = {"loss_sum": 0.0, "count": 0.0}
            for batch_index, (waveforms, target, _) in enumerate(training_loader):
                if batch_index < epoch_start_batch:
                    continue
                waveforms = waveforms.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    prediction = model(waveforms)
                    loss = torch.mean((prediction - target) ** 2)
                scaler.scale(loss).backward()
                if gradient_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(gradient_clip)
                    )
                scaler.step(optimizer)
                scaler.update()

                batch_count = int(target.shape[0])
                accumulator["loss_sum"] += float(loss.item()) * batch_count
                accumulator["count"] += batch_count
                current_next_batch = batch_index + 1
                if current_next_batch % log_every == 0:
                    logger.info(
                        "Epoch %d/%d batch %d/%d loss %.6g",
                        epoch + 1,
                        epochs,
                        current_next_batch,
                        len(training_loader),
                        accumulator["loss_sum"] / max(accumulator["count"], 1.0),
                    )
                if checkpoint_every > 0 and current_next_batch % checkpoint_every == 0:
                    payload = _checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        next_batch=current_next_batch,
                        accumulator=accumulator,
                        history=history,
                        best_value=best_value,
                        bad_epochs=bad_epochs,
                        context=context,
                    )
                    _atomic_torch_save(last_path, payload)

            train_metrics = predict_loader(model, train_metric_loader, device)
            validation_metrics = predict_loader(model, validation_loader, device)
            compute_fit = (epoch + 1) % fit_every == 0 or epoch + 1 == epochs
            train_ctr = (
                _fit_ctr(
                    train_metrics["corrected_ps"],
                    "Train LED corrected",
                    pipeline_config["fit"],
                )
                if compute_fit
                else np.nan
            )
            validation_ctr = (
                _fit_ctr(
                    validation_metrics["corrected_ps"],
                    "Validation LED corrected",
                    pipeline_config["fit"],
                )
                if compute_fit
                else np.nan
            )
            row = {
                "epoch": epoch + 1,
                "train_loss": float(train_metrics["loss"]),
                "validation_loss": float(validation_metrics["loss"]),
                "train_corrected_std_ps": float(train_metrics["corrected_std_ps"]),
                "validation_corrected_std_ps": float(
                    validation_metrics["corrected_std_ps"]
                ),
                "train_ctr_ps": float(train_ctr),
                "validation_ctr_ps": float(validation_ctr),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "elapsed_seconds": float(time.time() - started),
            }
            history.append(row)
            _write_history(checkpoint_dir / "training_history.csv", history)
            logger.info(
                "Epoch %d/%d complete | train loss %.6g | val loss %.6g | "
                "val σ %.3f ps | val CTR %s ps",
                epoch + 1,
                epochs,
                row["train_loss"],
                row["validation_loss"],
                row["validation_corrected_std_ps"],
                "nan" if not np.isfinite(validation_ctr) else f"{validation_ctr:.3f}",
            )

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(row["validation_loss"])
                else:
                    scheduler.step()

            candidate = row.get(monitor, np.nan)
            improved = bool(
                np.isfinite(candidate)
                and float(candidate) < best_value - min_delta
            )
            if improved:
                best_value = float(candidate)
                bad_epochs = 0
            else:
                bad_epochs += 1

            completed_payload = _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch + 1,
                next_batch=0,
                accumulator={"loss_sum": 0.0, "count": 0.0},
                history=history,
                best_value=best_value,
                bad_epochs=bad_epochs,
                context=context,
            )
            _atomic_torch_save(last_path, completed_payload)
            if improved:
                _atomic_torch_save(best_path, completed_payload)
                logger.info("Saved new best validation checkpoint: %s", best_path)
            start_batch = 0
            current_next_batch = 0
            if patience > 0 and bad_epochs >= patience:
                logger.info("Early stopping after %d non-improving epochs", bad_epochs)
                break
    except KeyboardInterrupt:
        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=current_epoch,
            next_batch=current_next_batch,
            accumulator=accumulator,
            history=history,
            best_value=best_value,
            bad_epochs=bad_epochs,
            context=context,
        )
        _atomic_torch_save(interrupted_path, payload)
        logger.warning("Training interrupted; resumable checkpoint saved to %s", interrupted_path)
        raise

    if not best_path.is_file() and last_path.is_file():
        # A fit-based monitor can be unavailable for very small debug datasets.
        # Preserve a usable final checkpoint rather than failing evaluation.
        shutil.copy2(last_path, best_path)
        logger.warning(
            "No finite monitored metric was available; using the last checkpoint as best"
        )
    plot_training_history(
        history,
        plot_dir,
        dpi=int(pipeline_config["plotting"].get("dpi", 180)),
    )
    result = {
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "epochs_completed": len(history),
        "best_monitor_value": best_value,
        "monitor": monitor,
        "history": history,
        "context": context,
    }
    atomic_json(checkpoint_dir / "training_summary.json", result)
    return result
