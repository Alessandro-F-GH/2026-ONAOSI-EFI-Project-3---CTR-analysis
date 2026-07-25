from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .common import canonical_hash


class MLConfigError(ValueError):
    pass


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MLConfigError(f"{path} must be a JSON object")
    return value


def _positive(value: Any, path: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise MLConfigError(f"{path} must be numeric") from exc
    if number <= 0:
        raise MLConfigError(f"{path} must be positive")
    return number


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise MLConfigError(f"{path} must contain a JSON object")
    return value


def _resolve_path(project_root: Path, value: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return str(path.resolve())


def load_pipeline_config(path: str | Path, project_root: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    root = Path(project_root).resolve()
    config = load_json(source)
    validate_pipeline_config(config)
    result = copy.deepcopy(config)
    result["_config_path"] = str(source)
    result["_config_hash"] = canonical_hash(config)
    result["_project_root"] = str(root)
    result["data"]["input_root"] = _resolve_path(root, result["data"]["input_root"])
    for key in ("work_dir", "dataset_cache_dir", "split_dir", "checkpoint_dir", "plot_dir", "log_dir"):
        result["paths"][key] = _resolve_path(root, result["paths"][key])
    return result


def load_cnn_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    config = load_json(source)
    validate_cnn_config(config)
    result = copy.deepcopy(config)
    result["_config_path"] = str(source)
    result["_config_hash"] = canonical_hash(config)
    return result


def validate_pipeline_config(config: dict[str, Any]) -> None:
    root = _mapping(config, "config")
    required = (
        "data",
        "channels",
        "waveform",
        "selection",
        "photopeak",
        "split",
        "parallelization",
        "paths",
        "cache",
        "fit",
        "plotting",
        "logging",
        "evaluation",
    )
    for section in required:
        _mapping(root.get(section), section)

    data = root["data"]
    if not isinstance(data.get("input_root"), str) or not data["input_root"]:
        raise MLConfigError("data.input_root must be a non-empty path string")
    float(data["true_tof_ps"])

    channels = root["channels"]
    energy = channels.get("energy")
    polarities = channels.get("polarities")
    if not isinstance(energy, list) or len(energy) != 2:
        raise MLConfigError("channels.energy must contain two one-based channel numbers")
    if len(set(int(item) for item in energy)) != 2 or any(
        int(item) not in (1, 2, 3, 4) for item in energy
    ):
        raise MLConfigError("channels.energy must contain two distinct values in [1, 4]")
    if not isinstance(polarities, list) or len(polarities) != 2:
        raise MLConfigError("channels.polarities must contain two entries")
    if any(int(item) not in (-1, 1) for item in polarities):
        raise MLConfigError("channels.polarities values must be +1 or -1")

    waveform = root["waveform"]
    for name in (
        "baseline_samples",
        "search_trigger_threshold_mV",
        "upsample_step_ps",
        "led_threshold_mV",
        "cfd_fraction",
        "subsample_factor",
    ):
        _positive(waveform[name], f"waveform.{name}")
    if not 0 < float(waveform["cfd_fraction"]) < 1:
        raise MLConfigError("waveform.cfd_fraction must lie strictly between 0 and 1")
    if int(waveform["subsample_factor"]) < 1:
        raise MLConfigError("waveform.subsample_factor must be at least 1")
    for section in ("analysis_crop_ns", "ml_window_ns"):
        window = _mapping(waveform.get(section), f"waveform.{section}")
        _positive(window["before"], f"waveform.{section}.before")
        _positive(window["after"], f"waveform.{section}.after")

    split = root["split"]
    fractions = [float(split[name]) for name in ("train_fraction", "validation_fraction", "test_fraction")]
    if any(value <= 0 for value in fractions) or abs(sum(fractions) - 1.0) > 1e-9:
        raise MLConfigError("split fractions must be positive and sum to 1")
    if split.get("strategy", "event") not in ("event", "source_file"):
        raise MLConfigError("split.strategy must be 'event' or 'source_file'")

    parallel = root["parallelization"]
    if parallel.get("preprocessing_backend", "process") not in ("process", "thread", "serial"):
        raise MLConfigError("parallelization.preprocessing_backend is invalid")
    for name in ("preprocessing_workers", "training_num_workers", "torch_num_threads"):
        if int(parallel.get(name, 0)) < 0:
            raise MLConfigError(f"parallelization.{name} cannot be negative")

    fit = root["fit"]
    if len(fit["histogram_range_ps"]) != 2:
        raise MLConfigError("fit.histogram_range_ps must contain two values")
    if float(fit["histogram_range_ps"][1]) <= float(fit["histogram_range_ps"][0]):
        raise MLConfigError("fit.histogram_range_ps must be increasing")
    for name in ("histogram_bin_ps", "initial_half_width_ps", "iteration_sigma"):
        _positive(fit[name], f"fit.{name}")


def validate_cnn_config(config: dict[str, Any]) -> None:
    root = _mapping(config, "cnn_config")
    for section in ("architecture", "optimizer", "scheduler", "training", "checkpointing"):
        _mapping(root.get(section), section)
    architecture = root["architecture"]
    channels = architecture.get("conv_channels")
    kernels = architecture.get("kernel_sizes")
    pools = architecture.get("pool_sizes")
    if not isinstance(channels, list) or not channels:
        raise MLConfigError("architecture.conv_channels must be a non-empty list")
    if not isinstance(kernels, list) or len(kernels) != len(channels):
        raise MLConfigError("architecture.kernel_sizes must match conv_channels")
    if not isinstance(pools, list) or len(pools) != len(channels):
        raise MLConfigError("architecture.pool_sizes must match conv_channels")
    if any(int(value) <= 0 for value in channels + kernels + pools):
        raise MLConfigError("CNN channels, kernels, and pool sizes must be positive")
    if architecture.get("activation", "relu") not in ("relu", "gelu", "silu"):
        raise MLConfigError("architecture.activation must be relu, gelu, or silu")
    training = root["training"]
    for name in ("epochs", "batch_size"):
        if int(training[name]) <= 0:
            raise MLConfigError(f"training.{name} must be positive")
    _positive(root["optimizer"]["learning_rate"], "optimizer.learning_rate")
