from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class CNNConfigError(ValueError):
    """Raised when a CNN experiment configuration is invalid."""


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise CNNConfigError(f"{path} must contain a JSON object")
    return value


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CNNConfigError(f"{name} must be a JSON object")
    return value


def _require_positive(value: Any, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise CNNConfigError(f"{name} must be numeric") from exc
    if numeric <= 0:
        raise CNNConfigError(f"{name} must be positive")
    return numeric


def resolve_path(value: str | Path, *, relative_to: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (relative_to / path).resolve()


def load_preprocessing_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    config = _load_json(source)
    validate_preprocessing_config(config)
    config["_config_path"] = str(source)
    return config


def validate_preprocessing_config(config: dict[str, Any]) -> None:
    root = _require_mapping(config, "preprocessing config")
    crop = _require_mapping(root.get("crop"), "crop")
    _require_positive(crop.get("threshold_mV"), "crop.threshold_mV")
    width_ns = _require_positive(crop.get("width_ns"), "crop.width_ns")
    step_ps = _require_positive(crop.get("resample_step_ps"), "crop.resample_step_ps")
    samples = int(round(width_ns * 1000.0 / step_ps))
    if samples < 16:
        raise CNNConfigError("crop must contain at least 16 samples")
    if str(crop.get("interpolation", "cubic")) not in {"linear", "cubic"}:
        raise CNNConfigError("crop.interpolation must be 'linear' or 'cubic'")

    augmentation = _require_mapping(root.get("augmentation"), "augmentation")
    shifts = augmentation.get("discrete_shifts_ps")
    if not isinstance(shifts, list) or len(shifts) < 2:
        raise CNNConfigError("augmentation.discrete_shifts_ps must contain at least two values")
    try:
        numeric_shifts = sorted({int(item) for item in shifts})
    except (TypeError, ValueError) as exc:
        raise CNNConfigError("augmentation.discrete_shifts_ps must contain integers") from exc
    if len(numeric_shifts) != len(shifts):
        raise CNNConfigError("augmentation.discrete_shifts_ps must not contain duplicates")
    if 0 not in numeric_shifts:
        raise CNNConfigError("augmentation.discrete_shifts_ps must include zero")
    shifted_channel = int(augmentation.get("shifted_timing_channel", 3))
    if shifted_channel not in {3, 4}:
        raise CNNConfigError("augmentation.shifted_timing_channel must be 3 or 4")

    uniform = _require_mapping(augmentation.get("uniform_test"), "augmentation.uniform_test")
    lower = int(uniform.get("min_ps"))
    upper = int(uniform.get("max_ps"))
    step = int(uniform.get("step_ps"))
    if lower >= upper:
        raise CNNConfigError("augmentation.uniform_test min_ps must be smaller than max_ps")
    if step <= 0:
        raise CNNConfigError("augmentation.uniform_test.step_ps must be positive")

    split = _require_mapping(root.get("split"), "split")
    train_fraction = float(split.get("train_fraction", 0))
    validation_fraction = float(split.get("validation_fraction", 0))
    test_fraction = float(split.get("test_fraction", 0))
    if min(train_fraction, validation_fraction, test_fraction) <= 0:
        raise CNNConfigError("all split fractions must be positive")
    if abs(train_fraction + validation_fraction + test_fraction - 1.0) > 1e-9:
        raise CNNConfigError("split fractions must sum to one")

    selection = _require_mapping(root.get("selection"), "selection")
    mad_threshold = selection.get("led_mad_threshold")
    if mad_threshold is not None and float(mad_threshold) <= 0:
        raise CNNConfigError("selection.led_mad_threshold must be positive or null")

    cache = _require_mapping(root.get("cache"), "cache")
    if not str(cache.get("filename", "")).strip():
        raise CNNConfigError("cache.filename must be non-empty")

    normalization = _require_mapping(root.get("normalization"), "normalization")
    if str(normalization.get("type", "train_channel_zscore")) not in {
        "none",
        "train_channel_zscore",
    }:
        raise CNNConfigError("normalization.type must be 'none' or 'train_channel_zscore'")
    _require_positive(normalization.get("epsilon", 1e-6), "normalization.epsilon")


def load_model_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    config = _load_json(source)
    validate_model_config(config)
    config["_config_path"] = str(source)
    return config


def _validate_architecture(section: dict[str, Any], name: str) -> None:
    channels = section.get("conv_channels")
    kernels = section.get("kernel_sizes")
    if not isinstance(channels, list) or not channels:
        raise CNNConfigError(f"{name}.conv_channels must be a non-empty list")
    if not isinstance(kernels, list) or len(kernels) != len(channels):
        raise CNNConfigError(f"{name}.kernel_sizes must match conv_channels")
    if any(int(item) <= 0 for item in channels):
        raise CNNConfigError(f"{name}.conv_channels must be positive")
    if any(int(item) <= 0 or int(item) % 2 == 0 for item in kernels):
        raise CNNConfigError(f"{name}.kernel_sizes must be positive odd integers")
    pools = section.get("pool_after", [])
    if not isinstance(pools, list) or any(int(item) < 0 or int(item) >= len(channels) for item in pools):
        raise CNNConfigError(f"{name}.pool_after contains an invalid layer index")
    _require_positive(section.get("pool_size", 2), f"{name}.pool_size")
    _require_positive(section.get("adaptive_pool_bins", 4), f"{name}.adaptive_pool_bins")
    dropout = float(section.get("dropout", 0.0))
    if not 0.0 <= dropout < 1.0:
        raise CNNConfigError(f"{name}.dropout must be in [0, 1)")


def validate_model_config(config: dict[str, Any]) -> None:
    root = _require_mapping(config, "model config")
    _validate_architecture(_require_mapping(root.get("direct_model"), "direct_model"), "direct_model")
    _validate_architecture(
        _require_mapping(root.get("correction_model"), "correction_model"),
        "correction_model",
    )

    training = _require_mapping(root.get("training"), "training")
    for key in ("epochs", "batch_size", "early_stopping_patience", "lr_patience"):
        if int(training.get(key, 0)) < 1:
            raise CNNConfigError(f"training.{key} must be positive")
    for key in ("learning_rate", "min_learning_rate"):
        _require_positive(training.get(key), f"training.{key}")
    if float(training.get("weight_decay", 0.0)) < 0:
        raise CNNConfigError("training.weight_decay must be non-negative")
    factor = float(training.get("lr_factor", 0.5))
    if not 0.0 < factor < 1.0:
        raise CNNConfigError("training.lr_factor must be between zero and one")
    if str(training.get("loss", "mse")) not in {"mse", "huber"}:
        raise CNNConfigError("training.loss must be 'mse' or 'huber'")
    seeds = training.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise CNNConfigError("training.seeds must be a non-empty list")
    if int(training.get("num_workers", 0)) < 0:
        raise CNNConfigError("training.num_workers must be non-negative")
    if int(training.get("evaluation_num_workers", 0)) < 0:
        raise CNNConfigError("training.evaluation_num_workers must be non-negative")
    if int(training.get("epoch_log_every", 1)) < 1:
        raise CNNConfigError("training.epoch_log_every must be positive")
    if int(training.get("eta_smoothing_epochs", 5)) < 1:
        raise CNNConfigError("training.eta_smoothing_epochs must be positive")

    parallel = _require_mapping(root.get("parallel"), "parallel")
    if int(parallel.get("max_parallel_runs", 1)) < 1:
        raise CNNConfigError("parallel.max_parallel_runs must be positive")
    device_pool = parallel.get("device_pool", ["auto"])
    if not isinstance(device_pool, list) or not device_pool:
        raise CNNConfigError("parallel.device_pool must be a non-empty list")


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    config = _load_json(source)
    validate_experiment_config(config)
    config["_config_path"] = str(source)
    base = source.parent
    for key in (
        "input_root",
        "output_dir",
        "cache_dir",
        "analysis_config",
        "preprocessing_config",
        "model_config",
    ):
        config[key] = str(resolve_path(config[key], relative_to=base))
    return config


def validate_experiment_config(config: dict[str, Any]) -> None:
    root = _require_mapping(config, "experiment config")
    for key in (
        "input_root",
        "output_dir",
        "cache_dir",
        "analysis_config",
        "preprocessing_config",
        "model_config",
    ):
        if not str(root.get(key, "")).strip():
            raise CNNConfigError(f"{key} must be non-empty")
    if int(root.get("max_events", 0)) < 0:
        raise CNNConfigError("max_events must be non-negative")
    if str(root.get("logging_level", "INFO")).upper() not in {
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
    }:
        raise CNNConfigError("logging_level is invalid")
