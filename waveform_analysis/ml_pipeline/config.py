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


def load_model_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    config = load_json(source)
    validate_model_config(config)
    result = copy.deepcopy(config)
    result["_config_path"] = str(source)
    result["_config_hash"] = canonical_hash(config)
    # Import lazily to avoid importing torch while merely validating JSON.
    from .model import model_type

    result["_model_type"] = model_type(result)
    return result


def load_cnn_config(path: str | Path) -> dict[str, Any]:
    """Backward-compatible alias for older terminal commands."""
    result = load_model_config(path)
    if result["_model_type"] != "cnn":
        raise MLConfigError(
            f"Expected a CNN config, found model_type={result['_model_type']!r}"
        )
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

    augmentation = root.get("channel_swap_augmentation", {})
    if augmentation is not None:
        augmentation = _mapping(augmentation, "channel_swap_augmentation")
        for name in ("enabled", "paired_batches"):
            if name in augmentation and not isinstance(augmentation[name], bool):
                raise MLConfigError(
                    f"channel_swap_augmentation.{name} must be boolean"
                )

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


def _validate_common_model_sections(root: dict[str, Any]) -> None:
    for section in ("architecture", "optimizer", "scheduler", "training", "checkpointing"):
        _mapping(root.get(section), section)
    architecture = root["architecture"]
    if architecture.get("activation", "relu") not in ("relu", "gelu", "silu"):
        raise MLConfigError("architecture.activation must be relu, gelu, or silu")
    dropout = float(architecture.get("dropout", 0.0))
    if not 0.0 <= dropout < 1.0:
        raise MLConfigError("architecture.dropout must lie in [0, 1)")
    max_abs = architecture.get("max_abs_single_channel_output_ps")
    if max_abs is not None:
        _positive(max_abs, "architecture.max_abs_single_channel_output_ps")
    training = root["training"]
    for name in ("epochs", "batch_size"):
        if int(training[name]) <= 0:
            raise MLConfigError(f"training.{name} must be positive")
    _positive(root["optimizer"]["learning_rate"], "optimizer.learning_rate")


def validate_model_config(config: dict[str, Any]) -> None:
    root = _mapping(config, "model_config")
    from .model import model_type

    kind = model_type(root)
    if kind == "catch22_random_forest":
        for section in ("features", "random_forest", "training", "checkpointing"):
            _mapping(root.get(section), section)
        features = root["features"]
        if str(features.get("implementation", "aeon")).lower() != "aeon":
            raise MLConfigError("features.implementation currently supports only aeon")
        if int(features.get("chunk_events", 512)) <= 0:
            raise MLConfigError("features.chunk_events must be positive")
        n_jobs = int(features.get("n_jobs", 1))
        if n_jobs == 0 or n_jobs < -1:
            raise MLConfigError("features.n_jobs must be -1 or a positive integer")
        backend = features.get("parallel_backend", "threading")
        if backend not in (None, "threading", "loky", "multiprocessing"):
            raise MLConfigError("features.parallel_backend is invalid")

        forest = root["random_forest"]
        if int(forest.get("n_estimators", 0)) <= 0:
            raise MLConfigError("random_forest.n_estimators must be positive")
        if forest.get("criterion", "squared_error") not in (
            "squared_error",
            "friedman_mse",
            "absolute_error",
        ):
            raise MLConfigError("random_forest.criterion is invalid")
        max_depth = forest.get("max_depth")
        if max_depth is not None and int(max_depth) <= 0:
            raise MLConfigError("random_forest.max_depth must be null or positive")
        if int(forest.get("n_jobs", -1)) == 0 or int(forest.get("n_jobs", -1)) < -1:
            raise MLConfigError("random_forest.n_jobs must be -1 or a positive integer")
        if bool(forest.get("oob_score", False)) and not bool(
            forest.get("bootstrap", True)
        ):
            raise MLConfigError("random_forest.oob_score requires bootstrap=true")
        if forest.get("max_samples") is not None and not bool(
            forest.get("bootstrap", True)
        ):
            raise MLConfigError("random_forest.max_samples requires bootstrap=true")

        training = root["training"]
        if int(training.get("stages", 0)) <= 0:
            raise MLConfigError("training.stages must be positive")
        rate = float(training.get("stage_learning_rate", 0.5))
        if not 0 < rate <= 1:
            raise MLConfigError("training.stage_learning_rate must lie in (0, 1]")
        if training.get("monitor", "validation_loss") not in (
            "validation_loss",
            "validation_ctr_ps",
            "validation_corrected_std_ps",
        ):
            raise MLConfigError("training.monitor is invalid")
        max_abs = training.get("max_abs_single_channel_output_ps")
        if max_abs is not None:
            _positive(max_abs, "training.max_abs_single_channel_output_ps")
        if int(root["checkpointing"].get("every_trees", 1)) <= 0:
            raise MLConfigError("checkpointing.every_trees must be positive")
        return

    _validate_common_model_sections(root)
    architecture = root["architecture"]
    if kind == "cnn":
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
        return
    if kind == "time_series_mlp":
        hidden = architecture.get("hidden_units", [])
        if not isinstance(hidden, list):
            raise MLConfigError("architecture.hidden_units must be a list")
        if any(int(value) <= 0 for value in hidden):
            raise MLConfigError("architecture.hidden_units values must be positive")
        return
    raise MLConfigError(f"Unsupported model type: {kind}")


def validate_cnn_config(config: dict[str, Any]) -> None:
    """Backward-compatible validator."""
    validate_model_config(config)
    from .model import model_type

    if model_type(config) != "cnn":
        raise MLConfigError("validate_cnn_config received a non-CNN model config")
