from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .common import canonical_hash
from .input_transform import resolve_input_transform
from .prediction import resolve_prediction_config


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


def _resolve_path(project_root: Path, value: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return str(path.resolve())


def load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    with source.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise MLConfigError(f"{source} must contain a JSON object")
    return value


def _finish(config: dict[str, Any], source: Path, root: Path) -> dict[str, Any]:
    result = copy.deepcopy(config)
    result["_config_path"] = str(source)
    result["_config_hash"] = canonical_hash(config)
    result["_project_root"] = str(root)
    return result



def resolve_fit_config(config: dict[str, Any] | None) -> dict[str, Any]:
    fit = copy.deepcopy(config or {})
    fit.setdefault("histogram_bin_ps", 10.0)
    fit.setdefault("initial_half_width_ps", 500.0)
    fit.setdefault("iteration_sigma", 2.5)
    fit.setdefault("max_iterations", 3)
    fit.setdefault("min_events", 10)
    fit.setdefault("minimum_fit_bins", 5)
    fit.setdefault("convergence_tolerance_ps", 0.1)
    fit.setdefault("minimum_sigma_bins", 1.0)
    return fit

def _validate_fit(config: dict[str, Any]) -> None:
    fit = _mapping(config, "fit")
    for name in ("histogram_bin_ps", "initial_half_width_ps", "iteration_sigma"):
        _positive(fit[name], f"fit.{name}")
    for name in ("max_iterations", "min_events", "minimum_fit_bins"):
        if int(fit[name]) <= 0:
            raise MLConfigError(f"fit.{name} must be positive")




def _validate_denoising_config(value: Any, path: str) -> None:
    if value is None:
        return
    denoising = _mapping(value, path)
    enabled = denoising.get("enabled", False)
    if not isinstance(enabled, bool):
        raise MLConfigError(f"{path}.enabled must be boolean")
    method = str(denoising.get("method", "butterworth_lowpass"))
    if method != "butterworth_lowpass":
        raise MLConfigError(f"{path}.method must be 'butterworth_lowpass'")
    if enabled or "cutoff_GHz" in denoising:
        _positive(denoising.get("cutoff_GHz"), f"{path}.cutoff_GHz")
    if enabled or "order" in denoising:
        order_value = denoising.get("order", 4)
        try:
            order = int(order_value)
        except (TypeError, ValueError) as exc:
            raise MLConfigError(f"{path}.order must be an integer") from exc
        if float(order_value) != float(order) or not 1 <= order <= 12:
            raise MLConfigError(f"{path}.order must be an integer from 1 to 12")


def validate_preprocess_config(config: dict[str, Any]) -> None:
    for section in (
        "dataset",
        "data",
        "channels",
        "waveform",
        "selection",
        "photopeak",
        "split",
        "parallelization",
        "cache",
        "logging",
    ):
        _mapping(config.get(section), section)

    dataset = config["dataset"]
    if not str(dataset.get("name", "")).strip():
        raise MLConfigError("dataset.name must be non-empty")
    if dataset.get("role", "training") not in ("training", "blind"):
        raise MLConfigError("dataset.role must be 'training' or 'blind'")
    if not str(dataset.get("output_dir", "")).strip():
        raise MLConfigError("dataset.output_dir must be non-empty")
    blind_test = dataset.get("blind_test")
    if blind_test is not None:
        blind_test = _mapping(blind_test, "dataset.blind_test")
        if dataset.get("role", "training") != "training":
            raise MLConfigError("dataset.blind_test is allowed only for training datasets")
        if not str(blind_test.get("name", "")).strip():
            raise MLConfigError("dataset.blind_test.name must be non-empty")
        if not str(blind_test.get("output_dir", "")).strip():
            raise MLConfigError("dataset.blind_test.output_dir must be non-empty")
        if str(blind_test["output_dir"]) == str(dataset["output_dir"]):
            raise MLConfigError("Training and blind-test output directories must differ")

    if not str(config["data"].get("input_root", "")).strip():
        raise MLConfigError("data.input_root must be non-empty")
    float(config["data"]["true_tof_ps"])
    channels = config["channels"]
    if not isinstance(channels.get("energy"), list) or len(channels["energy"]) != 2:
        raise MLConfigError("channels.energy must contain exactly two channels")
    energy_channels = [int(value) for value in channels["energy"]]
    if any(value <= 0 for value in energy_channels) or len(set(energy_channels)) != 2:
        raise MLConfigError("channels.energy must contain two distinct positive channels")
    if not isinstance(channels.get("polarities"), list) or len(channels["polarities"]) != 2:
        raise MLConfigError("channels.polarities must contain exactly two values")
    if any(int(value) not in (-1, 1) for value in channels["polarities"]):
        raise MLConfigError("channels.polarities values must be +1 or -1")

    waveform = config["waveform"]
    for name in (
        "baseline_samples",
        "search_trigger_threshold_mV",
        "led_threshold_mV",
        "cfd_fraction",
    ):
        _positive(waveform[name], f"waveform.{name}")
    if float(waveform["cfd_fraction"]) > 1.0:
        raise MLConfigError("waveform.cfd_fraction must lie in (0, 1]")
    # Deprecated preprocessing options are accepted for old configs but ignored.
    # Canonical waveform windows are now kept on the native acquisition grid.
    if "upsample_step_ps" in waveform:
        _positive(waveform["upsample_step_ps"], "waveform.upsample_step_ps")
    if "subsample_factor" in waveform:
        _positive(waveform["subsample_factor"], "waveform.subsample_factor")
    for name in ("analysis_crop_ns", "ml_window_ns"):
        window = _mapping(waveform[name], f"waveform.{name}")
        _positive(window["before"], f"waveform.{name}.before")
        _positive(window["after"], f"waveform.{name}.after")

    _validate_denoising_config(waveform.get("denoising"), "waveform.denoising")

    timing_led = waveform.get("timing_channel_led", {"enabled": False})
    timing_led = _mapping(timing_led, "waveform.timing_channel_led")
    timing_enabled = timing_led.get("enabled", False)
    if not isinstance(timing_enabled, bool):
        raise MLConfigError("waveform.timing_channel_led.enabled must be boolean")
    if timing_enabled:
        timing_channels = channels.get("timing")
        if not isinstance(timing_channels, list) or len(timing_channels) != 2:
            raise MLConfigError(
                "channels.timing must contain exactly two channels when timing-channel LED is enabled"
            )
        timing_channels = [int(value) for value in timing_channels]
        if any(value <= 0 for value in timing_channels) or len(set(timing_channels)) != 2:
            raise MLConfigError("channels.timing must contain two distinct positive channels")
        if set(timing_channels).intersection(energy_channels):
            raise MLConfigError("channels.timing must be distinct from channels.energy")
        timing_polarities = channels.get("timing_polarities")
        if not isinstance(timing_polarities, list) or len(timing_polarities) != 2:
            raise MLConfigError(
                "channels.timing_polarities must contain exactly two values when timing-channel LED is enabled"
            )
        if any(int(value) not in (-1, 1) for value in timing_polarities):
            raise MLConfigError("channels.timing_polarities values must be +1 or -1")

    for name in (
        "baseline_samples",
        "search_trigger_threshold_mV",
        "led_threshold_mV",
        "cfd_fraction",
    ):
        if name in timing_led:
            _positive(timing_led[name], f"waveform.timing_channel_led.{name}")
    if "analysis_crop_ns" in timing_led:
        timing_crop = _mapping(
            timing_led["analysis_crop_ns"],
            "waveform.timing_channel_led.analysis_crop_ns",
        )
        _positive(
            timing_crop["before"],
            "waveform.timing_channel_led.analysis_crop_ns.before",
        )
        _positive(
            timing_crop["after"],
            "waveform.timing_channel_led.analysis_crop_ns.after",
        )
    if "cfd_fraction" in timing_led and float(timing_led["cfd_fraction"]) > 1.0:
        raise MLConfigError(
            "waveform.timing_channel_led.cfd_fraction must lie in (0, 1]"
        )
    if "ml_window_ns" in timing_led:
        timing_window = _mapping(
            timing_led["ml_window_ns"],
            "waveform.timing_channel_led.ml_window_ns",
        )
        _positive(
            timing_window["before"],
            "waveform.timing_channel_led.ml_window_ns.before",
        )
        _positive(
            timing_window["after"],
            "waveform.timing_channel_led.ml_window_ns.after",
        )
    _validate_denoising_config(
        timing_led.get("denoising"),
        "waveform.timing_channel_led.denoising",
    )

    split = config["split"]
    fractions = [float(split[name]) for name in ("train_fraction", "validation_fraction", "test_fraction")]
    if any(value <= 0 for value in fractions) or abs(sum(fractions) - 1.0) > 1e-9:
        raise MLConfigError("split fractions must be positive and sum to 1")
    strategy = str(split.get("strategy", "event"))
    allowed = {"event", "stratified_event", "source_file", "contiguous_blocks"}
    if strategy not in allowed:
        raise MLConfigError(f"split.strategy must be one of {sorted(allowed)}")
    if strategy == "contiguous_blocks" and int(split.get("guard_gap_events", 0)) < 0:
        raise MLConfigError("split.guard_gap_events must be non-negative")

    cache = config["cache"]
    for name in ("raw_cache_dir", "selection_cache_dir"):
        if not str(cache.get(name, "")).strip():
            raise MLConfigError(f"cache.{name} must be non-empty")


def load_preprocess_config(path: str | Path, project_root: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    root = Path(project_root).resolve()
    config = load_json(source)
    validate_preprocess_config(config)
    result = _finish(config, source, root)
    result["data"]["input_root"] = _resolve_path(root, result["data"]["input_root"])
    if not Path(result["data"]["input_root"]).is_file():
        raise MLConfigError(f"Input ROOT file does not exist: {result['data']['input_root']}")
    result["dataset"]["output_dir"] = _resolve_path(root, result["dataset"]["output_dir"])
    if "blind_test" in result["dataset"]:
        result["dataset"]["blind_test"]["output_dir"] = _resolve_path(
            root, result["dataset"]["blind_test"]["output_dir"]
        )
    result["cache"]["raw_cache_dir"] = _resolve_path(root, result["cache"]["raw_cache_dir"])
    result["cache"]["selection_cache_dir"] = _resolve_path(
        root, result["cache"]["selection_cache_dir"]
    )
    return result


def validate_train_config(config: dict[str, Any]) -> None:
    for section in ("model", "training", "output", "fit", "plotting", "logging"):
        _mapping(config.get(section), section)
    datasets = config.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise MLConfigError("datasets must be a non-empty list")
    for value in datasets:
        if isinstance(value, str) and value:
            continue
        if isinstance(value, dict) and str(value.get("dataset", "")).strip():
            continue
        raise MLConfigError(
            "Each dataset must be a path string or an object containing dataset"
        )

    model = config["model"]
    try:
        resolve_input_transform(config)
        resolve_prediction_config(config)
    except ValueError as exc:
        raise MLConfigError(str(exc)) from exc
    if not str(model.get("name", "")).strip():
        raise MLConfigError("model.name must be non-empty")
    model_type = str(model.get("type", "")).strip()
    model_options = {
        key: value
        for key, value in model.items()
        if key not in ("type", "name", "input_transform")
    }
    try:
        from .models import validate_model, validate_model_training

        validate_model(model_type, model_options)
        validate_model_training(model_type, config)
    except ValueError as exc:
        raise MLConfigError(f"Invalid {model_type or 'model'} configuration: {exc}") from exc

    if not str(config["output"].get("train_dir", "")).strip():
        raise MLConfigError("output.train_dir must be non-empty")
    _validate_fit(config["fit"])


def load_train_config(path: str | Path, project_root: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    root = Path(project_root).resolve()
    config = load_json(source)
    validate_train_config(config)
    result = _finish(config, source, root)
    result["fit"] = resolve_fit_config(result.get("fit"))
    result["input_transform"] = resolve_input_transform(result)
    result["prediction"] = resolve_prediction_config(result)
    result["model"].pop("input_transform", None)
    resolved_datasets = []
    for value in result["datasets"]:
        if isinstance(value, str):
            resolved_datasets.append(_resolve_path(root, value))
        else:
            item = dict(value)
            item["dataset"] = _resolve_path(root, item["dataset"])
            resolved_datasets.append(item)
    result["datasets"] = resolved_datasets
    result["output"]["train_dir"] = _resolve_path(root, result["output"]["train_dir"])
    return result


def validate_evaluate_config(config: dict[str, Any]) -> None:
    for section in ("output", "fit", "plotting", "logging"):
        _mapping(config.get(section), section)
    blind_tests = config.get("blind_tests")
    if not isinstance(blind_tests, list) or not blind_tests:
        raise MLConfigError("blind_tests must be a non-empty list")
    for item in blind_tests:
        if isinstance(item, str):
            continue
        if not isinstance(item, dict) or not str(item.get("dataset", "")).strip():
            raise MLConfigError("Each blind test must be a path string or an object with dataset")
    models = config.get("models", [])
    if models is not None and (
        not isinstance(models, list) or not all(isinstance(value, str) for value in models)
    ):
        raise MLConfigError("models must be a list of model run/checkpoint paths")
    standard_methods = config.get("standard_methods", {})
    has_standard = isinstance(standard_methods, dict) and bool(standard_methods)
    if not models and not str(config.get("model_search_dir", "")).strip() and not has_standard:
        raise MLConfigError("Provide models, model_search_dir, or standard_methods")
    if not str(config["output"].get("evaluation_dir", "")).strip():
        raise MLConfigError("output.evaluation_dir must be non-empty")
    correlation = config.get("model_output_correlation", {})
    if correlation is not None:
        _mapping(correlation, "model_output_correlation")
        if not isinstance(correlation.get("enabled", True), bool):
            raise MLConfigError("model_output_correlation.enabled must be boolean")
        if not isinstance(correlation.get("annotate", True), bool):
            raise MLConfigError("model_output_correlation.annotate must be boolean")

    correction_analysis = config.get("correction_analysis", {})
    if correction_analysis is not None:
        _mapping(correction_analysis, "correction_analysis")
        if not isinstance(correction_analysis.get("enabled", False), bool):
            raise MLConfigError("correction_analysis.enabled must be boolean")
        if not isinstance(correction_analysis.get("save_waveform_plots", True), bool):
            raise MLConfigError(
                "correction_analysis.save_waveform_plots must be boolean"
            )
        try:
            top_n = int(correction_analysis.get("top_n", 10))
        except (TypeError, ValueError) as exc:
            raise MLConfigError("correction_analysis.top_n must be an integer") from exc
        if top_n <= 0:
            raise MLConfigError("correction_analysis.top_n must be positive")
        try:
            minimum = float(
                correction_analysis.get("minimum_improvement_ps", 0.0)
            )
        except (TypeError, ValueError) as exc:
            raise MLConfigError(
                "correction_analysis.minimum_improvement_ps must be numeric"
            ) from exc
        if minimum < 0.0:
            raise MLConfigError(
                "correction_analysis.minimum_improvement_ps must be non-negative"
            )
    _validate_fit(config["fit"])


def load_evaluate_config(path: str | Path, project_root: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    root = Path(project_root).resolve()
    config = load_json(source)
    validate_evaluate_config(config)
    result = _finish(config, source, root)
    result["fit"] = resolve_fit_config(result.get("fit"))
    resolved_tests = []
    for item in result["blind_tests"]:
        if isinstance(item, str):
            resolved_tests.append({"name": Path(item).name, "dataset": _resolve_path(root, item)})
        else:
            value = dict(item)
            value["dataset"] = _resolve_path(root, value["dataset"])
            value.setdefault("name", Path(value["dataset"]).name)
            resolved_tests.append(value)
    result["blind_tests"] = resolved_tests
    result["models"] = [_resolve_path(root, value) for value in (result.get("models") or [])]
    if str(result.get("model_search_dir", "")).strip():
        result["model_search_dir"] = _resolve_path(root, result["model_search_dir"])
    if isinstance(result.get("standard_methods"), dict):
        spline = result["standard_methods"].get("linear_spline")
        if isinstance(spline, dict) and str(spline.get("artifact", "")).strip():
            spline["artifact"] = _resolve_path(root, spline["artifact"])
    result["output"]["evaluation_dir"] = _resolve_path(root, result["output"]["evaluation_dir"])
    return result
