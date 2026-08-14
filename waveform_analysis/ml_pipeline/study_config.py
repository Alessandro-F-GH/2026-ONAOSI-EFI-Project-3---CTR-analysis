from __future__ import annotations

import copy
import itertools
import json
import math
from pathlib import Path
from typing import Any

from .common import canonical_hash
from .config import MLConfigError, resolve_fit_config
from .models import model_registry

CHANNEL_MODES: dict[str, tuple[str, str]] = {
    "energy_to_energy": ("energy", "energy_led"),
    "energy_to_timing": ("energy", "timing_led"),
    "timing_to_timing": ("timing", "timing_led"),
}
RETAINED_MODELS = {"linear_svr", "constructive_mlp_encoder", "cnn_regressor"}


def _finite(value: Any, name: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise MLConfigError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise MLConfigError(f"{name} must be finite")
    if positive and number <= 0.0:
        raise MLConfigError(f"{name} must be > 0")
    if nonnegative and number < 0.0:
        raise MLConfigError(f"{name} must be >= 0")
    return number


def _validate_pair(values: Any, name: str, *, polarity: bool = False) -> list[int]:
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise MLConfigError(f"{name} must contain exactly two values")
    result = [int(v) for v in values]
    if polarity:
        if any(v not in {-1, 1} for v in result):
            raise MLConfigError(f"{name} entries must be -1 or +1")
    else:
        if any(v <= 0 for v in result) or result[0] == result[1]:
            raise MLConfigError(f"{name} must contain two distinct positive channel numbers")
    return result


def _validate_waveform_config(config: dict[str, Any], name: str) -> None:
    if not isinstance(config, dict):
        raise MLConfigError(f"{name} must be an object")
    if "baseline_samples" in config and int(config["baseline_samples"]) <= 0:
        raise MLConfigError(f"{name}.baseline_samples must be positive")
    for key in ("search_trigger_threshold_mV", "led_threshold_mV"):
        if key in config:
            _finite(config[key], f"{name}.{key}", positive=True)
    if "cfd_fraction" in config:
        value = _finite(config["cfd_fraction"], f"{name}.cfd_fraction", positive=True)
        if value > 1.0:
            raise MLConfigError(f"{name}.cfd_fraction must be <= 1")
    crop = config.get("analysis_crop_ns")
    if crop is not None:
        if not isinstance(crop, dict):
            raise MLConfigError(f"{name}.analysis_crop_ns must be an object")
        _finite(crop.get("before"), f"{name}.analysis_crop_ns.before", positive=True)
        _finite(crop.get("after"), f"{name}.analysis_crop_ns.after", positive=True)


def _validate_search_parameter(spec: Any, name: str, *, grid: bool) -> None:
    if not isinstance(spec, dict):
        raise MLConfigError(f"{name} must be an object")
    typ = str(spec.get("type", "categorical"))
    if grid and typ != "categorical":
        raise MLConfigError(f"{name}: grid search supports categorical parameters only")
    if typ == "categorical":
        values = spec.get("values")
        if not isinstance(values, list) or not values:
            raise MLConfigError(f"{name}.values must be a non-empty list")
    elif typ in {"uniform", "loguniform", "int"}:
        if "low" not in spec or "high" not in spec:
            raise MLConfigError(f"{name} requires low/high")
        low = _finite(spec["low"], f"{name}.low")
        high = _finite(spec["high"], f"{name}.high")
        if high < low or (typ != "int" and high == low):
            raise MLConfigError(f"{name}.high must be greater than low")
        if typ == "loguniform" and low <= 0.0:
            raise MLConfigError(f"{name}.low must be > 0 for loguniform")
    else:
        raise MLConfigError(f"{name}.type {typ!r} is unsupported")


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def discover_root_files(config: dict[str, Any]) -> list[Path]:
    data = config["data"]
    folder = Path(data["root_folder"])
    pattern = str(data.get("root_glob", "*.root"))
    files = sorted(folder.rglob(pattern) if bool(data.get("recursive", False)) else folder.glob(pattern))
    return [p.resolve() for p in files if p.is_file()]


def load_model_space(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise MLConfigError(f"Model space {path} must contain a JSON object")
    for key in ("id", "model_type", "base_train_config", "search"):
        if key not in value:
            raise MLConfigError(f"Model space {path} is missing {key!r}")
    model_type = str(value["model_type"])
    if model_type not in RETAINED_MODELS:
        raise MLConfigError(
            f"Model space {path.name} uses obsolete model type {model_type!r}; "
            f"retained waveform models are {sorted(RETAINED_MODELS)}"
        )
    if model_type not in model_registry():
        raise MLConfigError(f"Model type {model_type!r} is not registered")
    return value


def _window(value: dict[str, Any], index: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MLConfigError("Every windows_ns entry must be an object")
    if "before_ns" in value or "after_ns" in value:
        before = float(value["before_ns"])
        after = float(value["after_ns"])
    else:
        start = float(value["start_ns"])
        end = float(value["end_ns"])
        if start > 0 or end <= 0 or end <= start:
            raise MLConfigError("Window start_ns/end_ns must straddle zero (e.g. -4, 20)")
        before, after = -start, end
    if before <= 0 or after <= 0:
        raise MLConfigError("Window before_ns and after_ns must be positive")
    return {"id": str(value.get("id", f"w{index}")), "before_ns": before, "after_ns": after}


def _default_fit(raw: dict[str, Any] | None) -> dict[str, Any]:
    return resolve_fit_config(raw)


def _validate_search(space: dict[str, Any]) -> None:
    search = space["search"]
    if not isinstance(search, dict):
        raise MLConfigError(f"{space['id']}.search must be an object")
    method = str(search.get("method", "grid"))
    if method not in {"grid", "random"}:
        raise MLConfigError(f"{space['id']}.search.method must be grid or random")
    parameters = search.get("parameters", {})
    if not isinstance(parameters, dict):
        raise MLConfigError(f"{space['id']}.search.parameters must be an object")
    for parameter, spec in parameters.items():
        _validate_search_parameter(spec, f"{space['id']}.search.parameters.{parameter}", grid=method == "grid")
    if method == "random" and int(search.get("n_trials", 0)) <= 0:
        raise MLConfigError(f"{space['id']}.search.n_trials must be positive")
    if not isinstance(space.get("base_train_config"), dict):
        raise MLConfigError(f"{space['id']}.base_train_config must be an object")
    model = space["base_train_config"].get("model", {})
    if str(model.get("type", "")) != str(space["model_type"]):
        raise MLConfigError(f"{space['id']} base model.type must equal model_type")


def load_study_config(path: str | Path, project_root: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    root = Path(project_root).resolve()
    with source.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, dict):
        raise MLConfigError("Study config must be a JSON object")
    cfg = copy.deepcopy(raw)

    experiment = cfg.setdefault("experiment", {})
    if not str(experiment.get("name", "")).strip():
        raise MLConfigError("experiment.name is required")
    experiment["output_dir"] = str(_resolve(root, experiment.get("output_dir", f"results/studies/{experiment['name']}")))

    data = cfg.setdefault("data", {})
    if "root_folder" not in data:
        raise MLConfigError("data.root_folder is required")
    data["root_folder"] = str(_resolve(root, data["root_folder"]))
    data.setdefault("root_glob", "*.root")
    data.setdefault("recursive", False)
    data.setdefault("true_tof_ps", 0.0)
    data["true_tof_ps"] = _finite(data["true_tof_ps"], "data.true_tof_ps")
    channels = data.get("channels")
    if not isinstance(channels, dict):
        raise MLConfigError("data.channels must be an object")
    channels["energy"] = _validate_pair(channels.get("energy"), "data.channels.energy")
    channels["polarities"] = _validate_pair(channels.get("polarities", [1, 1]), "data.channels.polarities", polarity=True)
    modes = [str(v) for v in cfg.get("channel_modes", list(CHANNEL_MODES))]
    unknown_modes = sorted(set(modes) - set(CHANNEL_MODES))
    if unknown_modes:
        raise MLConfigError(f"Unsupported channel modes: {unknown_modes}; available: {sorted(CHANNEL_MODES)}")
    if any(CHANNEL_MODES[m][0] == "timing" or CHANNEL_MODES[m][1] == "timing_led" for m in modes):
        channels["timing"] = _validate_pair(channels.get("timing"), "data.channels.timing")
        channels["timing_polarities"] = _validate_pair(
            channels.get("timing_polarities", [1, 1]), "data.channels.timing_polarities", polarity=True
        )
    cfg["channel_modes"] = modes

    raw_windows = cfg.get("windows_ns") or [{"id": "default", "start_ns": -4.0, "end_ns": 20.0}]
    cfg["windows_ns"] = [_window(v, i) for i, v in enumerate(raw_windows)]
    window_ids = [window["id"] for window in cfg["windows_ns"]]
    if len(window_ids) != len(set(window_ids)):
        raise MLConfigError("windows_ns IDs must be unique")

    preprocessing = cfg.setdefault("preprocessing", {})
    preprocessing["prepared_dir"] = str(_resolve(root, preprocessing.get("prepared_dir", "processed_data/ml_prepared")))
    preprocessing.setdefault("materialization_chunk_size", 2048)
    if int(preprocessing["materialization_chunk_size"]) <= 0:
        raise MLConfigError("preprocessing.materialization_chunk_size must be positive")
    preprocessing.setdefault("cleanup_raw_cache", True)
    common = preprocessing.setdefault("common", {})
    _validate_waveform_config(common, "preprocessing.common")
    energy_cfg = copy.deepcopy(common)
    energy_cfg.update(copy.deepcopy(preprocessing.setdefault("energy", {})))
    _validate_waveform_config(energy_cfg, "preprocessing.energy")
    timing_cfg = copy.deepcopy(common)
    timing_cfg.update(copy.deepcopy(preprocessing.setdefault("timing", {})))
    _validate_waveform_config(timing_cfg, "preprocessing.timing")
    selection = preprocessing.setdefault("selection", {})
    if not isinstance(selection, dict):
        raise MLConfigError("preprocessing.selection must be an object")
    minimum_events = int(selection.get("minimum_events", 100))
    if minimum_events < 3:
        raise MLConfigError("preprocessing.selection.minimum_events must be >= 3")
    led_outlier = selection.get("led_outlier_rejection", {}) or {}
    if not isinstance(led_outlier, dict):
        raise MLConfigError("preprocessing.selection.led_outlier_rejection must be an object")
    if bool(led_outlier.get("enabled", False)):
        if "max_distance_ps" in led_outlier:
            raise MLConfigError(
                "preprocessing.selection.led_outlier_rejection.max_distance_ps is obsolete; "
                "use zscore_limit for robust dataset-level LED mismatch rejection"
            )
        _finite(
            led_outlier.get("zscore_limit", 6.0),
            "preprocessing.selection.led_outlier_rejection.zscore_limit",
            positive=True,
        )

    photopeak = preprocessing.setdefault("photopeak", {"enabled": False})
    if not isinstance(photopeak, dict):
        raise MLConfigError("preprocessing.photopeak must be an object")
    if bool(photopeak.get("enabled", False)):
        _finite(photopeak.get("histogram_bin_mV", 1.0), "preprocessing.photopeak.histogram_bin_mV", positive=True)
        quantile = _finite(photopeak.get("search_quantile_min", 0.2), "preprocessing.photopeak.search_quantile_min", nonnegative=True)
        if quantile >= 1.0:
            raise MLConfigError("preprocessing.photopeak.search_quantile_min must be < 1")
        _finite(photopeak.get("smoothing_sigma_bins", 2.0), "preprocessing.photopeak.smoothing_sigma_bins", nonnegative=True)
        _finite(photopeak.get("initial_half_width_mV", 15.0), "preprocessing.photopeak.initial_half_width_mV", positive=True)
        _finite(photopeak.get("iteration_sigma", 2.5), "preprocessing.photopeak.iteration_sigma", positive=True)
        if int(photopeak.get("max_iterations", 6)) < 1:
            raise MLConfigError("preprocessing.photopeak.max_iterations must be >= 1")
        if float(photopeak.get("selection_sigma_low", -2.0)) >= float(photopeak.get("selection_sigma_high", 4.0)):
            raise MLConfigError("photopeak selection_sigma_low must be below selection_sigma_high")

    if "input_variants" in preprocessing:
        raise MLConfigError(
            "preprocessing.input_variants is obsolete because it duplicates the ML search. "
            "Use preprocessing.input_variant_by_channel with explicit energy/timing choices."
        )
    variants = preprocessing.setdefault(
        "input_variant_by_channel", {"energy": "raw", "timing": "raw"}
    )
    if not isinstance(variants, dict):
        raise MLConfigError("preprocessing.input_variant_by_channel must be an object")
    unknown_variant_channels = sorted(set(variants) - {"energy", "timing"})
    if unknown_variant_channels:
        raise MLConfigError(
            "preprocessing.input_variant_by_channel has unsupported keys: "
            + ", ".join(unknown_variant_channels)
        )
    for channel_type in ("energy", "timing"):
        value = str(variants.get(channel_type, "raw")).strip().lower()
        if value not in {"raw", "denoised"}:
            raise MLConfigError(
                f"preprocessing.input_variant_by_channel.{channel_type} must be raw or denoised"
            )
        variants[channel_type] = value
    preprocessing["input_variant_by_channel"] = variants

    denoising = preprocessing.setdefault("denoising", {})
    needs_denoising = any(value == "denoised" for value in variants.values())
    # ``enabled`` is derived from the channel policy, not a second independent knob.
    denoising["enabled"] = needs_denoising
    if needs_denoising:
        denoising.setdefault("method", "butterworth_lowpass")
        denoising.setdefault("cutoff_GHz", 1.0)
        denoising.setdefault("order", 4)
        if str(denoising["method"]) != "butterworth_lowpass":
            raise MLConfigError("Only butterworth_lowpass denoising is supported")
        _finite(denoising["cutoff_GHz"], "preprocessing.denoising.cutoff_GHz", positive=True)
        if int(denoising["order"]) < 1:
            raise MLConfigError("preprocessing.denoising.order must be >= 1")
    factors = [int(v) for v in preprocessing.get("subsampling_factors", [1])]
    if any(v <= 0 for v in factors):
        raise MLConfigError("preprocessing.subsampling_factors must be positive")
    preprocessing["subsampling_factors"] = factors

    cv = cfg.setdefault("cross_validation", {})
    cv.setdefault("n_splits", 5)
    cv.setdefault("seed", 20260813)
    cv.setdefault("blind_fraction", 0.2)
    cv.setdefault("early_stop_fraction", 0.15)
    if int(cv["n_splits"]) < 2:
        raise MLConfigError("cross_validation.n_splits must be >= 2")
    if not 0.0 < float(cv["blind_fraction"]) < 0.5:
        raise MLConfigError("cross_validation.blind_fraction must be in (0, 0.5)")
    if not 0.0 < float(cv["early_stop_fraction"]) < 0.5:
        raise MLConfigError("cross_validation.early_stop_fraction must be in (0, 0.5)")

    cfg["fit"] = _default_fit(cfg.get("fit"))
    if int(cfg["fit"]["min_events"]) > minimum_events:
        raise MLConfigError(
            "fit.min_events cannot exceed preprocessing.selection.minimum_events; "
            "otherwise a successfully prepared dataset may be impossible to evaluate"
        )
    reporting = cfg.setdefault("reporting", {})
    reporting.setdefault("dpi", 180)
    if int(reporting["dpi"]) <= 0:
        raise MLConfigError("reporting.dpi must be positive")
    reporting.setdefault("voltage_pattern", r"(?P<voltage>\d+(?:\.\d+)?)V")
    reporting.setdefault("save_final_fit_plots", True)
    xai = reporting.setdefault("xai", {"enabled": True, "max_events": 512})
    if bool(xai.get("enabled", True)):
        if int(xai.get("max_events", 512)) <= 0 or int(xai.get("integrated_gradient_steps", 16)) <= 0:
            raise MLConfigError("reporting.xai max_events and integrated_gradient_steps must be positive")

    model_dir = _resolve(root, cfg.get("model_spaces_dir", "config/model_spaces"))
    cfg["model_spaces_dir"] = str(model_dir)
    requested = [str(v) for v in cfg.get("models", ["linear_svr", "constructive_mlp", "cnn"])]
    if len(requested) != len(set(requested)):
        raise MLConfigError("models entries must be unique")
    model_spaces: list[dict[str, Any]] = []
    for name in requested:
        candidate = model_dir / f"{name}.json"
        if not candidate.is_file():
            raise MLConfigError(f"Model-space config not found: {candidate}")
        space = load_model_space(candidate)
        _validate_search(space)
        model_spaces.append(space)
    if len({str(space["id"]) for space in model_spaces}) != len(model_spaces):
        raise MLConfigError("model-space IDs must be unique")
    cfg["models"] = requested
    cfg["_model_spaces"] = model_spaces

    multithreshold = cfg.setdefault("multithreshold", {"enabled": False})
    multithreshold.setdefault("enabled", False)
    if multithreshold["enabled"]:
        thresholds = [_finite(v, "multithreshold.thresholds_mV[]", positive=True) for v in multithreshold.get("thresholds_mV", [])]
        if not thresholds:
            raise MLConfigError("multithreshold.thresholds_mV must be non-empty")
        multithreshold["thresholds_mV"] = sorted(set(thresholds))
        multithreshold.setdefault("min_thresholds", 1)
        multithreshold.setdefault("max_thresholds", min(4, len(thresholds)))
        min_thresholds = int(multithreshold["min_thresholds"])
        max_thresholds = int(multithreshold["max_thresholds"])
        if min_thresholds < 1 or max_thresholds < min_thresholds or max_thresholds > len(multithreshold["thresholds_mV"]):
            raise MLConfigError("multithreshold min/max_thresholds are inconsistent with thresholds_mV")
        kernels = [str(v).lower() for v in multithreshold.setdefault("kernels", ["linear", "rbf"])]
        if not kernels or any(v not in {"linear", "rbf"} for v in kernels):
            raise MLConfigError("multithreshold.kernels supports only linear/rbf")
        multithreshold["kernels"] = list(dict.fromkeys(kernels))
        multithreshold.setdefault("C_values", [1.0, 10.0, 100.0])
        multithreshold["C_values"] = [_finite(v, "multithreshold.C_values[]", positive=True) for v in multithreshold["C_values"]]
        multithreshold.setdefault("epsilon_values_ps", [0.0, 10.0, 30.0])
        multithreshold["epsilon_values_ps"] = [_finite(v, "multithreshold.epsilon_values_ps[]", nonnegative=True) for v in multithreshold["epsilon_values_ps"]]
        multithreshold.setdefault("gamma_values", ["scale", "auto"])
        gamma_values: list[Any] = []
        for value in multithreshold["gamma_values"]:
            if isinstance(value, str):
                if value not in {"scale", "auto"}:
                    raise MLConfigError("multithreshold.gamma_values strings must be scale/auto")
                gamma_values.append(value)
            else:
                gamma_values.append(_finite(value, "multithreshold.gamma_values[]", positive=True))
        multithreshold["gamma_values"] = gamma_values
        window_id = str(multithreshold.get("window_id", cfg["windows_ns"][0]["id"]))
        if window_id not in window_ids:
            raise MLConfigError(f"multithreshold.window_id {window_id!r} is not in windows_ns")
        multithreshold["window_id"] = window_id

    cfg.setdefault("logging", {"level": "INFO"})
    cfg["_config_path"] = str(source)
    cfg["_project_root"] = str(root)
    cfg["_config_hash"] = canonical_hash(raw)
    return cfg


def set_nested(mapping: dict[str, Any], dotted: str, value: Any) -> None:
    current = mapping
    parts = dotted.split(".")
    for name in parts[:-1]:
        current = current.setdefault(name, {})
    current[parts[-1]] = value


def candidate_overrides(space: dict[str, Any], *, seed: int) -> list[dict[str, Any]]:
    """Resolve one model-space search into deterministic candidate override dictionaries."""
    import numpy as np

    search = space["search"]
    params = search.get("parameters", {})
    method = str(search.get("method", "grid"))
    if not params:
        return [{}]

    def grid_values(spec: dict[str, Any]) -> list[Any]:
        typ = str(spec.get("type", "categorical"))
        if typ != "categorical":
            raise MLConfigError("Grid search supports categorical parameter values only")
        return list(spec.get("values", []))

    names = list(params)
    if method == "grid":
        values = [grid_values(params[name]) for name in names]
        return [dict(zip(names, combo)) for combo in itertools.product(*values)]

    rng = np.random.default_rng(seed)
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    attempts = 0
    n_trials = int(search["n_trials"])
    while len(output) < n_trials and attempts < max(1000, n_trials * 100):
        attempts += 1
        row: dict[str, Any] = {}
        for name in names:
            spec = params[name]
            typ = str(spec.get("type", "categorical"))
            if typ == "categorical":
                values = list(spec["values"])
                row[name] = copy.deepcopy(values[int(rng.integers(0, len(values)))])
            elif typ == "uniform":
                row[name] = float(rng.uniform(float(spec["low"]), float(spec["high"])))
            elif typ == "loguniform":
                row[name] = float(np.exp(rng.uniform(np.log(float(spec["low"])), np.log(float(spec["high"])))))
            elif typ == "int":
                row[name] = int(rng.integers(int(spec["low"]), int(spec["high"]) + 1))
            else:
                raise MLConfigError(f"Unsupported search parameter type {typ!r}")
        key = canonical_hash(row)
        if key not in seen:
            output.append(row)
            seen.add(key)
    if len(output) < n_trials:
        raise MLConfigError(f"Could generate only {len(output)} unique random candidates for {space['id']}")
    return output
