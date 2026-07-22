from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    pass


def _require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{path} must be a JSON object")
    return value


def _require_pair(value: Any, path: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ConfigError(f"{path} must contain exactly two values")
    try:
        return [float(value[0]), float(value[1])]
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{path} must contain numeric values") from exc


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    root = _require_mapping(config, "config")
    for section in (
        "channels",
        "io",
        "waveform",
        "timing_scan",
        "photopeak",
        "selection",
        "fit",
        "plot",
        "cache",
    ):
        _require_mapping(root.get(section), section)

    channels = root["channels"]
    energy = channels.get("energy")
    timing = channels.get("timing")
    polarities = channels.get("polarities")
    if not isinstance(energy, list) or len(energy) != 2:
        raise ConfigError("channels.energy must contain two one-based channel numbers")
    if not isinstance(timing, list) or len(timing) != 2:
        raise ConfigError("channels.timing must contain two one-based channel numbers")
    if not isinstance(polarities, list) or len(polarities) != 4:
        raise ConfigError("channels.polarities must contain four entries")
    if any(int(item) not in (1, 2, 3, 4) for item in energy + timing):
        raise ConfigError("channel numbers must be between 1 and 4")
    if any(int(item) not in (-1, 1) for item in polarities):
        raise ConfigError("channel polarities must be +1 or -1")

    waveform = root["waveform"]
    if int(waveform["baseline_samples"]) <= 0:
        raise ConfigError("waveform.baseline_samples must be positive")
    if float(waveform["trigger_threshold_mV"]) <= 0:
        raise ConfigError("waveform.trigger_threshold_mV must be positive")
    crop = _require_mapping(waveform.get("timing_crop_ns"), "waveform.timing_crop_ns")
    if float(crop["before"]) <= 0 or float(crop["after"]) <= 0:
        raise ConfigError("timing crop widths must be positive")
    if float(waveform["upsample_step_ps"]) <= 0:
        raise ConfigError("waveform.upsample_step_ps must be positive")

    for grid_name in ("led_thresholds_mV", "cfd_fractions"):
        grid = _require_mapping(root["timing_scan"].get(grid_name), f"timing_scan.{grid_name}")
        start, stop, step = float(grid["start"]), float(grid["stop"]), float(grid["step"])
        if step <= 0 or stop < start:
            raise ConfigError(f"invalid timing grid: {grid_name}")

    photopeak = root["photopeak"]
    if float(photopeak["histogram_bin_mV"]) <= 0:
        raise ConfigError("photopeak.histogram_bin_mV must be positive")
    if float(photopeak["selection_sigma_low"]) >= float(photopeak["selection_sigma_high"]):
        raise ConfigError("photopeak selection sigma limits are invalid")

    selection = root["selection"]
    trigger_range = selection.get("energy_trigger_index_range")
    if trigger_range is not None:
        bounds = _require_pair(trigger_range, "selection.energy_trigger_index_range")
        if bounds[1] <= bounds[0]:
            raise ConfigError("energy trigger index range must be increasing")
    noise_max = selection.get("timing_noise_max_mV")
    if noise_max is not None and float(noise_max) <= 0:
        raise ConfigError("selection.timing_noise_max_mV must be positive or null")

    fit = root["fit"]
    fit_range = _require_pair(fit["histogram_range_ps"], "fit.histogram_range_ps")
    if fit_range[1] <= fit_range[0]:
        raise ConfigError("fit histogram range must be increasing")
    for name in ("histogram_bin_ps", "initial_half_width_ps", "iteration_sigma"):
        if float(fit[name]) <= 0:
            raise ConfigError(f"fit.{name} must be positive")
    if int(fit["max_iterations"]) < 1:
        raise ConfigError("fit.max_iterations must be at least one")
    if int(fit["min_events"]) < 3:
        raise ConfigError("fit.min_events must be at least three")

    ml_dataset = root.get("ml_dataset")
    if ml_dataset is not None:
        dataset = _require_mapping(ml_dataset, "ml_dataset")
        if float(dataset["led_threshold_mV"]) <= 0:
            raise ConfigError("ml_dataset.led_threshold_mV must be positive")
        if float(dataset["crossing_step_ps"]) <= 0:
            raise ConfigError("ml_dataset.crossing_step_ps must be positive")
        if float(dataset["timing_window_width_ns"]) <= 0:
            raise ConfigError("ml_dataset.timing_window_width_ns must be positive")

        for name in ("led_search_ns", "energy_search_ns"):
            search = _require_mapping(dataset.get(name), f"ml_dataset.{name}")
            if float(search["before"]) <= 0 or float(search["after"]) <= 0:
                raise ConfigError(f"ml_dataset.{name} widths must be positive")

        polynomial = _require_mapping(dataset.get("polynomial"), "ml_dataset.polynomial")
        if int(polynomial["degree"]) < 0:
            raise ConfigError("ml_dataset.polynomial.degree must be non-negative")
        if float(polynomial["l2_regularization"]) < 0:
            raise ConfigError(
                "ml_dataset.polynomial.l2_regularization must be non-negative"
            )
        if float(polynomial["resample_step_ps"]) <= 0:
            raise ConfigError("ml_dataset.polynomial.resample_step_ps must be positive")
        number_of_samples = (
            float(dataset["timing_window_width_ns"]) * 1000.0
            / float(polynomial["resample_step_ps"])
        ) + 1.0
        if number_of_samples < int(polynomial["degree"]) + 1:
            raise ConfigError(
                "timing window/resampling does not provide enough samples for "
                "the configured polynomial degree"
            )

        fractions = dataset.get("energy_fractions")
        if not isinstance(fractions, list) or len(fractions) < 1:
            raise ConfigError("ml_dataset.energy_fractions must be a non-empty list")
        numeric_fractions = [float(item) for item in fractions]
        if any(not 0.0 < item < 1.0 for item in numeric_fractions):
            raise ConfigError(
                "ml_dataset.energy_fractions values must be strictly between 0 and 1"
            )
        if len(set(numeric_fractions)) != len(numeric_fractions):
            raise ConfigError("ml_dataset.energy_fractions must not contain duplicates")

        parallel = _require_mapping(dataset.get("parallel"), "ml_dataset.parallel")
        if int(parallel.get("workers", 0)) < 0:
            raise ConfigError("ml_dataset.parallel.workers must be non-negative")
        if int(parallel.get("max_auto_workers", 1)) < 1:
            raise ConfigError("ml_dataset.parallel.max_auto_workers must be positive")
        if int(parallel.get("map_chunksize", 1)) < 1:
            raise ConfigError("ml_dataset.parallel.map_chunksize must be positive")
        if int(parallel.get("progress_every", 1)) < 1:
            raise ConfigError("ml_dataset.parallel.progress_every must be positive")

        target = _require_mapping(dataset.get("target"), "ml_dataset.target")
        if str(target.get("center", "mean")).lower() not in {"mean", "median"}:
            raise ConfigError("ml_dataset.target.center must be 'mean' or 'median'")
        if not str(target.get("column_name", "")).strip():
            raise ConfigError("ml_dataset.target.column_name must be non-empty")


def grid_from_config(grid: dict[str, Any]) -> list[float]:
    start = float(grid["start"])
    stop = float(grid["stop"])
    step = float(grid["step"])
    count = int(round((stop - start) / step)) + 1
    values = [start + i * step for i in range(count)]
    if values[-1] < stop - max(1e-12, abs(step) * 1e-9):
        values.append(stop)
    values[-1] = stop if abs(values[-1] - stop) < abs(step) * 1e-8 else values[-1]
    return values


def extraction_fingerprint(config: dict[str, Any]) -> str:
    relevant = {
        "channels": config["channels"],
        "waveform": config["waveform"],
        "timing_scan": config["timing_scan"],
        "max_events": int(config["io"].get("max_events", 0)),
    }
    payload = json.dumps(relevant, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def config_copy(config: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(config)
