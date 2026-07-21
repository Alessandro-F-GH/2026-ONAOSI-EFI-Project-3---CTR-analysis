from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    pass


def _positive_int(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{name} must be an integer >= {minimum}")
    return value


def _positive_number(value: Any, name: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be numeric")
    number = float(value)
    if number < minimum:
        raise ConfigError(f"{name} must be >= {minimum}")
    return number



def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigError(f"{name} must be finite")
    return number



def _apply_defaults_and_migrations(cfg: dict[str, Any]) -> None:
    output = cfg.setdefault("analysis_output", {})
    output.setdefault("large_table_format", "csv")
    output.setdefault("diagnostic_mode", "compact")

    matching = cfg.setdefault("matching_model", {})
    matching.setdefault("method", "average_delay")
    if "average_delay" not in matching:
        legacy = matching.get("ridge_polynomial", {})
        legacy_filter = legacy.get("outlier_filter", {}) if isinstance(legacy, dict) else {}
        matching["average_delay"] = {
            "outlier_filter": {
                "enabled": bool(legacy_filter.get("enabled", True)),
                "z_threshold": float(legacy_filter.get("z_threshold", 3.5)),
                "max_iterations": int(legacy_filter.get("max_iterations", 8)),
                "minimum_scale_lsb": float(legacy_filter.get("minimum_scale_lsb", 2.0)),
            }
        }
    inference = matching.setdefault("inference", {})
    if "maximum_deviation_ns" not in inference:
        inference["maximum_deviation_ns"] = inference.get("maximum_prediction_error_ns")

    streaming = cfg.setdefault("preprocessing", {}).setdefault(
        "streaming_physical_time", {}
    )
    streaming.setdefault("enabled", True)
    streaming.setdefault("require_same_board", True)
    streaming.setdefault("energy_leading_pair_window_ns", 5.0)

    reconstruction = cfg.setdefault("preprocessing", {}).setdefault(
        "pulse_reconstruction", {}
    )
    reconstruction.setdefault("tot_lead_match_tolerance_ns", 0.05)
    reconstruction.setdefault("fallback_when_tot_missing", True)
    reconstruction.setdefault("fallback_when_tot_inconsistent", False)

    bootstrap = cfg.setdefault("bootstrap", {})
    bootstrap.setdefault("mode", "full_analysis")

def load_config(path: str | Path, root: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    if not isinstance(cfg, dict):
        raise ConfigError("Configuration root must be a JSON object")
    _apply_defaults_and_migrations(cfg)
    validate_config(cfg)
    project_root = Path(root).resolve()
    cfg["paths"]["input_dir"] = str(
        (project_root / cfg["paths"]["input_dir"]).resolve()
    )
    cfg["paths"]["output_dir"] = str(
        (project_root / cfg["paths"]["output_dir"]).resolve()
    )
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    required = (
        "paths",
        "files",
        "runs",
        "tasks",
        "preprocessing",
        "matching_model",
        "channels",
        "peak_selection",
        "alignment_filter",
        "fit",
        "bootstrap",
        "plots",
        "thresholds",
        "analysis_output",
    )
    for section in required:
        if not isinstance(cfg.get(section), dict):
            raise ConfigError(f"Missing configuration section: {section}")

    for key in ("input_dir", "output_dir"):
        if not str(cfg["paths"].get(key, "")).strip():
            raise ConfigError(f"paths.{key} is required")
    if not str(cfg["files"].get("data_pattern", "")).strip():
        raise ConfigError("files.data_pattern is required")
    if not isinstance(cfg["files"].get("recursive"), bool):
        raise ConfigError("files.recursive must be true or false")
    for key in ("include", "overwrite"):
        if not isinstance(cfg["runs"].get(key), list):
            raise ConfigError(f"runs.{key} must be a list")

    for key in (
        "streaming_pulse_decode",
        "candidate_preprocessing",
        "energy_selection",
        "matching_training",
        "matching_model",
        "preprocessing",
        "selection",
        "fit",
    ):
        task = cfg["tasks"].get(key)
        if not isinstance(task, dict):
            raise ConfigError(f"tasks.{key} is required")
        for flag in ("enabled", "overwrite"):
            if not isinstance(task.get(flag), bool):
                raise ConfigError(f"tasks.{key}.{flag} must be true or false")

    preprocessing = cfg["preprocessing"]
    streaming_physical = preprocessing.get("streaming_physical_time")
    if not isinstance(streaming_physical, dict):
        raise ConfigError(
            "preprocessing.streaming_physical_time is required"
        )
    for flag in ("enabled", "require_same_board"):
        if not isinstance(streaming_physical.get(flag), bool):
            raise ConfigError(
                f"preprocessing.streaming_physical_time.{flag} must be true or false"
            )
    if not bool(streaming_physical["enabled"]):
        raise ConfigError("STREAMING physical-time preprocessing must be enabled")
    _positive_number(
        streaming_physical.get("energy_leading_pair_window_ns"),
        "preprocessing.streaming_physical_time.energy_leading_pair_window_ns",
        1e-12,
    )

    reconstruction = preprocessing.get("pulse_reconstruction")
    if not isinstance(reconstruction, dict):
        raise ConfigError("preprocessing.pulse_reconstruction is required")
    if not isinstance(reconstruction.get("deduplicate_exact_edges"), bool):
        raise ConfigError(
            "preprocessing.pulse_reconstruction.deduplicate_exact_edges "
            "must be true or false"
        )
    for flag in (
        "fallback_when_tot_missing",
        "fallback_when_tot_inconsistent",
    ):
        if not isinstance(reconstruction.get(flag), bool):
            raise ConfigError(
                f"preprocessing.pulse_reconstruction.{flag} must be true or false"
            )
    _positive_number(
        reconstruction.get("tot_lead_match_tolerance_ns"),
        "preprocessing.pulse_reconstruction.tot_lead_match_tolerance_ns",
        0.0,
    )
    minimum_tot_ns = _positive_number(
        reconstruction.get("minimum_tot_ns"),
        "preprocessing.pulse_reconstruction.minimum_tot_ns",
        0.0,
    )
    maximum_energy_tot_ns = _positive_number(
        reconstruction.get("maximum_energy_tot_ns"),
        "preprocessing.pulse_reconstruction.maximum_energy_tot_ns",
        1e-12,
    )
    if minimum_tot_ns >= maximum_energy_tot_ns:
        raise ConfigError(
            "minimum_tot_ns must be smaller than maximum_energy_tot_ns"
        )

    for key in ("signal_a", "time_a", "signal_b", "time_b"):
        _positive_int(cfg["channels"].get(key), f"channels.{key}")
    if len(
        {
            int(cfg["channels"][key])
            for key in ("signal_a", "time_a", "signal_b", "time_b")
        }
    ) != 4:
        raise ConfigError("The four analysis channels must be distinct")

    matching = cfg["matching_model"]
    for section in ("training", "average_delay", "inference"):
        if not isinstance(matching.get(section), dict):
            raise ConfigError(f"matching_model.{section} is required")
    if str(matching.get("method", "")).lower() != "average_delay":
        raise ConfigError("matching_model.method must be average_delay")
    training = matching["training"]
    _positive_number(training.get("window_ns"), "matching_model.training.window_ns", 1e-12)
    _positive_int(training.get("minimum_samples"), "matching_model.training.minimum_samples", 3)
    average_delay = matching["average_delay"]
    outlier_filter = average_delay.get("outlier_filter")
    if not isinstance(outlier_filter, dict):
        raise ConfigError("matching_model.average_delay.outlier_filter is required")
    if not isinstance(outlier_filter.get("enabled"), bool):
        raise ConfigError(
            "matching_model.average_delay.outlier_filter.enabled must be true or false"
        )
    _positive_number(
        outlier_filter.get("z_threshold"),
        "matching_model.average_delay.outlier_filter.z_threshold",
        1e-9,
    )
    _positive_int(
        outlier_filter.get("max_iterations"),
        "matching_model.average_delay.outlier_filter.max_iterations",
        1,
    )
    _positive_number(
        outlier_filter.get("minimum_scale_lsb"),
        "matching_model.average_delay.outlier_filter.minimum_scale_lsb",
        1e-12,
    )
    inference = matching["inference"]
    _positive_number(
        inference.get("candidate_window_ns"),
        "matching_model.inference.candidate_window_ns",
        1e-12,
    )
    if float(training["window_ns"]) > float(inference["candidate_window_ns"]):
        raise ConfigError(
            "matching_model.training.window_ns must be <= "
            "matching_model.inference.candidate_window_ns"
        )
    maximum_deviation = inference.get("maximum_deviation_ns")
    if maximum_deviation is not None:
        _positive_number(
            maximum_deviation,
            "matching_model.inference.maximum_deviation_ns",
            0.0,
        )

    output = cfg["analysis_output"]
    if str(output.get("large_table_format", "")).lower() not in {"csv", "parquet"}:
        raise ConfigError("analysis_output.large_table_format must be csv or parquet")
    if str(output.get("diagnostic_mode", "")).lower() not in {"compact", "debug", "summary"}:
        raise ConfigError(
            "analysis_output.diagnostic_mode must be compact, debug or summary"
        )

    peak = cfg["peak_selection"]
    _positive_int(peak.get("min_events"), "peak_selection.min_events", 2)
    _positive_int(
        peak.get("kde_grid_points"), "peak_selection.kde_grid_points", 256
    )
    _positive_number(
        peak.get("kde_bandwidth_factor"),
        "peak_selection.kde_bandwidth_factor",
        1e-9,
    )
    _positive_number(
        peak.get("min_peak_height_fraction"),
        "peak_selection.min_peak_height_fraction",
        0.0,
    )
    _positive_number(
        peak.get("left_sigma_multiplier"),
        "peak_selection.left_sigma_multiplier",
        0.0,
    )
    _positive_number(
        peak.get("right_sigma_multiplier"),
        "peak_selection.right_sigma_multiplier",
        0.0,
    )

    alignment = cfg["alignment_filter"]
    if not isinstance(alignment.get("enabled"), bool):
        raise ConfigError("alignment_filter.enabled must be true or false")
    if str(alignment.get("method", "")).lower() not in {
        "robust_mad",
        "standard",
    }:
        raise ConfigError("alignment_filter.method must be robust_mad or standard")
    _positive_number(
        alignment.get("z_threshold"), "alignment_filter.z_threshold", 1e-9
    )
    _positive_number(
        alignment.get("minimum_scale_lsb"),
        "alignment_filter.minimum_scale_lsb",
        1e-9,
    )
    _positive_int(
        alignment.get("minimum_events"), "alignment_filter.minimum_events", 2
    )

    fit = cfg["fit"]
    _positive_int(fit.get("min_events"), "fit.min_events", 4)
    _positive_int(fit.get("min_bin_width_lsb"), "fit.min_bin_width_lsb", 1)
    _positive_int(
        fit.get("min_histogram_bins"), "fit.min_histogram_bins", 4
    )
    _positive_int(
        fit.get("max_histogram_bins"), "fit.max_histogram_bins", 4
    )
    if fit["max_histogram_bins"] < fit["min_histogram_bins"]:
        raise ConfigError(
            "fit.max_histogram_bins must be >= fit.min_histogram_bins"
        )
    _positive_number(
        fit.get("target_events_per_bin"), "fit.target_events_per_bin", 1.0
    )
    _positive_int(fit.get("iterations"), "fit.iterations", 1)
    _positive_number(
        fit.get("initial_half_width_sigma"),
        "fit.initial_half_width_sigma",
        0.5,
    )
    _positive_number(
        fit.get("refit_half_width_sigma"),
        "fit.refit_half_width_sigma",
        0.5,
    )
    _positive_int(fit.get("minimum_fit_bins"), "fit.minimum_fit_bins", 4)
    _positive_int(
        fit.get("minimum_fit_events"), "fit.minimum_fit_events", 4
    )
    _positive_number(
        fit.get("outlier_z_threshold"), "fit.outlier_z_threshold", 1.0
    )
    _positive_number(
        fit.get("outlier_minimum_scale_lsb"),
        "fit.outlier_minimum_scale_lsb",
        1e-9,
    )

    bootstrap = cfg["bootstrap"]
    for flag in ("enabled", "overwrite", "plots"):
        if not isinstance(bootstrap.get(flag), bool):
            raise ConfigError(f"bootstrap.{flag} must be true or false")
    _positive_int(bootstrap.get("replicates"), "bootstrap.replicates", 1)
    _positive_int(bootstrap.get("workers"), "bootstrap.workers", 0)
    _positive_int(bootstrap.get("chunksize"), "bootstrap.chunksize", 1)
    _positive_int(bootstrap.get("seed"), "bootstrap.seed", 0)
    if str(bootstrap.get("mode", "")).strip().lower() not in {
        "full_analysis",
        "fixed_photopeak",
    }:
        raise ConfigError(
            "bootstrap.mode must be full_analysis or fixed_photopeak"
        )

    plots = cfg["plots"]
    for key in (
        "matching_train",
        "matching_total",
        "peak_selection",
        "timing_fit",
        "bootstrap",
    ):
        item = plots.get(key)
        if not isinstance(item, dict):
            raise ConfigError(f"plots.{key} is required")
        for flag in ("enabled", "overwrite"):
            if not isinstance(item.get(flag), bool):
                raise ConfigError(f"plots.{key}.{flag} must be true or false")
    _positive_int(plots.get("dpi"), "plots.dpi", 72)
    _positive_int(
        plots.get("max_histogram_bins"), "plots.max_histogram_bins", 10
    )
    _positive_int(
        plots.get("matching_curve_points"), "plots.matching_curve_points", 100
    )

    if str(cfg["thresholds"].get("consistency", "")).lower() not in {
        "error",
        "warning",
    }:
        raise ConfigError("thresholds.consistency must be error or warning")



def _stage_output_config(cfg: dict[str, Any], *, supports_summary: bool = False) -> dict[str, str]:
    mode = str(cfg["analysis_output"]["diagnostic_mode"]).lower()
    if mode == "summary" and not supports_summary:
        mode = "compact"
    return {
        "large_table_format": str(cfg["analysis_output"]["large_table_format"]).lower(),
        "diagnostic_mode": mode,
    }

def stage_config(cfg: dict[str, Any], stage: str) -> dict[str, Any]:
    if stage == "streaming_pulse_decode":
        return {
            "channels": cfg["channels"],
            "streaming_physical_time": cfg["preprocessing"][
                "streaming_physical_time"
            ],
            "pulse_reconstruction": cfg["preprocessing"][
                "pulse_reconstruction"
            ],
            "training_window_ns": cfg["matching_model"]["training"]["window_ns"],
            "candidate_window_ns": cfg["matching_model"]["inference"][
                "candidate_window_ns"
            ],
            "version": 4,
        }
    if stage == "candidate_preprocessing":
        return {
            "channels": cfg["channels"],
            "candidate_window_ns": cfg["matching_model"]["inference"][
                "candidate_window_ns"
            ],
            "streaming_physical_time": cfg["preprocessing"][
                "streaming_physical_time"
            ],
            "version": 6,
        }
    if stage == "energy_selection":
        return {
            "channels": cfg["channels"],
            "peak_selection": cfg["peak_selection"],
            "analysis_output": _stage_output_config(cfg),
            "version": 2,
        }
    if stage == "matching_training":
        return {
            "channels": cfg["channels"],
            "training": cfg["matching_model"]["training"],
            "analysis_output": _stage_output_config(cfg),
            "version": 8,
        }
    if stage == "matching_model":
        return {
            "method": cfg["matching_model"]["method"],
            "average_delay": cfg["matching_model"]["average_delay"],
            "minimum_samples": cfg["matching_model"]["training"][
                "minimum_samples"
            ],
            "version": 5,
        }
    if stage == "preprocessing":
        return {
            "channels": cfg["channels"],
            "inference": cfg["matching_model"]["inference"],
            "analysis_output": _stage_output_config(cfg, supports_summary=True),
            "version": 11,
        }
    if stage == "selection":
        return {
            "channels": cfg["channels"],
            "alignment_filter": cfg["alignment_filter"],
            "analysis_output": _stage_output_config(cfg),
            "version": 4,
        }
    if stage == "fit":
        return {"fit": cfg["fit"], "analysis_output": _stage_output_config(cfg)}
    if stage == "bootstrap":
        return {
            "peak_selection": cfg["peak_selection"],
            "alignment_filter": cfg["alignment_filter"],
            "fit": cfg["fit"],
            "analysis_output": _stage_output_config(cfg),
            "bootstrap": {
                key: cfg["bootstrap"][key]
                for key in ("replicates", "seed", "mode")
            },
            "version": 2,
        }
    if stage.startswith("plot_"):
        versions = {
            "plot_matching_train": 8,
            "plot_matching_total": 9,
            "plot_peak_selection": 3,
            "plot_timing_fit": 2,
            "plot_bootstrap": 1,
        }
        plot_key = stage.removeprefix("plot_")
        result = {
            "plot": cfg["plots"][plot_key],
            "dpi": cfg["plots"]["dpi"],
            "max_histogram_bins": cfg["plots"]["max_histogram_bins"],
            "matching_curve_points": cfg["plots"]["matching_curve_points"],
            "renderer_version": versions[stage],
        }
        if stage == "plot_matching_train":
            result["delay_window_ns"] = cfg["matching_model"]["training"][
                "window_ns"
            ]
        if stage == "plot_matching_total":
            result["delay_window_ns"] = cfg["matching_model"]["inference"][
                "candidate_window_ns"
            ]
        return result
    raise ConfigError(f"Unknown stage: {stage}")
