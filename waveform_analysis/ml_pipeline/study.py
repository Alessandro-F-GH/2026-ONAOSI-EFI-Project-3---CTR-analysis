from __future__ import annotations

import copy
import gc
import csv
import itertools
import json
import math
import random
import re
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .common import (
    atomic_json,
    canonical_hash,
    read_json,
    restrict_to_study_progress,
    setup_logging,
    write_csv_rows,
)
from .dataset import PreparedDataset, load_prepared_dataset
from .evaluation import evaluate_trained_model, load_trained_model
from .input_transform import normalize_input_transform
from .metrics import fit_times_ps
from .prediction import prediction_window_dataset_view
from .robust_selection import RobustLocationScale, fit_median_mad_z, robust_z_mask
from .study_config import CHANNEL_MODES
from .training import train_model
from .training_utils import resolve_device


_PROGRESS_EXTRA = {"study_progress": True}


def _progress(logger: Any, message: str, *args: Any) -> None:
    logger.info(message, *args, extra=_PROGRESS_EXTRA)


def _format_duration(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "unknown"
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _format_eta(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "ETA unknown"
    finish = time.strftime("%H:%M", time.localtime(time.time() + seconds))
    return f"ETA {_format_duration(seconds)} (finish ~{finish})"


def _compact_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_compact_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(
            f"{key}:{_compact_value(item)}" for key, item in sorted(value.items())
        ) + "}"
    return str(value)


def _compact_parameters(parameters: dict[str, Any], limit: int = 180) -> str:
    if not parameters:
        return "fixed defaults"
    text = ", ".join(
        f"{key}={_compact_value(value)}" for key, value in sorted(parameters.items())
    )
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _metric_text(value: Any, suffix: str = "") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:.3f}{suffix}" if np.isfinite(number) else "n/a"


INTERNAL_RESULT_FIELDS = [
    "row_key",
    "record_type",
    "experiment_id",
    "root_id",
    "root_file",
    "channel_mode",
    "model_id",
    "model_type",
    "loss_id",
    "loss_type",
    "input_transform",
    "subsampling_factor",
    "window_id",
    "window_start_ns",
    "window_end_ns",
    "trial_id",
    "fold_id",
    "split",
    "statistic",
    "is_selected_hyperparameters",
    "is_selected_window",
    "status",
    "n_events",
    "loss",
    "rmse_ps",
    "bias_ps",
    "ctr_ps",
    "baseline_ctr_ps",
    "relative_improvement_pct",
    "outlier_center_ps",
    "outlier_scale_ps",
    "outlier_scale_method",
    "outlier_z_threshold",
    "runtime_seconds",
    "pearson_cv_blind",
    "spearman_cv_blind",
    "mean_cv_blind_gap_ps",
    "blind_rank_of_cv_selected_window",
    "blind_regret_ps",
    "params_json",
    "error",
]

# all_results.csv is intentionally numeric-only. Repeated strings, absolute paths,
# parameter dictionaries, and error messages are stored once in the sidecar
# results_metadata.json and reconstructed transparently when a study is resumed.
CSV_RESULT_FIELDS = [
    "row_uid_hi",
    "row_uid_lo",
    "record_type_code",
    "root_code",
    "channel_mode_code",
    "model_code",
    "loss_code",
    "transform_code",
    "subsampling_factor",
    "window_start_ns",
    "window_end_ns",
    "trial_code",
    "fold_id",
    "split_code",
    "statistic_code",
    "is_selected_hyperparameters",
    "is_selected_window",
    "status_code",
    "n_events",
    "loss",
    "rmse_ps",
    "bias_ps",
    "ctr_ps",
    "baseline_ctr_ps",
    "relative_improvement_pct",
    "outlier_center_ps",
    "outlier_scale_ps",
    "outlier_method_code",
    "outlier_z_threshold",
    "runtime_seconds",
    "pearson_cv_blind",
    "spearman_cv_blind",
    "mean_cv_blind_gap_ps",
    "blind_rank_of_cv_selected_window",
    "blind_regret_ps",
]


REPORT_SCHEMA_VERSION = 6
CV_DATA_PREPARATION_PROTOCOL = "development_blind_anchor_factored_v2"


SUMMARY_RESULT_FIELDS = [
    "file_name",
    "voltage_V",
    "channel_mode",
    "model_id",
    "model_type",
    "loss_id",
    "loss_type",
    "input_transform",
    "subsampling_factor",
    "window_start_ns",
    "window_end_ns",
    "trial_id",
    "validation_n_events",
    "validation_loss_mean",
    "validation_rmse_mean_ps",
    "validation_rmse_sem_ps",
    "validation_bias_mean_ps",
    "validation_ctr_mean_ps",
    "validation_ctr_sem_ps",
    "validation_gaussian_mean_ps",
    "validation_gaussian_sigma_ps",
    "validation_gaussian_ctr_ps",
    "validation_gaussian_chi2_ndof",
    "blind_n_events",
    "blind_loss_mean",
    "blind_rmse_mean_ps",
    "blind_rmse_sem_ps",
    "blind_bias_mean_ps",
    "blind_ctr_mean_ps",
    "blind_ctr_sem_ps",
    "blind_gaussian_mean_ps",
    "blind_gaussian_sigma_ps",
    "blind_gaussian_ctr_ps",
    "blind_gaussian_chi2_ndof",
    "baseline_validation_ctr_mean_ps",
    "baseline_blind_ctr_mean_ps",
    "validation_relative_improvement_mean_pct",
    "blind_relative_improvement_mean_pct",
    "shapelet_count",
    "shapelet_distance_metric",
    "shapelet_dtw_radius_points",
    "shapelet_ridge_alpha",
]


MODEL_LOSS_RESULT_FIELDS = [
    "file_name",
    "voltage_V",
    "channel_mode",
    "model_id",
    "model_type",
    "loss_id",
    "loss_type",
    "input_transform",
    "subsampling_factor",
    "window_id",
    "window_start_ns",
    "window_end_ns",
    "window_size_ns",
    "trial_id",
    "validation_n_events",
    "validation_loss_mean",
    "validation_rmse_mean_ps",
    "validation_rmse_sem_ps",
    "validation_bias_mean_ps",
    "validation_ctr_mean_ps",
    "validation_ctr_sem_ps",
    "blind_n_events",
    "blind_loss_mean",
    "blind_rmse_mean_ps",
    "blind_rmse_sem_ps",
    "blind_bias_mean_ps",
    "blind_ctr_mean_ps",
    "blind_ctr_sem_ps",
    "baseline_validation_ctr_mean_ps",
    "baseline_blind_ctr_mean_ps",
    "validation_relative_improvement_mean_pct",
    "blind_relative_improvement_mean_pct",
    "shapelet_count",
    "shapelet_distance_metric",
    "shapelet_dtw_radius_points",
    "shapelet_ridge_alpha",
]



def _safe_name(value: str) -> str:
    text = "".join(c.lower() if c.isalnum() else "_" for c in str(value))
    return "_".join(part for part in text.split("_") if part) or "item"


def _cv_data_preparation_signature(config: dict[str, Any]) -> str:
    """Fingerprint the parts of a study that determine its event set and arrays."""

    split = copy.deepcopy(config.get("split", {}))
    protocol = (
        "legacy_initial_train_validation_blind"
        if "initial_validation_fraction" in split
        else CV_DATA_PREPARATION_PROTOCOL
    )
    preprocessing = copy.deepcopy(config.get("preprocessing", {}))
    # This grid controls an on-the-fly model-input view, not ROOT conversion.
    preprocessing.pop("subsampling_factors", None)
    preprocessing.pop("subsampling_factor", None)
    max_before = max(float(window["before_ns"]) for window in config["windows_ns"])
    max_after = max(float(window["after_ns"]) for window in config["windows_ns"])
    return canonical_hash(
        {
            "protocol": protocol,
            "data": config.get("data", {}),
            "preprocessing": preprocessing,
            "split": split,
            "materialized_window_ns": {
                "before": max_before,
                "after": max_after,
            },
        }
    )


def _assert_resume_data_compatibility(config: dict[str, Any], output: Path) -> None:
    resolved_path = output / "resolved_study_config.json"
    if not resolved_path.is_file():
        return
    previous = read_json(resolved_path)
    if _cv_data_preparation_signature(previous) == _cv_data_preparation_signature(config):
        return
    raise RuntimeError(
        "Cannot resume this study because the data/target preparation protocol changed. "
        "The current protocol uses a direct development/blind split and factors the "
        "interpolated-LED/native-window-anchor shift analytically. Old rows and "
        "checkpoints are not comparable. Run the experiment with --restart."
    )


def _deep_update(target: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
    return target


def _set_nested(target: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    current = target
    for part in parts[:-1]:
        current = current.setdefault(part, {})
        if not isinstance(current, dict):
            raise ValueError(f"Cannot set nested parameter {dotted!r}")
    current[parts[-1]] = copy.deepcopy(value)


def _results_metadata_path(path: Path) -> Path:
    return path.with_name("results_metadata.json")


def _string_value(value: Any) -> str:
    return "" if value is None else str(value)


def _make_codebook(rows: list[dict[str, Any]], field: str) -> tuple[dict[str, int], list[str]]:
    values = sorted(
        {
            _string_value(row.get(field, ""))
            for row in rows
            if _string_value(row.get(field, "")) != ""
        }
    )
    return {value: index for index, value in enumerate(values)}, values


def _row_uid_parts(row: dict[str, Any]) -> tuple[int, int, str]:
    key = _string_value(row.get("row_key", ""))
    if len(key) != 24 or any(char not in "0123456789abcdefABCDEF" for char in key):
        key = canonical_hash({
            field: row.get(field, "")
            for field in INTERNAL_RESULT_FIELDS
            if field not in {"row_key", "error"}
        })[:24]
    key = key.lower()
    return int(key[:12], 16), int(key[12:], 16), key


def _build_results_metadata(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, dict[str, int]]]:
    categorical_fields = (
        "record_type",
        "channel_mode",
        "input_transform",
        "split",
        "statistic",
        "status",
        "outlier_scale_method",
    )
    code_maps: dict[str, dict[str, int]] = {}
    codebooks: dict[str, list[str]] = {}
    for field in categorical_fields:
        mapping, values = _make_codebook(rows, field)
        code_maps[field] = mapping
        codebooks[field] = values

    root_ids = sorted({_string_value(row.get("root_id", "")) for row in rows if row.get("root_id", "") != ""})
    root_map = {value: index for index, value in enumerate(root_ids)}
    code_maps["root_id"] = root_map
    roots = []
    for root_id in root_ids:
        representative = next(row for row in rows if _string_value(row.get("root_id")) == root_id)
        roots.append({
            "code": root_map[root_id],
            "id": root_id,
            # Stored once here, never repeated in the numeric CSV.
            "path": _string_value(representative.get("root_file", "")),
        })

    model_ids = sorted({_string_value(row.get("model_id", "")) for row in rows if row.get("model_id", "") != ""})
    model_map = {value: index for index, value in enumerate(model_ids)}
    code_maps["model_id"] = model_map
    models = []
    for model_id in model_ids:
        representative = next(row for row in rows if _string_value(row.get("model_id")) == model_id)
        models.append({
            "code": model_map[model_id],
            "id": model_id,
            "type": _string_value(representative.get("model_type", "")),
        })

    loss_ids = sorted({_string_value(row.get("loss_id", "")) for row in rows if row.get("loss_id", "") != ""})
    loss_map = {value: index for index, value in enumerate(loss_ids)}
    code_maps["loss_id"] = loss_map
    losses = []
    for loss_id in loss_ids:
        representative = next(row for row in rows if _string_value(row.get("loss_id")) == loss_id)
        losses.append({
            "code": loss_map[loss_id],
            "id": loss_id,
            "type": _string_value(representative.get("loss_type", "")),
        })

    trial_ids = sorted({_string_value(row.get("trial_id", "")) for row in rows if row.get("trial_id", "") != ""})
    trial_map = {value: index for index, value in enumerate(trial_ids)}
    code_maps["trial_id"] = trial_map
    trial_parameters: dict[str, Any] = {}
    for row in rows:
        trial_id = _string_value(row.get("trial_id", ""))
        payload = _string_value(row.get("params_json", ""))
        if not trial_id or not payload or trial_id in trial_parameters:
            continue
        try:
            trial_parameters[trial_id] = json.loads(payload)
        except json.JSONDecodeError:
            trial_parameters[trial_id] = payload
    trials = [
        {
            "code": trial_map[trial_id],
            "id": trial_id,
            "parameters": trial_parameters.get(trial_id, {}),
        }
        for trial_id in trial_ids
    ]

    windows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        start = _string_value(row.get("window_start_ns", ""))
        end = _string_value(row.get("window_end_ns", ""))
        if start == "" and end == "":
            continue
        windows_by_key.setdefault(
            (start, end),
            {
                "id": _string_value(row.get("window_id", "")),
                "start_ns": row.get("window_start_ns", ""),
                "end_ns": row.get("window_end_ns", ""),
            },
        )

    errors: dict[str, str] = {}
    for row in rows:
        message = _string_value(row.get("error", ""))
        if not message:
            continue
        _, _, key = _row_uid_parts(row)
        errors[key] = message

    experiment_ids = sorted(
        {_string_value(row.get("experiment_id", "")) for row in rows if row.get("experiment_id", "") != ""}
    )
    metadata = {
        "schema_version": 2,
        "csv_format": "numeric_long_format",
        "experiment_id": experiment_ids[0] if len(experiment_ids) == 1 else experiment_ids,
        "codebooks": codebooks,
        "roots": roots,
        "models": models,
        "losses": losses,
        "trials": trials,
        "windows": list(windows_by_key.values()),
        "errors": errors,
        "notes": {
            "paths": "Stored once in roots; omitted from all_results.csv.",
            "parameters": "Stored once in trials; omitted from all_results.csv.",
            "categorical_values": "Integer-coded in all_results.csv and decoded through this file.",
        },
    }
    return metadata, code_maps


def _code_or_empty(mapping: dict[str, int], value: Any) -> int | str:
    text = _string_value(value)
    return "" if text == "" else mapping[text]


def _encode_result_row(row: dict[str, Any], code_maps: dict[str, dict[str, int]]) -> dict[str, Any]:
    hi, lo, _ = _row_uid_parts(row)
    encoded = {
        "row_uid_hi": hi,
        "row_uid_lo": lo,
        "record_type_code": _code_or_empty(code_maps["record_type"], row.get("record_type", "")),
        "root_code": _code_or_empty(code_maps["root_id"], row.get("root_id", "")),
        "channel_mode_code": _code_or_empty(code_maps["channel_mode"], row.get("channel_mode", "")),
        "model_code": _code_or_empty(code_maps["model_id"], row.get("model_id", "")),
        "loss_code": _code_or_empty(code_maps["loss_id"], row.get("loss_id", "")),
        "transform_code": _code_or_empty(code_maps["input_transform"], row.get("input_transform", "")),
        "subsampling_factor": row.get("subsampling_factor", ""),
        "window_start_ns": row.get("window_start_ns", ""),
        "window_end_ns": row.get("window_end_ns", ""),
        "trial_code": _code_or_empty(code_maps["trial_id"], row.get("trial_id", "")),
        "fold_id": row.get("fold_id", ""),
        "split_code": _code_or_empty(code_maps["split"], row.get("split", "")),
        "statistic_code": _code_or_empty(code_maps["statistic"], row.get("statistic", "")),
        "is_selected_hyperparameters": row.get("is_selected_hyperparameters", ""),
        "is_selected_window": row.get("is_selected_window", ""),
        "status_code": _code_or_empty(code_maps["status"], row.get("status", "")),
        "n_events": row.get("n_events", ""),
        "loss": row.get("loss", ""),
        "rmse_ps": row.get("rmse_ps", ""),
        "bias_ps": row.get("bias_ps", ""),
        "ctr_ps": row.get("ctr_ps", ""),
        "baseline_ctr_ps": row.get("baseline_ctr_ps", ""),
        "relative_improvement_pct": row.get("relative_improvement_pct", ""),
        "outlier_center_ps": row.get("outlier_center_ps", ""),
        "outlier_scale_ps": row.get("outlier_scale_ps", ""),
        "outlier_method_code": _code_or_empty(
            code_maps["outlier_scale_method"], row.get("outlier_scale_method", "")
        ),
        "outlier_z_threshold": row.get("outlier_z_threshold", ""),
        "runtime_seconds": row.get("runtime_seconds", ""),
        "pearson_cv_blind": row.get("pearson_cv_blind", ""),
        "spearman_cv_blind": row.get("spearman_cv_blind", ""),
        "mean_cv_blind_gap_ps": row.get("mean_cv_blind_gap_ps", ""),
        "blind_rank_of_cv_selected_window": row.get("blind_rank_of_cv_selected_window", ""),
        "blind_regret_ps": row.get("blind_regret_ps", ""),
    }
    return {field: encoded.get(field, "") for field in CSV_RESULT_FIELDS}


def _metadata_code_lookup(metadata: dict[str, Any], field: str) -> dict[str, str]:
    values = metadata.get("codebooks", {}).get(field, [])
    return {str(index): _string_value(value) for index, value in enumerate(values)}


def _read_results(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    metadata_path = _results_metadata_path(path)
    if not metadata_path.is_file():
        # Accept an old study CSV only to produce a clear transition path. The next
        # write converts it to the compact numeric schema.
        with path.open("r", encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    metadata = read_json(metadata_path)
    if int(metadata.get("schema_version", 0)) != 2:
        raise ValueError(f"Unsupported results metadata schema in {metadata_path}")

    lookups = {
        field: _metadata_code_lookup(metadata, field)
        for field in (
            "record_type",
            "channel_mode",
            "input_transform",
            "split",
            "statistic",
            "status",
            "outlier_scale_method",
        )
    }
    roots = {str(item["code"]): item for item in metadata.get("roots", [])}
    models = {str(item["code"]): item for item in metadata.get("models", [])}
    losses = {str(item["code"]): item for item in metadata.get("losses", [])}
    trials = {str(item["code"]): item for item in metadata.get("trials", [])}
    window_lookup = {
        (_string_value(item.get("start_ns", "")), _string_value(item.get("end_ns", ""))): item
        for item in metadata.get("windows", [])
    }
    experiment = metadata.get("experiment_id", "")
    experiment_id = experiment if isinstance(experiment, str) else ""
    errors = metadata.get("errors", {})

    decoded: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        for compact in csv.DictReader(stream):
            root = roots.get(compact.get("root_code", ""), {})
            model = models.get(compact.get("model_code", ""), {})
            loss = losses.get(compact.get("loss_code", ""), {})
            trial = trials.get(compact.get("trial_code", ""), {})
            window = window_lookup.get(
                (compact.get("window_start_ns", ""), compact.get("window_end_ns", "")), {}
            )
            hi = int(compact.get("row_uid_hi", "0") or 0)
            lo = int(compact.get("row_uid_lo", "0") or 0)
            row_key = f"{hi:012x}{lo:012x}"
            row = {
                "row_key": row_key,
                "record_type": lookups["record_type"].get(compact.get("record_type_code", ""), ""),
                "experiment_id": experiment_id,
                "root_id": _string_value(root.get("id", "")),
                "root_file": _string_value(root.get("path", "")),
                "channel_mode": lookups["channel_mode"].get(compact.get("channel_mode_code", ""), ""),
                "model_id": _string_value(model.get("id", "")),
                "model_type": _string_value(model.get("type", "")),
                "loss_id": _string_value(loss.get("id", "")),
                "loss_type": _string_value(loss.get("type", "")),
                "input_transform": lookups["input_transform"].get(compact.get("transform_code", ""), ""),
                "subsampling_factor": compact.get("subsampling_factor", "1"),
                "window_id": _string_value(window.get("id", "")),
                "window_start_ns": compact.get("window_start_ns", ""),
                "window_end_ns": compact.get("window_end_ns", ""),
                "trial_id": _string_value(trial.get("id", "")),
                "fold_id": compact.get("fold_id", ""),
                "split": lookups["split"].get(compact.get("split_code", ""), ""),
                "statistic": lookups["statistic"].get(compact.get("statistic_code", ""), ""),
                "is_selected_hyperparameters": compact.get("is_selected_hyperparameters", ""),
                "is_selected_window": compact.get("is_selected_window", ""),
                "status": lookups["status"].get(compact.get("status_code", ""), ""),
                "n_events": compact.get("n_events", ""),
                "loss": compact.get("loss", ""),
                "rmse_ps": compact.get("rmse_ps", ""),
                "bias_ps": compact.get("bias_ps", ""),
                "ctr_ps": compact.get("ctr_ps", ""),
                "baseline_ctr_ps": compact.get("baseline_ctr_ps", ""),
                "relative_improvement_pct": compact.get("relative_improvement_pct", ""),
                "outlier_center_ps": compact.get("outlier_center_ps", ""),
                "outlier_scale_ps": compact.get("outlier_scale_ps", ""),
                "outlier_scale_method": lookups["outlier_scale_method"].get(
                    compact.get("outlier_method_code", ""), ""
                ),
                "outlier_z_threshold": compact.get("outlier_z_threshold", ""),
                "runtime_seconds": compact.get("runtime_seconds", ""),
                "pearson_cv_blind": compact.get("pearson_cv_blind", ""),
                "spearman_cv_blind": compact.get("spearman_cv_blind", ""),
                "mean_cv_blind_gap_ps": compact.get("mean_cv_blind_gap_ps", ""),
                "blind_rank_of_cv_selected_window": compact.get(
                    "blind_rank_of_cv_selected_window", ""
                ),
                "blind_regret_ps": compact.get("blind_regret_ps", ""),
                "params_json": (
                    json.dumps(trial.get("parameters", {}), sort_keys=True, separators=(",", ":"))
                    if lookups["record_type"].get(compact.get("record_type_code", ""), "")
                    == "trial_definition"
                    else ""
                ),
                "error": _string_value(errors.get(row_key, "")),
            }
            decoded.append({field: _string_value(row.get(field, "")) for field in INTERNAL_RESULT_FIELDS})
    return decoded


def _write_results(path: Path, rows: list[dict[str, Any]]) -> None:
    metadata, code_maps = _build_results_metadata(rows)
    atomic_json(_results_metadata_path(path), metadata)
    write_csv_rows(path, [_encode_result_row(row, code_maps) for row in rows])



def _state_results_path(output: Path) -> Path:
    return output / "_state" / "all_results.csv"


def _summary_results_path(output: Path) -> Path:
    return output / "summary_results.csv"


def _model_loss_results_path(output: Path) -> Path:
    return output / "model_loss_results.csv"


def _migrate_legacy_results(output: Path) -> None:
    """Move the old root-level fold table into the private resumable state area."""
    state_path = _state_results_path(output)
    if state_path.is_file():
        return
    legacy_path = output / "all_results.csv"
    legacy_metadata = output / "results_metadata.json"
    if not legacy_path.is_file():
        return
    state_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(legacy_path), str(state_path))
    if legacy_metadata.is_file():
        shutil.move(str(legacy_metadata), str(_results_metadata_path(state_path)))


def _finite_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def _extract_voltage(root_file: Path, reporting: dict[str, Any]) -> float:
    naming = reporting.get("voltage_from_filename", {})
    if not bool(naming.get("enabled", False)):
        return float("nan")
    pattern = str(naming.get("pattern", r"^(?P<voltage>\d+(?:\.\d+)?)V"))
    group = naming.get("group", "voltage")
    match = re.search(pattern, root_file.name, flags=re.IGNORECASE)
    if match is None:
        return float("nan")
    try:
        raw = match.group(group) if isinstance(group, str) else match.group(int(group))
        return float(raw)
    except (IndexError, KeyError, TypeError, ValueError):
        return float("nan")


def _row_matches_configuration(row: dict[str, Any], selected: dict[str, Any]) -> bool:
    return all(
        str(row.get(field, "")) == str(selected.get(field, ""))
        for field in (
            "root_id", "channel_mode", "model_id", "loss_id",
            "input_transform", "subsampling_factor", "window_id", "trial_id",
        )
    )


def _best_configuration_row(
    rows: list[dict[str, Any]], root_id: str, mode_id: str, metric: str,
) -> dict[str, Any] | None:
    metric = _canonical_study_metric(metric)
    candidates = [
        row for row in rows
        if row.get("record_type") == "summary"
        and row.get("split") == "validation"
        and row.get("statistic") == "mean"
        and row.get("status") == "completed"
        and str(row.get("root_id", "")) == root_id
        and str(row.get("channel_mode", "")) == mode_id
        and str(row.get("is_selected_hyperparameters", "")) == "1"
        and str(row.get("is_selected_window", "")) == "1"
        and np.isfinite(_row_metric_value(row, metric))
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda row: _row_metric_value(row, metric))


def _best_model_loss_configuration_row(
    rows: list[dict[str, Any]],
    root_id: str,
    mode_id: str,
    model_id: str,
    loss_id: str,
    metric: str,
) -> dict[str, Any] | None:
    """Select one compact result per model/loss using CV only.

    Hyperparameters and windows have already been selected independently inside
    each input transform. This final comparison chooses the best transform for
    the requested model/loss pair, without consulting blind metrics.
    """

    metric = _canonical_study_metric(metric)
    candidates = [
        row for row in rows
        if row.get("record_type") == "summary"
        and row.get("split") == "validation"
        and row.get("statistic") == "mean"
        and row.get("status") == "completed"
        and str(row.get("root_id", "")) == root_id
        and str(row.get("channel_mode", "")) == mode_id
        and str(row.get("model_id", "")) == model_id
        and str(row.get("loss_id", "")) == loss_id
        and str(row.get("is_selected_hyperparameters", "")) == "1"
        and str(row.get("is_selected_window", "")) == "1"
        and np.isfinite(_row_metric_value(row, metric))
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda row: _row_metric_value(row, metric))


def _configuration_summary_row(
    rows: list[dict[str, Any]], selected: dict[str, Any], split: str, statistic: str,
) -> dict[str, Any] | None:
    return next(
        (
            row for row in rows
            if row.get("record_type") == "summary"
            and row.get("split") == split
            and row.get("statistic") == statistic
            and row.get("status") == "completed"
            and _row_matches_configuration(row, selected)
        ),
        None,
    )


def _evaluation_config_for_summary(
    config: dict[str, Any], root_id: str, mode_id: str,
    selected: dict[str, Any], run_dir: Path,
) -> dict[str, Any]:
    return {
        "device": config.get("evaluation", {}).get("device", "auto"),
        "batch_size": int(config.get("evaluation", {}).get("batch_size", 512)),
        "num_workers": int(config.get("evaluation", {}).get("num_workers", 0)),
        "pin_memory": bool(config.get("evaluation", {}).get("pin_memory", False)),
        "input_transform_cache_dir": str(
            _shared_input_cache_root(
                config,
                root_id,
                mode_id,
                str(selected["window_id"]),
                str(selected["input_transform"]),
            )
        ),
        "output": {"evaluation_dir": str(run_dir / "summary_evaluation")},
    }


def _pooled_best_fits(
    *,
    config: dict[str, Any],
    root_id: str,
    mode_id: str,
    selected: dict[str, Any],
    development: PreparedDataset,
    blind: PreparedDataset,
    folds: list[dict[str, Any]],
    run_root: Path,
) -> tuple[Any, Any]:
    validation_values: list[np.ndarray] = []
    blind_values: list[np.ndarray] = []
    if str(selected.get("model_type", "")) == "standard_method":
        method_id = str(selected.get("model_id", ""))
        target = CHANNEL_MODES[mode_id]["target"]
        for fold in folds:
            validation_values.append(
                _standard_method_delta_ps(
                    development, target, method_id,
                    np.asarray(fold["validation"], dtype=np.int64),
                )
            )
            blind_values.append(
                _standard_method_delta_ps(
                    blind, target, method_id,
                    np.asarray(fold["blind"], dtype=np.int64),
                )
            )
        validation = np.concatenate(validation_values) if validation_values else np.empty(0)
        blind_pooled = np.concatenate(blind_values) if blind_values else np.empty(0)
        return (
            fit_times_ps(validation, f"{method_id.upper()} validation", config["fit"]),
            fit_times_ps(blind_pooled, f"{method_id.upper()} blind", config["fit"]),
        )

    device = resolve_device(config.get("evaluation", {}).get("device", "auto"))
    for fold in folds:
        run_dir = (
            run_root / root_id / mode_id / str(selected["model_id"])
            / str(selected["loss_id"]) / str(selected["input_transform"])
            / str(selected["window_id"]) / str(selected["trial_id"])
            / f"fold_{fold['fold_id']}"
        )
        trained = load_trained_model(run_dir)
        eval_config = _evaluation_config_for_summary(
            config, root_id, mode_id, selected, run_dir
        )
        validation_source = replace(
            development,
            evaluation=np.asarray(fold["validation"], dtype=np.int64),
        )
        blind_source = replace(
            blind,
            evaluation=np.asarray(fold["blind"], dtype=np.int64),
        )
        validation_prediction = evaluate_trained_model(
            trained, validation_source, eval_config, device
        )
        blind_prediction = evaluate_trained_model(
            trained, blind_source, eval_config, device
        )
        validation_values.append(
            np.asarray(validation_prediction.corrected_ps, dtype=np.float64)
        )
        blind_values.append(np.asarray(blind_prediction.corrected_ps, dtype=np.float64))

    validation = (
        np.concatenate(validation_values) if validation_values else np.empty(0, dtype=np.float64)
    )
    # Blind predictions are intentionally pooled across fold-trained models. This
    # is a reporting-only distribution; selection remains based exclusively on CV.
    blind_pooled = (
        np.concatenate(blind_values) if blind_values else np.empty(0, dtype=np.float64)
    )
    fit_config = config["fit"]
    return (
        fit_times_ps(validation, "best model validation", fit_config),
        fit_times_ps(blind_pooled, "best model blind", fit_config),
    )


def _fit_value(fit: Any, field: str) -> float:
    if fit is None or not bool(getattr(fit, "success", False)):
        return float("nan")
    return _finite_float(getattr(fit, field, float("nan")))


def _fit_value_or_previous(
    fit: Any, field: str, previous: dict[str, Any], column: str
) -> float:
    value = _fit_value(fit, field)
    if np.isfinite(value):
        return value
    return _finite_float(previous.get(column))


def _draw_fit_panel(axis: Any, fit: Any, title: str, subtitle: str) -> None:
    if fit is None or not bool(getattr(fit, "success", False)) or fit.edges_ps.size < 2:
        axis.text(0.5, 0.5, "Gaussian fit unavailable", ha="center", va="center")
        axis.set_title(title)
        axis.set_axis_off()
        return
    edges = np.asarray(fit.edges_ps, dtype=np.float64)
    counts = np.asarray(fit.counts, dtype=np.float64)
    expected = np.asarray(fit.expected, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    axis.bar(edges[:-1], counts, width=np.diff(edges), align="edge", alpha=0.55, label="Data")
    axis.plot(centers, expected, linewidth=2.0, label="Gaussian fit")
    axis.axvspan(fit.fit_low_ps, fit.fit_high_ps, alpha=0.10, label="Fit range")
    axis.set_title(title)
    axis.set_xlabel("Time difference (ps)")
    axis.set_ylabel("Events")
    axis.grid(True, alpha=0.20)
    axis.legend(fontsize=8)
    axis.text(
        0.98,
        0.96,
        f"{subtitle}\nμ={fit.mean_ps:.2f} ps\nσ={fit.sigma_ps:.2f} ps\n"
        f"CTR={fit.ctr_ps:.2f} ps\nχ²/ndof={fit.chi2_ndof:.2f}",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )


def _plot_file_best_results(
    root_file: Path,
    fits_by_mode: dict[str, tuple[Any, Any, dict[str, Any]]],
    destination: Path,
    dpi: int,
) -> None:
    if not fits_by_mode:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    modes = [mode for mode in CHANNEL_MODES if mode in fits_by_mode]
    figure, axes = plt.subplots(
        len(modes), 2,
        figsize=(12.5, max(4.8, 4.2 * len(modes))),
        squeeze=False,
    )
    for row_index, mode_id in enumerate(modes):
        validation_fit, blind_fit, record = fits_by_mode[mode_id]
        if str(record.get("model_type", "")) == "standard_method":
            subtitle = f"{str(record['model_id']).upper()} | standard method"
        else:
            subtitle = (
                f"{record['model_id']} | [{record['window_start_ns']:g},"
                f"{record['window_end_ns']:g}] ns | {record['input_transform']}"
            )
        _draw_fit_panel(axes[row_index, 0], validation_fit, f"{mode_id} — validation", subtitle)
        _draw_fit_panel(axes[row_index, 1], blind_fit, f"{mode_id} — blind", subtitle)
    figure.suptitle(f"Best CV-selected method/configuration — {root_file.name}")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)


def _read_summary_results(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_summary_results(path: Path, records: list[dict[str, Any]]) -> None:
    ordered = [
        {field: record.get(field, "") for field in SUMMARY_RESULT_FIELDS}
        for record in records
    ]
    write_csv_rows(path, ordered)


def _update_summary_results(
    path: Path, root_file: Path, records: list[dict[str, Any]], root_order: list[str]
) -> None:
    existing = [
        row for row in _read_summary_results(path)
        if str(row.get("file_name", "")) != root_file.name
    ]
    merged: list[dict[str, Any]] = [*existing, *records]
    order = {Path(value).name: index for index, value in enumerate(root_order)}
    merged.sort(
        key=lambda row: (
            order.get(str(row.get("file_name", "")), len(order)),
            str(row.get("channel_mode", "")),
        )
    )
    _write_summary_results(path, merged)


def _write_model_loss_results(path: Path, records: list[dict[str, Any]]) -> None:
    ordered = [
        {field: record.get(field, "") for field in MODEL_LOSS_RESULT_FIELDS}
        for record in records
    ]
    write_csv_rows(path, ordered)


def _update_model_loss_results(
    path: Path,
    root_file: Path,
    records: list[dict[str, Any]],
    root_order: list[str],
) -> None:
    existing = [
        row for row in _read_summary_results(path)
        if str(row.get("file_name", "")) != root_file.name
    ]
    merged: list[dict[str, Any]] = [*existing, *records]
    order = {Path(value).name: index for index, value in enumerate(root_order)}
    merged.sort(
        key=lambda row: (
            order.get(str(row.get("file_name", "")), len(order)),
            str(row.get("channel_mode", "")),
            str(row.get("model_id", "")),
            str(row.get("loss_id", "")),
        )
    )
    _write_model_loss_results(path, merged)


def _compact_model_record(
    *, root_file: Path, voltage: float, selected: dict[str, Any],
    validation_mean: dict[str, Any], validation_sem: dict[str, Any] | None,
    blind_mean: dict[str, Any], blind_sem: dict[str, Any] | None,
) -> dict[str, Any]:
    start_ns = _finite_float(selected.get("window_start_ns"))
    end_ns = _finite_float(selected.get("window_end_ns"))
    window_size = end_ns - start_ns if np.isfinite(start_ns) and np.isfinite(end_ns) else float("nan")
    return {
        "file_name": root_file.name,
        "voltage_V": voltage,
        "channel_mode": selected.get("channel_mode", ""),
        "model_id": selected.get("model_id", ""),
        "model_type": selected.get("model_type", ""),
        "loss_id": selected.get("loss_id", ""),
        "loss_type": selected.get("loss_type", ""),
        "input_transform": selected.get("input_transform", ""),
        "subsampling_factor": int(float(selected.get("subsampling_factor", 1) or 1)),
        "window_id": selected.get("window_id", ""),
        "window_start_ns": start_ns,
        "window_end_ns": end_ns,
        "window_size_ns": window_size,
        "trial_id": selected.get("trial_id", ""),
        "validation_n_events": int(round(_finite_float(validation_mean.get("n_events")))),
        "validation_loss_mean": _finite_float(validation_mean.get("loss")),
        "validation_rmse_mean_ps": _finite_float(validation_mean.get("rmse_ps")),
        "validation_rmse_sem_ps": _finite_float((validation_sem or {}).get("rmse_ps")),
        "validation_bias_mean_ps": _finite_float(validation_mean.get("bias_ps")),
        "validation_ctr_mean_ps": _finite_float(validation_mean.get("ctr_ps")),
        "validation_ctr_sem_ps": _finite_float((validation_sem or {}).get("ctr_ps")),
        "blind_n_events": int(round(_finite_float(blind_mean.get("n_events")))),
        "blind_loss_mean": _finite_float(blind_mean.get("loss")),
        "blind_rmse_mean_ps": _finite_float(blind_mean.get("rmse_ps")),
        "blind_rmse_sem_ps": _finite_float((blind_sem or {}).get("rmse_ps")),
        "blind_bias_mean_ps": _finite_float(blind_mean.get("bias_ps")),
        "blind_ctr_mean_ps": _finite_float(blind_mean.get("ctr_ps")),
        "blind_ctr_sem_ps": _finite_float((blind_sem or {}).get("ctr_ps")),
        "baseline_validation_ctr_mean_ps": _finite_float(validation_mean.get("baseline_ctr_ps")),
        "baseline_blind_ctr_mean_ps": _finite_float(blind_mean.get("baseline_ctr_ps")),
        "validation_relative_improvement_mean_pct": _finite_float(validation_mean.get("relative_improvement_pct")),
        "blind_relative_improvement_mean_pct": _finite_float(blind_mean.get("relative_improvement_pct")),
        "shapelet_count": selected.get("shapelet_count", ""),
        "shapelet_distance_metric": selected.get("shapelet_distance_metric", ""),
        "shapelet_dtw_radius_points": selected.get("shapelet_dtw_radius_points", ""),
        "shapelet_ridge_alpha": selected.get("shapelet_ridge_alpha", ""),
    }


def _generate_model_loss_records(
    *, config: dict[str, Any], rows: list[dict[str, Any]],
    root_file: Path, root_id: str,
) -> list[dict[str, Any]]:
    metric = str(config["selection"].get("window_metric", "ctr_ps"))
    voltage = _extract_voltage(root_file, config.get("reporting", {}))
    records: list[dict[str, Any]] = []
    pairs: list[tuple[str, str]] = [
        (model_id, str(loss["id"]))
        for model_id in config["models"]
        for loss in config["losses"]
    ]
    pairs.extend(
        (str(method).lower(), "evaluation_mse")
        for method in config.get("standard_methods", [])
    )
    for mode_id in config["channel_modes"]:
        for model_id, loss_id in pairs:
            selected = _best_model_loss_configuration_row(
                rows, root_id, mode_id, model_id, loss_id, metric
            )
            if selected is None:
                continue
            selected = {
                **selected,
                **_resolved_shapelet_metadata(config, rows, selected),
            }
            validation_mean = _configuration_summary_row(rows, selected, "validation", "mean")
            validation_sem = _configuration_summary_row(rows, selected, "validation", "sem")
            blind_mean = _configuration_summary_row(rows, selected, "blind", "mean")
            blind_sem = _configuration_summary_row(rows, selected, "blind", "sem")
            if validation_mean is None or blind_mean is None:
                continue
            records.append(
                _compact_model_record(
                    root_file=root_file, voltage=voltage, selected=selected,
                    validation_mean=validation_mean, validation_sem=validation_sem,
                    blind_mean=blind_mean, blind_sem=blind_sem,
                )
            )
    return records


def _plot_file_best_ctr_vs_window(
    *,
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    root_file: Path,
    root_id: str,
    output: Path,
    logger: Any,
) -> None:
    """Plot the lowest mean CV CTR for each model at every physical window size.

    For a fixed file, channel mode, model, and window, the plotted value is the
    minimum completed mean CV CTR over compatible losses and input transforms.
    Hyperparameters have already been selected by CV inside each window.
    """

    reporting = config.get("reporting", {})
    if not bool(reporting.get("plot_best_ctr_vs_window", True)):
        return
    metric = str(config["selection"].get("window_metric", "ctr_ps"))
    candidates = [
        row
        for row in rows
        if row.get("record_type") == "summary"
        and row.get("split") == "validation"
        and row.get("statistic") == "mean"
        and row.get("status") == "completed"
        and str(row.get("root_id", "")) == root_id
        and str(row.get("is_selected_hyperparameters", "")) == "1"
        and np.isfinite(_finite_float(row.get(metric)))
    ]
    if not candidates:
        logger.warning("Best-CTR-vs-window plot skipped for %s: no CV summaries", root_file.name)
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    modes = [
        mode_id
        for mode_id in config["channel_modes"]
        if any(str(row.get("channel_mode", "")) == mode_id for row in candidates)
    ]
    if not modes:
        return
    figure, axes = plt.subplots(
        len(modes),
        1,
        figsize=(9.0, max(4.6, 3.9 * len(modes))),
        squeeze=False,
    )
    for mode_index, mode_id in enumerate(modes):
        axis = axes[mode_index, 0]
        mode_rows = [
            row for row in candidates if str(row.get("channel_mode", "")) == mode_id
        ]
        for model_id in config["models"]:
            model_points: list[tuple[float, float]] = []
            for window in config["windows_ns"]:
                window_rows = [
                    row
                    for row in mode_rows
                    if str(row.get("model_id", "")) == model_id
                    and str(row.get("window_id", "")) == str(window["id"])
                ]
                if not window_rows:
                    continue
                best = min(window_rows, key=lambda row: _finite_float(row.get(metric)))
                size_ns = float(window["end_ns"]) - float(window["start_ns"])
                model_points.append((size_ns, _finite_float(best.get(metric))))
            if not model_points:
                continue
            model_points.sort(key=lambda item: item[0])
            x = np.asarray([item[0] for item in model_points], dtype=np.float64)
            y = np.asarray([item[1] for item in model_points], dtype=np.float64)
            axis.plot(x, y, marker="o", label=model_id)
        axis.set_title(mode_id)
        axis.set_xlabel("Window size (ns)")
        axis.set_ylabel("Best mean CV CTR (ps)")
        axis.grid(True, alpha=0.20)
        axis.legend()
    figure.suptitle(
        "Best mean cross-validation CTR versus window size\n"
        f"{root_file.name} — minimum over loss and input transform"
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    destination = (
        output
        / "summary_plots"
        / f"{_safe_name(root_file.stem)}_best_ctr_vs_window.png"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        destination,
        dpi=int(reporting.get("dpi", 180)),
        bbox_inches="tight",
    )
    plt.close(figure)


def _generate_file_secondary_reports(
    *,
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    root_file: Path,
    root_id: str,
    output: Path,
    logger: Any,
) -> None:
    model_loss_records = _generate_model_loss_records(
        config=config,
        rows=rows,
        root_file=root_file,
        root_id=root_id,
    )
    _update_model_loss_results(
        _model_loss_results_path(output),
        root_file,
        model_loss_records,
        config["root_files"],
    )
    _plot_file_best_ctr_vs_window(
        config=config,
        rows=rows,
        root_file=root_file,
        root_id=root_id,
        output=output,
        logger=logger,
    )


def _summary_contains_file(output: Path, root_file: Path) -> bool:
    return any(
        str(row.get("file_name", "")) == root_file.name
        for row in _read_summary_results(_summary_results_path(output))
    )


def _report_marker_is_current(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        marker = read_json(path)
    except (OSError, ValueError, TypeError):
        return False
    return int(marker.get("report_schema_version", 0)) >= REPORT_SCHEMA_VERSION


def _root_has_complete_requested_blocks(
    config: dict[str, Any], rows: list[dict[str, Any]], root_id: str
) -> bool:
    """Return whether every currently requested experiment block is complete.

    A completed-file marker from an older grid is insufficient: adding a model,
    loss, transform, or window must make ``--resume`` execute only the missing
    blocks rather than silently treating the file as complete.
    """

    for mode_id in config["channel_modes"]:
        for model_id in config["models"]:
            space = config["model_spaces"][model_id]
            supported = {str(value) for value in space["supported_losses"]}
            for loss in config["losses"]:
                if str(loss["type"]) not in supported:
                    continue
                for transform in config["input_transforms"]:
                    for window in config["windows_ns"]:
                        preprocessing_grid = config.get("preprocessing", {})
                        if "subsampling_factors" in preprocessing_grid:
                            requested_factors = {
                                int(value)
                                for value in preprocessing_grid.get(
                                    "subsampling_factors", [1]
                                )
                            }
                            observed_factors: set[int] = set()
                            for definition in rows:
                                if not (
                                    definition.get("record_type") == "trial_definition"
                                    and definition.get("status") == "completed"
                                    and str(definition.get("root_id", "")) == root_id
                                    and str(definition.get("channel_mode", "")) == mode_id
                                    and str(definition.get("model_id", "")) == model_id
                                    and str(definition.get("loss_id", "")) == str(loss["id"])
                                    and str(definition.get("input_transform", "")) == str(transform)
                                    and str(definition.get("window_id", "")) == str(window["id"])
                                ):
                                    continue
                                try:
                                    parameters = json.loads(
                                        str(definition.get("params_json", "{}") or "{}")
                                    )
                                    observed_factors.add(
                                        int(parameters.get("preprocessing.subsampling_factor", 1))
                                    )
                                except (TypeError, ValueError, json.JSONDecodeError):
                                    continue
                            if not requested_factors.issubset(observed_factors):
                                return False
                        validation = next(
                            (
                                row
                                for row in rows
                                if row.get("record_type") == "summary"
                                and row.get("split") == "validation"
                                and row.get("statistic") == "mean"
                                and row.get("status") == "completed"
                                and str(row.get("root_id", "")) == root_id
                                and str(row.get("channel_mode", "")) == mode_id
                                and str(row.get("model_id", "")) == model_id
                                and str(row.get("loss_id", "")) == str(loss["id"])
                                and str(row.get("input_transform", "")) == str(transform)
                                and str(row.get("window_id", "")) == str(window["id"])
                                and str(row.get("is_selected_hyperparameters", "")) == "1"
                            ),
                            None,
                        )
                        if validation is None:
                            return False
                        blind = _configuration_summary_row(
                            rows, validation, "blind", "mean"
                        )
                        if blind is None:
                            return False
    for mode_id in config["channel_modes"]:
        for method_id in config.get("standard_methods", []):
            validation = next(
                (
                    row for row in rows
                    if row.get("record_type") == "summary"
                    and row.get("split") == "validation"
                    and row.get("statistic") == "mean"
                    and row.get("status") == "completed"
                    and str(row.get("root_id", "")) == root_id
                    and str(row.get("channel_mode", "")) == mode_id
                    and str(row.get("model_id", "")) == str(method_id).lower()
                    and str(row.get("model_type", "")) == "standard_method"
                ),
                None,
            )
            if validation is None or _configuration_summary_row(
                rows, validation, "blind", "mean"
            ) is None:
                return False
    return True


def _generate_file_summary(
    *,
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    root_file: Path,
    root_id: str,
    development: PreparedDataset,
    blind: PreparedDataset,
    output: Path,
    run_root: Path,
    logger: Any,
) -> list[dict[str, Any]]:
    reporting = config.get("reporting", {})
    metric = str(config["selection"].get("window_metric", "ctr_ps"))
    voltage = _extract_voltage(root_file, reporting)
    records: list[dict[str, Any]] = []
    existing_records = {
        str(row.get("channel_mode", "")): row
        for row in _read_summary_results(_summary_results_path(output))
        if str(row.get("file_name", "")) == root_file.name
    }
    fits_by_mode: dict[str, tuple[Any, Any, dict[str, Any]]] = {}
    for mode_id in config["channel_modes"]:
        selected = _best_configuration_row(rows, root_id, mode_id, metric)
        if selected is None:
            logger.warning("No completed selected configuration for %s | %s", root_file.name, mode_id)
            continue
        validation_mean = _configuration_summary_row(rows, selected, "validation", "mean")
        validation_sem = _configuration_summary_row(rows, selected, "validation", "sem")
        blind_mean = _configuration_summary_row(rows, selected, "blind", "mean")
        blind_sem = _configuration_summary_row(rows, selected, "blind", "sem")
        if validation_mean is None or blind_mean is None:
            logger.warning("Missing validation/blind summary for %s | %s", root_file.name, mode_id)
            continue
        folds = _fold_masks(
            development,
            blind,
            CHANNEL_MODES[mode_id]["target"],
            config["cross_validation"],
            config["selection"],
        )
        try:
            validation_fit, blind_fit = _pooled_best_fits(
                config=config,
                root_id=root_id,
                mode_id=mode_id,
                selected=selected,
                development=development,
                blind=blind,
                folds=folds,
                run_root=run_root,
            )
        except (FileNotFoundError, ValueError) as exc:
            # Numeric fold summaries are sufficient to rebuild the compact CSV.
            # Pooled Gaussian fits require the temporary winning checkpoints; an
            # old results-only resume may legitimately no longer have them.
            logger.warning(
                "Pooled Gaussian fit unavailable for %s | %s: %s",
                root_file.name,
                mode_id,
                exc,
            )
            validation_fit, blind_fit = None, None
        previous = existing_records.get(mode_id, {})
        record = {
            "file_name": root_file.name,
            "voltage_V": voltage,
            "channel_mode": mode_id,
            "model_id": selected.get("model_id", ""),
            "model_type": selected.get("model_type", ""),
            "loss_id": selected.get("loss_id", ""),
            "loss_type": selected.get("loss_type", ""),
            "input_transform": selected.get("input_transform", ""),
            "subsampling_factor": int(float(selected.get("subsampling_factor", 1) or 1)),
            "window_start_ns": _finite_float(selected.get("window_start_ns")),
            "window_end_ns": _finite_float(selected.get("window_end_ns")),
            "trial_id": selected.get("trial_id", ""),
            "validation_n_events": int(round(_finite_float(validation_mean.get("n_events")))),
            "validation_loss_mean": _finite_float(validation_mean.get("loss")),
            "validation_rmse_mean_ps": _finite_float(validation_mean.get("rmse_ps")),
            "validation_rmse_sem_ps": _finite_float((validation_sem or {}).get("rmse_ps")),
            "validation_bias_mean_ps": _finite_float(validation_mean.get("bias_ps")),
            "validation_ctr_mean_ps": _finite_float(validation_mean.get("ctr_ps")),
            "validation_ctr_sem_ps": _finite_float((validation_sem or {}).get("ctr_ps")),
            "validation_gaussian_mean_ps": _fit_value_or_previous(
                validation_fit, "mean_ps", previous, "validation_gaussian_mean_ps"
            ),
            "validation_gaussian_sigma_ps": _fit_value_or_previous(
                validation_fit, "sigma_ps", previous, "validation_gaussian_sigma_ps"
            ),
            "validation_gaussian_ctr_ps": _fit_value_or_previous(
                validation_fit, "ctr_ps", previous, "validation_gaussian_ctr_ps"
            ),
            "validation_gaussian_chi2_ndof": _fit_value_or_previous(
                validation_fit, "chi2_ndof", previous,
                "validation_gaussian_chi2_ndof",
            ),
            "blind_n_events": int(round(_finite_float(blind_mean.get("n_events")))),
            "blind_loss_mean": _finite_float(blind_mean.get("loss")),
            "blind_rmse_mean_ps": _finite_float(blind_mean.get("rmse_ps")),
            "blind_rmse_sem_ps": _finite_float((blind_sem or {}).get("rmse_ps")),
            "blind_bias_mean_ps": _finite_float(blind_mean.get("bias_ps")),
            "blind_ctr_mean_ps": _finite_float(blind_mean.get("ctr_ps")),
            "blind_ctr_sem_ps": _finite_float((blind_sem or {}).get("ctr_ps")),
            "blind_gaussian_mean_ps": _fit_value_or_previous(
                blind_fit, "mean_ps", previous, "blind_gaussian_mean_ps"
            ),
            "blind_gaussian_sigma_ps": _fit_value_or_previous(
                blind_fit, "sigma_ps", previous, "blind_gaussian_sigma_ps"
            ),
            "blind_gaussian_ctr_ps": _fit_value_or_previous(
                blind_fit, "ctr_ps", previous, "blind_gaussian_ctr_ps"
            ),
            "blind_gaussian_chi2_ndof": _fit_value_or_previous(
                blind_fit, "chi2_ndof", previous, "blind_gaussian_chi2_ndof"
            ),
            "baseline_validation_ctr_mean_ps": _finite_float(validation_mean.get("baseline_ctr_ps")),
            "baseline_blind_ctr_mean_ps": _finite_float(blind_mean.get("baseline_ctr_ps")),
            "validation_relative_improvement_mean_pct": _finite_float(validation_mean.get("relative_improvement_pct")),
            "blind_relative_improvement_mean_pct": _finite_float(blind_mean.get("relative_improvement_pct")),
            **_resolved_shapelet_metadata(config, rows, selected),
        }
        records.append(record)
        fits_by_mode[mode_id] = (validation_fit, blind_fit, record)

    _update_summary_results(
        _summary_results_path(output), root_file, records, config["root_files"]
    )
    if bool(reporting.get("plot_best_gaussian_fits", True)):
        _plot_file_best_results(
            root_file,
            fits_by_mode,
            output / "summary_plots" / f"{_safe_name(root_file.stem)}_best.png",
            int(reporting.get("dpi", 180)),
        )
    _generate_file_secondary_reports(
        config=config,
        rows=rows,
        root_file=root_file,
        root_id=root_id,
        output=output,
        logger=logger,
    )
    marker = output / "_state" / "reported_files" / f"{root_id}.json"
    atomic_json(
        marker,
        {
            "file_name": root_file.name,
            "completed": True,
            "report_schema_version": REPORT_SCHEMA_VERSION,
        },
    )
    return records




def _refresh_summary_voltages(config: dict[str, Any], output: Path) -> None:
    """Backfill voltage values in existing compact summaries from file names."""
    path = _summary_results_path(output)
    rows = _read_summary_results(path)
    if not rows:
        return
    reporting = config.get("reporting", {})
    changed = False
    for row in rows:
        file_name = str(row.get("file_name", "")).strip()
        if not file_name:
            continue
        voltage = _extract_voltage(Path(file_name), reporting)
        current = _finite_float(row.get("voltage_V"))
        if np.isfinite(voltage) and (not np.isfinite(current) or current != voltage):
            row["voltage_V"] = voltage
            changed = True
    if changed:
        _write_summary_results(path, rows)

def _plot_ctr_vs_voltage(config: dict[str, Any], output: Path, logger: Any) -> None:
    reporting = config.get("reporting", {})
    naming = reporting.get("voltage_from_filename", {})
    if not bool(naming.get("enabled", False)) or not bool(naming.get("plot_ctr_vs_voltage", True)):
        return
    _refresh_summary_voltages(config, output)
    rows = _read_summary_results(_summary_results_path(output))
    usable = [
        row for row in rows
        if np.isfinite(_finite_float(row.get("voltage_V")))
        and np.isfinite(_finite_float(row.get("validation_gaussian_ctr_ps")))
        and np.isfinite(_finite_float(row.get("blind_gaussian_ctr_ps")))
    ]
    if not usable:
        logger.warning("CTR-vs-voltage plot skipped: no filename matched the configured voltage pattern")
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    modes = [mode for mode in CHANNEL_MODES if any(row.get("channel_mode") == mode for row in usable)]
    figure, axes = plt.subplots(
        len(modes), 1,
        figsize=(8.5, max(4.5, 3.8 * len(modes))),
        squeeze=False,
    )
    for index, mode_id in enumerate(modes):
        axis = axes[index, 0]
        group = sorted(
            (row for row in usable if row.get("channel_mode") == mode_id),
            key=lambda row: _finite_float(row.get("voltage_V")),
        )
        x = np.asarray([_finite_float(row["voltage_V"]) for row in group])
        validation = np.asarray([_finite_float(row["validation_gaussian_ctr_ps"]) for row in group])
        blind = np.asarray([_finite_float(row["blind_gaussian_ctr_ps"]) for row in group])
        axis.plot(x, validation, marker="o", label="Validation")
        axis.plot(x, blind, marker="s", label="Blind")
        axis.set_title(mode_id)
        axis.set_xlabel("Voltage (V)")
        axis.set_ylabel("CTR (ps)")
        axis.grid(True, alpha=0.20)
        axis.legend()
    figure.suptitle("Best CV-selected result versus voltage")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    destination = output / "summary_plots" / "ctr_vs_voltage.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=int(reporting.get("dpi", 180)), bbox_inches="tight")
    plt.close(figure)


def _root_id(path: Path) -> str:
    return f"{_safe_name(path.stem)}_{canonical_hash(str(path.resolve()))[:8]}"


def _shared_input_cache_root(
    config: dict[str, Any], root_id: str, mode_id: str,
    window_id: str, transform: str,
) -> Path:
    return (
        Path(config["experiment"]["output_dir"])
        / "shared_input_cache"
        / root_id
        / mode_id
        / str(window_id)
        / str(transform)
    )


def _close_dataset_memmaps(dataset: PreparedDataset) -> None:
    for value in dataset.__dict__.values():
        mmap = getattr(value, "_mmap", None)
        if mmap is not None:
            try:
                mmap.close()
            except OSError:
                pass


def _remove_tree(path: Path, logger: Any, label: str) -> None:
    if not path.exists():
        return
    try:
        shutil.rmtree(path)
    except OSError as exc:
        logger.warning("Could not remove %s at %s: %s", label, path, exc)


def _keep_study_checkpoints(config: dict[str, Any]) -> bool:
    """Return whether study model checkpoints should persist after reporting.

    The study always creates a best checkpoint temporarily because the common
    evaluation path rebuilds predictions from it.  With the default ``False``
    policy, losing trials and windows are pruned during the run and the final
    winning checkpoints are deleted after the compact CSV/plots are written.
    """

    return bool(config.get("storage", {}).get("keep_checkpoints", False))


def _trial_run_directory(
    run_root: Path, root_id: str, mode_id: str, model_id: str, loss_id: str,
    transform: str, window_id: str, trial_id: str,
) -> Path:
    return (
        run_root / root_id / mode_id / model_id / loss_id / transform
        / window_id / trial_id
    )


def _fold_run_directory(trial_directory: Path, fold_id: int) -> Path:
    return trial_directory / f"fold_{fold_id}"


def _has_best_checkpoint(run_dir: Path) -> bool:
    return (run_dir / "checkpoints" / "best.pt").is_file()


def _linear_weights_path(output: Path) -> Path:
    return output / "linear_model_weights.csv"


def _read_plain_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _linear_feature_metadata(
    dataset_view: PreparedDataset,
    input_transform: str,
    subsampling_factor: int,
) -> list[dict[str, Any]]:
    """Describe every transformed linear feature in model-vector order."""

    transform = normalize_input_transform(input_transform)
    values = np.asarray(dataset_view.relative_time_ps, dtype=np.float64)
    raw_lengths = dataset_view.manifest.get("input_component_lengths")
    lengths = (
        [int(value) for value in raw_lengths]
        if isinstance(raw_lengths, list)
        else [int(values.size)]
    )
    raw_components = dataset_view.manifest.get("input_components")
    components = (
        [str(value) for value in raw_components]
        if isinstance(raw_components, list)
        else ["waveform"]
    )
    if len(components) != len(lengths) or sum(lengths) != int(values.size):
        raise ValueError("Invalid input component metadata for linear-weight export")

    boundaries = np.cumsum([0, *lengths])
    rows: list[dict[str, Any]] = []
    feature_index = 0
    for component_index, (component, length) in enumerate(zip(components, lengths)):
        time = values[boundaries[component_index] : boundaries[component_index + 1]]
        if transform in {"none", "normalize"}:
            pieces = [("raw", time)]
        elif transform == "differentiate":
            pieces = [("first_difference", 0.5 * (time[1:] + time[:-1]))]
        elif transform == "concatenate_diff":
            pieces = [
                ("raw", time),
                ("first_difference", 0.5 * (time[1:] + time[:-1])),
            ]
        else:  # guarded by normalize_input_transform
            raise ValueError(f"Unsupported input transform: {transform}")
        for feature_kind, feature_times in pieces:
            selected_feature_indices = np.arange(
                0, len(feature_times), int(subsampling_factor), dtype=np.int64
            )
            for component_feature_index in selected_feature_indices:
                relative_time_ps = feature_times[int(component_feature_index)]
                rows.append(
                    {
                        "feature_index": feature_index,
                        "component_index": component_index,
                        "component": component,
                        "feature_kind": feature_kind,
                        "component_feature_index": component_feature_index,
                        "relative_time_ps": float(relative_time_ps),
                    }
                )
                feature_index += 1
    return rows


def _persist_selected_linear_weights(
    *,
    output: Path,
    config: dict[str, Any],
    development: PreparedDataset,
    root_id: str,
    root_file: Path,
    mode_id: str,
    mode: dict[str, str],
    model_id: str,
    model_type: str,
    loss: dict[str, Any],
    transform: str,
    subsampling_factor: int,
    window: dict[str, Any],
    trial_id: str,
    folds: list[dict[str, Any]],
    run_root: Path,
    cv_rows: list[dict[str, Any]],
) -> bool:
    """Persist compact fold coefficients for the selected trial of one window.

    Run/checkpoint directories may be deleted later.  This CSV is intentionally
    the only durable coefficient artifact and contains both coefficients in the
    normalized model space and coefficients converted back to physical feature
    units.
    """

    if str(model_type) != "linear_regression":
        return True

    dataset_view = prediction_window_dataset_view(
        development,
        input_waveforms=mode["input_waveforms"],
        target=mode["target"],
        before_ns=float(window["before_ns"]),
        after_ns=float(window["after_ns"]),
    )
    metadata = _linear_feature_metadata(
        dataset_view, transform, subsampling_factor
    )
    expected_length = len(metadata)
    weight_rows: list[dict[str, Any]] = []

    cv_mean_ctr = _selection_value(cv_rows, "ctr_ps")
    cv_mean_rmse = _selection_value(cv_rows, "rmse_ps")
    cv_mean_loss = _selection_value(cv_rows, "loss")

    for fold in folds:
        fold_id = int(fold["fold_id"])
        run_dir = (
            run_root / root_id / mode_id / model_id / str(loss["id"])
            / transform / str(window["id"]) / trial_id / f"fold_{fold_id}"
        )
        coefficient_path = run_dir / "linear_regression_weight.npy"
        summary_path = run_dir / "training_summary.json"
        if not coefficient_path.is_file() or not summary_path.is_file():
            return False
        coefficient = np.asarray(np.load(coefficient_path), dtype=np.float64).reshape(-1)
        if coefficient.size != expected_length:
            raise ValueError(
                "Linear coefficient length does not match transformed feature metadata: "
                f"{coefficient.size} != {expected_length}"
            )
        summary = read_json(summary_path)
        normalization = summary.get("normalization", {})
        std = np.asarray(normalization.get("std_mV", 1.0), dtype=np.float64)
        if std.ndim == 0:
            std = np.full(expected_length, float(std), dtype=np.float64)
        if std.shape != (expected_length,) or np.any(std <= 0.0):
            raise ValueError("Invalid normalization scale in linear-regression summary")
        physical = coefficient / std
        regularization = str(summary.get("regularization", "none"))
        alpha = float(summary.get("alpha", 0.0))
        pair_bias_ps = float(summary.get("pair_output_bias_ps", 0.0))
        for feature, normalized_weight, physical_weight in zip(
            metadata, coefficient, physical
        ):
            identity = {
                "root": root_id,
                "mode": mode_id,
                "model": model_id,
                "loss": str(loss["id"]),
                "transform": transform,
                "window": str(window["id"]),
                "trial": trial_id,
                "fold": fold_id,
                "feature": int(feature["feature_index"]),
            }
            weight_rows.append(
                {
                    "row_key": canonical_hash(identity)[:24],
                    "experiment_id": config["experiment"]["name"],
                    "root_id": root_id,
                    "root_file": str(root_file),
                    "channel_mode": mode_id,
                    "model_id": model_id,
                    "model_type": model_type,
                    "loss_id": str(loss["id"]),
                    "loss_type": str(loss["type"]),
                    "input_transform": transform,
                    "subsampling_factor": int(subsampling_factor),
                    "window_id": str(window["id"]),
                    "window_start_ns": float(window["start_ns"]),
                    "window_end_ns": float(window["end_ns"]),
                    "window_length_ns": float(window["end_ns"]) - float(window["start_ns"]),
                    "trial_id": trial_id,
                    "fold_id": fold_id,
                    "regularization": regularization,
                    "alpha": alpha,
                    "pair_output_bias_ps": pair_bias_ps,
                    "feature_index": int(feature["feature_index"]),
                    "component_index": int(feature["component_index"]),
                    "component": str(feature["component"]),
                    "feature_kind": str(feature["feature_kind"]),
                    "component_feature_index": int(feature["component_feature_index"]),
                    "relative_time_ps": float(feature["relative_time_ps"]),
                    "relative_time_ns": float(feature["relative_time_ps"]) / 1000.0,
                    "weight_normalized": float(normalized_weight),
                    "weight_physical_ps_per_mV": float(physical_weight),
                    "abs_weight_normalized": float(abs(normalized_weight)),
                    "abs_weight_physical_ps_per_mV": float(abs(physical_weight)),
                    "cv_mean_loss": float(cv_mean_loss),
                    "cv_mean_rmse_ps": float(cv_mean_rmse),
                    "cv_mean_ctr_ps": float(cv_mean_ctr),
                    "is_selected_hyperparameters": 1,
                    "is_selected_window": 0,
                }
            )

    path = _linear_weights_path(output)
    existing = _read_plain_csv(path)
    replacements = {str(row["row_key"]): row for row in weight_rows}
    kept = [row for row in existing if str(row.get("row_key", "")) not in replacements]
    write_csv_rows(path, [*kept, *weight_rows])
    return True


def _mark_selected_linear_weight_window(
    *,
    output: Path,
    root_id: str,
    mode_id: str,
    model_id: str,
    loss_id: str,
    transform: str,
    selected_window_id: str,
) -> None:
    path = _linear_weights_path(output)
    rows = _read_plain_csv(path)
    if not rows:
        return
    changed = False
    for row in rows:
        if (
            row.get("root_id") == root_id
            and row.get("channel_mode") == mode_id
            and row.get("model_id") == model_id
            and row.get("loss_id") == loss_id
            and row.get("input_transform") == transform
        ):
            value = "1" if row.get("window_id") == selected_window_id else "0"
            if row.get("is_selected_window") != value:
                row["is_selected_window"] = value
                changed = True
    if changed:
        write_csv_rows(path, rows)


def _shapelet_models_path(output: Path) -> Path:
    return output / "shapelet_models.csv"


def _shapelet_artifact_is_current(
    config: dict[str, Any], output: Path, root_id: str
) -> bool:
    requested = any(
        str(config["model_spaces"][model_id].get("model_type", ""))
        == "shapelet_regressor"
        for model_id in config.get("models", [])
    )
    if not requested:
        return True
    return any(
        str(row.get("root_id", "")) == root_id
        for row in _read_plain_csv(_shapelet_models_path(output))
    )


def _trial_parameters_for_selected_row(
    rows: list[dict[str, Any]], selected: dict[str, Any]
) -> dict[str, Any]:
    definition = next(
        (
            row
            for row in rows
            if row.get("record_type") == "trial_definition"
            and _row_matches_configuration(row, selected)
        ),
        None,
    )
    if definition is None:
        return {}
    raw = definition.get("params_json", "")
    if not raw:
        return {}
    value = json.loads(str(raw))
    if not isinstance(value, dict):
        raise ValueError("trial_definition params_json must decode to an object")
    return value


def _resolved_shapelet_metadata(
    config: dict[str, Any], rows: list[dict[str, Any]], selected: dict[str, Any]
) -> dict[str, Any]:
    if str(selected.get("model_type", "")) != "shapelet_regressor":
        return {
            "shapelet_count": "",
            "shapelet_distance_metric": "",
            "shapelet_dtw_radius_points": "",
            "shapelet_ridge_alpha": "",
        }
    model_id = str(selected.get("model_id", ""))
    space = config["model_spaces"][model_id]
    model = copy.deepcopy(space["base_train_config"].get("model", {}))
    for key, value in _trial_parameters_for_selected_row(rows, selected).items():
        if str(key).startswith("model."):
            _set_nested({"model": model}, str(key), copy.deepcopy(value))
    return {
        "shapelet_count": int(model.get("n_shapelets", 0)),
        "shapelet_distance_metric": str(model.get("distance_metric", "")),
        "shapelet_dtw_radius_points": int(model.get("dtw_radius_points", 0)),
        "shapelet_ridge_alpha": float(model.get("ridge_alpha", float("nan"))),
    }


def _best_shapelet_configuration_row(
    config: dict[str, Any], rows: list[dict[str, Any]], root_id: str
) -> dict[str, Any] | None:
    metric = _canonical_study_metric(
        str(config["selection"].get("window_metric", "ctr_ps"))
    )
    candidates = [
        row
        for row in rows
        if row.get("record_type") == "summary"
        and row.get("split") == "validation"
        and row.get("statistic") == "mean"
        and row.get("status") == "completed"
        and str(row.get("root_id", "")) == root_id
        and str(row.get("model_type", "")) == "shapelet_regressor"
        and str(row.get("is_selected_hyperparameters", "")) == "1"
        and str(row.get("is_selected_window", "")) == "1"
        and np.isfinite(_row_metric_value(row, metric))
    ]
    return min(candidates, key=lambda row: _row_metric_value(row, metric)) if candidates else None


def _plot_compact_shapelet_model(
    *, rows: list[dict[str, Any]], destination: Path, title: str, dpi: int
) -> None:
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    ordered = sorted(
        rows,
        key=lambda row: (int(row["fold_id"]), int(row["rank"])),
    )
    count = len(ordered)
    height = max(5.0, 0.28 * count + 2.6)
    figure, (trace_axis, importance_axis) = plt.subplots(
        1,
        2,
        figsize=(12.5, height),
        gridspec_kw={"width_ratios": [4.6, 1.35]},
    )
    y_positions = np.arange(count, 0, -1, dtype=np.float64)
    labels: list[str] = []
    importance: list[float] = []
    seen_folds: set[int] = set()
    for y, row in zip(y_positions, ordered):
        fold_id = int(row["fold_id"])
        rank = int(row["rank"])
        values = np.fromstring(str(row.get("values", "")), sep=" ", dtype=np.float64)
        if values.size == 0:
            continue
        centered = values - float(np.mean(values))
        scale = float(np.max(np.abs(centered)))
        display_values = centered / scale if scale > 0 else centered
        display_values = y + 0.32 * display_values
        times = np.linspace(
            float(row["start_time_ns"]),
            float(row["end_time_ns"]),
            values.size,
        )
        label = f"Fold {fold_id}" if fold_id not in seen_folds else None
        seen_folds.add(fold_id)
        color = f"C{fold_id % 10}"
        trace_axis.plot(
            times,
            display_values,
            linewidth=1.25,
            label=label,
            color=color,
        )
        trace_axis.hlines(
            y,
            float(row["start_time_ns"]),
            float(row["end_time_ns"]),
            linewidth=0.45,
            alpha=0.35,
            color=color,
        )
        labels.append(f"F{fold_id} S{rank}")
        importance.append(float(row.get("mean_abs_contribution_ps", 0.0)))

    trace_axis.axvline(0.0, linestyle="--", linewidth=0.9, label="LED anchor")
    trace_axis.set_yticks(y_positions)
    trace_axis.set_yticklabels(labels, fontsize=7)
    trace_axis.set_xlabel("Time relative to LED anchor (ns)")
    trace_axis.set_ylabel("Fold and selected shapelet rank")
    trace_axis.set_title("Used shapelets at their fixed physical positions")
    trace_axis.grid(True, axis="x", alpha=0.18)
    trace_axis.set_ylim(0.25, count + 0.75)

    importance_axis.barh(y_positions, np.asarray(importance, dtype=np.float64))
    importance_axis.set_yticks([])
    importance_axis.set_xlabel("Mean |linear contribution| (ps)")
    importance_axis.set_title("Training-fold importance")
    importance_axis.grid(True, axis="x", alpha=0.18)
    importance_axis.set_ylim(trace_axis.get_ylim())

    figure.suptitle(title, y=0.995)
    handles, legend_labels = trace_axis.get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=max(2, min(6, len(legend_labels))),
        frameon=False,
    )
    # Reserve independent vertical bands for title, legend, and axes. This
    # prevents the figure-level legend from covering traces or panel titles.
    figure.subplots_adjust(left=0.12, right=0.97, bottom=0.07, top=0.88, wspace=0.24)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)


def _persist_best_shapelet_model_for_file(
    *,
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    root_file: Path,
    root_id: str,
    development: PreparedDataset,
    blind: PreparedDataset,
    output: Path,
    run_root: Path,
    logger: Any,
) -> None:
    """Save only the CV-best shapelet model for one ROOT file.

    The durable artifact is one compact CSV row per actually used shapelet and
    fold. Trial checkpoints, candidate pools, and feature matrices remain
    temporary and are removed by the normal results-only cleanup path.
    """

    selected = _best_shapelet_configuration_row(config, rows, root_id)
    path = _shapelet_models_path(output)
    existing = [
        row for row in _read_plain_csv(path) if str(row.get("root_id", "")) != root_id
    ]
    if selected is None:
        if path.is_file() and len(existing) != len(_read_plain_csv(path)):
            write_csv_rows(path, existing)
        return

    mode_id = str(selected["channel_mode"])
    model_id = str(selected["model_id"])
    loss_id = str(selected["loss_id"])
    transform = str(selected["input_transform"])
    window_id = str(selected["window_id"])
    trial_id = str(selected["trial_id"])
    space = config["model_spaces"][model_id]
    loss = next(item for item in config["losses"] if str(item["id"]) == loss_id)
    window = next(item for item in config["windows_ns"] if str(item["id"]) == window_id)
    mode = CHANNEL_MODES[mode_id]
    parameters = _trial_parameters_for_selected_row(rows, selected)
    folds = _fold_masks(
        development,
        blind,
        mode["target"],
        config["cross_validation"],
        config["selection"],
    )
    artifact_rows: list[dict[str, Any]] = []
    for fold in folds:
        fold_id = int(fold["fold_id"])
        run_dir = (
            run_root
            / root_id
            / mode_id
            / model_id
            / loss_id
            / transform
            / window_id
            / trial_id
            / f"fold_{fold_id}"
        )
        source = run_dir / "shapelet_model.csv"
        if not source.is_file():
            rebuilt_cv_row, _ = _run_fold(
                config=config,
                development=development,
                blind=blind,
                root_id=root_id,
                mode_id=mode_id,
                mode=mode,
                model_id=model_id,
                space=space,
                loss=loss,
                transform=transform,
                window=window,
                parameters=parameters,
                trial_id=trial_id,
                fold=fold,
                run_dir=run_dir,
                evaluate_blind=False,
                logger=logger,
            )
            rebuilt_cv_row.update(selected)
            _upsert_result_row(rows, rebuilt_cv_row)
        for raw in _read_plain_csv(source):
            identity = {
                "root": root_id,
                "fold": fold_id,
                "candidate": raw.get("candidate_id", ""),
            }
            artifact_rows.append(
                {
                    "row_key": canonical_hash(identity)[:24],
                    "root_id": root_id,
                    "root_file": str(root_file),
                    "file_name": root_file.name,
                    "channel_mode": mode_id,
                    "model_id": model_id,
                    "loss_id": loss_id,
                    "input_transform": transform,
                    "subsampling_factor": int(float(selected.get("subsampling_factor", 1) or 1)),
                    "window_id": window_id,
                    "window_start_ns": float(window["start_ns"]),
                    "window_end_ns": float(window["end_ns"]),
                    "trial_id": trial_id,
                    "fold_id": fold_id,
                    **raw,
                }
            )
    write_csv_rows(path, [*existing, *artifact_rows])
    metadata = _resolved_shapelet_metadata(config, rows, selected)
    title = (
        f"CV-best shapelet regressor — {root_file.name}\n"
        f"{mode_id}, {transform}, window {window_id}, "
        f"factor={int(float(selected.get('subsampling_factor', 1) or 1))}, "
        f"K={metadata['shapelet_count']}, "
        f"{metadata['shapelet_distance_metric']}"
    )
    _plot_compact_shapelet_model(
        rows=artifact_rows,
        destination=output / "summary_plots" / f"{_safe_name(root_file.stem)}_shapelets.png",
        title=title,
        dpi=int(config.get("reporting", {}).get("dpi", 180)),
    )


def _prune_window_trials(
    *, run_root: Path, root_id: str, mode_id: str, model_id: str, loss_id: str,
    transform: str, window_id: str, keep_trial_id: str, logger: Any,
) -> None:
    """Delete every trial directory for one window except the CV winner."""

    window_directory = (
        run_root / root_id / mode_id / model_id / loss_id / transform / window_id
    )
    if not window_directory.is_dir():
        return
    for candidate in window_directory.iterdir():
        if candidate.is_dir() and candidate.name != keep_trial_id:
            _remove_tree(candidate, logger, "non-selected trial artifacts")


def _prune_configuration_windows(
    *, run_root: Path, root_id: str, mode_id: str, model_id: str, loss_id: str,
    transform: str, keep_window_id: str, logger: Any,
) -> None:
    """Keep only the CV-selected window for one model/loss/transform."""

    transform_directory = (
        run_root / root_id / mode_id / model_id / loss_id / transform
    )
    if not transform_directory.is_dir():
        return
    for candidate in transform_directory.iterdir():
        if candidate.is_dir() and candidate.name != keep_window_id:
            _remove_tree(candidate, logger, "non-selected window artifacts")


def _prune_file_runs_to_summary_winners(
    *, config: dict[str, Any], rows: list[dict[str, Any]], root_id: str,
    run_root: Path, logger: Any,
) -> None:
    """Retain only the single overall CV winner needed for each channel mode."""

    root_directory = run_root / root_id
    if not root_directory.is_dir():
        return
    metric = str(config["selection"].get("window_metric", "ctr_ps"))
    keep: set[Path] = set()
    for mode_id in config["channel_modes"]:
        selected = _best_configuration_row(rows, root_id, mode_id, metric)
        if selected is None:
            continue
        keep.add(
            _trial_run_directory(
                run_root, root_id, mode_id, str(selected["model_id"]),
                str(selected["loss_id"]), str(selected["input_transform"]),
                str(selected["window_id"]), str(selected["trial_id"]),
            ).resolve()
        )

    # Trial directories have the fixed hierarchy
    # mode/model/loss/transform/window/trial below the file root.
    for trial_directory in list(root_directory.glob("*/*/*/*/*/*")):
        if trial_directory.is_dir() and trial_directory.resolve() not in keep:
            _remove_tree(trial_directory, logger, "non-winning model artifacts")

    # Remove now-empty hierarchy directories without touching retained trials.
    for directory in sorted(
        (path for path in root_directory.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        try:
            directory.rmdir()
        except OSError:
            pass


def _build_preprocess_config(
    config: dict[str, Any], root_file: Path, root_id: str, output: Path
) -> dict[str, Any]:
    preprocessing = config["preprocessing"]
    common = copy.deepcopy(preprocessing.get("common", {}))
    energy = _deep_update(copy.deepcopy(common), preprocessing.get("energy", {}))
    timing = _deep_update(copy.deepcopy(common), preprocessing.get("timing", {}))
    # Study invariant: timing-channel waveforms are never low-pass filtered.
    # This is intentionally not a tunable experiment axis.
    timing["denoising"] = {"enabled": False}
    max_before = max(float(window["before_ns"]) for window in config["windows_ns"])
    max_after = max(float(window["after_ns"]) for window in config["windows_ns"])
    energy["ml_window_ns"] = {"before": max_before, "after": max_after}
    timing["ml_window_ns"] = {"before": max_before, "after": max_after}
    timing["enabled"] = True
    energy["timing_channel_led"] = timing

    prepared_root = output / "prepared" / root_id
    cache_root = output / "cache" / root_id
    split = copy.deepcopy(config.get("split", {}))
    blind_fraction = float(split.pop("blind_fraction", 0.15))
    # Kept as an ignored migration key so older study configs remain loadable.
    # Cross-validation creates all validation folds from the full development set.
    split.pop("initial_validation_fraction", None)
    if not 0.0 < blind_fraction < 1.0:
        raise ValueError("split.blind_fraction must lie strictly between 0 and 1")
    development_fraction = 1.0 - blind_fraction

    selection = copy.deepcopy(preprocessing.get("selection", {}))
    # Robust target-specific outlier rejection belongs to each CV fold, not the
    # preprocessing split.  Preprocessing only freezes waveform validity.
    selection["led_outlier_rejection"] = {"enabled": False}
    selection.setdefault("minimum_events_per_split", 1)

    return {
        "dataset": {
            "name": f"{root_id}_development",
            "role": "training",
            "output_dir": str(prepared_root / "development"),
            "blind_test": {
                "name": f"{root_id}_blind",
                "output_dir": str(prepared_root / "blind"),
            },
        },
        "data": {
            "input_root": str(root_file),
            "true_tof_ps": float(config["data"].get("true_tof_ps", 0.0)),
        },
        "channels": copy.deepcopy(config["data"]["channels"]),
        "io": copy.deepcopy(preprocessing.get("io", {
            "step_size": "128 MB", "max_events": 0, "progress_every": 1000
        })),
        "waveform": energy,
        "selection": selection,
        "photopeak": copy.deepcopy(preprocessing.get("photopeak", {"enabled": False})),
        "split": {
            "strategy": str(split.get("strategy", "contiguous_blocks")),
            "seed": int(split.get("seed", 20260804)),
            "development_blind": True,
            "train_fraction": development_fraction,
            "validation_fraction": 0.0,
            "test_fraction": blind_fraction,
            "guard_gap_events": int(split.get("guard_gap_events", 0)),
        },
        "parallelization": copy.deepcopy(preprocessing.get("parallelization", {
            "preprocessing_backend": "process",
            "preprocessing_workers": 0,
            "preprocessing_chunksize": 8,
        })),
        "cache": {
            "reuse": True,
            "raw_cache_dir": str(cache_root / "raw"),
            "selection_cache_dir": str(cache_root / "selection"),
            "materialization_chunk_size": int(
                preprocessing.get("materialization_chunk_size", 2048)
            ),
        },
        "logging": copy.deepcopy(config.get("logging", {"level": "INFO"})),
    }


def _ensure_preprocessed(
    config: dict[str, Any], root_file: Path, root_id: str, output: Path,
    *, rebuild: bool, logger: Any
) -> tuple[PreparedDataset, PreparedDataset]:
    from .preprocessing import preprocess_dataset

    generated = _build_preprocess_config(config, root_file, root_id, output)
    config_path = output / "resolved_preprocessing" / f"{root_id}.json"

    # Programmatically generated preprocessing configurations do not pass through
    # ml_pipeline.config.load_preprocess_config(), which normally injects the
    # provenance fields required by dataset materialization. Persist the clean
    # resolved JSON first, then add the in-memory metadata used by manifests.
    atomic_json(config_path, generated)
    generated["_config_path"] = str(config_path.resolve())
    generated["_config_hash"] = canonical_hash(generated)

    preprocess_dataset(generated, rebuild=rebuild, logger=logger)
    development = load_prepared_dataset(generated["dataset"]["output_dir"])
    blind = load_prepared_dataset(generated["dataset"]["blind_test"]["output_dir"])

    # The raw cache is only an intermediate source for materializing the prepared
    # development/blind datasets. Keeping it for every ROOT file roughly doubles
    # storage without helping training or resume.
    storage = config.get("storage", {})
    if bool(storage.get("cleanup_raw_cache_after_materialization", True)):
        _remove_tree(
            Path(generated["cache"]["raw_cache_dir"]),
            logger,
            "raw preprocessing cache",
        )
    return development, blind


def _kfold(indices: np.ndarray, n_splits: int, seed: int, shuffle: bool) -> list[tuple[np.ndarray, np.ndarray]]:
    values = np.asarray(indices, dtype=np.int64).copy()
    if not 2 <= int(n_splits) <= values.size:
        raise ValueError(f"n_splits must be in [2, {values.size}]")
    if shuffle:
        np.random.default_rng(seed).shuffle(values)
    chunks = np.array_split(values, int(n_splits))
    output = []
    for fold, validation in enumerate(chunks):
        training = np.concatenate([chunk for index, chunk in enumerate(chunks) if index != fold])
        output.append((training.astype(np.int64), validation.astype(np.int64)))
    return output


def _target_times(dataset: PreparedDataset, target: str) -> np.ndarray:
    if target == "energy_led":
        array = dataset.energy_led_time_fs
    elif target == "timing_led":
        array = dataset.timing_led_time_fs
    else:
        array = dataset.led_time_fs
    if array is None:
        raise ValueError(f"Dataset {dataset.directory} lacks target {target}")
    return np.asarray(array)


def _delta_ps(dataset: PreparedDataset, target: str, indices: np.ndarray) -> np.ndarray:
    values = _target_times(dataset, target)[np.asarray(indices, dtype=np.int64)]
    return (values[:, 0].astype(np.float64) - values[:, 1].astype(np.float64)) / 1000.0


def _standard_method_time_array(
    dataset: PreparedDataset, target: str, method_id: str
) -> np.ndarray:
    method = str(method_id).strip().lower()
    if method == "led":
        return _target_times(dataset, target)
    if method != "cfd":
        raise ValueError(f"Unsupported standard method: {method_id!r}")

    if target == "energy_led":
        array = dataset.energy_cfd_time_fs
        family = "energy"
    elif target == "timing_led":
        array = dataset.timing_cfd_time_fs
        family = "timing"
    else:
        array = dataset.cfd_time_fs
        family = "prepared"
    if array is None:
        raise ValueError(
            f"Dataset {dataset.directory} lacks {family}-channel CFD timestamps; "
            "rebuild preprocessing with the target-specific CFD format"
        )
    return np.asarray(array)


def _standard_method_delta_ps(
    dataset: PreparedDataset, target: str, method_id: str, indices: np.ndarray
) -> np.ndarray:
    values = _standard_method_time_array(dataset, target, method_id)[
        np.asarray(indices, dtype=np.int64)
    ]
    return (values[:, 0].astype(np.float64) - values[:, 1].astype(np.float64)) / 1000.0


def _standard_method_base(
    *, config: dict[str, Any], root_id: str, root_file: Path, mode_id: str,
    method_id: str,
) -> dict[str, Any]:
    return {
        "experiment_id": config["experiment"]["name"],
        "root_id": root_id,
        "root_file": str(root_file),
        "channel_mode": mode_id,
        "model_id": str(method_id).lower(),
        "model_type": "standard_method",
        # Standard methods are not optimized. The loss column is an evaluation
        # MSE so it remains numerically interpretable beside RMSE, bias and CTR.
        "loss_id": "evaluation_mse",
        "loss_type": "mse",
        "input_transform": "not_applicable",
        "window_id": "not_applicable",
        "window_start_ns": float("nan"),
        "window_end_ns": float("nan"),
        "trial_id": "not_applicable",
        "is_selected_hyperparameters": 1,
        "is_selected_window": 1,
    }


def _evaluate_standard_methods(
    *, config: dict[str, Any], rows: list[dict[str, Any]],
    development: PreparedDataset, blind: PreparedDataset, root_id: str,
    root_file: Path, mode_id: str, mode: dict[str, str],
    folds: list[dict[str, Any]], logger: Any,
) -> None:
    methods = [str(value).lower() for value in config.get("standard_methods", [])]
    if not methods:
        return
    evaluation_loss = {"type": "mse"}
    true_development = float(development.true_tof_ps)
    true_blind = float(blind.true_tof_ps)

    for method_id in methods:
        base = _standard_method_base(
            config=config, root_id=root_id, root_file=root_file,
            mode_id=mode_id, method_id=method_id,
        )
        validation_rows: list[dict[str, Any]] = []
        blind_rows: list[dict[str, Any]] = []
        for fold in folds:
            fold_id = int(fold["fold_id"])
            train_indices = np.asarray(fold["train"], dtype=np.int64)
            validation_indices = np.asarray(fold["validation"], dtype=np.int64)
            blind_indices = np.asarray(fold["blind"], dtype=np.int64)
            train_target = _delta_ps(development, mode["target"], train_indices)
            target_scale = max(float(np.std(train_target, ddof=0)), 1e-8)

            validation_values = _standard_method_delta_ps(
                development, mode["target"], method_id, validation_indices
            )
            validation_baseline = _delta_ps(
                development, mode["baseline"], validation_indices
            )
            validation_metrics = _metrics(
                validation_values,
                np.full(validation_values.shape, true_development, dtype=np.float64),
                validation_baseline,
                config["fit"],
                evaluation_loss,
                target_scale,
            )
            robust: RobustLocationScale = fold["robust"]
            validation_row = {
                **base,
                "record_type": "cv_fold",
                "fold_id": fold_id,
                "split": "validation",
                "statistic": "raw",
                "status": "completed",
                "n_events": int(validation_indices.size),
                **validation_metrics,
                "outlier_center_ps": robust.center_ps,
                "outlier_scale_ps": robust.scale_ps,
                "outlier_scale_method": robust.method,
                "outlier_z_threshold": fold["z_threshold"],
                "runtime_seconds": 0.0,
            }
            validation_row["row_key"] = canonical_hash({
                "type": "standard_cv_fold", "root": root_id, "mode": mode_id,
                "method": method_id, "fold": fold_id,
            })[:24]
            validation_rows.append(_upsert_result_row(rows, validation_row))

            blind_values = _standard_method_delta_ps(
                blind, mode["target"], method_id, blind_indices
            )
            blind_baseline = _delta_ps(blind, mode["baseline"], blind_indices)
            blind_metrics = _metrics(
                blind_values,
                np.full(blind_values.shape, true_blind, dtype=np.float64),
                blind_baseline,
                config["fit"],
                evaluation_loss,
                target_scale,
            )
            blind_row = {
                **base,
                "record_type": "blind_fold",
                "fold_id": fold_id,
                "split": "blind",
                "statistic": "raw",
                "status": "completed",
                "n_events": int(blind_indices.size),
                **blind_metrics,
                "outlier_center_ps": robust.center_ps,
                "outlier_scale_ps": robust.scale_ps,
                "outlier_scale_method": robust.method,
                "outlier_z_threshold": fold["z_threshold"],
                "runtime_seconds": 0.0,
            }
            blind_row["row_key"] = canonical_hash({
                "type": "standard_blind_fold", "root": root_id, "mode": mode_id,
                "method": method_id, "fold": fold_id,
            })[:24]
            blind_rows.append(_upsert_result_row(rows, blind_row))

        _append_summary_rows(rows, validation_rows, base, "validation", True)
        _append_summary_rows(rows, blind_rows, base, "blind", True)
        logger.info(
            "Standard method | file=%s | mode=%s | method=%s | CV CTR %.3f ps | blind CTR %.3f ps",
            root_file.name, mode_id, method_id.upper(),
            _selection_value(validation_rows, "ctr_ps"),
            _selection_value(blind_rows, "ctr_ps"),
        )


def _fold_masks(
    development: PreparedDataset,
    blind: PreparedDataset,
    target: str,
    cv: dict[str, Any],
    selection: dict[str, Any],
) -> list[dict[str, Any]]:
    development_indices = np.asarray(development.train, dtype=np.int64)
    if development.validation.size:
        raise RuntimeError(
            "Cross-validation study preprocessing must not contain a preliminary "
            "validation split"
        )
    folds = _kfold(
        development_indices,
        int(cv.get("n_splits", 5)),
        int(cv.get("seed", 20260804)),
        bool(cv.get("shuffle", True)),
    )
    z_threshold = float(selection.get("z_threshold", 4.0))
    minimum = int(selection.get("minimum_events_per_fold", 100))
    blind_candidates = np.asarray(blind.evaluation, dtype=np.int64)
    output = []
    for fold_id, (train_candidate, validation_candidate) in enumerate(folds):
        train_values = _delta_ps(development, target, train_candidate)
        fitted = fit_median_mad_z(train_values)
        train = train_candidate[robust_z_mask(train_values, fitted, z_threshold)]
        validation_values = _delta_ps(development, target, validation_candidate)
        validation = validation_candidate[
            robust_z_mask(validation_values, fitted, z_threshold)
        ]
        blind_values = _delta_ps(blind, target, blind_candidates)
        blind_selected = blind_candidates[
            robust_z_mask(blind_values, fitted, z_threshold)
        ]
        for name, indices in (
            ("training", train), ("validation", validation), ("blind", blind_selected)
        ):
            if indices.size < minimum:
                raise RuntimeError(
                    f"Fold {fold_id} has only {indices.size} {name} events after robust z selection; "
                    f"minimum is {minimum}"
                )
        output.append({
            "fold_id": fold_id,
            "train": train,
            "validation": validation,
            "blind": blind_selected,
            "robust": fitted,
            "z_threshold": z_threshold,
        })
    return output


def _sample_value(spec: Any, rng: random.Random) -> Any:
    if not isinstance(spec, dict) or "type" not in spec:
        return copy.deepcopy(spec)
    kind = str(spec["type"])
    if kind == "categorical":
        return copy.deepcopy(rng.choice(list(spec["values"])))
    if kind == "integer":
        return rng.randint(int(spec["low"]), int(spec["high"]))
    if kind == "uniform":
        return rng.uniform(float(spec["low"]), float(spec["high"]))
    if kind in {"loguniform", "log_uniform"}:
        return math.exp(rng.uniform(math.log(float(spec["low"])), math.log(float(spec["high"]))))
    raise ValueError(f"Unsupported search parameter kind: {kind}")


def _parameter_sets(space: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    search = space["search"]
    method = str(search.get("method", "grid"))
    specs = dict(search.get("parameters", {}))
    if method == "grid":
        keys = list(specs)
        values = [value if isinstance(value, list) else [value] for value in specs.values()]
        return [dict(zip(keys, combination)) for combination in itertools.product(*values)] or [{}]
    if method == "random":
        rng = random.Random(seed)
        count = int(search.get("n_trials", 1))
        return [
            {key: _sample_value(spec, rng) for key, spec in specs.items()}
            for _ in range(count)
        ]
    if method == "optuna_tpe":
        # Optuna proposals are generated by the context runner because the
        # objective is the CV metric for one physical window.
        return []
    raise ValueError(f"Unsupported search method: {method}")


def _suggest_optuna(trial: Any, specs: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for key, spec in specs.items():
        if not isinstance(spec, dict) or "type" not in spec:
            output[key] = copy.deepcopy(spec)
            continue
        kind = str(spec["type"])
        if kind == "categorical":
            choices = list(spec["values"])
            encoded = [json.dumps(value, sort_keys=True) for value in choices]
            chosen = trial.suggest_categorical(key, encoded)
            output[key] = choices[encoded.index(chosen)]
        elif kind == "integer":
            output[key] = trial.suggest_int(key, int(spec["low"]), int(spec["high"]))
        elif kind == "uniform":
            output[key] = trial.suggest_float(key, float(spec["low"]), float(spec["high"]))
        elif kind in {"loguniform", "log_uniform"}:
            output[key] = trial.suggest_float(key, float(spec["low"]), float(spec["high"]), log=True)
        else:
            raise ValueError(f"Unsupported Optuna parameter kind: {kind}")
    return output


def _effective_train_config(
    space: dict[str, Any], loss: dict[str, Any], transform: str,
    mode: dict[str, str], model_name: str, output_dir: Path, seed: int,
) -> dict[str, Any]:
    config = copy.deepcopy(space["base_train_config"])
    config.setdefault("model", {})["type"] = str(space["model_type"])
    config["model"]["name"] = model_name
    mapping = space.get("study_loss_mapping", {})
    mapped_loss = mapping.get(str(loss["type"]))
    if mapped_loss is None:
        model_loss = {
            key: copy.deepcopy(value) for key, value in loss.items() if key != "id"
        }
    else:
        model_loss = copy.deepcopy(mapped_loss)
        # Preserve the common penalty parameters while allowing the model space
        # to translate only the loss name (for example mse -> rmse for SVR
        # candidate ranking, which is monotonic-equivalent).
        for key, value in loss.items():
            if key not in {"id", "type"}:
                model_loss[key] = copy.deepcopy(value)
    config["model"]["loss"] = model_loss
    config["input_transform"] = transform
    config["prediction"] = {
        "input_waveforms": mode["input_waveforms"],
        "target": mode["target"],
    }
    config["datasets"] = ["injected_by_study"]
    config.setdefault("training", {})["seed"] = int(seed)
    config["training"]["data_seed"] = int(seed)
    config["training"]["selection_metric"] = "validation_loss"
    config.setdefault("output", {})["train_dir"] = str(output_dir)
    config.setdefault("artifacts", {}).update({
        "save_config": True,
        "save_history": False,
        "save_plots": False,
        "save_last_checkpoint": False,
        "save_summary": True,
    })
    return config


def _objective_loss(
    corrected_ps: np.ndarray,
    true_ps: np.ndarray,
    loss: dict[str, Any],
    target_scale_ps: float,
) -> tuple[float, float]:
    residual = np.asarray(corrected_ps, dtype=np.float64) - np.asarray(true_ps, dtype=np.float64)
    bias = float(np.mean(residual))
    if str(loss["type"]) == "mse":
        return float(np.mean(residual * residual)), bias
    variance = float(np.mean((residual - bias) ** 2))
    normalization = str(loss.get("bias_normalization", "target_std"))
    raw_scale = float(target_scale_ps) if normalization == "target_std" else 1.0
    scale = max(raw_scale, float(loss.get("minimum_scale", 1e-8)))
    penalty = float(loss.get("bias_weight", 0.0)) * (bias / scale) ** 2
    return variance + penalty, bias


def _metrics(
    corrected_ps: np.ndarray,
    true_ps: np.ndarray,
    baseline_ps: np.ndarray,
    fit_config: dict[str, Any],
    loss: dict[str, Any],
    target_scale_ps: float,
) -> dict[str, float]:
    objective, bias = _objective_loss(corrected_ps, true_ps, loss, target_scale_ps)
    residual = (
        np.asarray(corrected_ps, dtype=np.float64)
        - np.asarray(true_ps, dtype=np.float64)
    )
    rmse = float(np.sqrt(np.mean(residual * residual)))
    fit = fit_times_ps(corrected_ps, "model", fit_config)
    baseline_fit = fit_times_ps(baseline_ps, "LED baseline", fit_config)
    ctr = float(fit.ctr_ps) if fit.success else float("nan")
    baseline_ctr = float(baseline_fit.ctr_ps) if baseline_fit.success else float("nan")
    improvement = (
        100.0 * (baseline_ctr - ctr) / baseline_ctr
        if np.isfinite(ctr) and np.isfinite(baseline_ctr) and baseline_ctr != 0.0
        else float("nan")
    )
    return {
        "loss": objective,
        "rmse_ps": rmse,
        "bias_ps": bias,
        "ctr_ps": ctr,
        "baseline_ctr_ps": baseline_ctr,
        "relative_improvement_pct": improvement,
    }


def _aggregate(values: list[dict[str, Any]], statistic: str) -> dict[str, float]:
    result = {}
    for metric in ("loss", "rmse_ps", "bias_ps", "ctr_ps", "baseline_ctr_ps", "relative_improvement_pct"):
        array = np.asarray([float(row[metric]) for row in values], dtype=np.float64)
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            result[metric] = float("nan")
        elif statistic == "mean":
            result[metric] = float(np.mean(finite))
        elif statistic == "std":
            result[metric] = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
        elif statistic == "sem":
            result[metric] = (
                float(np.std(finite, ddof=1) / math.sqrt(finite.size))
                if finite.size > 1 else 0.0
            )
        else:
            raise ValueError(statistic)
    return result


_STUDY_METRIC_ALIASES = {
    "ctr": "ctr_ps",
    "ctr_ps": "ctr_ps",
    "validation_ctr": "ctr_ps",
    "validation_ctr_ps": "ctr_ps",
    "rmse": "rmse_ps",
    "rmse_ps": "rmse_ps",
    "validation_rmse": "rmse_ps",
    "validation_rmse_ps": "rmse_ps",
    "loss": "loss",
    "objective": "loss",
    "validation_loss": "loss",
    "bias": "bias_ps",
    "bias_ps": "bias_ps",
    "validation_bias": "bias_ps",
    "validation_bias_ps": "bias_ps",
}


def _canonical_study_metric(metric: str) -> str:
    normalized = str(metric).strip().lower()
    try:
        return _STUDY_METRIC_ALIASES[normalized]
    except KeyError as exc:
        allowed = ", ".join(sorted(_STUDY_METRIC_ALIASES))
        raise ValueError(
            f"Unsupported study selection metric {metric!r}. Supported names: {allowed}"
        ) from exc


def _row_metric_value(row: dict[str, Any], metric: str) -> float:
    key = _canonical_study_metric(metric)
    value = _finite_float(row.get(key))
    if not np.isfinite(value) and key == "rmse_ps":
        # Backward compatibility for old MSE rows created before rmse_ps was
        # persisted explicitly. For MSE, sqrt(loss) is exactly RMSE. This
        # fallback is deliberately not applied to var_bias rows because their
        # objective is not mean squared error.
        if str(row.get("loss_type", "")) == "mse":
            mse = _finite_float(row.get("loss"))
            if np.isfinite(mse) and mse >= 0.0:
                value = float(math.sqrt(mse))
    return abs(value) if key == "bias_ps" and np.isfinite(value) else value


def _selection_value(rows: list[dict[str, Any]], metric: str) -> float:
    values = np.asarray(
        [_row_metric_value(row, metric) for row in rows], dtype=np.float64
    )
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("inf")


def _base_row(
    *, config: dict[str, Any], root_id: str, root_file: Path, mode_id: str,
    model_id: str, model_type: str, loss: dict[str, Any], transform: str,
    subsampling_factor: int, window: dict[str, Any], trial_id: str
) -> dict[str, Any]:
    return {
        "experiment_id": config["experiment"]["name"],
        "root_id": root_id,
        "root_file": str(root_file),
        "channel_mode": mode_id,
        "model_id": model_id,
        "model_type": model_type,
        "loss_id": loss["id"],
        "loss_type": loss["type"],
        "input_transform": transform,
        "subsampling_factor": int(subsampling_factor),
        "window_id": window["id"],
        "window_start_ns": window["start_ns"],
        "window_end_ns": window["end_ns"],
        "trial_id": trial_id,
    }



def _upsert_result_row(
    rows: list[dict[str, Any]], row: dict[str, Any]
) -> dict[str, Any]:
    """Insert a result row or replace the previous row with the same identity."""

    key = str(row.get("row_key", ""))
    if key:
        for index, existing in enumerate(rows):
            if str(existing.get("row_key", "")) == key:
                rows[index] = row
                return row
    rows.append(row)
    return row

def _run_fold(
    *, config: dict[str, Any], development: PreparedDataset, blind: PreparedDataset,
    root_id: str, mode_id: str, mode: dict[str, str], model_id: str, space: dict[str, Any],
    loss: dict[str, Any], transform: str, window: dict[str, Any], parameters: dict[str, Any],
    trial_id: str, fold: dict[str, Any], run_dir: Path, evaluate_blind: bool,
    logger: Any
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    fold_id = int(fold["fold_id"])
    seed = int(config["cross_validation"].get("seed", 20260804)) + fold_id
    view = prediction_window_dataset_view(
        development,
        input_waveforms=mode["input_waveforms"],
        target=mode["target"],
        before_ns=float(window["before_ns"]),
        after_ns=float(window["after_ns"]),
    )
    view = replace(
        view,
        train=np.asarray(fold["train"], dtype=np.int64),
        validation=np.asarray(fold["validation"], dtype=np.int64),
    )
    model_name = f"{model_id}_{mode_id}_{window['id']}_{trial_id}_f{fold_id}"
    train_config = _effective_train_config(
        space, loss, transform, mode, model_name, run_dir, seed
    )
    shared_input_cache = _shared_input_cache_root(
        config, root_id, mode_id, str(window["id"]), transform
    )
    train_config.setdefault("training", {})["input_transform_cache_dir"] = str(
        shared_input_cache
    )
    for key, value in parameters.items():
        _set_nested(train_config, key, value)
    # The study-level fit protocol is common to every model.
    if "fit" in config:
        train_config["fit"] = copy.deepcopy(config["fit"])
    started = time.time()
    summary = train_model(
        train_config,
        restart=True,
        logger=logger,
        prepared_datasets=[view],
        data_view={
            "window_id": window["id"],
            "window_before_ns": float(window["before_ns"]),
            "window_after_ns": float(window["after_ns"]),
            "channel_mode": mode_id,
        },
    )
    eval_config = {
        "device": config.get("evaluation", {}).get("device", "auto"),
        "batch_size": int(config.get("evaluation", {}).get("batch_size", 512)),
        "num_workers": int(config.get("evaluation", {}).get("num_workers", 0)),
        "pin_memory": bool(config.get("evaluation", {}).get("pin_memory", False)),
        "input_transform_cache_dir": str(shared_input_cache),
        "output": {"evaluation_dir": str(run_dir / "evaluation")},
    }
    device = resolve_device(eval_config["device"])
    train_target = _delta_ps(development, mode["target"], np.asarray(fold["train"]))
    target_scale = max(float(np.std(train_target, ddof=0)), 1e-8)
    validation_baseline = _delta_ps(
        development, mode["baseline"], np.asarray(fold["validation"])
    )
    if (
        str(space.get("model_type", "")) == "shapelet_regressor"
        and "_validation_prediction_ps" in summary
    ):
        validation_prediction = np.asarray(
            summary["_validation_prediction_ps"], dtype=np.float64
        )
        validation_target = np.asarray(
            summary["_validation_target_ps"], dtype=np.float64
        )
        true_tof = np.full(
            validation_target.size,
            float(development.true_tof_ps),
            dtype=np.float64,
        )
        corrected_ps = true_tof + validation_target - validation_prediction
        prediction_true_tof = true_tof
        trained = load_trained_model(run_dir) if evaluate_blind else None
    else:
        trained = load_trained_model(run_dir)
        validation_source = replace(
            development, evaluation=np.asarray(fold["validation"], dtype=np.int64)
        )
        prediction = evaluate_trained_model(trained, validation_source, eval_config, device)
        corrected_ps = prediction.corrected_ps
        prediction_true_tof = prediction.true_tof_ps
    validation_metrics = _metrics(
        corrected_ps,
        prediction_true_tof,
        validation_baseline,
        train_config["fit"],
        loss,
        target_scale,
    )
    robust: RobustLocationScale = fold["robust"]
    cv_row = {
        "record_type": "cv_fold",
        "fold_id": fold_id,
        "split": "validation",
        "statistic": "raw",
        "status": "completed",
        "n_events": int(fold["validation"].size),
        **validation_metrics,
        "outlier_center_ps": robust.center_ps,
        "outlier_scale_ps": robust.scale_ps,
        "outlier_scale_method": robust.method,
        "outlier_z_threshold": fold["z_threshold"],
        "runtime_seconds": float(time.time() - started),
    }
    cv_row["row_key"] = canonical_hash({"type": "cv_fold", "run": str(run_dir)})[:24]

    blind_row = None
    if evaluate_blind:
        if trained is None:
            trained = load_trained_model(run_dir)
        blind_source = replace(blind, evaluation=np.asarray(fold["blind"], dtype=np.int64))
        blind_prediction = evaluate_trained_model(trained, blind_source, eval_config, device)
        blind_baseline = _delta_ps(blind, mode["baseline"], np.asarray(fold["blind"]))
        blind_metrics = _metrics(
            blind_prediction.corrected_ps,
            blind_prediction.true_tof_ps,
            blind_baseline,
            train_config["fit"],
            loss,
            target_scale,
        )
        blind_row = {
            "record_type": "blind_fold",
            "fold_id": fold_id,
            "split": "blind",
            "statistic": "raw",
            "status": "completed",
            "n_events": int(fold["blind"].size),
            **blind_metrics,
            "outlier_center_ps": robust.center_ps,
            "outlier_scale_ps": robust.scale_ps,
            "outlier_scale_method": robust.method,
            "outlier_z_threshold": fold["z_threshold"],
            "runtime_seconds": 0.0,
        }
        blind_row["row_key"] = canonical_hash({"type": "blind_fold", "run": str(run_dir)})[:24]
    return cv_row, blind_row


def _append_summary_rows(
    rows: list[dict[str, Any]], fold_rows: list[dict[str, Any]], base: dict[str, Any],
    split: str, selected_hyperparameters: bool
) -> list[dict[str, Any]]:
    appended = []
    for statistic in ("mean", "std", "sem"):
        metric_values = _aggregate(fold_rows, statistic)
        row = {
            **base,
            "record_type": "summary",
            "fold_id": "",
            "split": split,
            "statistic": statistic,
            "is_selected_hyperparameters": int(selected_hyperparameters),
            "status": "completed",
            "n_events": int(round(np.mean([int(item["n_events"]) for item in fold_rows]))),
            **metric_values,
        }
        row["row_key"] = canonical_hash({
            "type": "summary", "base": base, "split": split, "statistic": statistic
        })[:24]
        existing = next((item for item in rows if item.get("row_key") == row["row_key"]), None)
        if existing is None:
            rows.append(row)
            appended.append(row)
        else:
            existing.update(row)
            appended.append(existing)
    return appended


def _plot_results(rows: list[dict[str, Any]], output: Path, logger: Any) -> None:
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        logger.warning("Plotting skipped because matplotlib/pandas are unavailable")
        return
    # ``rows`` already uses the decoded internal result schema.  The compact
    # numeric CSV refactor removed the old ``_normalize_row`` helper, but the
    # plotting path still referenced it.  Build the frame directly from the
    # internal fields so finalization works both after a fresh run and after
    # ``--resume`` decodes all_results.csv through results_metadata.json.
    frame = pd.DataFrame(
        [
            {field: _string_value(row.get(field, "")) for field in INTERNAL_RESULT_FIELDS}
            for row in rows
        ],
        columns=INTERNAL_RESULT_FIELDS,
    )
    if frame.empty:
        return
    numeric = [
        "window_start_ns", "window_end_ns", "ctr_ps", "relative_improvement_pct"
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    selected = frame[
        (frame["record_type"] == "summary")
        & (frame["statistic"].isin(["mean", "sem"]))
        & (frame["is_selected_hyperparameters"].astype(str) == "1")
    ]
    plot_root = output / "plots"
    for keys, group in selected.groupby(
        ["root_id", "channel_mode", "model_id", "loss_id", "input_transform"],
        dropna=False,
    ):
        root_id, mode, model, loss_id, transform = map(str, keys)
        mean = group[group["statistic"] == "mean"].copy()
        sem = group[group["statistic"] == "sem"].copy()
        if mean.empty:
            continue
        for current_frame in (mean, sem):
            current_frame["window_label"] = current_frame.apply(
                lambda row: f"[{row.window_start_ns:g},{row.window_end_ns:g}]", axis=1
            )
        labels = list(dict.fromkeys(mean["window_label"].tolist()))
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(max(7.0, 0.9 * len(labels)), 4.8))
        for split, marker in (("validation", "o"), ("blind", "s")):
            current = mean[mean["split"] == split].set_index("window_label")
            error = sem[sem["split"] == split].set_index("window_label")
            y = np.asarray([current.loc[label, "ctr_ps"] if label in current.index else np.nan for label in labels], dtype=float)
            e = np.asarray([error.loc[label, "ctr_ps"] if label in error.index else np.nan for label in labels], dtype=float)
            ax.errorbar(x, y, yerr=e, marker=marker, label=split)
        chosen = mean[
            (mean["split"] == "validation")
            & (mean["is_selected_window"].astype(str) == "1")
        ]
        if not chosen.empty:
            label = chosen.iloc[0]["window_label"]
            ax.axvline(labels.index(label), linestyle="--", linewidth=1.0, label="CV-selected window")
        ax.set_xticks(x, labels, rotation=35, ha="right")
        ax.set_ylabel("CTR (ps)")
        ax.set_xlabel("LED-relative window [start,end] ns")
        ax.set_title(f"{root_id} | {mode} | {model} | {loss_id} | {transform}")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        destination = plot_root / mode
        destination.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination / f"{root_id}_{model}_{loss_id}_{transform}_windows.png", dpi=180)
        plt.close(fig)

        validation_points = mean[mean["split"] == "validation"].set_index("window_label")
        blind_points = mean[mean["split"] == "blind"].set_index("window_label")
        common_labels = [label for label in labels if label in validation_points.index and label in blind_points.index]
        if common_labels:
            cv_values = np.asarray([validation_points.loc[label, "ctr_ps"] for label in common_labels], dtype=float)
            blind_values = np.asarray([blind_points.loc[label, "ctr_ps"] for label in common_labels], dtype=float)
            finite = np.isfinite(cv_values) & np.isfinite(blind_values)
            cv_values = cv_values[finite]
            blind_values = blind_values[finite]
            point_labels = [label for label, keep in zip(common_labels, finite) if keep]
            if cv_values.size:
                fig, ax = plt.subplots(figsize=(5.4, 5.0))
                ax.scatter(cv_values, blind_values)
                lower = float(min(np.min(cv_values), np.min(blind_values)))
                upper = float(max(np.max(cv_values), np.max(blind_values)))
                padding = max((upper - lower) * 0.08, 1.0)
                ax.plot([lower - padding, upper + padding], [lower - padding, upper + padding], linestyle="--", linewidth=1.0)
                for x_value, y_value, label in zip(cv_values, blind_values, point_labels):
                    ax.annotate(label, (x_value, y_value), xytext=(4, 4), textcoords="offset points", fontsize=8)
                pearson = (
                    float(np.corrcoef(cv_values, blind_values)[0, 1])
                    if cv_values.size >= 2 and np.std(cv_values) > 0 and np.std(blind_values) > 0
                    else float("nan")
                )
                cv_rank = np.argsort(np.argsort(cv_values))
                blind_rank = np.argsort(np.argsort(blind_values))
                spearman = (
                    float(np.corrcoef(cv_rank, blind_rank)[0, 1])
                    if cv_values.size >= 2 and np.std(cv_rank) > 0 and np.std(blind_rank) > 0
                    else float("nan")
                )
                ax.set_xlabel("Mean cross-validation CTR (ps)")
                ax.set_ylabel("Mean blind CTR across fold models (ps)")
                ax.set_title(f"Validation quality | Pearson={pearson:.3f} | Spearman={spearman:.3f}")
                ax.grid(True, alpha=0.25)
                fig.tight_layout()
                fig.savefig(destination / f"{root_id}_{model}_{loss_id}_{transform}_cv_vs_blind.png", dpi=180)
                plt.close(fig)

    best = selected[
        (selected["statistic"] == "mean")
        & (selected["split"] == "blind")
        & (selected["is_selected_window"].astype(str) == "1")
    ].copy()
    for keys, group in best.groupby(["root_id", "channel_mode", "loss_id", "input_transform"]):
        root_id, mode, loss_id, transform = map(str, keys)
        group = group.sort_values("ctr_ps")
        fig, ax = plt.subplots(figsize=(max(6.0, 0.9 * len(group)), 4.6))
        x = np.arange(len(group))
        ax.bar(x, group["relative_improvement_pct"].astype(float))
        ax.set_xticks(x, group["model_id"].astype(str), rotation=25, ha="right")
        ax.axhline(0.0, linewidth=1.0)
        ax.set_ylabel("CTR improvement over target LED (%)")
        ax.set_title(f"CV-selected windows | {root_id} | {mode} | {loss_id} | {transform}")
        fig.tight_layout()
        destination = plot_root / mode
        destination.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination / f"{root_id}_{loss_id}_{transform}_best_models.png", dpi=180)
        plt.close(fig)


def run_study(
    config: dict[str, Any], *, dry_run: bool, resume: bool, restart: bool,
    rebuild_preprocessing: bool, logger: Any
) -> dict[str, Any]:
    output = Path(config["experiment"]["output_dir"])
    if restart and output.exists():
        # run_study can also be invoked directly. Close every handler before
        # deleting the directory so Windows releases study.log, then recreate
        # the configured logger for the new study directory.
        for handler in list(getattr(logger, "handlers", [])):
            try:
                handler.flush()
            finally:
                handler.close()
                logger.removeHandler(handler)
        shutil.rmtree(output)
        logger = restrict_to_study_progress(
            setup_logging(
                output / "study.log", config.get("logging", {}).get("level", "INFO")
            )
        )
    output.mkdir(parents=True, exist_ok=True)
    if resume:
        _migrate_legacy_results(output)
        _assert_resume_data_compatibility(config, output)
    results_path = _state_results_path(output)
    rows: list[dict[str, Any]] = [dict(row) for row in (_read_results(results_path) if resume else [])]
    completed = {str(row.get("row_key", "")) for row in rows if row.get("status") == "completed"}
    atomic_json(output / "resolved_study_config.json", config)

    planned_contexts = (
        len(config["root_files"])
        * len(config["channel_modes"])
        * len(config["models"])
        * len(config["losses"])
        * len(config["input_transforms"])
        * len(config["windows_ns"])
        * len(config.get("preprocessing", {}).get("subsampling_factors", [1]))
    )
    progress_total_blocks = (
        len(config["root_files"])
        * len(config["channel_modes"])
        * len(config["input_transforms"])
        * len(config["windows_ns"])
        * sum(
            1
            for model_id in config["models"]
            for loss in config["losses"]
            if loss["type"] in config["model_spaces"][model_id]["supported_losses"]
        )
    )
    study_started = time.perf_counter()
    completed_blocks = 0
    _progress(
        logger,
        "Study start | %s | files=%d | model-window blocks=%d | selection=CV only",
        config["experiment"]["name"],
        len(config["root_files"]),
        progress_total_blocks,
    )
    if dry_run:
        return {"root_files": len(config["root_files"]), "planned_contexts": planned_contexts}

    cv_metric = str(config["selection"].get("hyperparameter_metric", "ctr_ps"))
    window_metric = str(config["selection"].get("window_metric", "ctr_ps"))
    run_root = output / "runs"
    keep_checkpoints = _keep_study_checkpoints(config)

    for root_index, root_text in enumerate(config["root_files"], start=1):
        root_file = Path(root_text)
        root_id = _root_id(root_file)
        completed_file_marker = output / "completed_files" / f"{root_id}.json"
        report_marker = output / "_state" / "reported_files" / f"{root_id}.json"
        report_only = False
        requested_blocks_complete = _root_has_complete_requested_blocks(
            config, rows, root_id
        )
        if resume and completed_file_marker.is_file() and requested_blocks_complete:
            blocks_per_file = (
                progress_total_blocks // max(len(config["root_files"]), 1)
            )
            completed_blocks += blocks_per_file
            if (
                _report_marker_is_current(report_marker)
                and _shapelet_artifact_is_current(config, output, root_id)
            ):
                _progress(
                    logger,
                    "File %d/%d | already completed and summarized | %s",
                    root_index,
                    len(config["root_files"]),
                    root_file.name,
                )
                continue
            # A stale report marker means the compact schema or durable
            # model-specific artifacts changed. Rebuild the lightweight report
            # path from the canonical prepared data instead of only touching
            # secondary tables; this also backfills newly added summary columns.
            report_only = True
            _progress(
                logger,
                "File %d/%d | rebuilding data for compact summary only | %s",
                root_index,
                len(config["root_files"]),
                root_file.name,
            )
        elif resume and completed_file_marker.is_file():
            _progress(
                logger,
                "File %d/%d | completed marker predates current experiment grid; "
                "running missing blocks only | %s",
                root_index,
                len(config["root_files"]),
                root_file.name,
            )
        preprocessing_started = time.perf_counter()
        _progress(
            logger,
            "File %d/%d | preprocessing | %s",
            root_index,
            len(config["root_files"]),
            root_file.name,
        )
        development, blind = _ensure_preprocessed(
            config, root_file, root_id, output,
            rebuild=rebuild_preprocessing, logger=logger,
        )
        _progress(
            logger,
            "File %d/%d | preprocessing ready | development=%d blind=%d | %s",
            root_index,
            len(config["root_files"]),
            int(development.event_id.shape[0]),
            int(blind.event_id.shape[0]),
            _format_duration(time.perf_counter() - preprocessing_started),
        )
        for mode_id in ([] if report_only else config["channel_modes"]):
            mode = CHANNEL_MODES[mode_id]
            folds = _fold_masks(
                development, blind, mode["target"],
                config["cross_validation"], config["selection"],
            )
            split_manifest = {
                "root_file": str(root_file),
                "channel_mode": mode_id,
                "target": mode["target"],
                "z_threshold": float(config["selection"].get("z_threshold", 4.0)),
                "folds": [
                    {
                        "fold_id": int(fold["fold_id"]),
                        "train_count": int(fold["train"].size),
                        "validation_count": int(fold["validation"].size),
                        "blind_count": int(fold["blind"].size),
                        "center_ps": float(fold["robust"].center_ps),
                        "scale_ps": float(fold["robust"].scale_ps),
                        "scale_method": fold["robust"].method,
                    }
                    for fold in folds
                ],
            }
            atomic_json(output / "folds" / root_id / f"{mode_id}.json", split_manifest)

            _evaluate_standard_methods(
                config=config, rows=rows, development=development, blind=blind,
                root_id=root_id, root_file=root_file, mode_id=mode_id, mode=mode,
                folds=folds, logger=logger,
            )
            _write_results(results_path, rows)

            for model_id in config["models"]:
                space = config["model_spaces"][model_id]
                for loss in config["losses"]:
                    if loss["type"] not in space["supported_losses"]:
                        logger.warning("Skipping unsupported loss %s for model %s", loss["type"], model_id)
                        continue
                    for transform in config["input_transforms"]:
                        selected_by_window: dict[str, dict[str, Any]] = {}
                        for window in config["windows_ns"]:
                            search = space["search"]
                            method = str(search.get("method", "grid"))
                            parameter_sets = _parameter_sets(
                                space,
                                int(config["cross_validation"].get("seed", 20260804)),
                            )
                            subsampling_factors = list(
                                config.get("preprocessing", {}).get(
                                    "subsampling_factors", [1]
                                )
                            )
                            study = None
                            if method == "optuna_tpe":
                                try:
                                    import optuna
                                    optuna.logging.set_verbosity(optuna.logging.WARNING)
                                except ImportError as exc:
                                    raise RuntimeError("Optuna is required by this model space") from exc
                                storage = output / "search_state" / root_id / mode_id / model_id / loss["id"] / transform
                                storage.mkdir(parents=True, exist_ok=True)
                                study = optuna.create_study(
                                    direction="minimize",
                                    study_name=f"{root_id}_{mode_id}_{model_id}_{loss['id']}_{transform}_{window['id']}",
                                    storage=f"sqlite:///{storage / (window['id'] + '.db')}",
                                    load_if_exists=True,
                                    sampler=optuna.samplers.TPESampler(
                                        seed=int(config["cross_validation"].get("seed", 20260804))
                                    ),
                                )
                                parameter_sets = []
                                optuna_specs = dict(search.get("parameters", {}))
                                optuna_specs["preprocessing.subsampling_factor"] = {
                                    "type": "categorical",
                                    "values": subsampling_factors,
                                }
                                for _ in range(int(search.get("n_trials", 10))):
                                    trial = study.ask()
                                    parameter_sets.append(
                                        (trial, _suggest_optuna(trial, optuna_specs))
                                    )
                            else:
                                parameter_sets = [
                                    (
                                        None,
                                        {
                                            **parameters,
                                            "preprocessing.subsampling_factor": int(factor),
                                        },
                                    )
                                    for parameters in parameter_sets
                                    for factor in subsampling_factors
                                ]

                            block_started = time.perf_counter()
                            total_fold_runs = len(parameter_sets) * len(folds)
                            completed_fold_runs = 0
                            observed_fold_seconds = 0.0
                            _progress(
                                logger,
                                "Block %d/%d | file=%s | mode=%s | model=%s | loss=%s | transform=%s | "
                                "window=[%g,%g] ns | trials=%d folds=%d",
                                completed_blocks + 1,
                                progress_total_blocks,
                                root_file.name,
                                mode_id,
                                model_id,
                                loss["id"],
                                transform,
                                float(window["before_ns"]),
                                float(window["after_ns"]),
                                len(parameter_sets),
                                len(folds),
                            )

                            trial_summaries: list[tuple[str, float, list[dict[str, Any]], dict[str, Any]]] = []
                            retained_trial_id: str | None = None
                            retained_trial_objective = float("inf")
                            for trial_index, (optuna_trial, parameters) in enumerate(parameter_sets, start=1):
                                trial_id = f"t{trial_index:04d}_{canonical_hash(parameters)[:8]}"
                                _progress(
                                    logger,
                                    "  Trial %d/%d | %s",
                                    trial_index,
                                    len(parameter_sets),
                                    _compact_parameters(parameters),
                                )
                                base = _base_row(
                                    config=config, root_id=root_id, root_file=root_file,
                                    mode_id=mode_id, model_id=model_id,
                                    model_type=space["model_type"], loss=loss,
                                    transform=transform,
                                    subsampling_factor=int(parameters.get("preprocessing.subsampling_factor", 1)),
                                    window=window, trial_id=trial_id,
                                )
                                definition_key = canonical_hash({"type": "trial_definition", **base})[:24]
                                if definition_key not in completed:
                                    definition = {
                                        **base,
                                        "row_key": definition_key,
                                        "record_type": "trial_definition",
                                        "status": "completed",
                                        "params_json": json.dumps(parameters, sort_keys=True, separators=(",", ":")),
                                    }
                                    _upsert_result_row(rows, definition)
                                    completed.add(definition_key)

                                cv_rows: list[dict[str, Any]] = []
                                for fold in folds:
                                    run_dir = (
                                        run_root / root_id / mode_id / model_id / str(loss["id"])
                                        / transform / str(window["id"]) / trial_id / f"fold_{fold['fold_id']}"
                                    )
                                    expected_key = canonical_hash({"type": "cv_fold", "run": str(run_dir)})[:24]
                                    existing = next((row for row in rows if row.get("row_key") == expected_key), None)
                                    if (
                                        existing is not None
                                        and existing.get("status") == "completed"
                                        and np.isfinite(_selection_value([existing], cv_metric))
                                    ):
                                        cv_rows.append(existing)
                                        completed_fold_runs += 1
                                        observed_fold_seconds += float(existing.get("runtime_seconds") or 0.0)
                                        remaining = total_fold_runs - completed_fold_runs
                                        mean_seconds = (
                                            observed_fold_seconds / completed_fold_runs
                                            if completed_fold_runs
                                            else float("nan")
                                        )
                                        _progress(
                                            logger,
                                            "    Fold %d/%d | cached | val CTR=%s | bias=%s | block %s",
                                            int(fold["fold_id"]) + 1,
                                            len(folds),
                                            _metric_text(existing.get("ctr_ps"), " ps"),
                                            _metric_text(existing.get("bias_ps"), " ps"),
                                            _format_eta(mean_seconds * remaining),
                                        )
                                        continue
                                    _progress(
                                        logger,
                                        "    Fold %d/%d | training",
                                        int(fold["fold_id"]) + 1,
                                        len(folds),
                                    )
                                    try:
                                        cv_row, _ = _run_fold(
                                            config=config, development=development, blind=blind,
                                            root_id=root_id, mode_id=mode_id, mode=mode, model_id=model_id,
                                            space=space, loss=loss, transform=transform,
                                            window=window, parameters=parameters,
                                            trial_id=trial_id, fold=fold, run_dir=run_dir,
                                            evaluate_blind=False, logger=logger,
                                        )
                                        cv_row.update(base)
                                    except Exception as exc:
                                        logger.exception(
                                            "Study fold failed | root=%s mode=%s model=%s window=%s trial=%s fold=%s",
                                            root_id, mode_id, model_id, window["id"], trial_id, fold["fold_id"],
                                        )
                                        cv_row = {
                                            **base,
                                            "row_key": expected_key,
                                            "record_type": "cv_fold",
                                            "fold_id": fold["fold_id"],
                                            "split": "validation",
                                            "statistic": "raw",
                                            "status": "failed",
                                            "error": f"{type(exc).__name__}: {exc}",
                                        }
                                    _upsert_result_row(rows, cv_row)
                                    _write_results(results_path, rows)
                                    completed_fold_runs += 1
                                    fold_runtime = float(cv_row.get("runtime_seconds") or 0.0)
                                    observed_fold_seconds += fold_runtime
                                    remaining = total_fold_runs - completed_fold_runs
                                    mean_seconds = (
                                        observed_fold_seconds / completed_fold_runs
                                        if completed_fold_runs
                                        else float("nan")
                                    )
                                    if cv_row["status"] == "completed":
                                        cv_rows.append(cv_row)
                                        _progress(
                                            logger,
                                            "    Fold %d/%d | val CTR=%s | bias=%s | loss=%s | %s | block %s",
                                            int(fold["fold_id"]) + 1,
                                            len(folds),
                                            _metric_text(cv_row.get("ctr_ps"), " ps"),
                                            _metric_text(cv_row.get("bias_ps"), " ps"),
                                            _metric_text(cv_row.get("loss")),
                                            _format_duration(fold_runtime),
                                            _format_eta(mean_seconds * remaining),
                                        )
                                if len(cv_rows) != len(folds):
                                    objective = float("inf")
                                else:
                                    objective = _selection_value(cv_rows, cv_metric)
                                    _append_summary_rows(rows, cv_rows, base, "validation", False)
                                if study is not None and optuna_trial is not None:
                                    study.tell(optuna_trial, objective)
                                trial_summaries.append((trial_id, objective, cv_rows, parameters))
                                _write_results(results_path, rows)
                                if not keep_checkpoints:
                                    current_trial_directory = _trial_run_directory(
                                        run_root, root_id, mode_id, model_id,
                                        str(loss["id"]), transform,
                                        str(window["id"]), trial_id,
                                    )
                                    if (
                                        np.isfinite(objective)
                                        and objective < retained_trial_objective
                                    ):
                                        if (
                                            retained_trial_id is not None
                                            and retained_trial_id != trial_id
                                        ):
                                            _remove_tree(
                                                _trial_run_directory(
                                                    run_root, root_id, mode_id, model_id,
                                                    str(loss["id"]), transform,
                                                    str(window["id"]), retained_trial_id,
                                                ),
                                                logger,
                                                "superseded trial artifacts",
                                            )
                                        retained_trial_id = trial_id
                                        retained_trial_objective = objective
                                    else:
                                        _remove_tree(
                                            current_trial_directory,
                                            logger,
                                            "non-selected trial artifacts",
                                        )
                                _progress(
                                    logger,
                                    "  Trial %d/%d complete | mean CV %s=%s",
                                    trial_index,
                                    len(parameter_sets),
                                    cv_metric,
                                    _metric_text(objective, " ps" if "ctr" in cv_metric else ""),
                                )

                            complete_trials = [item for item in trial_summaries if np.isfinite(item[1])]
                            if not complete_trials:
                                completed_blocks += 1
                                overall_elapsed = time.perf_counter() - study_started
                                overall_eta = (
                                    overall_elapsed / completed_blocks
                                    * (progress_total_blocks - completed_blocks)
                                    if completed_blocks and progress_total_blocks > completed_blocks
                                    else 0.0
                                )
                                _progress(
                                    logger,
                                    "Block %d/%d failed | no complete CV trial | elapsed=%s | overall %s",
                                    completed_blocks,
                                    progress_total_blocks,
                                    _format_duration(time.perf_counter() - block_started),
                                    _format_eta(overall_eta),
                                )
                                continue
                            best_trial_id, _best_value, best_cv_rows, best_parameters = min(
                                complete_trials, key=lambda item: item[1]
                            )
                            selected_by_window[str(window["id"])] = {
                                "window": window,
                                "trial_id": best_trial_id,
                                "cv_rows": best_cv_rows,
                                "parameters": best_parameters,
                                "cv_value": _selection_value(best_cv_rows, window_metric),
                            }

                            # Linear-regression checkpoints are temporary, but their
                            # selected fold coefficients are useful scientific results.
                            # Preserve only the best hyperparameter trial for each
                            # window in one compact study-level CSV before pruning.
                            if str(space["model_type"]) == "linear_regression":
                                selected_base = _base_row(
                                    config=config,
                                    root_id=root_id,
                                    root_file=root_file,
                                    mode_id=mode_id,
                                    model_id=model_id,
                                    model_type=space["model_type"],
                                    loss=loss,
                                    transform=transform,
                                    subsampling_factor=int(best_parameters.get("preprocessing.subsampling_factor", 1)),
                                    window=window,
                                    trial_id=best_trial_id,
                                )
                                by_fold = {
                                    int(row["fold_id"]): row for row in best_cv_rows
                                }
                                for fold in folds:
                                    fold_id = int(fold["fold_id"])
                                    selected_run_dir = (
                                        run_root / root_id / mode_id / model_id
                                        / str(loss["id"]) / transform
                                        / str(window["id"]) / best_trial_id
                                        / f"fold_{fold_id}"
                                    )
                                    if not (
                                        selected_run_dir / "linear_regression_weight.npy"
                                    ).is_file():
                                        # Results-only resume: rebuild only the selected
                                        # fold long enough to export its coefficients.
                                        rebuilt_cv_row, _ = _run_fold(
                                            config=config,
                                            development=development,
                                            blind=blind,
                                            root_id=root_id,
                                            mode_id=mode_id,
                                            mode=mode,
                                            model_id=model_id,
                                            space=space,
                                            loss=loss,
                                            transform=transform,
                                            window=window,
                                            parameters=best_parameters,
                                            trial_id=best_trial_id,
                                            fold=fold,
                                            run_dir=selected_run_dir,
                                            evaluate_blind=False,
                                            logger=logger,
                                        )
                                        rebuilt_cv_row.update(selected_base)
                                        _upsert_result_row(rows, rebuilt_cv_row)
                                        by_fold[fold_id] = rebuilt_cv_row
                                best_cv_rows = [by_fold[index] for index in sorted(by_fold)]
                                selected_by_window[str(window["id"])]["cv_rows"] = best_cv_rows
                                selected_by_window[str(window["id"])]["cv_value"] = (
                                    _selection_value(best_cv_rows, window_metric)
                                )
                                if not _persist_selected_linear_weights(
                                    output=output,
                                    config=config,
                                    development=development,
                                    root_id=root_id,
                                    root_file=root_file,
                                    mode_id=mode_id,
                                    mode=mode,
                                    model_id=model_id,
                                    model_type=space["model_type"],
                                    loss=loss,
                                    transform=transform,
                                    subsampling_factor=int(best_parameters.get("preprocessing.subsampling_factor", 1)),
                                    window=window,
                                    trial_id=best_trial_id,
                                    folds=folds,
                                    run_root=run_root,
                                    cv_rows=best_cv_rows,
                                ):
                                    raise RuntimeError(
                                        "Unable to persist selected linear-regression "
                                        "coefficients before checkpoint cleanup"
                                    )
                                _write_results(results_path, rows)

                            if not keep_checkpoints:
                                _prune_window_trials(
                                    run_root=run_root,
                                    root_id=root_id,
                                    mode_id=mode_id,
                                    model_id=model_id,
                                    loss_id=str(loss["id"]),
                                    transform=transform,
                                    window_id=str(window["id"]),
                                    keep_trial_id=best_trial_id,
                                    logger=logger,
                                )
                            # Mark and summarize the hyperparameter choice for this window.
                            for row in rows:
                                if (
                                    row.get("root_id") == root_id
                                    and row.get("channel_mode") == mode_id
                                    and row.get("model_id") == model_id
                                    and row.get("loss_id") == loss["id"]
                                    and row.get("input_transform") == transform
                                    and row.get("window_id") == window["id"]
                                    and row.get("trial_id") == best_trial_id
                                ):
                                    row["is_selected_hyperparameters"] = 1
                            base = _base_row(
                                config=config, root_id=root_id, root_file=root_file,
                                mode_id=mode_id, model_id=model_id,
                                model_type=space["model_type"], loss=loss,
                                transform=transform,
                                subsampling_factor=int(best_parameters.get("preprocessing.subsampling_factor", 1)),
                                window=window, trial_id=best_trial_id,
                            )
                            blind_rows = []
                            for fold in folds:
                                run_dir = (
                                    run_root / root_id / mode_id / model_id / str(loss["id"])
                                    / transform / str(window["id"]) / best_trial_id / f"fold_{fold['fold_id']}"
                                )
                                expected_key = canonical_hash({"type": "blind_fold", "run": str(run_dir)})[:24]
                                existing = next((row for row in rows if row.get("row_key") == expected_key), None)
                                if existing is not None and existing.get("status") == "completed":
                                    blind_rows.append(existing)
                                    continue
                                try:
                                    if not _has_best_checkpoint(run_dir):
                                        # A results-only resume may have numeric CV rows but no
                                        # persisted model. Retrain only the selected fold needed
                                        # to finish its blind audit, then prune it later.
                                        retrained_cv_row, retrained_blind_row = _run_fold(
                                            config=config,
                                            development=development,
                                            blind=blind,
                                            root_id=root_id,
                                            mode_id=mode_id,
                                            mode=mode,
                                            model_id=model_id,
                                            space=space,
                                            loss=loss,
                                            transform=transform,
                                            window=window,
                                            parameters=best_parameters,
                                            trial_id=best_trial_id,
                                            fold=fold,
                                            run_dir=run_dir,
                                            evaluate_blind=True,
                                            logger=logger,
                                        )
                                        retrained_cv_row.update(base)
                                        _upsert_result_row(rows, retrained_cv_row)
                                        if retrained_blind_row is None:
                                            raise RuntimeError(
                                                "Selected fold retraining did not return blind metrics"
                                            )
                                        retrained_blind_row.update(base)
                                        retrained_blind_row["is_selected_hyperparameters"] = 1
                                        blind_row = retrained_blind_row
                                    else:
                                        # Reuse the temporary CV winner; blind data never
                                        # participates in selecting the trial.
                                        trained = load_trained_model(run_dir)
                                        eval_config = {
                                            "device": config.get("evaluation", {}).get("device", "auto"),
                                            "batch_size": int(config.get("evaluation", {}).get("batch_size", 512)),
                                            "num_workers": int(config.get("evaluation", {}).get("num_workers", 0)),
                                            "pin_memory": bool(config.get("evaluation", {}).get("pin_memory", False)),
                                            "input_transform_cache_dir": str(
                                                _shared_input_cache_root(
                                                    config, root_id, mode_id,
                                                    str(window["id"]), transform,
                                                )
                                            ),
                                            "output": {"evaluation_dir": str(run_dir / "evaluation")},
                                        }
                                        blind_source = replace(
                                            blind,
                                            evaluation=np.asarray(
                                                fold["blind"], dtype=np.int64
                                            ),
                                        )
                                        prediction = evaluate_trained_model(
                                            trained, blind_source, eval_config,
                                            resolve_device(eval_config["device"]),
                                        )
                                        train_target = _delta_ps(
                                            development, mode["target"],
                                            np.asarray(fold["train"]),
                                        )
                                        target_scale = max(
                                            float(np.std(train_target, ddof=0)), 1e-8
                                        )
                                        baseline = _delta_ps(
                                            blind, mode["baseline"],
                                            np.asarray(fold["blind"]),
                                        )
                                        metrics = _metrics(
                                            prediction.corrected_ps,
                                            prediction.true_tof_ps,
                                            baseline,
                                            config.get(
                                                "fit", space["base_train_config"]["fit"]
                                            ),
                                            loss,
                                            target_scale,
                                        )
                                        robust = fold["robust"]
                                        blind_row = {
                                            **base,
                                            "row_key": expected_key,
                                            "record_type": "blind_fold",
                                            "fold_id": fold["fold_id"],
                                            "split": "blind",
                                            "statistic": "raw",
                                            "is_selected_hyperparameters": 1,
                                            "status": "completed",
                                            "n_events": int(fold["blind"].size),
                                            **metrics,
                                            "outlier_center_ps": robust.center_ps,
                                            "outlier_scale_ps": robust.scale_ps,
                                            "outlier_scale_method": robust.method,
                                            "outlier_z_threshold": fold["z_threshold"],
                                        }
                                except Exception as exc:
                                    logger.exception("Blind audit failed for %s", run_dir)
                                    blind_row = {
                                        **base,
                                        "row_key": expected_key,
                                        "record_type": "blind_fold",
                                        "fold_id": fold["fold_id"],
                                        "split": "blind",
                                        "statistic": "raw",
                                        "is_selected_hyperparameters": 1,
                                        "status": "failed",
                                        "error": f"{type(exc).__name__}: {exc}",
                                    }
                                _upsert_result_row(rows, blind_row)
                                if blind_row["status"] == "completed":
                                    blind_rows.append(blind_row)
                                _write_results(results_path, rows)
                            if len(blind_rows) == len(folds):
                                _append_summary_rows(rows, blind_rows, base, "blind", True)
                            _write_results(results_path, rows)

                            completed_blocks += 1
                            block_elapsed = time.perf_counter() - block_started
                            overall_elapsed = time.perf_counter() - study_started
                            overall_eta = (
                                overall_elapsed / completed_blocks
                                * (progress_total_blocks - completed_blocks)
                                if completed_blocks and progress_total_blocks > completed_blocks
                                else 0.0
                            )
                            cv_value = float(selected_by_window[str(window["id"])]["cv_value"])
                            blind_value = (
                                _selection_value(blind_rows, "ctr_ps")
                                if blind_rows
                                else float("nan")
                            )
                            percent = (
                                100.0 * completed_blocks / progress_total_blocks
                                if progress_total_blocks
                                else 100.0
                            )
                            _progress(
                                logger,
                                "Block %d/%d complete (%.1f%%) | best trial=%s | CV CTR=%s | blind CTR=%s | "
                                "elapsed=%s | overall %s",
                                completed_blocks,
                                progress_total_blocks,
                                percent,
                                best_trial_id,
                                _metric_text(cv_value, " ps"),
                                _metric_text(blind_value, " ps"),
                                _format_duration(block_elapsed),
                                _format_eta(overall_eta),
                            )

                        if selected_by_window:
                            best_window_id, best_info = min(
                                selected_by_window.items(), key=lambda item: float(item[1]["cv_value"])
                            )
                            for row in rows:
                                if (
                                    row.get("root_id") == root_id
                                    and row.get("channel_mode") == mode_id
                                    and row.get("model_id") == model_id
                                    and row.get("loss_id") == loss["id"]
                                    and row.get("input_transform") == transform
                                    and row.get("window_id") == best_window_id
                                    and str(row.get("is_selected_hyperparameters", "")) == "1"
                                ):
                                    row["is_selected_window"] = 1
                            atomic_json(
                                output / "selected" / root_id / mode_id / model_id
                                / str(loss["id"]) / f"{transform}.json",
                                {
                                    "selection_source": "cross_validation_only",
                                    "window_metric": window_metric,
                                    "selected_window_id": best_window_id,
                                    "selected_window": best_info["window"],
                                    "selected_trial_id": best_info["trial_id"],
                                    "blind_used_for_selection": False,
                                },
                            )
                            if str(space["model_type"]) == "linear_regression":
                                _mark_selected_linear_weight_window(
                                    output=output,
                                    root_id=root_id,
                                    mode_id=mode_id,
                                    model_id=model_id,
                                    loss_id=str(loss["id"]),
                                    transform=transform,
                                    selected_window_id=str(best_window_id),
                                )

                            validation_means = []
                            blind_means = []
                            labels = []
                            for candidate_window_id, info in selected_by_window.items():
                                cv_value = _selection_value(info["cv_rows"], "ctr_ps")
                                blind_raw = [
                                    row for row in rows
                                    if row.get("record_type") == "blind_fold"
                                    and row.get("root_id") == root_id
                                    and row.get("channel_mode") == mode_id
                                    and row.get("model_id") == model_id
                                    and row.get("loss_id") == loss["id"]
                                    and row.get("input_transform") == transform
                                    and row.get("window_id") == candidate_window_id
                                    and row.get("trial_id") == info["trial_id"]
                                    and row.get("status") == "completed"
                                ]
                                if blind_raw:
                                    validation_means.append(cv_value)
                                    blind_means.append(_selection_value(blind_raw, "ctr_ps"))
                                    labels.append(candidate_window_id)
                            if validation_means:
                                cv_array = np.asarray(validation_means, dtype=np.float64)
                                blind_array = np.asarray(blind_means, dtype=np.float64)
                                pearson = (
                                    float(np.corrcoef(cv_array, blind_array)[0, 1])
                                    if cv_array.size >= 2 and np.std(cv_array) > 0 and np.std(blind_array) > 0
                                    else float("nan")
                                )
                                if cv_array.size >= 2:
                                    cv_rank = np.argsort(np.argsort(cv_array))
                                    blind_rank = np.argsort(np.argsort(blind_array))
                                    spearman = (
                                        float(np.corrcoef(cv_rank, blind_rank)[0, 1])
                                        if np.std(cv_rank) > 0 and np.std(blind_rank) > 0
                                        else float("nan")
                                    )
                                else:
                                    spearman = float("nan")
                                selected_position = labels.index(best_window_id)
                                blind_order = np.argsort(blind_array)
                                blind_rank_selected = int(np.where(blind_order == selected_position)[0][0]) + 1
                                blind_regret = float(
                                    blind_array[selected_position] - np.min(blind_array)
                                )
                                diagnostic_base = _base_row(
                                    config=config, root_id=root_id, root_file=root_file,
                                    mode_id=mode_id, model_id=model_id,
                                    model_type=space["model_type"], loss=loss,
                                    transform=transform,
                                    subsampling_factor=int(best_info["parameters"].get("preprocessing.subsampling_factor", 1)),
                                    window=best_info["window"],
                                    trial_id=best_info["trial_id"],
                                )
                                diagnostic = {
                                    **diagnostic_base,
                                    "record_type": "validation_quality",
                                    "split": "cv_vs_blind",
                                    "statistic": "diagnostic",
                                    "is_selected_hyperparameters": 1,
                                    "is_selected_window": 1,
                                    "status": "completed",
                                    "pearson_cv_blind": pearson,
                                    "spearman_cv_blind": spearman,
                                    "mean_cv_blind_gap_ps": float(np.mean(blind_array - cv_array)),
                                    "blind_rank_of_cv_selected_window": blind_rank_selected,
                                    "blind_regret_ps": blind_regret,
                                }
                                diagnostic["row_key"] = canonical_hash({
                                    "type": "validation_quality",
                                    "root": root_id,
                                    "mode": mode_id,
                                    "model": model_id,
                                    "loss": loss["id"],
                                    "transform": transform,
                                })[:24]
                                existing = next(
                                    (row for row in rows if row.get("row_key") == diagnostic["row_key"]),
                                    None,
                                )
                                if existing is None:
                                    rows.append(diagnostic)
                                else:
                                    existing.update(diagnostic)
                            _write_results(results_path, rows)
                            if not keep_checkpoints:
                                _prune_configuration_windows(
                                    run_root=run_root,
                                    root_id=root_id,
                                    mode_id=mode_id,
                                    model_id=model_id,
                                    loss_id=str(loss["id"]),
                                    transform=transform,
                                    keep_window_id=str(best_window_id),
                                    logger=logger,
                                )

        _persist_best_shapelet_model_for_file(
            config=config,
            rows=rows,
            root_file=root_file,
            root_id=root_id,
            development=development,
            blind=blind,
            output=output,
            run_root=run_root,
            logger=logger,
        )
        _write_results(results_path, rows)

        if not keep_checkpoints:
            _prune_file_runs_to_summary_winners(
                config=config,
                rows=rows,
                root_id=root_id,
                run_root=run_root,
                logger=logger,
            )
        _progress(logger, "File %d/%d | generating compact best-result summary", root_index, len(config["root_files"]))
        _generate_file_summary(
            config=config,
            rows=rows,
            root_file=root_file,
            root_id=root_id,
            development=development,
            blind=blind,
            output=output,
            run_root=run_root,
            logger=logger,
        )
        if not keep_checkpoints:
            _remove_tree(
                run_root / root_id,
                logger,
                "temporary model checkpoints and run artifacts",
            )
            _progress(
                logger,
                "File %d/%d | temporary model checkpoints released",
                root_index,
                len(config["root_files"]),
            )

        root_has_failures = any(
            row.get("root_id") == root_id and row.get("status") == "failed"
            for row in rows
        )
        if not root_has_failures:
            atomic_json(
                completed_file_marker,
                {
                    "root_id": root_id,
                    "root_file": str(root_file),
                    "completed": True,
                },
            )
            storage = config.get("storage", {})
            if bool(storage.get("cleanup_after_completed_file", True)):
                _close_dataset_memmaps(development)
                _close_dataset_memmaps(blind)
                del development, blind
                gc.collect()
                _remove_tree(output / "prepared" / root_id, logger, "prepared datasets")
                _remove_tree(output / "cache" / root_id, logger, "file cache")
                _remove_tree(
                    output / "shared_input_cache" / root_id,
                    logger,
                    "shared transform cache",
                )
                _progress(
                    logger,
                    "File %d/%d | temporary waveform caches released",
                    root_index,
                    len(config["root_files"]),
                )
        try:
            from .models.shapelet_regressor import clear_runtime_cache

            clear_runtime_cache()
        except ImportError:
            pass

    _progress(logger, "Finalizing compact summaries")
    _remove_tree(output / "plots", logger, "legacy plot directory")
    _plot_ctr_vs_voltage(config, output, logger)
    _progress(
        logger,
        "Study finished | blocks=%d/%d | elapsed=%s",
        completed_blocks,
        progress_total_blocks,
        _format_duration(time.perf_counter() - study_started),
    )
    return {
        "output_dir": str(output),
        "results_csv": str(_summary_results_path(output)),
        "model_loss_results_csv": str(_model_loss_results_path(output)),
        "shapelet_models_csv": (
            str(_shapelet_models_path(output))
            if _shapelet_models_path(output).is_file()
            else ""
        ),
        "state_results_csv": str(results_path),
        "row_count": len(_read_summary_results(_summary_results_path(output))),
    }
