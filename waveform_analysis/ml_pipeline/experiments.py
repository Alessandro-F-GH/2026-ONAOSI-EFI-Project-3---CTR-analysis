from __future__ import annotations

import copy
import csv
import itertools
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from .common import canonical_hash, read_json, write_csv_rows
from .config import validate_train_config
from .dataset import (
    PreparedDataset,
    load_prepared_dataset,
    prepared_dataset_view,
    window_slice_indices,
)
from .input_transform import resolve_input_transform, transformed_input_length
from .models import count_model_parameters
from .prediction import prediction_dataset_view, resolve_prediction_config
from .training import train_model
import logging
from pathlib import Path


def close_log_handlers_under(directory: Path) -> None:
    """Close logging handlers whose files are inside directory."""

    directory = directory.resolve()

    loggers = [logging.getLogger()]
    loggers.extend(
        logger
        for logger in logging.Logger.manager.loggerDict.values()
        if isinstance(logger, logging.Logger)
    )

    for current_logger in loggers:
        for handler in current_logger.handlers[:]:
            if not isinstance(handler, logging.FileHandler):
                continue

            handler_path = Path(handler.baseFilename).resolve()

            try:
                handler_path.relative_to(directory)
            except ValueError:
                continue

            try:
                handler.flush()
                handler.close()
            finally:
                current_logger.removeHandler(handler)


def load_experiment_config(path: str | Path, project_root: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    root = Path(project_root).resolve()
    config = read_json(source)
    if not str(config.get("dataset", "")).strip():
        raise ValueError("experiment config requires dataset")
    base = config.get("base_train_config")
    if not isinstance(base, dict):
        raise ValueError("experiment config requires base_train_config")
    model = base.get("model")
    if not isinstance(model, dict) or not str(model.get("type", "")).strip():
        raise ValueError("base_train_config.model.type must be non-empty")

    result = copy.deepcopy(config)
    dataset = Path(result["dataset"])
    if not dataset.is_absolute():
        dataset = root / dataset
    result["dataset"] = str(dataset.resolve())
    output = Path(result.get("output_dir", root / "results" / "experiments" / source.stem))
    if not output.is_absolute():
        output = root / output
    result["output_dir"] = str(output.resolve())
    result["_config_path"] = str(source)
    result["_config_hash"] = canonical_hash(config)
    result["_project_root"] = str(root)
    return result


def _set_nested(target: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    current = target
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = copy.deepcopy(value)


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    output: dict[str, Any] = {}
    if isinstance(value, dict):
        for key in sorted(value):
            path = f"{prefix}.{key}" if prefix else str(key)
            output.update(_flatten(value[key], path))
    else:
        output[prefix] = value
    return output


def _compact(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (list, tuple)):
        return "|".join(str(_compact(item)) for item in value)
    return value


def _write_key_value_csv(path: Path, values: dict[str, Any]) -> None:
    rows = [{"key": key, "value": _compact(value)} for key, value in sorted(values.items())]
    write_csv_rows(path, rows)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))



def _read_key_value_csv(path: Path) -> dict[str, str]:
    return {
        str(row["key"]): str(row["value"])
        for row in _read_csv_rows(path)
        if row.get("key") is not None
    }

def _grid_values(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else [value]


def _product_parameters(parameters: dict[str, Any]) -> list[dict[str, Any]]:
    if not parameters:
        return [{}]
    keys = list(parameters)
    values = [_grid_values(parameters[key]) for key in keys]
    return [dict(zip(keys, combination)) for combination in itertools.product(*values)]


def _sample_parameter(spec: Any, rng: random.Random) -> Any:
    if not isinstance(spec, dict) or "type" not in spec:
        return copy.deepcopy(spec)
    kind = str(spec["type"])
    if kind == "categorical":
        return copy.deepcopy(rng.choice(list(spec["values"])))
    if kind == "integer":
        return rng.randint(int(spec["low"]), int(spec["high"]))
    if kind == "uniform":
        return rng.uniform(float(spec["low"]), float(spec["high"]))
    if kind == "loguniform":
        return math.exp(
            rng.uniform(math.log(float(spec["low"])), math.log(float(spec["high"])))
        )
    raise ValueError(f"Unsupported random parameter type: {kind}")


def _static_parameter_sets(config: dict[str, Any]) -> list[dict[str, Any]]:
    search = dict(config.get("search", {}))
    method = str(search.get("method", "grid"))
    parameters = dict(search.get("parameters", {}))
    if method == "grid":
        raw = _product_parameters(parameters)
    elif method == "random":
        rng = random.Random(int(search.get("random_state", 12345)))
        raw = [
            {key: _sample_parameter(spec, rng) for key, spec in parameters.items()}
            for _ in range(int(search.get("n_trials", 1)))
        ]
    else:
        raise ValueError(f"Static parameter generation does not support {method!r}")

    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for parameters_set in raw:
        fingerprint = canonical_hash(parameters_set)
        if fingerprint not in seen:
            seen.add(fingerprint)
            unique.append(parameters_set)
    return unique


def _suggest_optuna_parameters(trial: Any, specifications: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key, spec in specifications.items():
        if not isinstance(spec, dict) or "type" not in spec:
            values[key] = copy.deepcopy(spec)
            continue
        kind = str(spec["type"])
        if kind == "categorical":
            choices = list(spec["values"])
            encoded = [canonical_hash(value) for value in choices]
            selected = trial.suggest_categorical(key, encoded)
            values[key] = copy.deepcopy(choices[encoded.index(selected)])
        elif kind == "integer":
            values[key] = trial.suggest_int(key, int(spec["low"]), int(spec["high"]))
        elif kind == "uniform":
            values[key] = trial.suggest_float(key, float(spec["low"]), float(spec["high"]))
        elif kind == "loguniform":
            values[key] = trial.suggest_float(
                key, float(spec["low"]), float(spec["high"]), log=True
            )
        else:
            raise ValueError(f"Unsupported Optuna parameter type for {key}: {kind}")
    return values


def _effective_train_config(base: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(base)
    for key, value in parameters.items():
        _set_nested(config, key, value)
    config.setdefault("plotting", {})
    config.setdefault("logging", {})
    config.setdefault("artifacts", {})
    return config


def _window_rows(config: dict[str, Any], dataset: PreparedDataset) -> list[dict[str, Any]]:
    search = config.get("window_search", {})
    if not isinstance(search, dict) or not bool(search.get("enabled", False)):
        combinations = [
            (
                max(0.0, -float(dataset.relative_time_ps[0]) / 1000.0),
                max(0.0, float(dataset.relative_time_ps[-1]) / 1000.0),
            )
        ]
    else:
        before_values = _grid_values(search.get("window_before_ns", [2.0]))
        after_values = _grid_values(search.get("window_after_ns", [5.0]))
        combinations = [
            (float(before), float(after))
            for before in before_values
            for after in after_values
        ]
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for before, after in combinations:
        start, stop = window_slice_indices(dataset, before, after)
        if (start, stop) in seen:
            continue
        seen.add((start, stop))
        rows.append(
            {
                "window_id": len(rows) + 1,
                "before_ns": before,
                "after_ns": after,
                "start_index": start,
                "stop_index": stop,
                "input_length": stop - start,
                "time_start_ps": float(dataset.relative_time_ps[start]),
                "time_stop_ps": float(dataset.relative_time_ps[stop - 1]),
            }
        )
    return rows


def _kfold_indices(
    indices: np.ndarray,
    n_splits: int,
    shuffle: bool,
    random_state: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    values = np.asarray(indices, dtype=np.int64).copy()
    if not 2 <= int(n_splits) <= values.size:
        raise ValueError(
            f"cross_validation.n_splits must lie in [2, {values.size}], got {n_splits}"
        )
    if shuffle:
        rng = np.random.default_rng(int(random_state))
        rng.shuffle(values)
    fold_sizes = np.full(int(n_splits), values.size // int(n_splits), dtype=int)
    fold_sizes[: values.size % int(n_splits)] += 1
    output: list[tuple[np.ndarray, np.ndarray]] = []
    cursor = 0
    for fold_size in fold_sizes:
        validation = values[cursor : cursor + fold_size]
        training = np.concatenate([values[:cursor], values[cursor + fold_size :]])
        output.append((training, validation))
        cursor += fold_size
    return output


def _resolve_folds(
    dataset: PreparedDataset, cv_config: dict[str, Any]
) -> list[tuple[np.ndarray, np.ndarray]]:
    if bool(cv_config.get("enabled", True)):
        development = np.concatenate([dataset.train, dataset.validation]).astype(np.int64)
        return _kfold_indices(
            development,
            int(cv_config.get("n_splits", 5)),
            bool(cv_config.get("shuffle", True)),
            int(cv_config.get("random_state", 12345)),
        )
    if dataset.train.size == 0 or dataset.validation.size == 0:
        raise ValueError("Prepared dataset requires non-empty train and validation splits")
    return [(np.asarray(dataset.train), np.asarray(dataset.validation))]


def _indices_hash(values: np.ndarray) -> str:
    return canonical_hash(np.asarray(values, dtype=np.int64))


def _trial_key(
    *,
    dataset_fingerprint: str,
    config_hash: str,
    window: dict[str, Any],
    fold: int,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    seed: int,
) -> str:
    return canonical_hash(
        {
            "dataset": dataset_fingerprint,
            "config": config_hash,
            "window_start": int(window["start_index"]),
            "window_stop": int(window["stop_index"]),
            "fold": int(fold),
            "train": _indices_hash(train_indices),
            "validation": _indices_hash(validation_indices),
            "seed": int(seed),
        }
    )[:16]


def _validate_effective_config(config: dict[str, Any], dataset_path: str, output: Path) -> None:
    candidate = copy.deepcopy(config)
    candidate["datasets"] = [dataset_path]
    candidate.setdefault("output", {})["train_dir"] = str(output)
    validate_train_config(candidate)


def _summary_rows(
    trials: list[dict[str, Any]],
    configurations: dict[int, dict[str, Any]],
    windows: dict[int, dict[str, Any]],
    required_runs: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in trials:
        if int(row["status_code"]) != 1:
            continue
        grouped.setdefault((int(row["config_id"]), int(row["window_id"])), []).append(row)

    output: list[dict[str, Any]] = []
    for (config_id, window_id), group in grouped.items():
        rmses = np.asarray([float(row["validation_rmse_ps"]) for row in group])
        biases = np.asarray([float(row["validation_bias_ps"]) for row in group])
        ctrs = np.asarray([float(row["validation_ctr_ps"]) for row in group])
        complete = len(group) == required_runs
        effective_config = configurations[config_id]["effective_config"]
        model_cfg = dict(effective_config["model"])
        model_type = str(model_cfg.pop("type"))
        model_cfg.pop("name", None)
        model_cfg.pop("input_transform", None)
        model_input_length = transformed_input_length(
            int(windows[window_id]["input_length"]),
            resolve_input_transform(effective_config),
        )
        observed_parameter_counts = [
            int(float(row.get("parameter_count", 0)))
            for row in group
            if float(row.get("parameter_count", 0) or 0) > 0
        ]
        parameter_count = (
            max(observed_parameter_counts)
            if observed_parameter_counts
            else count_model_parameters(
                model_type, model_cfg, model_input_length
            )
        )
        output.append(
            {
                "config_id": config_id,
                "window_id": window_id,
                "completed_runs": len(group),
                "required_runs": required_runs,
                "complete": int(complete),
                "parameter_count": parameter_count,
                "mean_validation_rmse_ps": float(np.mean(rmses)),
                "std_validation_rmse_ps": float(np.std(rmses, ddof=0)),
                "se_validation_rmse_ps": float(
                    np.std(rmses, ddof=0) / math.sqrt(len(group))
                ),
                "median_validation_rmse_ps": float(np.median(rmses)),
                "worst_validation_rmse_ps": float(np.max(rmses)),
                "mean_validation_bias_ps": float(np.mean(biases)),
                "mean_abs_validation_bias_ps": float(np.mean(np.abs(biases))),
                "mean_validation_ctr_ps": float(np.mean(ctrs)),
                "std_validation_ctr_ps": float(np.std(ctrs, ddof=0)),
                "mean_runtime_seconds": float(
                    np.mean([float(row["runtime_seconds"]) for row in group])
                ),
                "rank": 0,
                "selected": 0,
            }
        )
    complete_rows = [row for row in output if int(row["complete"]) == 1]
    complete_rows.sort(key=lambda row: float(row["mean_validation_rmse_ps"]))
    if complete_rows:
        best = complete_rows[0]
        threshold = float(best["mean_validation_rmse_ps"]) + float(
            best["se_validation_rmse_ps"]
        )
        eligible = [
            row
            for row in complete_rows
            if float(row["mean_validation_rmse_ps"]) <= threshold
        ]
        eligible.sort(
            key=lambda row: (
                int(row["parameter_count"]),
                float(row["mean_runtime_seconds"]),
                float(row["mean_validation_rmse_ps"]),
            )
        )
        selected = eligible[0]
        for rank, row in enumerate(complete_rows, start=1):
            row["rank"] = rank
            row["selected"] = int(row is selected)
    output.sort(
        key=lambda row: (
            0 if int(row["complete"]) else 1,
            float(row["mean_validation_rmse_ps"]),
        )
    )
    return output


def run_experiment(
    config: dict[str, Any],
    *,
    dry_run: bool,
    resume: bool,
    restart: bool,
    logger: Any,
) -> dict[str, Any]:
    output = Path(config["output_dir"])
    if restart and output.exists():
        close_log_handlers_under(output)
        shutil.rmtree(output)
    existing_settings = _read_key_value_csv(output / "experiment_settings.csv")
    if output.exists() and not restart and not resume and (output / "trials.csv").is_file():
        raise FileExistsError(
            f"Experiment output already contains trials: {output}; use --resume or --restart"
        )
    if resume and existing_settings:
        if existing_settings.get("config_hash") != str(config["_config_hash"]):
            raise ValueError(
                "Cannot resume: experiment configuration hash changed; use --restart"
            )
    output.mkdir(parents=True, exist_ok=True)

    dataset = load_prepared_dataset(config["dataset"])
    prediction = resolve_prediction_config(config["base_train_config"])
    dataset = prediction_dataset_view(
        dataset,
        input_waveforms=prediction["input_waveforms"],
        target=prediction["target"],
    )
    if resume and existing_settings:
        previous_dataset = existing_settings.get("dataset_fingerprint")
        if previous_dataset and previous_dataset != str(dataset.manifest["fingerprint"]):
            raise ValueError("Cannot resume: dataset fingerprint changed; use --restart")
    windows_list = _window_rows(config, dataset)
    windows = {int(row["window_id"]): row for row in windows_list}
    folds = _resolve_folds(dataset, dict(config.get("cross_validation", {})))
    seeds = [int(value) for value in config.get("repetitions", {}).get("seeds", [101])]
    if not seeds:
        raise ValueError("repetitions.seeds must be non-empty")

    search = dict(config.get("search", {}))
    search_method = str(search.get("method", "grid"))
    if search_method not in {"grid", "random", "optuna_tpe"}:
        raise ValueError("search.method must be grid, random, or optuna_tpe")

    settings = {
        "name": config.get("name", Path(config["_config_path"]).stem),
        "config_path": config["_config_path"],
        "config_hash": config["_config_hash"],
        "dataset": config["dataset"],
        "dataset_fingerprint": dataset.manifest["fingerprint"],
        "model_type": config["base_train_config"]["model"]["type"],
        "input_waveform_source": prediction["input_waveforms"],
        "prediction_target": prediction["target"],
        "search_method": search_method,
        "fold_count": len(folds),
        "seed_count": len(seeds),
        "window_count": len(windows_list),
    }
    _write_key_value_csv(output / "experiment_settings.csv", settings)
    _write_key_value_csv(
        output / "base_configuration.csv", _flatten(config["base_train_config"])
    )
    write_csv_rows(output / "windows.csv", windows_list)
    np.savez_compressed(
        output / "folds.npz",
        **{f"fold_{index}_train": train for index, (train, _) in enumerate(folds)},
        **{
            f"fold_{index}_validation": validation
            for index, (_, validation) in enumerate(folds)
        },
    )

    if search_method in {"grid", "random"}:
        parameter_sets = _static_parameter_sets(config)
        planned_configurations = len(parameter_sets)
    else:
        parameter_sets = []
        planned_configurations = int(search.get("n_trials", 10))

    stats = {
        "configurations": planned_configurations,
        "windows": len(windows_list),
        "folds": len(folds),
        "seeds": len(seeds),
        "training_runs": planned_configurations
        * len(windows_list)
        * len(folds)
        * len(seeds),
    }
    logger.info(
        "Experiment %s | model=%s | configs=%d | windows=%d | folds=%d | seeds=%d | runs=%d",
        settings["name"],
        settings["model_type"],
        stats["configurations"],
        stats["windows"],
        stats["folds"],
        stats["seeds"],
        stats["training_runs"],
    )
    if dry_run:
        return stats

    existing_trials_raw = _read_csv_rows(output / "trials.csv") if resume else []
    trials: list[dict[str, Any]] = [dict(row) for row in existing_trials_raw]
    completed_by_key = {
        str(row["trial_key"]): row
        for row in trials
        if int(row.get("status_code", 0)) == 1
    }
    errors: list[dict[str, Any]] = [
        dict(row) for row in (_read_csv_rows(output / "errors.csv") if resume else [])
    ]
    configurations: dict[int, dict[str, Any]] = {}
    configuration_rows: list[dict[str, Any]] = []
    search_columns = list(dict(search.get("parameters", {})).keys())

    def evaluate_parameters(parameters: dict[str, Any], config_id: int) -> float:
        effective = _effective_train_config(config["base_train_config"], parameters)
        config_hash = canonical_hash(effective)
        validation_output = output / ".validation"
        _validate_effective_config(effective, config["dataset"], validation_output)
        configurations[config_id] = {
            "parameters": copy.deepcopy(parameters),
            "effective_config": effective,
            "config_hash": config_hash,
        }
        row = {"config_id": config_id, "config_hash": config_hash[:16]}
        row.update({key: _compact(parameters.get(key, "")) for key in search_columns})
        configuration_rows.append(row)
        write_csv_rows(output / "configurations.csv", configuration_rows)

        total = len(windows_list) * len(folds) * len(seeds)
        current = 0
        for window in windows_list:
            for fold_index, (train_indices, validation_indices) in enumerate(folds):
                view = prepared_dataset_view(
                    dataset,
                    train_indices=train_indices,
                    validation_indices=validation_indices,
                    window_start=int(window["start_index"]),
                    window_stop=int(window["stop_index"]),
                )
                for seed in seeds:
                    current += 1
                    key = _trial_key(
                        dataset_fingerprint=dataset.manifest["fingerprint"],
                        config_hash=config_hash,
                        window=window,
                        fold=fold_index,
                        train_indices=train_indices,
                        validation_indices=validation_indices,
                        seed=seed,
                    )
                    if key in completed_by_key:
                        logger.info(
                            "[%d/%d] cached | config=%d window=%d fold=%d seed=%d",
                            current,
                            total,
                            config_id,
                            int(window["window_id"]),
                            fold_index,
                            seed,
                        )
                        continue

                    temporary = output / ".trial_tmp" / key
                    train_config = copy.deepcopy(effective)
                    train_config["datasets"] = [config["dataset"]]
                    if "data_contract" in config:
                        train_config["data_contract"] = copy.deepcopy(config["data_contract"])
                    train_config.setdefault("training", {})["seed"] = seed
                    train_config.setdefault("output", {})["train_dir"] = str(temporary)
                    train_config["artifacts"] = {
                        "save_config": False,
                        "save_history": False,
                        "save_plots": False,
                        "save_last_checkpoint": False,
                        "save_summary": False,
                    }
                    data_view = {
                        "window_id": int(window["window_id"]),
                        "window_start_index": int(window["start_index"]),
                        "window_stop_index": int(window["stop_index"]),
                        "window_before_ns": float(window["before_ns"]),
                        "window_after_ns": float(window["after_ns"]),
                    }
                    logger.info(
                        "[%d/%d] start | config=%d window=%d fold=%d seed=%d",
                        current,
                        total,
                        config_id,
                        int(window["window_id"]),
                        fold_index,
                        seed,
                    )
                    started = time.time()
                    trial_id = len(trials) + 1
                    try:
                        summary = train_model(
                            train_config,
                            restart=True,
                            logger=logger,
                            prepared_datasets=[view],
                            data_view=data_view,
                        )
                        trial_row = {
                            "trial_id": trial_id,
                            "trial_key": key,
                            "config_id": config_id,
                            "window_id": int(window["window_id"]),
                            "fold": fold_index,
                            "seed": seed,
                            "status_code": 1,
                            "train_n": int(train_indices.size),
                            "validation_n": int(validation_indices.size),
                            "best_epoch": int(summary["best_epoch"]),
                            "train_rmse_ps": float(summary["final_train_rmse_ps"]),
                            "train_bias_ps": float(summary["final_train_bias_ps"]),
                            "validation_rmse_ps": float(
                                summary["best_validation_rmse_ps"]
                            ),
                            "validation_bias_ps": float(
                                summary["best_validation_bias_ps"]
                            ),
                            "validation_ctr_ps": float(
                                summary["best_validation_ctr_ps"]
                            ),
                            "parameter_count": int(
                                summary.get("model_parameter_count", 0)
                            ),
                            "runtime_seconds": float(time.time() - started),
                        }
                        completed_by_key[key] = trial_row
                    except Exception as exc:
                        trial_row = {
                            "trial_id": trial_id,
                            "trial_key": key,
                            "config_id": config_id,
                            "window_id": int(window["window_id"]),
                            "fold": fold_index,
                            "seed": seed,
                            "status_code": -1,
                            "train_n": int(train_indices.size),
                            "validation_n": int(validation_indices.size),
                            "best_epoch": 0,
                            "train_rmse_ps": math.nan,
                            "train_bias_ps": math.nan,
                            "validation_rmse_ps": math.nan,
                            "validation_bias_ps": math.nan,
                            "validation_ctr_ps": math.nan,
                            "parameter_count": 0,
                            "runtime_seconds": float(time.time() - started),
                        }
                        errors.append(
                            {
                                "trial_id": trial_id,
                                "error_type": type(exc).__name__,
                                "message": str(exc),
                            }
                        )
                        write_csv_rows(output / "errors.csv", errors)
                        logger.exception(
                            "Trial failed | config=%d window=%d fold=%d seed=%d",
                            config_id,
                            int(window["window_id"]),
                            fold_index,
                            seed,
                        )
                    finally:
                        shutil.rmtree(temporary, ignore_errors=True)
                    trials.append(trial_row)
                    write_csv_rows(output / "trials.csv", trials)

        config_trials = [
            row
            for row in trials
            if int(row["config_id"]) == config_id and int(row["status_code"]) == 1
        ]
        per_window: list[float] = []
        required = len(folds) * len(seeds)
        for window in windows_list:
            values = [
                float(row["validation_rmse_ps"])
                for row in config_trials
                if int(row["window_id"]) == int(window["window_id"])
            ]
            if len(values) == required:
                per_window.append(float(np.mean(values)))
        return min(per_window) if per_window else math.inf

    if search_method in {"grid", "random"}:
        for config_id, parameters in enumerate(parameter_sets, start=1):
            evaluate_parameters(parameters, config_id)
    else:
        try:
            import optuna
        except ImportError as exc:
            raise RuntimeError("Optuna is required for search.method='optuna_tpe'") from exc
        storage_dir = output / "search_state"
        storage_dir.mkdir(parents=True, exist_ok=True)
        study = optuna.create_study(
            direction="minimize",
            study_name=str(config.get("name", "experiment")),
            storage=f"sqlite:///{storage_dir / 'optuna.db'}",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(
                seed=int(search.get("random_state", 12345))
            ),
        )
        target_trials = int(search.get("n_trials", 10))
        completed_trials = [
            trial for trial in study.trials if trial.state.name == "COMPLETE"
        ]
        next_config_id = 1
        for stored_trial in completed_trials[:target_trials]:
            parameters = stored_trial.user_attrs.get("resolved_parameters")
            if not isinstance(parameters, dict):
                raise RuntimeError(
                    "Existing Optuna state predates the modular experiment format; "
                    "restart the experiment output directory"
                )
            evaluate_parameters(parameters, next_config_id)
            next_config_id += 1
        for _ in range(max(0, target_trials - len(completed_trials))):
            trial = study.ask()
            parameters = _suggest_optuna_parameters(
                trial, dict(search.get("parameters", {}))
            )
            trial.set_user_attr("resolved_parameters", parameters)
            objective = evaluate_parameters(parameters, next_config_id)
            study.tell(trial, objective)
            next_config_id += 1

    required_runs = len(folds) * len(seeds)
    summary = _summary_rows(
        trials,
        configurations,
        windows,
        required_runs,
    )
    write_csv_rows(output / "summary.csv", summary)
    selected = next((row for row in summary if int(row["selected"]) == 1), None)
    if selected is None:
        raise RuntimeError("No complete experiment configuration is available for selection")

    selected_config_id = int(selected["config_id"])
    selected_window_id = int(selected["window_id"])
    selected_definition = configurations[selected_config_id]
    selected_row = {
        "config_id": selected_config_id,
        "window_id": selected_window_id,
        "config_hash": selected_definition["config_hash"][:16],
    }
    selected_row.update(
        {
            key: _compact(selected_definition["parameters"].get(key, ""))
            for key in search_columns
        }
    )
    write_csv_rows(output / "selected_configuration.csv", [selected_row])

    final_path = ""
    final_identity = ""
    if bool(config.get("final_refit", {}).get("enabled", True)):
        selected_trials = [
            row
            for row in trials
            if int(row["status_code"]) == 1
            and int(row["config_id"]) == selected_config_id
            and int(row["window_id"]) == selected_window_id
        ]
        epochs = max(
            1,
            int(
                round(
                    float(
                        np.median(
                            [int(row["best_epoch"]) for row in selected_trials]
                        )
                    )
                )
            ),
        )
        window = windows[selected_window_id]
        final_view = prepared_dataset_view(
            dataset,
            train_indices=dataset.train,
            validation_indices=dataset.validation,
            window_start=int(window["start_index"]),
            window_stop=int(window["stop_index"]),
        )
        final_config = copy.deepcopy(selected_definition["effective_config"])
        final_config["datasets"] = [config["dataset"]]
        final_config.setdefault("training", {})["epochs"] = epochs
        final_config["training"]["early_stopping_patience"] = epochs
        final_config.setdefault("output", {})["train_dir"] = str(output / "final_model")
        final_artifacts = {
            "save_config": False,
            "save_history": False,
            "save_plots": False,
            "save_last_checkpoint": False,
            "save_summary": True,
        }
        final_artifacts.update(
            dict(config.get("final_refit", {}).get("artifacts", {}))
        )
        final_config["artifacts"] = final_artifacts
        final_data_view = {
            "window_id": selected_window_id,
            "window_start_index": int(window["start_index"]),
            "window_stop_index": int(window["stop_index"]),
            "window_before_ns": float(window["before_ns"]),
            "window_after_ns": float(window["after_ns"]),
        }
        final_identity = canonical_hash(
            {
                "config_hash": selected_definition["config_hash"],
                "window_id": selected_window_id,
                "window_start": int(window["start_index"]),
                "window_stop": int(window["stop_index"]),
                "epochs": epochs,
                "train": _indices_hash(dataset.train),
                "validation": _indices_hash(dataset.validation),
            }
        )[:16]
        existing_result = _read_key_value_csv(output / "result.csv")
        final_checkpoint = output / "final_model" / "checkpoints" / "best.pt"
        if (
            resume
            and final_checkpoint.is_file()
            and existing_result.get("final_identity") == final_identity
        ):
            logger.info("Final refit cached | identity=%s", final_identity)
        else:
            logger.info(
                "Final refit | config=%d window=%d epochs=%d train_n=%d validation_n=%d",
                selected_config_id,
                selected_window_id,
                epochs,
                int(dataset.train.size),
                int(dataset.validation.size),
            )
            train_model(
                final_config,
                restart=True,
                logger=logger,
                prepared_datasets=[final_view],
                data_view=final_data_view,
            )
            write_csv_rows(output / "final_model" / "selected_configuration.csv", [selected_row])
        final_path = str((output / "final_model").resolve())

    _write_key_value_csv(
        output / "result.csv",
        {
            "selected_config_id": selected_config_id,
            "selected_window_id": selected_window_id,
            "final_model": final_path,
            "final_identity": final_identity,
            "trial_count": len(trials),
            "failed_trials": sum(int(row["status_code"]) != 1 for row in trials),
        },
    )
    shutil.rmtree(output / ".trial_tmp", ignore_errors=True)
    shutil.rmtree(output / ".validation", ignore_errors=True)
    return {"output_dir": str(output), "final_model": final_path}
