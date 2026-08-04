from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any

SCRIPT_VERSION = "3.0-separate-data-and-initialization-seeds"

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from ml_pipeline.common import setup_logging
from ml_pipeline.config import load_train_config
from ml_pipeline.dataset import load_prepared_dataset
from ml_pipeline.input_transform import (
    normalize_input_transform,
    transform_relative_time_ps,
    transformed_input_length,
)
from ml_pipeline.training import train_model
from ml_pipeline.models.mlp_regressor import build as build_mlp, model_state_hash
from ml_pipeline.prediction import prediction_dataset_view, resolve_prediction_config




def _array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("utf-8"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _preview_initialization(
    model_config: dict[str, Any],
    *,
    input_length: int,
    hidden_units: int,
    initialization_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Rebuild exactly the state used at the start of MLP training."""
    torch.manual_seed(int(initialization_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(initialization_seed))
    model = build_mlp(model_config, input_length)
    state = model.state_dict()

    first_candidates = [
        (name, tensor)
        for name, tensor in state.items()
        if tensor.ndim == 2 and tuple(tensor.shape) == (hidden_units, input_length)
    ]
    output_candidates = [
        (name, tensor)
        for name, tensor in state.items()
        if tensor.ndim == 2 and tuple(tensor.shape) == (1, hidden_units)
    ]
    if len(first_candidates) != 1 or len(output_candidates) != 1:
        raise RuntimeError("Could not identify MLP layers during initialization audit")

    first_name, first_tensor = first_candidates[0]
    output_name, output_tensor = output_candidates[0]
    first_bias = state.get(first_name.removesuffix("weight") + "bias")
    output_bias = state.get(output_name.removesuffix("weight") + "bias")

    first = first_tensor.detach().cpu().numpy().astype(np.float64)
    first_b = (
        np.zeros(hidden_units, dtype=np.float64)
        if first_bias is None
        else first_bias.detach().cpu().numpy().astype(np.float64)
    )
    output = output_tensor.detach().cpu().numpy().reshape(hidden_units).astype(np.float64)
    output_b = (
        np.zeros(1, dtype=np.float64)
        if output_bias is None
        else output_bias.detach().cpu().numpy().reshape(1).astype(np.float64)
    )
    return first, first_b, output, output_b, model_state_hash(model)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _save_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _load_existing(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        return {}
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def _extract_weights(
    checkpoint_path: Path,
    *,
    input_length: int,
    hidden_units: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state"]

    first_candidates = [
        (name, tensor)
        for name, tensor in state.items()
        if tensor.ndim == 2 and tuple(tensor.shape) == (hidden_units, input_length)
    ]
    output_candidates = [
        (name, tensor)
        for name, tensor in state.items()
        if tensor.ndim == 2 and tuple(tensor.shape) == (1, hidden_units)
    ]
    if len(first_candidates) != 1 or len(output_candidates) != 1:
        shapes = {name: tuple(value.shape) for name, value in state.items()}
        raise RuntimeError(
            "Could not identify the first and output linear layers uniquely. "
            f"Expected ({hidden_units}, {input_length}) and (1, {hidden_units}); "
            f"found state shapes: {shapes}"
        )

    first_name, first_tensor = first_candidates[0]
    output_name, output_tensor = output_candidates[0]
    first_bias_name = first_name.removesuffix("weight") + "bias"
    output_bias_name = output_name.removesuffix("weight") + "bias"
    first_bias = state.get(first_bias_name)
    output_bias = state.get(output_bias_name)

    context = checkpoint.get("context", {})
    normalization = context.get("normalization", {})
    std_mV = float(normalization.get("std_mV", 1.0))
    if not np.isfinite(std_mV) or std_mV <= 0.0:
        raise RuntimeError(f"Invalid normalization standard deviation: {std_mV}")

    return (
        first_tensor.detach().cpu().numpy().astype(np.float64),
        np.zeros(hidden_units, dtype=np.float64)
        if first_bias is None
        else first_bias.detach().cpu().numpy().astype(np.float64),
        output_tensor.detach().cpu().numpy().reshape(hidden_units).astype(np.float64),
        np.zeros(1, dtype=np.float64)
        if output_bias is None
        else output_bias.detach().cpu().numpy().reshape(1).astype(np.float64),
        std_mV,
    )


def _run_importance(
    first_weights: np.ndarray,
    output_weights: np.ndarray,
    normalization_std_mV: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    first_physical = first_weights / float(normalization_std_mV)
    first_group_l2 = np.linalg.norm(first_physical, axis=0)
    path_abs = np.sum(
        np.abs(output_weights[:, None] * first_physical),
        axis=0,
    )
    path_total = float(np.sum(path_abs))
    normalized_path = (
        path_abs / path_total
        if path_total > 0.0
        else np.zeros_like(path_abs)
    )
    signed_path = output_weights @ first_physical
    return first_physical, first_group_l2, path_abs, normalized_path, signed_path


def _quantiles(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.quantile(values, 0.05, axis=0),
        np.median(values, axis=0),
        np.quantile(values, 0.95, axis=0),
    )


def _top_frequency(values: np.ndarray, fraction: float) -> np.ndarray:
    count = values.shape[1]
    selected = max(1, int(math.ceil(float(fraction) * count)))
    mask = np.zeros_like(values, dtype=np.float64)
    for run_index, row in enumerate(values):
        indices = np.argpartition(row, -selected)[-selected:]
        mask[run_index, indices] = 1.0
    return np.mean(mask, axis=0)


def _sample_rows(
    relative_time_ps: np.ndarray,
    first_l2: np.ndarray,
    path_abs: np.ndarray,
    normalized_path: np.ndarray,
    signed_path: np.ndarray,
    top_fraction: float,
) -> list[dict[str, Any]]:
    first_q05, first_median, first_q95 = _quantiles(first_l2)
    path_q05, path_median, path_q95 = _quantiles(path_abs)
    norm_q05, norm_median, norm_q95 = _quantiles(normalized_path)
    signed_q05, signed_median, signed_q95 = _quantiles(signed_path)
    positive_fraction = np.mean(signed_path > 0.0, axis=0)
    negative_fraction = np.mean(signed_path < 0.0, axis=0)
    sign_consistency = np.maximum(positive_fraction, negative_fraction)
    top_frequency = _top_frequency(normalized_path, top_fraction)

    rows: list[dict[str, Any]] = []
    for index, time_ps in enumerate(relative_time_ps):
        rows.append(
            {
                "sample_index": index,
                "time_ps": float(time_ps),
                "first_l2_mean": float(np.mean(first_l2[:, index])),
                "first_l2_std": float(np.std(first_l2[:, index])),
                "first_l2_q05": float(first_q05[index]),
                "first_l2_median": float(first_median[index]),
                "first_l2_q95": float(first_q95[index]),
                "path_abs_mean": float(np.mean(path_abs[:, index])),
                "path_abs_std": float(np.std(path_abs[:, index])),
                "path_abs_q05": float(path_q05[index]),
                "path_abs_median": float(path_median[index]),
                "path_abs_q95": float(path_q95[index]),
                "importance_mean": float(np.mean(normalized_path[:, index])),
                "importance_std": float(np.std(normalized_path[:, index])),
                "importance_q05": float(norm_q05[index]),
                "importance_median": float(norm_median[index]),
                "importance_q95": float(norm_q95[index]),
                "top_selection_frequency": float(top_frequency[index]),
                "signed_path_mean": float(np.mean(signed_path[:, index])),
                "signed_path_std": float(np.std(signed_path[:, index])),
                "signed_path_q05": float(signed_q05[index]),
                "signed_path_median": float(signed_median[index]),
                "signed_path_q95": float(signed_q95[index]),
                "signed_positive_fraction": float(positive_fraction[index]),
                "signed_sign_consistency": float(sign_consistency[index]),
            }
        )
    return rows


def _block_arrays(values: np.ndarray, block_size: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    blocks: list[np.ndarray] = []
    bounds: list[tuple[int, int]] = []
    for start in range(0, values.shape[1], block_size):
        stop = min(start + block_size, values.shape[1])
        blocks.append(np.sum(values[:, start:stop], axis=1))
        bounds.append((start, stop))
    return np.stack(blocks, axis=1), bounds


def _block_rows(
    relative_time_ps: np.ndarray,
    normalized_path: np.ndarray,
    signed_path: np.ndarray,
    block_size: int,
    top_fraction: float,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    block_importance, bounds = _block_arrays(normalized_path, block_size)
    block_signed, _ = _block_arrays(signed_path, block_size)
    importance_q05, importance_median, importance_q95 = _quantiles(block_importance)
    signed_q05, signed_median, signed_q95 = _quantiles(block_signed)
    top_frequency = _top_frequency(block_importance, top_fraction)
    positive_fraction = np.mean(block_signed > 0.0, axis=0)
    negative_fraction = np.mean(block_signed < 0.0, axis=0)

    rows: list[dict[str, Any]] = []
    for block_id, (start, stop) in enumerate(bounds):
        rows.append(
            {
                "block_id": block_id,
                "start_index": start,
                "stop_index_exclusive": stop,
                "start_time_ps": float(relative_time_ps[start]),
                "stop_time_ps": float(relative_time_ps[stop - 1]),
                "center_time_ps": float(np.mean(relative_time_ps[start:stop])),
                "sample_count": stop - start,
                "importance_mean": float(np.mean(block_importance[:, block_id])),
                "importance_std": float(np.std(block_importance[:, block_id])),
                "importance_q05": float(importance_q05[block_id]),
                "importance_median": float(importance_median[block_id]),
                "importance_q95": float(importance_q95[block_id]),
                "top_selection_frequency": float(top_frequency[block_id]),
                "signed_path_mean": float(np.mean(block_signed[:, block_id])),
                "signed_path_std": float(np.std(block_signed[:, block_id])),
                "signed_path_q05": float(signed_q05[block_id]),
                "signed_path_median": float(signed_median[block_id]),
                "signed_path_q95": float(signed_q95[block_id]),
                "signed_positive_fraction": float(positive_fraction[block_id]),
                "signed_sign_consistency": float(
                    max(positive_fraction[block_id], negative_fraction[block_id])
                ),
            }
        )
    return rows, block_importance


def _make_plots(
    output_dir: Path,
    relative_time_ps: np.ndarray,
    normalized_path: np.ndarray,
    signed_path: np.ndarray,
    block_rows: list[dict[str, Any]],
    block_importance: np.ndarray,
    run_metrics: list[dict[str, Any]],
    dpi: int,
) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    q05, median, q95 = _quantiles(normalized_path)
    mean = np.mean(normalized_path, axis=0)
    figure, axis = plt.subplots(figsize=(11, 4.5))
    axis.plot(relative_time_ps, mean, label="Mean")
    axis.plot(relative_time_ps, median, label="Median")
    axis.fill_between(relative_time_ps, q05, q95, alpha=0.25, label="5-95%")
    axis.set_xlabel("Relative time [ps]")
    axis.set_ylabel("Normalized absolute path importance")
    axis.legend()
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(plot_dir / "sample_importance_distribution.png", dpi=dpi)
    plt.close(figure)

    signed_q05, signed_median, signed_q95 = _quantiles(signed_path)
    signed_mean = np.mean(signed_path, axis=0)
    figure, axis = plt.subplots(figsize=(11, 4.5))
    axis.plot(relative_time_ps, signed_mean, label="Mean")
    axis.plot(relative_time_ps, signed_median, label="Median")
    axis.fill_between(relative_time_ps, signed_q05, signed_q95, alpha=0.25, label="5-95%")
    axis.axhline(0.0, linewidth=1.0)
    axis.set_xlabel("Relative time [ps]")
    axis.set_ylabel("Signed output-weighted path coefficient")
    axis.legend()
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(plot_dir / "signed_path_distribution.png", dpi=dpi)
    plt.close(figure)

    centers = np.asarray([row["center_time_ps"] for row in block_rows], dtype=np.float64)
    figure, axis = plt.subplots(figsize=(11, 5.0))
    image = axis.imshow(
        block_importance,
        aspect="auto",
        origin="lower",
        extent=(centers[0], centers[-1], 0, block_importance.shape[0]),
    )
    axis.set_xlabel("Block center time [ps]")
    axis.set_ylabel("Run index")
    figure.colorbar(image, ax=axis, label="Block importance")
    figure.tight_layout()
    figure.savefig(plot_dir / "block_importance_heatmap.png", dpi=dpi)
    plt.close(figure)

    runs = np.asarray([row["run_id"] for row in run_metrics], dtype=np.int64)
    rmse = np.asarray([row["validation_rmse_ps"] for row in run_metrics], dtype=np.float64)
    ctr = np.asarray([row["validation_ctr_ps"] for row in run_metrics], dtype=np.float64)
    figure, axis = plt.subplots(figsize=(8.5, 4.5))
    axis.plot(runs, rmse, marker="o", label="Validation RMSE")
    axis.plot(runs, ctr, marker="o", label="Validation CTR")
    axis.set_xlabel("Run")
    axis.set_ylabel("Time [ps]")
    axis.legend()
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(plot_dir / "run_performance.png", dpi=dpi)
    plt.close(figure)


def _validate_dataset_grids(
    dataset_paths: list[str], input_transform: str, prediction: dict[str, str]
) -> tuple[int, np.ndarray]:
    datasets = [
        prediction_dataset_view(
            load_prepared_dataset(path),
            input_waveforms=prediction["input_waveforms"],
            target=prediction["target"],
        )
        for path in dataset_paths
    ]
    lengths = {dataset.input_length for dataset in datasets}
    if len(lengths) != 1:
        raise ValueError(f"Datasets have incompatible input lengths: {sorted(lengths)}")
    reference = np.asarray(datasets[0].relative_time_ps, dtype=np.float64)
    for dataset in datasets[1:]:
        values = np.asarray(dataset.relative_time_ps, dtype=np.float64)
        if values.shape != reference.shape or not np.allclose(values, reference):
            raise ValueError("All datasets must use the same relative_time_ps grid")
    transformed_time = transform_relative_time_ps(reference, input_transform)
    return transformed_input_length(reference.size, input_transform), transformed_time


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train repeated one-hidden-layer, four-neuron antisymmetric MLPs and "
            "summarize input-weight distributions across initialization seeds."
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="Base MLP training config")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--initialization-seed-start", type=int, default=10000)
    parser.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help="Fixed seed for split-independent minibatch order; defaults to training.seed",
    )
    parser.add_argument("--block-size", type=int, default=100)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-run-checkpoints", action="store_true")
    parser.add_argument(
        "--verify-initializations-only",
        action="store_true",
        help="Audit initial tensors for every seed without training",
    )
    args = parser.parse_args()

    if args.runs <= 0:
        raise ValueError("--runs must be positive")
    if args.block_size <= 0:
        raise ValueError("--block-size must be positive")
    if not 0.0 < args.top_fraction <= 1.0:
        raise ValueError("--top-fraction must lie in (0, 1]")

    config = load_train_config(args.config, PROJECT)
    if str(config["model"].get("type")) != "mlp_regressor":
        raise ValueError("The supplied configuration must use model.type='mlp_regressor'")
    if bool(config["model"].get("batch_norm", False)):
        raise ValueError("Batch normalization must be disabled for direct weight analysis")
    if float(config["model"].get("dropout", 0.0)) != 0.0:
        raise ValueError(
            "Dropout must be zero so repeated runs differ by initialization rather than masks"
        )

    config["model"]["hidden_units"] = [4]
    config["model"]["name"] = "mlp_4_initialization_weight_analysis"
    data_seed = int(
        config["training"].get("seed", 12345)
        if args.data_seed is None
        else args.data_seed
    )
    config["training"]["seed"] = data_seed
    config["training"]["data_seed"] = data_seed

    base_train_dir = Path(config["output"]["train_dir"])
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else base_train_dir.with_name(base_train_dir.name + "_initialization_weight_analysis")
    )
    if output_dir.exists() and not args.resume:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(
        output_dir / "analysis.log",
        config.get("logging", {}).get("level", "INFO"),
    )
    logger.info("Running %s version %s", Path(__file__).name, SCRIPT_VERSION)
    input_transform = normalize_input_transform(config.get("input_transform", "none"))
    prediction = resolve_prediction_config(config)
    input_length, relative_time_ps = _validate_dataset_grids(
        config["datasets"], input_transform, prediction
    )
    if args.block_size > input_length:
        raise ValueError("--block-size cannot exceed the waveform input length")

    initialization_rows: list[dict[str, Any]] = []
    initial_first_all: list[np.ndarray] = []
    initial_first_bias_all: list[np.ndarray] = []
    initial_output_all: list[np.ndarray] = []
    initial_output_bias_all: list[np.ndarray] = []
    initialization_hashes: list[str] = []
    reference_initial: np.ndarray | None = None
    for run_id in range(args.runs):
        initialization_seed = int(args.initialization_seed_start + run_id)
        initial_first, initial_first_bias, initial_output, initial_output_bias, state_hash = (
            _preview_initialization(
                config["model"],
                input_length=input_length,
                hidden_units=4,
                initialization_seed=initialization_seed,
            )
        )
        if reference_initial is None:
            reference_initial = initial_first.copy()
        initialization_rows.append(
            {
                "run_id": run_id,
                "initialization_seed": initialization_seed,
                "initial_state_hash": state_hash,
                "first_weight_0": float(initial_first.reshape(-1)[0]),
                "first_layer_mean": float(np.mean(initial_first)),
                "first_layer_std": float(np.std(initial_first)),
                "first_layer_l2": float(np.linalg.norm(initial_first)),
                "first_layer_max_abs_difference_from_run_0": float(
                    np.max(np.abs(initial_first - reference_initial))
                ),
            }
        )
        initial_first_all.append(initial_first)
        initial_first_bias_all.append(initial_first_bias)
        initial_output_all.append(initial_output)
        initial_output_bias_all.append(initial_output_bias)
        initialization_hashes.append(state_hash)

    duplicate_hashes = sorted(
        {value for value in initialization_hashes if initialization_hashes.count(value) > 1}
    )
    if duplicate_hashes:
        raise RuntimeError(
            "Initialization audit found identical initial model states for different seeds: "
            + ", ".join(duplicate_hashes)
        )
    _write_csv(output_dir / "initialization_audit.csv", initialization_rows)
    _save_npz_atomic(
        output_dir / "initial_weights.npz",
        first_weights=np.stack(initial_first_all),
        first_bias=np.stack(initial_first_bias_all),
        output_weights=np.stack(initial_output_all),
        output_bias=np.stack(initial_output_bias_all),
        initialization_seeds=np.arange(
            args.initialization_seed_start,
            args.initialization_seed_start + args.runs,
            dtype=np.int64,
        ),
    )
    logger.info(
        "Initialization audit passed: %d distinct model states for %d seeds",
        len(set(initialization_hashes)),
        args.runs,
    )
    if args.verify_initializations_only:
        logger.info("Initialization-only audit written to %s", output_dir)
        return

    weights_path = output_dir / "weights.npz"
    existing = _load_existing(weights_path) if args.resume else {}
    first_weights_runs = list(existing.get("first_weights", np.empty((0, 4, input_length))))
    first_bias_runs = list(existing.get("first_bias", np.empty((0, 4))))
    output_weights_runs = list(existing.get("output_weights", np.empty((0, 4))))
    output_bias_runs = list(existing.get("output_bias", np.empty((0, 1))))
    normalization_std_runs = list(existing.get("normalization_std_mV", np.empty(0)))
    seeds = [int(value) for value in existing.get("initialization_seeds", np.empty(0, dtype=np.int64))]

    run_metrics_path = output_dir / "run_metrics.csv"
    run_metrics: list[dict[str, Any]] = []
    if args.resume and run_metrics_path.is_file():
        with run_metrics_path.open("r", newline="", encoding="utf-8") as stream:
            run_metrics = list(csv.DictReader(stream))
        for row in run_metrics:
            for key in ("run_id", "initialization_seed", "best_epoch"):
                row[key] = int(row[key])
            for key in (
                "validation_rmse_ps",
                "validation_ctr_ps",
                "validation_bias_ps",
                "normalization_std_mV",
                "initial_to_final_first_layer_l2",
            ):
                row[key] = float(row[key])

    completed_seeds = set(seeds)
    working_root = output_dir / ".working"
    working_root.mkdir(parents=True, exist_ok=True)

    for run_id in range(args.runs):
        initialization_seed = int(args.initialization_seed_start + run_id)
        if initialization_seed in completed_seeds:
            logger.info("Skipping completed initialization seed %d", initialization_seed)
            continue

        run_dir = working_root / f"run_{run_id:04d}"
        expected_initial_hash = initialization_hashes[run_id]
        run_config = copy.deepcopy(config)
        run_config["training"]["initialization_seed"] = initialization_seed
        run_config["output"]["train_dir"] = str(run_dir)
        run_config["artifacts"] = {
            "save_config": False,
            "save_history": False,
            "save_plots": False,
            "save_last_checkpoint": False,
            "save_summary": False,
        }
        logger.info(
            "Run %d initialization hash %s | first weight %.9g",
            run_id + 1,
            expected_initial_hash[:12],
            float(initial_first_all[run_id].reshape(-1)[0]),
        )
        logger.info(
            "Run %d/%d | data seed %d | initialization seed %d",
            run_id + 1,
            args.runs,
            data_seed,
            initialization_seed,
        )
        summary = train_model(run_config, restart=True, logger=logger)
        actual_initial_hash = str(summary.get("initial_state_hash", ""))
        actual_initial_seed = int(summary.get("initialization_seed", -1))
        if actual_initial_seed != initialization_seed:
            raise RuntimeError(
                "Trainer used initialization seed "
                f"{actual_initial_seed}, expected {initialization_seed}."
            )
        if actual_initial_hash != expected_initial_hash:
            raise RuntimeError(
                "The model audited before training was not the model initialized by the trainer: "
                f"expected hash {expected_initial_hash[:12]}, actual {actual_initial_hash[:12]}. "
                "Update ml_pipeline/models/mlp_regressor.py and ml_pipeline/training_utils.py "
                "from the V3 package."
            )
        checkpoint_path = run_dir / "checkpoints" / "best.pt"
        first, first_bias, output, output_bias, std_mV = _extract_weights(
            checkpoint_path,
            input_length=input_length,
            hidden_units=4,
        )

        first_weights_runs.append(first)
        first_bias_runs.append(first_bias)
        output_weights_runs.append(output)
        output_bias_runs.append(output_bias)
        normalization_std_runs.append(std_mV)
        seeds.append(initialization_seed)
        completed_seeds.add(initialization_seed)
        final_state_hash = _array_hash(first, first_bias, output, output_bias)
        run_metrics.append(
            {
                "run_id": run_id,
                "initialization_seed": initialization_seed,
                "initial_state_hash": actual_initial_hash,
                "final_state_hash": final_state_hash,
                "initial_to_final_first_layer_l2": float(
                    np.linalg.norm(first - initial_first_all[run_id])
                ),
                "best_epoch": int(summary["best_epoch"]),
                "validation_rmse_ps": float(summary["best_validation_rmse_ps"]),
                "validation_ctr_ps": float(summary["best_validation_ctr_ps"]),
                "validation_bias_ps": float(summary["best_validation_bias_ps"]),
                "normalization_std_mV": float(std_mV),
            }
        )
        run_metrics.sort(key=lambda row: int(row["run_id"]))
        _write_csv(run_metrics_path, run_metrics)
        _save_npz_atomic(
            weights_path,
            first_weights=np.stack(first_weights_runs),
            first_bias=np.stack(first_bias_runs),
            output_weights=np.stack(output_weights_runs),
            output_bias=np.stack(output_bias_runs),
            normalization_std_mV=np.asarray(normalization_std_runs, dtype=np.float64),
            initialization_seeds=np.asarray(seeds, dtype=np.int64),
            relative_time_ps=relative_time_ps,
            input_transform=np.asarray([input_transform]),
            input_waveform_source=np.asarray([prediction["input_waveforms"]]),
            prediction_target=np.asarray([prediction["target"]]),
        )
        if not args.keep_run_checkpoints:
            shutil.rmtree(run_dir)

    if not first_weights_runs:
        raise RuntimeError("No completed training runs are available")

    first_weights = np.stack(first_weights_runs)
    output_weights = np.stack(output_weights_runs)
    normalization_std = np.asarray(normalization_std_runs, dtype=np.float64)
    first_physical: list[np.ndarray] = []
    first_l2: list[np.ndarray] = []
    path_abs: list[np.ndarray] = []
    normalized_path: list[np.ndarray] = []
    signed_path: list[np.ndarray] = []
    for first, output, std_mV in zip(first_weights, output_weights, normalization_std):
        values = _run_importance(first, output, float(std_mV))
        first_physical.append(values[0])
        first_l2.append(values[1])
        path_abs.append(values[2])
        normalized_path.append(values[3])
        signed_path.append(values[4])

    first_physical_array = np.stack(first_physical)
    first_l2_array = np.stack(first_l2)
    path_abs_array = np.stack(path_abs)
    normalized_path_array = np.stack(normalized_path)
    signed_path_array = np.stack(signed_path)

    _save_npz_atomic(
        weights_path,
        first_weights=first_weights,
        first_weights_physical=first_physical_array,
        first_bias=np.stack(first_bias_runs),
        output_weights=output_weights,
        output_bias=np.stack(output_bias_runs),
        first_layer_l2=first_l2_array,
        path_abs=path_abs_array,
        normalized_path_importance=normalized_path_array,
        signed_path=signed_path_array,
        normalization_std_mV=normalization_std,
        initialization_seeds=np.asarray(seeds, dtype=np.int64),
        relative_time_ps=relative_time_ps,
        input_transform=np.asarray([input_transform]),
        input_waveform_source=np.asarray([prediction["input_waveforms"]]),
        prediction_target=np.asarray([prediction["target"]]),
    )

    sample_rows = _sample_rows(
        relative_time_ps,
        first_l2_array,
        path_abs_array,
        normalized_path_array,
        signed_path_array,
        args.top_fraction,
    )
    block_rows, block_importance = _block_rows(
        relative_time_ps,
        normalized_path_array,
        signed_path_array,
        args.block_size,
        args.top_fraction,
    )
    _write_csv(output_dir / "sample_statistics.csv", sample_rows)
    _write_csv(output_dir / "block_statistics.csv", block_rows)

    activation = str(config["model"].get("activation", "relu"))
    output_bound = config["model"].get("max_abs_single_channel_output_ps")
    exact_linear = activation == "identity" and output_bound is None
    settings = [
        {
            "completed_runs": len(first_weights_runs),
            "requested_runs": args.runs,
            "data_seed": data_seed,
            "initialization_seed_start": args.initialization_seed_start,
            "hidden_units": 4,
            "activation": activation,
            "output_bound_ps": "" if output_bound is None else float(output_bound),
            "signed_path_is_exact_effective_weight": int(exact_linear),
            "input_length": input_length,
            "input_transform": input_transform,
            "input_waveform_source": prediction["input_waveforms"],
            "prediction_target": prediction["target"],
            "block_size": args.block_size,
            "top_fraction": args.top_fraction,
            "dataset_count": len(config["datasets"]),
        }
    ]
    _write_csv(output_dir / "analysis_settings.csv", settings)
    _make_plots(
        output_dir,
        relative_time_ps,
        normalized_path_array,
        signed_path_array,
        block_rows,
        block_importance,
        run_metrics,
        int(config.get("plotting", {}).get("dpi", 180)),
    )

    if not args.keep_run_checkpoints and working_root.exists():
        shutil.rmtree(working_root)
    logger.info("Weight-distribution analysis written to %s", output_dir)


if __name__ == "__main__":
    main()
