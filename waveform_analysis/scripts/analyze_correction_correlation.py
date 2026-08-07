#!/usr/bin/env python3
"""Train configured models once and compare their per-event timing corrections.

This is intentionally *not* a cross-validation study.  One ROOT file is
preprocessed once into a frozen train/validation/test split.  Every requested ML
model is configured from ``config/model_spaces/<model_id>.json`` using its
``base_train_config`` and trained on the same train/validation events.  Pairwise
correlations are then computed on the same held-out test events.

Standard timing methods can be included in the same correlation matrix.  Their
correction is expressed relative to the channel-mode LED baseline, e.g.

    correction_CFD = delta_t_LED - delta_t_CFD

so that ``delta_t_LED - correction_CFD == delta_t_CFD``.  LED itself therefore
has an identically-zero correction and off-diagonal Pearson/Spearman
correlations involving LED are undefined.

Example
-------
python scripts/analyze_correction_correlation.py \
  --file processed_data/experiments/47V-470mV.root \
  --base-config config/experiments/folder_window_channel_study.json \
  --models ridge_regression lasso_regression shapelet_regressor \
  --standard-methods cfd \
  --channel-mode energy_to_energy \
  --window-start-ns -4 --window-end-ns 60 \
  --input-transform normalize \
  --subsampling-factor 4
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import re
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from ml_pipeline.common import atomic_json, canonical_hash, setup_logging
from ml_pipeline.evaluation import evaluate_trained_model, load_trained_model
from ml_pipeline.input_transform import normalize_input_transform, normalize_subsampling_factor
from ml_pipeline.prediction import prediction_window_dataset_view
from ml_pipeline.preprocessing import preprocess_dataset
from ml_pipeline.study import _delta_ps, _effective_train_config, _standard_method_delta_ps
from ml_pipeline.study_config import CHANNEL_MODES, load_model_space
from ml_pipeline.training import train_model
from ml_pipeline.training_utils import resolve_device


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _deep_update(target: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
    return target


def _safe_name(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")
    return text or "item"


def _resolve(project: Path, value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else project / path).resolve()


def _build_preprocess_config(
    *,
    base: dict[str, Any],
    root_file: Path,
    output_dir: Path,
    window_start_ns: float,
    window_end_ns: float,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    split_strategy: str,
    split_seed: int,
    guard_gap_events: int,
) -> dict[str, Any]:
    """Build one canonical single-file train/validation/test preprocessing config."""

    if not window_start_ns <= 0.0 <= window_end_ns or not window_start_ns < window_end_ns:
        raise ValueError("Window must satisfy start_ns <= 0 <= end_ns and start_ns < end_ns")
    fractions = np.asarray(
        [train_fraction, validation_fraction, test_fraction], dtype=np.float64
    )
    if np.any(fractions <= 0.0) or not np.isclose(float(fractions.sum()), 1.0):
        raise ValueError("train/validation/test fractions must be positive and sum to 1")

    if "data" not in base or "preprocessing" not in base:
        raise ValueError(
            "--base-config must contain the study-style 'data' and 'preprocessing' sections"
        )

    preprocessing = base["preprocessing"]
    common = copy.deepcopy(preprocessing.get("common", {}))
    energy = _deep_update(copy.deepcopy(common), preprocessing.get("energy", {}))
    timing = _deep_update(copy.deepcopy(common), preprocessing.get("timing", {}))

    before_ns = -float(window_start_ns)
    after_ns = float(window_end_ns)
    energy["ml_window_ns"] = {"before": before_ns, "after": after_ns}
    timing["ml_window_ns"] = {"before": before_ns, "after": after_ns}

    # Preserve the study invariant and materialize timing data so every channel
    # mode in CHANNEL_MODES can be selected without re-preprocessing the file.
    timing["enabled"] = True
    timing["denoising"] = {"enabled": False}
    energy["timing_channel_led"] = timing

    selection = copy.deepcopy(preprocessing.get("selection", {}))
    # Any robust target-based filtering would otherwise make the compared methods
    # see different effective event populations.  Keep preprocessing selection
    # waveform-only and compare all methods on the exact same test event IDs.
    selection["led_outlier_rejection"] = {"enabled": False}
    selection.setdefault("minimum_events_per_split", 1)

    config = {
        "dataset": {
            "name": f"correction_correlation_{_safe_name(root_file.stem)}",
            "role": "training",
            "output_dir": str((output_dir / "prepared").resolve()),
        },
        "data": {
            "input_root": str(root_file.resolve()),
            "true_tof_ps": float(base["data"].get("true_tof_ps", 0.0)),
        },
        "channels": copy.deepcopy(base["data"]["channels"]),
        "io": copy.deepcopy(
            preprocessing.get(
                "io", {"step_size": "128 MB", "max_events": 0, "progress_every": 1000}
            )
        ),
        "waveform": energy,
        "selection": selection,
        "photopeak": copy.deepcopy(preprocessing.get("photopeak", {"enabled": False})),
        "split": {
            "strategy": str(split_strategy),
            "seed": int(split_seed),
            "train_fraction": float(train_fraction),
            "validation_fraction": float(validation_fraction),
            "test_fraction": float(test_fraction),
            "guard_gap_events": int(guard_gap_events),
        },
        "parallelization": copy.deepcopy(
            preprocessing.get(
                "parallelization",
                {
                    "preprocessing_backend": "process",
                    "preprocessing_workers": 0,
                    "preprocessing_chunksize": 8,
                },
            )
        ),
        "cache": {
            "reuse": True,
            "raw_cache_dir": str((output_dir / "cache" / "raw").resolve()),
            "selection_cache_dir": str((output_dir / "cache" / "selection").resolve()),
            "materialization_chunk_size": int(
                preprocessing.get("materialization_chunk_size", 2048)
            ),
        },
        "logging": copy.deepcopy(base.get("logging", {"level": "INFO"})),
    }
    config_path = output_dir / "resolved_preprocess_config.json"
    atomic_json(config_path, config)
    config["_config_path"] = str(config_path.resolve())
    config["_config_hash"] = canonical_hash(config)
    return config


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    """Average ranks for Spearman correlation, including ties."""

    x = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    start = 0
    while start < x.size:
        stop = start + 1
        while stop < x.size and x[order[stop]] == x[order[start]]:
            stop += 1
        average_rank = 0.5 * (start + stop - 1) + 1.0
        ranks[order[start:stop]] = average_rank
        start = stop
    return ranks


def _pairwise_correlation(
    outputs: list[tuple[str, np.ndarray]], *, method: str
) -> tuple[list[str], np.ndarray, np.ndarray]:
    labels = [str(label) for label, _ in outputs]
    arrays = [np.asarray(values, dtype=np.float64).reshape(-1) for _, values in outputs]
    if arrays:
        expected = arrays[0].size
        for label, values in zip(labels, arrays):
            if values.size != expected:
                raise ValueError(
                    f"Correction vector {label!r} has {values.size} events; expected {expected}"
                )

    n = len(arrays)
    matrix = np.full((n, n), np.nan, dtype=np.float64)
    counts = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(i, n):
            finite = np.isfinite(arrays[i]) & np.isfinite(arrays[j])
            count = int(np.count_nonzero(finite))
            counts[i, j] = counts[j, i] = count
            if count < 2:
                continue
            left = arrays[i][finite]
            right = arrays[j][finite]
            if method == "spearman":
                left = _rankdata_average(left)
                right = _rankdata_average(right)
            elif method != "pearson":
                raise ValueError(method)
            if i == j:
                value = 1.0
            elif float(np.std(left, ddof=0)) == 0.0 or float(np.std(right, ddof=0)) == 0.0:
                value = float("nan")
            else:
                value = float(np.corrcoef(left, right)[0, 1])
            matrix[i, j] = matrix[j, i] = value
    return labels, matrix, counts


def _write_matrix(path: Path, labels: list[str], matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", *labels])
        for label, row in zip(labels, np.asarray(matrix)):
            writer.writerow(
                [
                    label,
                    *[
                        "" if not np.isfinite(value) else f"{float(value):.12g}"
                        for value in row
                    ],
                ]
            )


def _write_counts(path: Path, labels: list[str], counts: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", *labels])
        for label, row in zip(labels, np.asarray(counts, dtype=np.int64)):
            writer.writerow([label, *[int(value) for value in row]])


def _plot_matrix(
    path: Path,
    labels: list[str],
    matrix: np.ndarray,
    *,
    title: str,
    dpi: int,
) -> None:
    count = max(1, len(labels))
    figure_size = max(6.0, 1.05 * count + 2.8)
    fig, ax = plt.subplots(figsize=(figure_size, figure_size))
    image = ax.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_xticks(np.arange(count), labels=labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(count), labels=labels)
    ax.set_title(title, pad=14)

    for row in range(count):
        for column in range(count):
            value = float(matrix[row, column])
            text = "—" if not np.isfinite(value) else f"{value:.2f}"
            ax.text(column, row, text, ha="center", va="center", fontsize=9)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Correlation")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def _write_event_corrections(
    path: Path,
    *,
    dataset: Any,
    indices: np.ndarray,
    baseline_ps: np.ndarray,
    required_correction_ps: np.ndarray,
    outputs: list[tuple[str, np.ndarray]],
) -> None:
    labels = [label for label, _ in outputs]
    arrays = [np.asarray(values, dtype=np.float64) for _, values in outputs]
    fields = [
        "dataset_row_index",
        "event_id",
        "event_index",
        "source_run_index",
        "led_baseline_ps",
        "required_correction_ps",
        *[f"correction__{_safe_name(label)}_ps" for label in labels],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for local, dataset_index in enumerate(np.asarray(indices, dtype=np.int64)):
            row: dict[str, Any] = {
                "dataset_row_index": int(dataset_index),
                "event_id": int(np.asarray(dataset.event_id)[dataset_index]),
                "event_index": int(np.asarray(dataset.event_index)[dataset_index]),
                "source_run_index": int(np.asarray(dataset.source_run_index)[dataset_index]),
                "led_baseline_ps": float(baseline_ps[local]),
                "required_correction_ps": float(required_correction_ps[local]),
            }
            for label, values in zip(labels, arrays):
                row[f"correction__{_safe_name(label)}_ps"] = float(values[local])
            writer.writerow(row)


def _correlation_with_required(values: np.ndarray, required: np.ndarray) -> tuple[float, float]:
    finite = np.isfinite(values) & np.isfinite(required)
    if int(np.count_nonzero(finite)) < 2:
        return float("nan"), float("nan")
    left = np.asarray(values[finite], dtype=np.float64)
    right = np.asarray(required[finite], dtype=np.float64)
    if float(np.std(left, ddof=0)) == 0.0 or float(np.std(right, ddof=0)) == 0.0:
        pearson = float("nan")
    else:
        pearson = float(np.corrcoef(left, right)[0, 1])
    left_rank = _rankdata_average(left)
    right_rank = _rankdata_average(right)
    if float(np.std(left_rank, ddof=0)) == 0.0 or float(np.std(right_rank, ddof=0)) == 0.0:
        spearman = float("nan")
    else:
        spearman = float(np.corrcoef(left_rank, right_rank)[0, 1])
    return pearson, spearman


def _write_summary(
    path: Path,
    *,
    outputs: list[tuple[str, np.ndarray]],
    required_correction_ps: np.ndarray,
    model_metadata: dict[str, dict[str, Any]],
) -> None:
    fields = [
        "method",
        "kind",
        "model_type",
        "model_space",
        "n_events",
        "correction_mean_ps",
        "correction_std_ps",
        "correction_rmse_vs_required_ps",
        "pearson_vs_required",
        "spearman_vs_required",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for label, raw_values in outputs:
            values = np.asarray(raw_values, dtype=np.float64)
            finite = np.isfinite(values) & np.isfinite(required_correction_ps)
            difference = values[finite] - required_correction_ps[finite]
            pearson, spearman = _correlation_with_required(values, required_correction_ps)
            meta = model_metadata.get(label, {})
            writer.writerow(
                {
                    "method": label,
                    "kind": meta.get("kind", "standard_method"),
                    "model_type": meta.get("model_type", ""),
                    "model_space": meta.get("model_space", ""),
                    "n_events": int(np.count_nonzero(finite)),
                    "correction_mean_ps": float(np.mean(values[finite])) if np.any(finite) else "",
                    "correction_std_ps": float(np.std(values[finite], ddof=0)) if np.any(finite) else "",
                    "correction_rmse_vs_required_ps": (
                        float(np.sqrt(np.mean(difference * difference))) if difference.size else ""
                    ),
                    "pearson_vs_required": pearson,
                    "spearman_vs_required": spearman,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train model-space base configurations on one frozen split of one ROOT file "
            "and compare held-out per-event correction correlations."
        )
    )
    parser.add_argument("--file", type=Path, required=True, help="Input ROOT file")
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("config/experiments/folder_window_channel_study.json"),
        help=(
            "Study-style JSON used only for channels, ROOT preprocessing, fit, logging, "
            "and evaluation defaults. ML model definitions come from --model-spaces-dir."
        ),
    )
    parser.add_argument(
        "--model-spaces-dir", type=Path, default=Path("config/model_spaces")
    )
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument(
        "--standard-methods",
        nargs="*",
        default=["cfd"],
        choices=["led", "cfd"],
    )
    parser.add_argument(
        "--channel-mode", choices=sorted(CHANNEL_MODES), default="energy_to_energy"
    )
    parser.add_argument("--window-start-ns", type=float, default=-4.0)
    parser.add_argument("--window-end-ns", type=float, default=60.0)
    parser.add_argument("--input-transform", default="none")
    parser.add_argument("--subsampling-factor", type=int, default=1)
    parser.add_argument(
        "--loss",
        choices=["mse", "var_bias"],
        default="mse",
        help="Common study loss mapped through each model-space configuration.",
    )
    parser.add_argument("--bias-weight", type=float, default=0.01)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument(
        "--split-strategy",
        choices=["event", "stratified_event", "source_file", "contiguous_blocks"],
        default=None,
    )
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--guard-gap-events", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/adhoc/correction_correlation"),
    )
    parser.add_argument("--rebuild-preprocessing", action="store_true")
    parser.add_argument(
        "--keep-work",
        action="store_true",
        help="Keep temporary per-model checkpoints/runs after correlations are written.",
    )
    args = parser.parse_args()

    root_file = _resolve(PROJECT, args.file)
    if not root_file.is_file():
        raise FileNotFoundError(root_file)
    base_config_path = _resolve(PROJECT, args.base_config)
    if not base_config_path.is_file():
        raise FileNotFoundError(base_config_path)
    model_spaces_dir = _resolve(PROJECT, args.model_spaces_dir)
    if not model_spaces_dir.is_dir():
        raise FileNotFoundError(model_spaces_dir)

    input_transform = normalize_input_transform(args.input_transform)
    subsampling_factor = normalize_subsampling_factor(args.subsampling_factor)
    base = _read_json(base_config_path)
    split_source = base.get("split", {}) if isinstance(base.get("split"), dict) else {}
    split_strategy = str(args.split_strategy or split_source.get("strategy", "contiguous_blocks"))
    guard_gap_events = int(
        split_source.get("guard_gap_events", 0)
        if args.guard_gap_events is None
        else args.guard_gap_events
    )

    output_dir = _resolve(PROJECT, args.output_dir) / _safe_name(root_file.stem)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(
        output_dir / "correction_correlation.log",
        base.get("logging", {}).get("level", "INFO"),
    )

    preprocess_config = _build_preprocess_config(
        base=base,
        root_file=root_file,
        output_dir=output_dir,
        window_start_ns=float(args.window_start_ns),
        window_end_ns=float(args.window_end_ns),
        train_fraction=float(args.train_fraction),
        validation_fraction=float(args.validation_fraction),
        test_fraction=float(args.test_fraction),
        split_strategy=split_strategy,
        split_seed=int(args.seed),
        guard_gap_events=guard_gap_events,
    )
    logger.info("Preprocessing %s once for correction-correlation analysis", root_file.name)
    dataset = preprocess_dataset(
        preprocess_config,
        rebuild=bool(args.rebuild_preprocessing),
        logger=logger,
    )
    if dataset.train.size == 0 or dataset.validation.size == 0 or dataset.test.size == 0:
        raise RuntimeError(
            "The prepared dataset must contain non-empty train, validation, and test splits"
        )

    mode = CHANNEL_MODES[str(args.channel_mode)]
    before_ns = -float(args.window_start_ns)
    after_ns = float(args.window_end_ns)
    train_view = prediction_window_dataset_view(
        dataset,
        input_waveforms=mode["input_waveforms"],
        target=mode["target"],
        before_ns=before_ns,
        after_ns=after_ns,
    )

    evaluation_indices = np.asarray(dataset.test, dtype=np.int64)
    evaluation_source = replace(dataset, evaluation=evaluation_indices)
    baseline_ps = _delta_ps(dataset, mode["baseline"], evaluation_indices)
    required_correction_ps = baseline_ps - float(dataset.true_tof_ps)

    outputs: list[tuple[str, np.ndarray]] = []
    metadata: dict[str, dict[str, Any]] = {}

    for method_id in args.standard_methods:
        method_id = str(method_id).lower()
        if method_id == "led":
            correction = np.zeros(evaluation_indices.size, dtype=np.float64)
            label = "LED"
        elif method_id == "cfd":
            cfd_ps = _standard_method_delta_ps(
                dataset, mode["target"], "cfd", evaluation_indices
            )
            correction = baseline_ps - cfd_ps
            label = "CFD"
        else:
            raise ValueError(method_id)
        outputs.append((label, np.asarray(correction, dtype=np.float64)))
        metadata[label] = {"kind": "standard_method", "model_type": method_id}
        logger.info(
            "Standard method %s | correction mean=%.3f ps | std=%.3f ps",
            label,
            float(np.mean(correction)),
            float(np.std(correction, ddof=0)),
        )

    if args.loss == "mse":
        loss = {"id": "mse", "type": "mse"}
    else:
        loss = {
            "id": "var_bias",
            "type": "var_bias",
            "bias_weight": float(args.bias_weight),
            "bias_normalization": "target_std",
            "minimum_scale": 1e-8,
        }

    shared_input_cache = work_dir / "shared_input_cache" / args.channel_mode / input_transform
    eval_config = {
        "device": base.get("evaluation", {}).get("device", "auto"),
        "batch_size": int(base.get("evaluation", {}).get("batch_size", 512)),
        "num_workers": int(base.get("evaluation", {}).get("num_workers", 0)),
        "pin_memory": bool(base.get("evaluation", {}).get("pin_memory", False)),
        "input_transform_cache_dir": str(shared_input_cache),
        "output": {"evaluation_dir": str(work_dir / "evaluation")},
    }
    device = resolve_device(eval_config["device"])

    for model_index, model_id in enumerate(args.models):
        space_path = model_spaces_dir / f"{model_id}.json"
        if not space_path.is_file():
            raise FileNotFoundError(f"Model-space config not found: {space_path}")
        space = load_model_space(space_path)
        if str(space["id"]) != str(model_id):
            raise ValueError(
                f"Model-space id mismatch: requested {model_id!r}, file contains {space['id']!r}"
            )
        if str(args.loss) not in {str(value) for value in space.get("supported_losses", [])}:
            raise ValueError(f"Model {model_id!r} does not support loss {args.loss!r}")

        run_dir = work_dir / "models" / _safe_name(model_id)
        model_name = f"correlation_{model_id}"
        train_config = _effective_train_config(
            space,
            loss,
            input_transform,
            mode,
            model_name,
            run_dir,
            int(args.seed) + model_index,
        )
        train_config.setdefault("preprocessing", {})["subsampling_factor"] = int(
            subsampling_factor
        )
        train_config.setdefault("training", {})["input_transform_cache_dir"] = str(
            shared_input_cache
        )
        if "fit" in base:
            train_config["fit"] = copy.deepcopy(base["fit"])

        logger.info(
            "Training %s from base_train_config | model_type=%s | transform=%s | factor=%d",
            model_id,
            space["model_type"],
            input_transform,
            subsampling_factor,
        )
        train_model(
            train_config,
            restart=True,
            logger=logger,
            prepared_datasets=[train_view],
            data_view={
                "window_id": "correlation_window",
                "window_before_ns": before_ns,
                "window_after_ns": after_ns,
                "channel_mode": str(args.channel_mode),
            },
        )
        trained = load_trained_model(run_dir)
        prediction = evaluate_trained_model(trained, evaluation_source, eval_config, device)
        correction = np.asarray(prediction.predicted_correction_ps, dtype=np.float64)
        if correction.size != evaluation_indices.size:
            raise RuntimeError(
                f"Model {model_id} returned {correction.size} corrections for "
                f"{evaluation_indices.size} test events"
            )
        outputs.append((str(model_id), correction))
        metadata[str(model_id)] = {
            "kind": "ml_model",
            "model_type": str(space["model_type"]),
            "model_space": str(space_path.relative_to(PROJECT))
            if space_path.is_relative_to(PROJECT)
            else str(space_path),
        }
        logger.info(
            "Model %s | correction mean=%.3f ps | std=%.3f ps",
            model_id,
            float(np.mean(correction)),
            float(np.std(correction, ddof=0)),
        )

    if len(outputs) < 2:
        raise ValueError(
            "At least two correction sources are required; add ML models and/or standard methods"
        )

    pearson_labels, pearson, counts = _pairwise_correlation(outputs, method="pearson")
    spearman_labels, spearman, spearman_counts = _pairwise_correlation(
        outputs, method="spearman"
    )
    if pearson_labels != spearman_labels or not np.array_equal(counts, spearman_counts):
        raise RuntimeError("Internal correlation output mismatch")

    _write_matrix(output_dir / "correlation_pearson.csv", pearson_labels, pearson)
    _write_matrix(output_dir / "correlation_spearman.csv", spearman_labels, spearman)
    _write_counts(output_dir / "correlation_event_counts.csv", pearson_labels, counts)
    _write_event_corrections(
        output_dir / "corrections.csv",
        dataset=dataset,
        indices=evaluation_indices,
        baseline_ps=baseline_ps,
        required_correction_ps=required_correction_ps,
        outputs=outputs,
    )
    _write_summary(
        output_dir / "method_summary.csv",
        outputs=outputs,
        required_correction_ps=required_correction_ps,
        model_metadata=metadata,
    )

    dpi = int(base.get("reporting", {}).get("dpi", base.get("plotting", {}).get("dpi", 180)))
    subtitle = (
        f"{root_file.name} | {args.channel_mode} | "
        f"[{args.window_start_ns:g}, {args.window_end_ns:g}] ns | "
        f"{input_transform} | subsampling={subsampling_factor} | test n={evaluation_indices.size}"
    )
    _plot_matrix(
        output_dir / "correlation_pearson.png",
        pearson_labels,
        pearson,
        title=f"Pearson correlation of applied timing corrections\n{subtitle}",
        dpi=dpi,
    )
    _plot_matrix(
        output_dir / "correlation_spearman.png",
        spearman_labels,
        spearman,
        title=f"Spearman correlation of applied timing corrections\n{subtitle}",
        dpi=dpi,
    )

    resolved = {
        "root_file": str(root_file),
        "base_config": str(base_config_path),
        "model_spaces_dir": str(model_spaces_dir),
        "models": list(args.models),
        "standard_methods": list(args.standard_methods),
        "channel_mode": str(args.channel_mode),
        "window_start_ns": float(args.window_start_ns),
        "window_end_ns": float(args.window_end_ns),
        "input_transform": input_transform,
        "subsampling_factor": int(subsampling_factor),
        "loss": loss,
        "split": {
            "strategy": split_strategy,
            "seed": int(args.seed),
            "train_fraction": float(args.train_fraction),
            "validation_fraction": float(args.validation_fraction),
            "test_fraction": float(args.test_fraction),
            "guard_gap_events": guard_gap_events,
        },
        "split_counts_after_preprocessing": {
            "train": int(dataset.train.size),
            "validation": int(dataset.validation.size),
            "test": int(dataset.test.size),
        },
        "correction_definition": {
            "ml": "prediction.predicted_correction_ps (total correction applied to LED)",
            "cfd": "target-specific LED delta - target-specific CFD delta",
            "led": "0 ps",
            "application": "final_delta_t = LED_delta_t - correction",
        },
    }
    atomic_json(output_dir / "resolved_correlation_config.json", resolved)

    if not args.keep_work:
        shutil.rmtree(work_dir, ignore_errors=True)

    logger.info("Correction-correlation analysis complete: %s", output_dir)
    print(output_dir)


if __name__ == "__main__":
    main()
