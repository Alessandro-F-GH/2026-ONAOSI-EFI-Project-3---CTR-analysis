#!/usr/bin/env python3
"""Ad-hoc matched-effective-degrees-of-freedom Ridge window study.

This script intentionally lives outside the main ``ml_pipeline`` experiment
runner. It reuses the repository's existing:

* ROOT preprocessing and direct development/blind split;
* channel-mode and physical-window views;
* CV fold construction and fold-local robust event selection;
* input transforms and fold-local normalization;
* anchor-factored correction target;
* CTR/RMSE/bias evaluation and Gaussian fitting.

For every file, channel mode, physical window, and CV fold, the Ridge penalty is
chosen from the *training design matrix only* so that the slope effective
degrees of freedom equal a requested target::

    d_eff(alpha) = sum_j s_j^2 / (s_j^2 + alpha),

where ``s_j`` are the singular values of the normalized detector-difference
matrix. The calibrated scalar pair bias adds one unpenalized degree of freedom
for every window, so total effective complexity is ``d_eff + 1`` and remains
matched across windows.

No checkpoints or model files are written. Persistent outputs are CSV/JSON
results and PNG plots only.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from ml_pipeline.common import atomic_json, canonical_hash, setup_logging
from ml_pipeline.input_transform import (
    INPUT_TRANSFORM_NORMALIZE,
    SUPPORTED_INPUT_TRANSFORMS,
    materialize_training_input_cache,
    normalize_input_transform,
)
from ml_pipeline.prediction import prediction_window_dataset_view
from ml_pipeline.study import (
    _delta_ps,
    _ensure_preprocessed,
    _extract_voltage,
    _fold_masks,
    _metrics,
    _root_id,
)
from ml_pipeline.study_config import CHANNEL_MODES, load_study_config
from ml_pipeline.torch_data import compute_normalization, factored_correction_target_ps


RESULT_COLUMNS = [
    "row_key",
    "root_id",
    "root_file",
    "file_name",
    "voltage_V",
    "channel_mode",
    "input_waveforms",
    "target",
    "input_transform",
    "window_id",
    "window_start_ns",
    "window_end_ns",
    "window_length_ns",
    "fold_id",
    "target_slope_effective_df",
    "target_total_effective_df",
    "actual_slope_effective_df",
    "actual_total_effective_df",
    "effective_df_error",
    "alpha",
    "design_rank",
    "feature_count",
    "train_event_count",
    "validation_event_count",
    "blind_event_count",
    "coefficient_l1_norm",
    "coefficient_l2_norm",
    "pair_bias_ps",
    "validation_loss",
    "validation_rmse_ps",
    "validation_bias_ps",
    "validation_ctr_ps",
    "validation_baseline_ctr_ps",
    "validation_relative_improvement_pct",
    "validation_ctr_minus_led_ps",
    "blind_loss",
    "blind_rmse_ps",
    "blind_bias_ps",
    "blind_ctr_ps",
    "blind_baseline_ctr_ps",
    "blind_relative_improvement_pct",
    "blind_ctr_minus_led_ps",
    "outlier_center_ps",
    "outlier_scale_ps",
    "outlier_scale_method",
    "outlier_z_threshold",
    "runtime_seconds",
]


def effective_df_from_singular_values(
    singular_values: np.ndarray | Iterable[float], alpha: float
) -> float:
    """Return Ridge slope effective degrees of freedom.

    The formula matches scikit-learn Ridge's objective
    ``||y - Xw||^2 + alpha ||w||^2`` with ``fit_intercept=False``.
    """

    values = np.asarray(singular_values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return 0.0
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("singular_values must be finite and non-negative")
    penalty = float(alpha)
    if not math.isfinite(penalty) or penalty < 0.0:
        raise ValueError("alpha must be finite and non-negative")
    squared = values * values
    if penalty == 0.0:
        return float(np.count_nonzero(squared > 0.0))
    return float(np.sum(squared / (squared + penalty)))


def numerical_rank_singular_values(
    singular_values: np.ndarray | Iterable[float],
    *,
    matrix_shape: tuple[int, int],
) -> int:
    values = np.asarray(singular_values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return 0
    tolerance = (
        max(int(matrix_shape[0]), int(matrix_shape[1]))
        * np.finfo(np.float64).eps
        * float(np.max(values))
    )
    return int(np.count_nonzero(values > tolerance))


def alpha_for_target_effective_df(
    singular_values: np.ndarray | Iterable[float],
    target_df: float,
    *,
    matrix_shape: tuple[int, int],
    absolute_tolerance: float = 1.0e-8,
    max_iterations: int = 160,
) -> tuple[float, float, int]:
    """Solve monotonically for alpha giving the requested slope effective df.

    Returns ``(alpha, achieved_df, numerical_rank)``. ``target_df`` must be in
    ``(0, rank]``. A target equal to rank selects ``alpha=0`` (OLS limit).
    """

    values = np.asarray(singular_values, dtype=np.float64).reshape(-1)
    rank = numerical_rank_singular_values(values, matrix_shape=matrix_shape)
    target = float(target_df)
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("target effective df must be finite and positive")
    if rank <= 0:
        raise ValueError("Training design matrix has zero numerical rank")
    if target > rank + absolute_tolerance:
        raise ValueError(
            f"target effective df {target:g} exceeds training design rank {rank}"
        )
    if target >= rank - absolute_tolerance:
        return 0.0, float(rank), rank

    positive = values[values > 0.0]
    squared_max = float(np.max(positive * positive))
    if squared_max <= 0.0 or not math.isfinite(squared_max):
        raise ValueError("Cannot bracket Ridge alpha from invalid singular values")

    # Work in log(alpha) because the solution may be many orders of magnitude
    # below/above the largest singular value squared.
    log_low = math.log(squared_max) - 50.0
    log_high = math.log(squared_max)
    while effective_df_from_singular_values(values, math.exp(log_high)) > target:
        log_high += math.log(10.0)
        if log_high > math.log(np.finfo(np.float64).max) - 2.0:
            raise RuntimeError("Could not bracket Ridge alpha for target effective df")

    for _ in range(int(max_iterations)):
        log_mid = 0.5 * (log_low + log_high)
        alpha = math.exp(log_mid)
        achieved = effective_df_from_singular_values(values, alpha)
        if abs(achieved - target) <= absolute_tolerance:
            return float(alpha), float(achieved), rank
        # d_eff decreases as alpha increases.
        if achieved > target:
            log_low = log_mid
        else:
            log_high = log_mid

    alpha = math.exp(0.5 * (log_low + log_high))
    achieved = effective_df_from_singular_values(values, alpha)
    return float(alpha), float(achieved), rank


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _materialize_feature_matrix(
    dataset: Any,
    indices: np.ndarray,
    std_mV: float | np.ndarray,
    *,
    chunk_size: int,
) -> np.ndarray:
    selected = np.asarray(indices, dtype=np.int64)
    feature_count = int(dataset.windows_mV.shape[-1])
    output = np.empty((selected.size, feature_count), dtype=np.float64)
    scale = np.asarray(std_mV, dtype=np.float64)
    if scale.ndim == 1 and scale.size != feature_count:
        raise ValueError(
            f"Normalization length {scale.size} does not match {feature_count} features"
        )
    if scale.ndim not in {0, 1} or np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("Invalid fold normalization scale")

    cursor = 0
    for start in range(0, selected.size, int(chunk_size)):
        block_indices = selected[start : start + int(chunk_size)]
        pair = np.asarray(dataset.windows_mV[block_indices], dtype=np.float64)
        size = int(block_indices.size)
        # The mean is shared by both detector channels and therefore cancels in
        # the antisymmetric detector difference, exactly as in the pipeline's
        # linear_regression implementation.
        output[cursor : cursor + size] = (pair[:, 0, :] - pair[:, 1, :]) / scale
        cursor += size
    return output


def _fit_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float,
    *,
    solver: str,
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, float]:
    try:
        from sklearn.linear_model import LinearRegression, Ridge
    except ImportError as exc:
        raise RuntimeError(
            "Matched-effective Ridge requires scikit-learn. Install it with "
            "'python -m pip install scikit-learn'."
        ) from exc

    if float(alpha) == 0.0:
        estimator = LinearRegression(fit_intercept=False)
    else:
        estimator = Ridge(
            alpha=float(alpha),
            fit_intercept=False,
            solver=str(solver),
            tol=float(tolerance),
            max_iter=int(max_iterations),
        )
    estimator.fit(x_train, y_train)
    coefficient = np.asarray(estimator.coef_, dtype=np.float64).reshape(-1)
    if coefficient.size != x_train.shape[1]:
        raise RuntimeError("Ridge estimator returned an unexpected coefficient shape")
    # Match the repository's linear model: fit slopes without an intercept, then
    # enforce zero arithmetic training residual bias exactly.
    pair_bias_ps = float(np.mean(y_train - x_train @ coefficient))
    return coefficient, pair_bias_ps


def _prediction_metrics(
    *,
    dataset: Any,
    indices: np.ndarray,
    features: np.ndarray,
    coefficient: np.ndarray,
    pair_bias_ps: float,
    fit_config: dict[str, Any],
    loss: dict[str, Any],
    target_scale_ps: float,
) -> dict[str, float]:
    selected = np.asarray(indices, dtype=np.int64)
    target = factored_correction_target_ps(dataset, selected)
    prediction = features @ coefficient + float(pair_bias_ps)
    residual = target - prediction
    true_tof = float(dataset.true_tof_ps)
    corrected = true_tof + residual
    true_values = np.full(selected.size, true_tof, dtype=np.float64)
    baseline = _delta_ps(dataset, "prepared_led", selected)
    return _metrics(
        corrected,
        true_values,
        baseline,
        fit_config,
        loss,
        target_scale_ps,
    )


def _load_existing_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    return pd.read_csv(path).to_dict(orient="records")


def _summary_frame(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    group_columns = [
        "root_id",
        "root_file",
        "file_name",
        "voltage_V",
        "channel_mode",
        "input_waveforms",
        "target",
        "input_transform",
        "window_id",
        "window_start_ns",
        "window_end_ns",
        "window_length_ns",
        "target_slope_effective_df",
        "target_total_effective_df",
        "feature_count",
    ]
    metric_columns = [
        "alpha",
        "actual_slope_effective_df",
        "actual_total_effective_df",
        "effective_df_error",
        "design_rank",
        "train_event_count",
        "validation_event_count",
        "blind_event_count",
        "coefficient_l1_norm",
        "coefficient_l2_norm",
        "pair_bias_ps",
        "validation_loss",
        "validation_rmse_ps",
        "validation_bias_ps",
        "validation_ctr_ps",
        "validation_baseline_ctr_ps",
        "validation_relative_improvement_pct",
        "validation_ctr_minus_led_ps",
        "blind_loss",
        "blind_rmse_ps",
        "blind_bias_ps",
        "blind_ctr_ps",
        "blind_baseline_ctr_ps",
        "blind_relative_improvement_pct",
        "blind_ctr_minus_led_ps",
        "runtime_seconds",
    ]

    records: list[dict[str, Any]] = []
    for keys, group in rows.groupby(group_columns, dropna=False, sort=True):
        base = dict(zip(group_columns, keys if isinstance(keys, tuple) else (keys,)))
        base["n_folds"] = int(group["fold_id"].nunique())
        for column in metric_columns:
            values = pd.to_numeric(group[column], errors="coerce").dropna().to_numpy(float)
            if values.size == 0:
                mean = std = sem = float("nan")
            else:
                mean = float(np.mean(values))
                std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
                sem = float(std / math.sqrt(values.size)) if values.size > 1 else 0.0
            base[f"{column}_mean"] = mean
            base[f"{column}_std"] = std
            base[f"{column}_sem"] = sem
        records.append(base)
    return pd.DataFrame.from_records(records)


def _safe_name(value: object) -> str:
    text = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in str(value)
    ).strip("_")
    return text or "result"


def _plot_group(group: pd.DataFrame, destination: Path, *, split: str, dpi: int) -> None:
    if group.empty:
        return
    metric = f"{split}_ctr_minus_led_ps_mean"
    sem = f"{split}_ctr_minus_led_ps_sem"
    if metric not in group or not np.any(np.isfinite(pd.to_numeric(group[metric], errors="coerce"))):
        return

    file_name = str(group.iloc[0]["file_name"])
    mode = str(group.iloc[0]["channel_mode"])
    transform = str(group.iloc[0]["input_transform"])

    figure, axis = plt.subplots(figsize=(9.0, 5.5))
    for target_df, series in group.groupby("target_slope_effective_df", sort=True):
        series = series.sort_values(["window_length_ns", "window_start_ns", "window_end_ns"])
        axis.errorbar(
            series["window_length_ns"],
            series[metric],
            yerr=series[sem],
            marker="o",
            capsize=3,
            label=f"slope d_eff={float(target_df):g} (total={float(target_df)+1:g})",
        )
        for _, row in series.iterrows():
            axis.annotate(
                str(row["window_id"]),
                (float(row["window_length_ns"]), float(row[metric])),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )
    axis.axhline(0.0, linestyle="--", linewidth=1.0, label="same CTR as LED")
    axis.set_xlabel("Window length [ns]")
    axis.set_ylabel(f"{split.capitalize()} CTR − LED CTR [ps]")
    axis.set_title(
        f"Matched-effective Ridge vs LED\n{file_name} | {mode} | {transform}"
    )
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)


def _plot_alpha(group: pd.DataFrame, destination: Path, *, dpi: int) -> None:
    if group.empty:
        return
    figure, axis = plt.subplots(figsize=(9.0, 5.5))
    for target_df, series in group.groupby("target_slope_effective_df", sort=True):
        series = series.sort_values(["window_length_ns", "window_start_ns", "window_end_ns"])
        axis.plot(
            series["window_length_ns"],
            series["alpha_mean"],
            marker="o",
            label=f"slope d_eff={float(target_df):g}",
        )
    axis.set_yscale("log")
    axis.set_xlabel("Window length [ns]")
    axis.set_ylabel("Fold-mean matched Ridge alpha")
    axis.set_title(
        "Penalty required to keep effective complexity fixed\n"
        f"{group.iloc[0]['file_name']} | {group.iloc[0]['channel_mode']} | "
        f"{group.iloc[0]['input_transform']}"
    )
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)


def _make_plots(summary: pd.DataFrame, output: Path, dpi: int) -> None:
    plot_root = output / "plots"
    if summary.empty:
        return
    for (file_name, mode, transform), group in summary.groupby(
        ["file_name", "channel_mode", "input_transform"], sort=True
    ):
        stem = "__".join(map(_safe_name, (file_name, mode, transform)))
        _plot_group(group, plot_root / f"{stem}__validation_delta_ctr.png", split="validation", dpi=dpi)
        if np.any(np.isfinite(pd.to_numeric(group["blind_ctr_minus_led_ps_mean"], errors="coerce"))):
            _plot_group(group, plot_root / f"{stem}__blind_delta_ctr.png", split="blind", dpi=dpi)
        _plot_alpha(group, plot_root / f"{stem}__matched_alpha.png", dpi=dpi)


def _filtered_values(requested: list[str] | None, available: list[str], label: str) -> list[str]:
    if not requested:
        return list(available)
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"Unknown {label}: {unknown}; available: {available}")
    requested_set = set(requested)
    return [value for value in available if value in requested_set]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test window-size effects with Ridge models matched by training-design "
            "effective degrees of freedom. This is an ad-hoc results-only script; "
            "it does not modify or run the main ML model grid."
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="Existing study JSON configuration")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/adhoc/matched_effective_ridge"),
    )
    parser.add_argument(
        "--effective-df",
        type=float,
        nargs="+",
        default=[5.0, 10.0, 20.0],
        help="Target slope effective degrees of freedom (bias adds +1 to total)",
    )
    parser.add_argument(
        "--input-transform",
        default="normalize",
        choices=sorted(SUPPORTED_INPUT_TRANSFORMS),
        help="Hold this representation fixed while varying physical window",
    )
    parser.add_argument("--file", action="append", dest="files", help="ROOT filename/path; repeatable")
    parser.add_argument(
        "--channel-mode",
        action="append",
        dest="channel_modes",
        choices=sorted(CHANNEL_MODES),
        help="Channel mode; repeatable. Default: all configured modes",
    )
    parser.add_argument("--window-id", action="append", dest="window_ids", help="Window ID; repeatable")
    parser.add_argument("--skip-blind", action="store_true")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--rebuild-preprocessing", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--ridge-solver", default="svd")
    parser.add_argument("--tolerance", type=float, default=1.0e-8)
    parser.add_argument("--max-iterations", type=int, default=100000)
    parser.add_argument("--df-tolerance", type=float, default=1.0e-7)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    target_dfs = sorted(set(float(value) for value in args.effective_df))
    if not target_dfs or any(not math.isfinite(value) or value <= 0.0 for value in target_dfs):
        raise ValueError("--effective-df values must be finite and positive")

    config = load_study_config(args.config, PROJECT)
    output = args.output_dir if args.output_dir.is_absolute() else PROJECT / args.output_dir
    output = output.resolve()
    if args.restart and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output / "matched_effective_ridge.log", config.get("logging", {}).get("level", "INFO"))

    transform = normalize_input_transform(args.input_transform)
    root_files = [Path(value) for value in config["root_files"]]
    if args.files:
        requested = set(args.files)
        root_files = [
            path for path in root_files
            if str(path) in requested or path.name in requested or path.stem in requested
        ]
        if not root_files:
            raise ValueError(f"None of --file {sorted(requested)} matched configured ROOT files")
    modes = _filtered_values(args.channel_modes, config["channel_modes"], "channel modes")
    window_ids = [str(window["id"]) for window in config["windows_ns"]]
    selected_window_ids = _filtered_values(args.window_ids, window_ids, "window IDs")
    windows = [window for window in config["windows_ns"] if str(window["id"]) in selected_window_ids]

    resolved = {
        "script": "run_matched_effective_ridge.py",
        "format_version": 1,
        "source_study_config": str(Path(args.config).resolve()),
        "source_study_config_hash": config["_config_hash"],
        "root_files": [str(path) for path in root_files],
        "channel_modes": modes,
        "windows": windows,
        "input_transform": transform,
        "target_slope_effective_df": target_dfs,
        "target_total_effective_df": [value + 1.0 for value in target_dfs],
        "skip_blind": bool(args.skip_blind),
        "ridge_solver": str(args.ridge_solver),
        "tolerance": float(args.tolerance),
        "max_iterations": int(args.max_iterations),
        "df_tolerance": float(args.df_tolerance),
        "cross_validation": config["cross_validation"],
        "selection": config["selection"],
        "fit": config["fit"],
        "split": config.get("split", {}),
    }
    resolved["fingerprint"] = canonical_hash(resolved)
    resolved_path = output / "resolved_matched_effective_ridge.json"
    if resolved_path.is_file() and not args.restart:
        previous = json.loads(resolved_path.read_text(encoding="utf-8"))
        if previous.get("fingerprint") != resolved["fingerprint"]:
            raise RuntimeError(
                "Existing ad-hoc output was created with different settings. "
                "Use --restart or choose another --output-dir."
            )
    atomic_json(resolved_path, resolved)

    results_path = output / "fold_results.csv"
    rows = _load_existing_rows(results_path)
    completed_keys = {str(row.get("row_key", "")) for row in rows}
    logger.info(
        "Matched-effective Ridge start | files=%d | modes=%d | windows=%d | d_eff=%s | transform=%s",
        len(root_files), len(modes), len(windows), target_dfs, transform,
    )

    common_loss = {"id": "mse", "type": "mse"}
    for file_position, root_file in enumerate(root_files, start=1):
        root_id = _root_id(root_file)
        logger.info("File %d/%d | %s", file_position, len(root_files), root_file.name)
        development, blind = _ensure_preprocessed(
            config,
            root_file,
            root_id,
            output,
            rebuild=bool(args.rebuild_preprocessing),
            logger=logger,
        )
        voltage = _extract_voltage(root_file, config.get("reporting", {}))

        for mode_id in modes:
            mode = CHANNEL_MODES[mode_id]
            folds = _fold_masks(
                development,
                blind,
                mode["target"],
                config["cross_validation"],
                config["selection"],
            )
            logger.info("Mode %s | folds=%d", mode_id, len(folds))

            for window_position, window in enumerate(windows, start=1):
                logger.info(
                    "Window %d/%d | %s [%.3f, %.3f] ns",
                    window_position,
                    len(windows),
                    window["id"],
                    window["start_ns"],
                    window["end_ns"],
                )
                development_view = prediction_window_dataset_view(
                    development,
                    input_waveforms=mode["input_waveforms"],
                    target=mode["target"],
                    before_ns=float(window["before_ns"]),
                    after_ns=float(window["after_ns"]),
                )
                blind_view = prediction_window_dataset_view(
                    blind,
                    input_waveforms=mode["input_waveforms"],
                    target=mode["target"],
                    before_ns=float(window["before_ns"]),
                    after_ns=float(window["after_ns"]),
                )
                transform_cache = output / "transform_cache" / root_id / mode_id / str(window["id"])
                transformed_development, _ = materialize_training_input_cache(
                    development_view,
                    transform,
                    transform_cache / "development",
                    chunk_size=int(args.chunk_size),
                    rebuild=False,
                    logger=logger,
                )
                transformed_blind, _ = materialize_training_input_cache(
                    blind_view,
                    transform,
                    transform_cache / "blind",
                    chunk_size=int(args.chunk_size),
                    rebuild=False,
                    logger=logger,
                )

                for fold in folds:
                    fold_id = int(fold["fold_id"])
                    train_indices = np.asarray(fold["train"], dtype=np.int64)
                    validation_indices = np.asarray(fold["validation"], dtype=np.int64)
                    blind_indices = np.asarray(fold["blind"], dtype=np.int64)
                    normalization = compute_normalization(
                        [(transformed_development, train_indices)],
                        chunk_size=int(args.chunk_size),
                        featurewise=transform == INPUT_TRANSFORM_NORMALIZE,
                    )
                    x_train = _materialize_feature_matrix(
                        transformed_development,
                        train_indices,
                        normalization.std_mV,
                        chunk_size=int(args.chunk_size),
                    )
                    x_validation = _materialize_feature_matrix(
                        transformed_development,
                        validation_indices,
                        normalization.std_mV,
                        chunk_size=int(args.chunk_size),
                    )
                    x_blind = (
                        None
                        if args.skip_blind
                        else _materialize_feature_matrix(
                            transformed_blind,
                            blind_indices,
                            normalization.std_mV,
                            chunk_size=int(args.chunk_size),
                        )
                    )
                    y_train = factored_correction_target_ps(
                        transformed_development, train_indices
                    )
                    singular_values = np.linalg.svd(x_train, compute_uv=False)
                    target_scale = max(float(np.std(y_train, ddof=0)), 1.0e-8)

                    for target_df in target_dfs:
                        row_key = canonical_hash(
                            {
                                "fingerprint": resolved["fingerprint"],
                                "root_id": root_id,
                                "mode": mode_id,
                                "window": window["id"],
                                "fold": fold_id,
                                "target_df": target_df,
                            }
                        )[:24]
                        if row_key in completed_keys:
                            continue
                        started = time.time()
                        try:
                            alpha, achieved_df, rank = alpha_for_target_effective_df(
                                singular_values,
                                target_df,
                                matrix_shape=x_train.shape,
                                absolute_tolerance=float(args.df_tolerance),
                            )
                        except ValueError as exc:
                            raise RuntimeError(
                                f"Cannot match slope d_eff={target_df:g} for "
                                f"{root_file.name}/{mode_id}/{window['id']}/fold {fold_id}: {exc}. "
                                "Choose targets no larger than the minimum design rank across all windows."
                            ) from exc
                        coefficient, pair_bias = _fit_ridge(
                            x_train,
                            y_train,
                            alpha,
                            solver=args.ridge_solver,
                            tolerance=float(args.tolerance),
                            max_iterations=int(args.max_iterations),
                        )
                        validation_metrics = _prediction_metrics(
                            dataset=transformed_development,
                            indices=validation_indices,
                            features=x_validation,
                            coefficient=coefficient,
                            pair_bias_ps=pair_bias,
                            fit_config=config["fit"],
                            loss=common_loss,
                            target_scale_ps=target_scale,
                        )
                        if args.skip_blind:
                            blind_metrics = {
                                "loss": float("nan"),
                                "rmse_ps": float("nan"),
                                "bias_ps": float("nan"),
                                "ctr_ps": float("nan"),
                                "baseline_ctr_ps": float("nan"),
                                "relative_improvement_pct": float("nan"),
                            }
                        else:
                            assert x_blind is not None
                            blind_metrics = _prediction_metrics(
                                dataset=transformed_blind,
                                indices=blind_indices,
                                features=x_blind,
                                coefficient=coefficient,
                                pair_bias_ps=pair_bias,
                                fit_config=config["fit"],
                                loss=common_loss,
                                target_scale_ps=target_scale,
                            )
                        robust = fold["robust"]
                        row = {
                            "row_key": row_key,
                            "root_id": root_id,
                            "root_file": str(root_file),
                            "file_name": root_file.name,
                            "voltage_V": voltage,
                            "channel_mode": mode_id,
                            "input_waveforms": mode["input_waveforms"],
                            "target": mode["target"],
                            "input_transform": transform,
                            "window_id": window["id"],
                            "window_start_ns": float(window["start_ns"]),
                            "window_end_ns": float(window["end_ns"]),
                            "window_length_ns": float(window["end_ns"] - window["start_ns"]),
                            "fold_id": fold_id,
                            "target_slope_effective_df": float(target_df),
                            "target_total_effective_df": float(target_df + 1.0),
                            "actual_slope_effective_df": float(achieved_df),
                            "actual_total_effective_df": float(achieved_df + 1.0),
                            "effective_df_error": float(achieved_df - target_df),
                            "alpha": float(alpha),
                            "design_rank": int(rank),
                            "feature_count": int(x_train.shape[1]),
                            "train_event_count": int(train_indices.size),
                            "validation_event_count": int(validation_indices.size),
                            "blind_event_count": 0 if args.skip_blind else int(blind_indices.size),
                            "coefficient_l1_norm": float(np.linalg.norm(coefficient, ord=1)),
                            "coefficient_l2_norm": float(np.linalg.norm(coefficient, ord=2)),
                            "pair_bias_ps": float(pair_bias),
                            "validation_loss": validation_metrics["loss"],
                            "validation_rmse_ps": validation_metrics["rmse_ps"],
                            "validation_bias_ps": validation_metrics["bias_ps"],
                            "validation_ctr_ps": validation_metrics["ctr_ps"],
                            "validation_baseline_ctr_ps": validation_metrics["baseline_ctr_ps"],
                            "validation_relative_improvement_pct": validation_metrics["relative_improvement_pct"],
                            "validation_ctr_minus_led_ps": validation_metrics["ctr_ps"] - validation_metrics["baseline_ctr_ps"],
                            "blind_loss": blind_metrics["loss"],
                            "blind_rmse_ps": blind_metrics["rmse_ps"],
                            "blind_bias_ps": blind_metrics["bias_ps"],
                            "blind_ctr_ps": blind_metrics["ctr_ps"],
                            "blind_baseline_ctr_ps": blind_metrics["baseline_ctr_ps"],
                            "blind_relative_improvement_pct": blind_metrics["relative_improvement_pct"],
                            "blind_ctr_minus_led_ps": blind_metrics["ctr_ps"] - blind_metrics["baseline_ctr_ps"],
                            "outlier_center_ps": float(robust.center_ps),
                            "outlier_scale_ps": float(robust.scale_ps),
                            "outlier_scale_method": str(robust.method),
                            "outlier_z_threshold": float(fold["z_threshold"]),
                            "runtime_seconds": float(time.time() - started),
                        }
                        rows.append(row)
                        completed_keys.add(row_key)
                        frame = pd.DataFrame.from_records(rows)
                        for column in RESULT_COLUMNS:
                            if column not in frame:
                                frame[column] = np.nan
                        _atomic_csv(results_path, frame[RESULT_COLUMNS])
                        logger.info(
                            "Result | %s | %s | window=%s | fold=%d | slope d_eff=%.3f | "
                            "alpha=%.6g | CV CTR=%.3f ps | LED=%.3f ps | delta=%+.3f ps",
                            root_file.name,
                            mode_id,
                            window["id"],
                            fold_id,
                            achieved_df,
                            alpha,
                            validation_metrics["ctr_ps"],
                            validation_metrics["baseline_ctr_ps"],
                            validation_metrics["ctr_ps"] - validation_metrics["baseline_ctr_ps"],
                        )

    fold_frame = pd.DataFrame.from_records(rows)
    if not fold_frame.empty:
        for column in RESULT_COLUMNS:
            if column not in fold_frame:
                fold_frame[column] = np.nan
        fold_frame = fold_frame[RESULT_COLUMNS].sort_values(
            [
                "file_name",
                "channel_mode",
                "window_length_ns",
                "window_start_ns",
                "target_slope_effective_df",
                "fold_id",
            ]
        )
        _atomic_csv(results_path, fold_frame)
    summary = _summary_frame(fold_frame)
    summary_path = output / "summary.csv"
    _atomic_csv(summary_path, summary)
    _make_plots(summary, output, int(config.get("reporting", {}).get("dpi", 180)))
    logger.info(
        "Complete | fold rows=%d | summary rows=%d | output=%s",
        len(fold_frame),
        len(summary),
        output,
    )


if __name__ == "__main__":
    main()
