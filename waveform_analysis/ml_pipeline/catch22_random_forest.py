from __future__ import annotations

import csv
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from utils.fit import fit_delta_times_integer_fs

from .catch22_features import Catch22FeatureCache, prepare_catch22_feature_cache
from .common import atomic_json, canonical_hash, json_safe, set_global_seed
if TYPE_CHECKING:
    from .data import EnergyCache, SplitData
from .model import model_label, model_output_path, model_type


RF_CHECKPOINT_FORMAT_VERSION = 2


@dataclass
class SharedCatch22RandomForest:
    """Shared single-channel map g built as an additive forest ensemble.

    Each base estimator receives the Catch22 representation of one waveform.
    The pair correction is always g(phi_1) - g(phi_2), so swapping channels
    reverses the sign exactly.
    """

    forests: list[RandomForestRegressor] = field(default_factory=list)
    stage_weights: list[float] = field(default_factory=list)
    max_abs_single_channel_output_ps: float | None = None

    def predict_single(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        output = np.zeros(x.shape[0], dtype=np.float64)
        for weight, forest in zip(self.stage_weights, self.forests, strict=True):
            output += float(weight) * np.asarray(forest.predict(x), dtype=np.float64)
        if self.max_abs_single_channel_output_ps is not None:
            limit = float(self.max_abs_single_channel_output_ps)
            output = np.clip(output, -limit, limit)
        return output

    def predict_pair(self, pair_features: np.ndarray) -> np.ndarray:
        pair = np.asarray(pair_features, dtype=np.float64)
        if pair.ndim != 3 or pair.shape[1] != 2:
            raise ValueError("Expected Catch22 pair features with shape [N, 2, F]")
        g1 = self.predict_single(pair[:, 0, :])
        g2 = self.predict_single(pair[:, 1, :])
        return g1 - g2

    def feature_importances(self, feature_count: int) -> np.ndarray:
        if not self.forests:
            return np.zeros(feature_count, dtype=np.float64)
        total = np.zeros(feature_count, dtype=np.float64)
        weight_sum = 0.0
        for weight, forest in zip(self.stage_weights, self.forests, strict=True):
            importance = np.asarray(forest.feature_importances_, dtype=np.float64)
            total += abs(float(weight)) * importance
            weight_sum += abs(float(weight))
        if weight_sum > 0:
            total /= weight_sum
        return total


def _led_deltas_ps(cache: EnergyCache, indices: np.ndarray) -> np.ndarray:
    values = np.asarray(cache.led_time_fs[indices], dtype=np.int64)
    return (values[:, 0] - values[:, 1]).astype(np.float64) / 1000.0


def _swap_augmentation_config(pipeline_config: dict[str, Any]) -> dict[str, bool]:
    config = pipeline_config.get("channel_swap_augmentation", {})
    return {
        "enabled": bool(config.get("enabled", False)),
        # Random forests have no mini-batches, so this field is recorded for
        # consistency but does not alter fitting.
        "paired_batches": bool(config.get("paired_batches", True)),
    }


def _duplicate_swapped_pairs(
    pair_features: np.ndarray,
    led_values_ps: np.ndarray,
    targets_ps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Append channel-swapped pairs with all signed timing values negated."""
    pair = np.asarray(pair_features)
    led = np.asarray(led_values_ps, dtype=np.float64)
    target = np.asarray(targets_ps, dtype=np.float64)
    swapped_pair = pair[:, [1, 0], :]
    return (
        np.concatenate([pair, swapped_pair], axis=0),
        np.concatenate([led, -led], axis=0),
        np.concatenate([target, -target], axis=0),
    )


def _fit_ctr(values_ps: np.ndarray, method: str, fit_config: dict[str, Any]) -> float:
    values_fs = np.rint(np.asarray(values_ps, dtype=np.float64) * 1000.0).astype(
        np.int64
    )
    result = fit_delta_times_integer_fs(
        values_fs,
        method=method,
        parameter=0.0,
        n_total=int(values_fs.size),
        n_selected=int(values_fs.size),
        config=fit_config,
    )
    return float(result.ctr_ps) if result.success else np.nan


def _atomic_joblib_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, temporary, compress=3)
    os.replace(temporary, path)


def _load_joblib(path: Path) -> dict[str, Any]:
    value = joblib.load(path)
    if not isinstance(value, dict):
        raise RuntimeError(f"Invalid random-forest checkpoint: {path}")
    return value


def _write_history(path: Path, history: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if not history:
        temporary.write_text("", encoding="utf-8")
    else:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows([json_safe(row) for row in history])
    os.replace(temporary, path)


def _rf_kwargs(model_config: dict[str, Any], *, stage_seed: int) -> dict[str, Any]:
    config = model_config["random_forest"]
    return {
        "criterion": str(config.get("criterion", "squared_error")),
        "max_depth": (
            None if config.get("max_depth") is None else int(config["max_depth"])
        ),
        "min_samples_split": config.get("min_samples_split", 2),
        "min_samples_leaf": config.get("min_samples_leaf", 1),
        "max_features": config.get("max_features", 1.0),
        "bootstrap": bool(config.get("bootstrap", True)),
        "oob_score": bool(config.get("oob_score", False)),
        "n_jobs": int(config.get("n_jobs", -1)),
        "random_state": int(stage_seed),
        "verbose": int(config.get("verbose", 0)),
        "warm_start": True,
        "max_samples": config.get("max_samples"),
    }


def _checkpoint_payload(
    *,
    model: SharedCatch22RandomForest,
    partial_forest: RandomForestRegressor | None,
    partial_stage_index: int | None,
    history: list[dict[str, Any]],
    best_value: float,
    bad_stages: int,
    context: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": RF_CHECKPOINT_FORMAT_VERSION,
        "model": model,
        "partial_forest": partial_forest,
        "partial_stage_index": partial_stage_index,
        "history": history,
        "best_value": float(best_value),
        "bad_stages": int(bad_stages),
        "context": context,
    }


def _predict_metrics(
    model: SharedCatch22RandomForest,
    features: np.ndarray,
    led_values: np.ndarray,
    target: np.ndarray,
    *,
    symmetric_objective: bool = False,
) -> dict[str, Any]:
    correction = model.predict_pair(features)
    error = correction - target
    corrected = led_values - correction
    std_loss_ps = float(np.std(error, ddof=0))
    rmse_loss_ps = float(np.sqrt(np.mean(error * error)))
    effective_loss_ps = rmse_loss_ps if symmetric_objective else std_loss_ps
    return {
        "loss": effective_loss_ps,
        "residual_rmse_ps": rmse_loss_ps,
        "correction_ps": correction,
        "corrected_ps": corrected,
        "corrected_std_ps": std_loss_ps,
        "correction_mean_ps": float(np.mean(correction)),
        "residual_mean_ps": float(np.mean(error)),
    }


def _plot_history(
    history: list[dict[str, Any]], output_dir: Path, *, dpi: int
) -> None:
    if not history:
        return
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    stages = [int(row["stage"]) for row in history]
    for train_key, validation_key, ylabel, filename in (
        (
            "train_loss",
            "validation_loss",
            "Optimization metric [ps]",
            "loss_vs_stage.png",
        ),
        (
            "train_corrected_std_ps",
            "validation_corrected_std_ps",
            "Corrected standard deviation [ps]",
            "std_vs_stage.png",
        ),
        ("train_ctr_ps", "validation_ctr_ps", "Gaussian CTR [ps]", "ctr_vs_stage.png"),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(stages, [row[train_key] for row in history], marker="o", label="Training")
        ax.plot(
            stages,
            [row[validation_key] for row in history],
            marker="o",
            label="Validation",
        )
        ax.set_xlabel("Random-forest residual stage")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=dpi)
        plt.close(fig)


def _write_feature_importance(
    model: SharedCatch22RandomForest,
    feature_cache: Catch22FeatureCache,
    output_dir: Path,
    *,
    dpi: int,
) -> None:
    importance = model.feature_importances(len(feature_cache.feature_names))
    order = np.argsort(importance)[::-1]
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "feature_importance.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["rank", "feature", "short_name", "importance"],
        )
        writer.writeheader()
        for rank, index in enumerate(order, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "feature": feature_cache.feature_names[int(index)],
                    "short_name": feature_cache.short_names[int(index)],
                    "importance": float(importance[int(index)]),
                }
            )

    import matplotlib.pyplot as plt

    top = order[: min(15, order.size)][::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        [feature_cache.short_names[int(index)] for index in top],
        [importance[int(index)] for index in top],
    )
    ax.set_xlabel("Mean random-forest feature importance")
    ax.set_ylabel("Catch22 feature")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importance.png", dpi=dpi)
    plt.close(fig)


def train_catch22_random_forest(
    cache: EnergyCache,
    splits: SplitData,
    pipeline_config: dict[str, Any],
    model_config: dict[str, Any],
    *,
    resume: bool,
    restart: bool,
    logger: Any,
) -> dict[str, Any]:
    if model_type(model_config) != "catch22_random_forest":
        raise ValueError("Expected model_type='catch22_random_forest'")

    checkpoint_dir = model_output_path(pipeline_config, "checkpoint_dir", model_config)
    plot_dir = model_output_path(pipeline_config, "plot_dir", model_config) / "training"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if restart:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(plot_dir, ignore_errors=True)

    last_path = checkpoint_dir / "last.joblib"
    interrupted_path = checkpoint_dir / "interrupted.joblib"
    best_path = checkpoint_dir / "best_validation.joblib"
    if not resume and not restart and (last_path.exists() or best_path.exists()):
        raise RuntimeError(
            "Random-forest checkpoints already exist. Use --resume or --restart."
        )

    seed = int(model_config["training"].get("seed", 12345))
    set_global_seed(seed)
    feature_cache = prepare_catch22_feature_cache(
        cache, splits, model_config, logger=logger
    )
    train_features_canonical = np.asarray(
        feature_cache.features[splits.train], dtype=np.float64
    )
    validation_features = np.asarray(
        feature_cache.features[splits.validation], dtype=np.float64
    )
    train_led_canonical = _led_deltas_ps(cache, splits.train)
    validation_led = _led_deltas_ps(cache, splits.validation)
    led_center_ps = float(np.mean(train_led_canonical))
    train_target_canonical = train_led_canonical - led_center_ps
    validation_target = validation_led - led_center_ps
    augmentation = _swap_augmentation_config(pipeline_config)
    if augmentation["enabled"]:
        train_features, train_led, train_target = _duplicate_swapped_pairs(
            train_features_canonical,
            train_led_canonical,
            train_target_canonical,
        )
    else:
        train_features = train_features_canonical
        train_led = train_led_canonical
        train_target = train_target_canonical
    logger.info(
        "Channel-swap training augmentation | enabled=%s | canonical events=%d | "
        "optimization pairs=%d | validation/test duplicated=False",
        augmentation["enabled"],
        int(train_features_canonical.shape[0]),
        int(train_features.shape[0]),
    )

    training_config = model_config["training"]
    stage_count = int(training_config.get("stages", 3))
    stage_learning_rate = float(training_config.get("stage_learning_rate", 0.5))
    patience = int(training_config.get("early_stopping_patience", 2))
    min_delta = float(training_config.get("early_stopping_min_delta", 0.0))
    monitor = str(training_config.get("monitor", "validation_loss"))
    max_abs = training_config.get("max_abs_single_channel_output_ps")
    max_abs = None if max_abs is None else float(max_abs)
    n_estimators = int(model_config["random_forest"].get("n_estimators", 300))
    tree_batch = max(
        1,
        int(model_config["checkpointing"].get("every_trees", n_estimators)),
    )

    context = {
        "pipeline_config_hash": pipeline_config["_config_hash"],
        "model_config_hash": model_config["_config_hash"],
        "dataset_fingerprint": cache.manifest["fingerprint"],
        "split_fingerprint": splits.manifest["fingerprint"],
        "feature_cache_fingerprint": feature_cache.manifest["fingerprint"],
        "training_context_fingerprint": canonical_hash(
            {
                "pipeline": pipeline_config["_config_hash"],
                "model": model_config["_config_hash"],
                "dataset": cache.manifest["fingerprint"],
                "split": splits.manifest["fingerprint"],
                "features": feature_cache.manifest["fingerprint"],
            }
        ),
        "led_center_ps": led_center_ps,
        "model_type": model_type(model_config),
        "model_label": model_label(model_config),
        "estimator": (
            "y_theta(s1,s2)=g_theta(catch22_or_catch24(s1))"
            "-g_theta(catch22_or_catch24(s2))"
        ),
        "objective": (
            "paired symmetric residual standard deviation (equal to canonical "
            "residual RMSE)" if augmentation["enabled"] else
            "sqrt(pairwise residual variance), i.e. residual standard deviation in ps"
        ),
        "training_method": (
            "staged residual fitting; each shared random forest is trained on "
            "+residual/2 for channel 1 and -residual/2 for channel 2"
        ),
        "feature_names": feature_cache.feature_names,
        "feature_short_names": feature_cache.short_names,
        "catch24": bool(model_config["features"].get("catch24", True)),
        "catch22_feature_scope": feature_cache.manifest.get("selection_scope"),
        "catch22_selected_event_count": int(
            feature_cache.manifest.get("selected_event_count", 0)
        ),
        "test_used_during_training": False,
        "channel_swap_augmentation": {
            "enabled": augmentation["enabled"],
            "training_only": True,
            "paired_batches": augmentation["paired_batches"],
            "canonical_training_events": int(train_features_canonical.shape[0]),
            "optimization_pairs": int(train_features.shape[0]),
            "validation_and_test_duplicated": False,
        },
    }
    atomic_json(checkpoint_dir / "training_context.json", context)

    model = SharedCatch22RandomForest(
        max_abs_single_channel_output_ps=max_abs
    )
    partial_forest: RandomForestRegressor | None = None
    partial_stage_index: int | None = None
    history: list[dict[str, Any]] = []
    best_value = float("inf")
    bad_stages = 0

    if resume:
        candidates = [path for path in (last_path, interrupted_path) if path.is_file()]
        if not candidates:
            raise FileNotFoundError("No resumable random-forest checkpoint was found")
        resume_path = max(candidates, key=lambda path: path.stat().st_mtime_ns)
        checkpoint = _load_joblib(resume_path)
        if checkpoint["context"]["training_context_fingerprint"] != context[
            "training_context_fingerprint"
        ]:
            raise RuntimeError("Random-forest checkpoint does not match current inputs")
        model = checkpoint["model"]
        partial_forest = checkpoint.get("partial_forest")
        partial_stage_index = checkpoint.get("partial_stage_index")
        history = list(checkpoint.get("history", []))
        best_value = float(checkpoint.get("best_value", np.inf))
        bad_stages = int(checkpoint.get("bad_stages", 0))
        logger.info(
            "Resuming catch22 random forest: %d completed stages%s",
            len(model.forests),
            " with a partial active stage" if partial_forest is not None else "",
        )

    started = time.time()
    current_partial: RandomForestRegressor | None = partial_forest
    current_stage_index: int | None = partial_stage_index
    try:
        for stage_index in range(len(model.forests), stage_count):
            if bad_stages >= patience > 0:
                break
            residual = train_target - model.predict_pair(train_features)
            # The external objective is calibration-invariant standard deviation.
            # Remove the current mean residual so each stage learns only event-wise
            # variation and cannot spend capacity on a constant calibration offset.
            residual = residual - float(np.mean(residual))
            x_stage = np.concatenate(
                [train_features[:, 0, :], train_features[:, 1, :]], axis=0
            )
            y_stage = np.concatenate([0.5 * residual, -0.5 * residual], axis=0)

            if current_partial is not None:
                if current_stage_index != stage_index:
                    raise RuntimeError("Partial forest belongs to a different stage")
                forest = current_partial
                current_tree_count = int(forest.n_estimators)
            else:
                forest = RandomForestRegressor(
                    n_estimators=min(tree_batch, n_estimators),
                    **_rf_kwargs(model_config, stage_seed=seed + stage_index),
                )
                current_tree_count = 0

            while current_tree_count < n_estimators:
                next_tree_count = min(
                    n_estimators,
                    current_tree_count + tree_batch,
                )
                forest.set_params(n_estimators=next_tree_count)
                forest.fit(x_stage, y_stage)
                current_tree_count = next_tree_count
                current_partial = forest
                current_stage_index = stage_index
                payload = _checkpoint_payload(
                    model=model,
                    partial_forest=forest,
                    partial_stage_index=stage_index,
                    history=history,
                    best_value=best_value,
                    bad_stages=bad_stages,
                    context=context,
                )
                _atomic_joblib_dump(last_path, payload)
                logger.info(
                    "Catch22 RF stage %d/%d | trees %d/%d",
                    stage_index + 1,
                    stage_count,
                    current_tree_count,
                    n_estimators,
                )

            model.forests.append(forest)
            model.stage_weights.append(stage_learning_rate)
            current_partial = None
            current_stage_index = None

            train_metrics = _predict_metrics(
                model,
                train_features_canonical,
                train_led_canonical,
                train_target_canonical,
                symmetric_objective=augmentation["enabled"],
            )
            validation_metrics = _predict_metrics(
                model,
                validation_features,
                validation_led,
                validation_target,
                symmetric_objective=augmentation["enabled"],
            )
            train_ctr = _fit_ctr(
                train_metrics["corrected_ps"],
                "Train LED corrected by catch22 RF",
                pipeline_config["fit"],
            )
            validation_ctr = _fit_ctr(
                validation_metrics["corrected_ps"],
                "Validation LED corrected by catch22 RF",
                pipeline_config["fit"],
            )
            row = {
                "stage": stage_index + 1,
                "total_trees": int((stage_index + 1) * n_estimators),
                "train_loss": float(train_metrics["loss"]),
                "validation_loss": float(validation_metrics["loss"]),
                "train_residual_rmse_ps": float(train_metrics["residual_rmse_ps"]),
                "validation_residual_rmse_ps": float(
                    validation_metrics["residual_rmse_ps"]
                ),
                "train_corrected_std_ps": float(
                    train_metrics["corrected_std_ps"]
                ),
                "validation_corrected_std_ps": float(
                    validation_metrics["corrected_std_ps"]
                ),
                "train_ctr_ps": float(train_ctr),
                "validation_ctr_ps": float(validation_ctr),
                "train_correction_mean_ps": float(
                    train_metrics["correction_mean_ps"]
                ),
                "validation_correction_mean_ps": float(
                    validation_metrics["correction_mean_ps"]
                ),
                "train_residual_mean_ps": float(
                    train_metrics["residual_mean_ps"]
                ),
                "validation_residual_mean_ps": float(
                    validation_metrics["residual_mean_ps"]
                ),
                "elapsed_seconds": float(time.time() - started),
            }
            history.append(row)
            _write_history(checkpoint_dir / "training_history.csv", history)
            logger.info(
                "%s stage %d/%d complete | train %s %.3f ps | "
                "val %s %.3f ps | val σ %.3f ps | val residual mean %.3f ps | "
                "val CTR %s ps",
                model_label(model_config),
                stage_index + 1,
                stage_count,
                "symmetric std/RMSE" if augmentation["enabled"] else "std loss",
                row["train_loss"],
                "symmetric std/RMSE" if augmentation["enabled"] else "std loss",
                row["validation_loss"],
                row["validation_corrected_std_ps"],
                row["validation_residual_mean_ps"],
                "nan" if not np.isfinite(validation_ctr) else f"{validation_ctr:.3f}",
            )

            candidate = row.get(monitor, np.nan)
            improved = bool(
                np.isfinite(candidate) and float(candidate) < best_value - min_delta
            )
            if improved:
                best_value = float(candidate)
                bad_stages = 0
            else:
                bad_stages += 1

            payload = _checkpoint_payload(
                model=model,
                partial_forest=None,
                partial_stage_index=None,
                history=history,
                best_value=best_value,
                bad_stages=bad_stages,
                context=context,
            )
            _atomic_joblib_dump(last_path, payload)
            if improved:
                _atomic_joblib_dump(best_path, payload)
                logger.info("Saved best catch22 RF checkpoint: %s", best_path)
    except KeyboardInterrupt:
        payload = _checkpoint_payload(
            model=model,
            partial_forest=current_partial,
            partial_stage_index=current_stage_index,
            history=history,
            best_value=best_value,
            bad_stages=bad_stages,
            context=context,
        )
        _atomic_joblib_dump(interrupted_path, payload)
        logger.warning(
            "Random-forest training interrupted; checkpoint saved to %s",
            interrupted_path,
        )
        raise

    if not best_path.is_file() and last_path.is_file():
        shutil.copy2(last_path, best_path)
    dpi = int(pipeline_config["plotting"].get("dpi", 180))
    _plot_history(history, plot_dir, dpi=dpi)
    importance_model = model
    if best_path.is_file():
        importance_model = _load_joblib(best_path)["model"]
    _write_feature_importance(importance_model, feature_cache, plot_dir, dpi=dpi)
    result = {
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "stages_completed": len(model.forests),
        "best_monitor_value": best_value,
        "monitor": monitor,
        "history": history,
        "context": context,
        "model_type": model_type(model_config),
        "model_label": model_label(model_config),
    }
    atomic_json(checkpoint_dir / "training_summary.json", result)
    return result


def load_catch22_random_forest_checkpoint(
    path: Path,
    *,
    expected_context: dict[str, str] | None = None,
) -> dict[str, Any]:
    checkpoint = _load_joblib(path)
    if int(checkpoint.get("format_version", -1)) != RF_CHECKPOINT_FORMAT_VERSION:
        raise RuntimeError("Unsupported catch22 random-forest checkpoint format")
    if expected_context is not None:
        context = checkpoint["context"]
        for key, value in expected_context.items():
            if context.get(key) != value:
                raise RuntimeError(f"Random-forest checkpoint mismatch for {key}")
    return checkpoint


def _fit_result(values_ps: np.ndarray, method: str, fit_config: dict[str, Any]) -> Any:
    values_fs = np.rint(np.asarray(values_ps, dtype=np.float64) * 1000.0).astype(
        np.int64
    )
    return fit_delta_times_integer_fs(
        values_fs,
        method=method,
        parameter=0.0,
        n_total=int(values_fs.size),
        n_selected=int(values_fs.size),
        config=fit_config,
    )


def _calibrate_training_fit(
    values_ps: np.ndarray,
    *,
    true_tof_ps: float,
    method: str,
    fit_config: dict[str, Any],
) -> tuple[float, Any]:
    result = _fit_result(values_ps, method + " calibration", fit_config)
    if not result.success:
        raise RuntimeError(f"Calibration Gaussian fit failed for {method}: {result.message}")
    return float(result.mean_ps - true_tof_ps), result


def _metric_row(
    method: str,
    fit: Any,
    true_tof_ps: float,
    offset_ps: float,
    calibration_source: str,
) -> dict[str, Any]:
    if not fit.success:
        raise RuntimeError(f"Final Gaussian fit failed for {method}: {fit.message}")
    return {
        "method": method,
        "bias_ps": float(fit.mean_ps - true_tof_ps),
        "ctr_ps": float(fit.ctr_ps),
        "ctr_error_ps": float(fit.ctr_error_ps),
        "fitted_mean_ps": float(fit.mean_ps),
        "fitted_mean_error_ps": float(fit.mean_error_ps),
        "fitted_sigma_ps": float(fit.sigma_ps),
        "calibration_offset_ps": float(offset_ps),
        "calibration_source": calibration_source,
        "true_tof_ps": float(true_tof_ps),
        "chi2": float(fit.chi2),
        "ndof": int(fit.ndof),
        "chi2_ndof": float(fit.chi2_ndof),
        "n_test": int(fit.n_valid),
        "fit_low_ps": float(fit.fit_low_ps),
        "fit_high_ps": float(fit.fit_high_ps),
    }


def evaluate_catch22_random_forest(
    cache: EnergyCache,
    splits: SplitData,
    pipeline_config: dict[str, Any],
    model_config: dict[str, Any],
    *,
    checkpoint_path: Path | None,
    logger: Any,
) -> dict[str, Any]:
    from .plots import plot_method_fit, plot_metric_comparison

    checkpoint_dir = model_output_path(pipeline_config, "checkpoint_dir", model_config)
    checkpoint_path = checkpoint_path or checkpoint_dir / "best_validation.joblib"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Catch22 random-forest checkpoint not found: {checkpoint_path}"
        )

    feature_cache = prepare_catch22_feature_cache(
        cache, splits, model_config, logger=logger
    )
    expected = {
        "pipeline_config_hash": pipeline_config["_config_hash"],
        "model_config_hash": model_config["_config_hash"],
        "dataset_fingerprint": cache.manifest["fingerprint"],
        "split_fingerprint": splits.manifest["fingerprint"],
        "feature_cache_fingerprint": feature_cache.manifest["fingerprint"],
    }
    checkpoint = load_catch22_random_forest_checkpoint(
        checkpoint_path, expected_context=expected
    )
    model: SharedCatch22RandomForest = checkpoint["model"]

    train_features = np.asarray(feature_cache.features[splits.train], dtype=np.float64)
    test_features = np.asarray(feature_cache.features[splits.test], dtype=np.float64)
    train_correction = model.predict_pair(train_features)
    test_correction = model.predict_pair(test_features)
    swapped_correction = model.predict_pair(test_features[:, [1, 0], :])

    train_led = _led_deltas_ps(cache, splits.train)
    test_led = _led_deltas_ps(cache, splits.test)
    train_cfd_values = np.asarray(cache.cfd_time_fs[splits.train], dtype=np.int64)
    test_cfd_values = np.asarray(cache.cfd_time_fs[splits.test], dtype=np.int64)
    train_cfd = (
        train_cfd_values[:, 0] - train_cfd_values[:, 1]
    ).astype(np.float64) / 1000.0
    test_cfd = (
        test_cfd_values[:, 0] - test_cfd_values[:, 1]
    ).astype(np.float64) / 1000.0
    train_corrected = train_led - train_correction
    test_corrected = test_led - test_correction

    true_tof_ps = float(pipeline_config["data"]["true_tof_ps"])
    fit_config = pipeline_config["fit"]
    led_offset, led_calibration_fit = _calibrate_training_fit(
        train_led,
        true_tof_ps=true_tof_ps,
        method="Energy LED standard",
        fit_config=fit_config,
    )
    cfd_offset, cfd_calibration_fit = _calibrate_training_fit(
        train_cfd,
        true_tof_ps=true_tof_ps,
        method="Energy CFD standard",
        fit_config=fit_config,
    )

    corrected_method = f"Energy LED + {model_label(model_config)} correction"
    methods = [
        (
            "Energy LED standard",
            test_led,
            led_offset,
            "training Energy LED Gaussian fit",
        ),
        (
            "Energy CFD standard",
            test_cfd,
            cfd_offset,
            "training Energy CFD Gaussian fit",
        ),
        (
            corrected_method,
            test_corrected,
            led_offset,
            "same frozen Energy LED calibration offset",
        ),
    ]

    metrics: list[dict[str, Any]] = []
    fits: dict[str, Any] = {}
    plot_dir = (
        model_output_path(pipeline_config, "plot_dir", model_config)
        / "final_evaluation"
    )
    dpi = int(pipeline_config["plotting"].get("dpi", 180))
    for method, values, offset, calibration_source in methods:
        calibrated = values - offset
        fit = _fit_result(calibrated, method, fit_config)
        row = _metric_row(
            method, fit, true_tof_ps, offset, calibration_source
        )
        metrics.append(row)
        fits[method] = fit
        filename = (
            method.lower().replace(" ", "_").replace("+", "plus").replace("-", "_")
        )
        plot_method_fit(
            fit,
            plot_dir / f"{filename}_gaussian_fit.png",
            true_tof_ps=true_tof_ps,
            bias_ps=float(row["bias_ps"]),
            dpi=dpi,
        )
        logger.info(
            "%s | bias %.3f ps | CTR %.3f +/- %.3f ps",
            method,
            row["bias_ps"],
            row["ctr_ps"],
            row["ctr_error_ps"],
        )
    plot_metric_comparison(metrics, plot_dir / "method_comparison.png", dpi)

    swapped_raw_led = -test_led
    swapped_corrected = swapped_raw_led - swapped_correction
    swapped_calibrated = swapped_corrected + led_offset
    swapped_true_tof = -true_tof_ps
    swapped_fit = _fit_result(
        swapped_calibrated, f"{corrected_method} (swapped)", fit_config
    )
    if not swapped_fit.success:
        raise RuntimeError(f"Swap-test Gaussian fit failed: {swapped_fit.message}")
    canonical_fit = fits[corrected_method]
    swap_test = {
        "correction_antisymmetry_mae_ps": float(
            np.mean(np.abs(test_correction + swapped_correction))
        ),
        "correction_antisymmetry_max_abs_ps": float(
            np.max(np.abs(test_correction + swapped_correction))
        ),
        "corrected_estimator_sign_mae_ps": float(
            np.mean(np.abs(test_corrected + swapped_corrected))
        ),
        "canonical_ctr_ps": float(canonical_fit.ctr_ps),
        "swapped_ctr_ps": float(swapped_fit.ctr_ps),
        "ctr_absolute_difference_ps": float(
            abs(canonical_fit.ctr_ps - swapped_fit.ctr_ps)
        ),
        "canonical_bias_ps": float(canonical_fit.mean_ps - true_tof_ps),
        "swapped_bias_ps": float(swapped_fit.mean_ps - swapped_true_tof),
        "swapped_true_tof_ps": float(swapped_true_tof),
        "passed": bool(
            np.mean(np.abs(test_correction + swapped_correction))
            <= float(pipeline_config["evaluation"].get("swap_tolerance_ps", 1e-5))
        ),
    }
    plot_method_fit(
        swapped_fit,
        plot_dir / "swap_test_gaussian_fit.png",
        true_tof_ps=swapped_true_tof,
        bias_ps=float(swap_test["swapped_bias_ps"]),
        dpi=dpi,
    )

    output_dir = model_output_path(pipeline_config, "work_dir", model_config)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "final_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows([json_safe(row) for row in metrics])

    summary = {
        "comparison": f"Energy LED standard vs Energy CFD standard vs {corrected_method}",
        "model_type": model_type(model_config),
        "model_label": model_label(model_config),
        "metrics_requested": ["bias_ps", "ctr_ps"],
        "all_methods_use_same_selected_blind_test_events": True,
        "all_ctr_values_use_same_iterative_gaussian_fit": True,
        "calibration": {
            "scope": "training split only",
            "strategy": (
                "LED offset from the training Energy LED Gaussian fit; CFD offset "
                "from the training Energy CFD Gaussian fit; corrected LED reuses "
                "the frozen LED offset"
            ),
            "test_recentering": False,
        },
        "true_tof_ps": true_tof_ps,
        "test_event_count": int(splits.test.size),
        "checkpoint": str(checkpoint_path),
        "feature_cache": str(feature_cache.directory),
        "feature_names": feature_cache.feature_names,
        "metrics": metrics,
        "calibration_fits": {
            "Energy LED standard": led_calibration_fit.as_dict(),
            "Energy CFD standard": cfd_calibration_fit.as_dict(),
        },
        "final_fits": {key: value.as_dict() for key, value in fits.items()},
        "swap_test": swap_test,
        "training_prediction_mean_ps": float(np.mean(train_correction)),
    }
    atomic_json(output_dir / "final_evaluation.json", summary)
    logger.info("Final catch22 RF evaluation written to %s", output_dir)
    return summary
