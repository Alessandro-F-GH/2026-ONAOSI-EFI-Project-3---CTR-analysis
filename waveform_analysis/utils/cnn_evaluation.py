from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch

from .cnn_cache import load_cnn_dataset_cache
from .cnn_training import load_trained_model, predict_array, resolve_device
from .fit import FWHM_FACTOR, fit_delta_times_integer_fs

LOGGER = logging.getLogger(__name__)


def _safe_nanmean(values: list[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _safe_nanstd(
    values: list[float] | np.ndarray,
    *,
    ddof: int = 1,
) -> float:
    """Return a finite-only standard deviation without NumPy warnings.

    No finite values -> NaN. One finite realization with ddof=1 -> 0, because
    there is no between-realization spread to estimate.
    """
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan")
    if finite.size <= ddof:
        return 0.0
    return float(np.std(finite, ddof=ddof))


def _plot_training_histories(training_results: list[dict[str, Any]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=False)
    model_to_axis = {"direct": axes[0], "correction": axes[1]}
    for result in training_results:
        history_path = Path(result["checkpoint"]).with_name("training_history.csv")
        if not history_path.is_file():
            continue
        data = np.genfromtxt(history_path, delimiter=",", names=True)
        if data.size == 0:
            continue
        axis = model_to_axis[str(result["model_type"])]
        axis.plot(data["epoch"], data["validation_loss"], label=f"seed {result['seed']}")
    axes[0].set_title("Direct CNN validation loss")
    axes[1].set_title("Invariant correction CNN validation loss")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Validation loss [ps²]")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(y_pred, dtype=np.float64)
    residual = prediction - truth
    return {
        "rmse_ps": float(np.sqrt(mean_squared_error(truth, prediction))),
        "mae_ps": float(mean_absolute_error(truth, prediction)),
        "r2": float(r2_score(truth, prediction)),
        "residual_mean_ps": float(np.mean(residual)),
        "residual_std_ps": float(np.std(residual, ddof=1)),
        "residual_fwhm_ps": float(FWHM_FACTOR * np.std(residual, ddof=1)),
    }


def per_position_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    method: str,
    seed: int,
    test_distribution: str,
    fit_config: dict[str, Any],
) -> list[dict[str, Any]]:
    truth = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(y_pred, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for position in np.unique(truth):
        selected = prediction[truth == position]
        mean = float(np.mean(selected))
        sigma = float(np.std(selected, ddof=1))
        fit = fit_delta_times_integer_fs(
            np.rint(selected * 1000.0).astype(np.int64),
            method=method,
            parameter=float(position),
            n_total=int(selected.size),
            n_selected=int(selected.size),
            config=fit_config,
        )
        rows.append(
            {
                "model": method,
                "seed": seed,
                "test_distribution": test_distribution,
                "target_position_ps": float(position),
                "events": int(selected.size),
                "prediction_mean_ps": mean,
                "bias_ps": mean - float(position),
                "absolute_bias_ps": abs(mean - float(position)),
                "prediction_std_ps": sigma,
                "empirical_fwhm_ps": FWHM_FACTOR * sigma,
                "gaussian_fit_success": bool(fit.success),
                "gaussian_ctr_ps": float(fit.ctr_ps),
                "gaussian_mean_ps": float(fit.mean_ps),
                "gaussian_bias_ps": float(fit.mean_ps - position),
                "chi2_ndof": float(fit.chi2_ndof),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["model"]), str(row["test_distribution"])), []).append(row)
    output: list[dict[str, Any]] = []
    numeric_keys = [
        "rmse_ps",
        "mae_ps",
        "r2",
        "residual_mean_ps",
        "residual_std_ps",
        "residual_fwhm_ps",
        "average_position_ctr_ps",
        "average_absolute_bias_ps",
    ]
    for (model, distribution), items in groups.items():
        row: dict[str, Any] = {
            "model": model,
            "test_distribution": distribution,
            "realizations": len(items),
        }
        for key in numeric_keys:
            values = np.asarray([float(item[key]) for item in items], dtype=np.float64)
            row[f"{key}_mean"] = _safe_nanmean(values)
            row[f"{key}_std"] = _safe_nanstd(values)
            row[f"{key}_valid_realizations"] = int(np.isfinite(values).sum())
        output.append(row)
    return output


def _position_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["model"]),
            str(row["test_distribution"]),
            float(row["target_position_ps"]),
        )
        groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for (model, distribution, position), items in sorted(groups.items()):
        row: dict[str, Any] = {
            "model": model,
            "test_distribution": distribution,
            "target_position_ps": position,
            "realizations": len(items),
        }
        for key in (
            "prediction_mean_ps",
            "bias_ps",
            "absolute_bias_ps",
            "empirical_fwhm_ps",
            "gaussian_ctr_ps",
            "gaussian_bias_ps",
        ):
            values = np.asarray([float(item[key]) for item in items], dtype=np.float64)
            row[f"{key}_mean"] = _safe_nanmean(values)
            row[f"{key}_std"] = _safe_nanstd(values)
            row[f"{key}_valid_realizations"] = int(np.isfinite(values).sum())
        output.append(row)
    return output


def _plot_metric_comparison(summary: list[dict[str, Any]], output_path: Path, metric: str, ylabel: str) -> None:
    models = ["LED baseline", "Direct CNN", "Invariant correction CNN"]
    distributions = ["discrete", "uniform"]
    lookup = {(row["model"], row["test_distribution"]): row for row in summary}
    x = np.arange(len(models), dtype=np.float64)
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for offset, distribution in zip((-width / 2, width / 2), distributions, strict=True):
        values = [lookup[(model, distribution)][f"{metric}_mean"] for model in models]
        errors = [lookup[(model, distribution)][f"{metric}_std"] for model in models]
        ax.bar(x + offset, values, width=width, yerr=errors, capsize=3, label=distribution)
    ax.set_xticks(x, models)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel}: same discrete positions vs uniform unseen positions")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_position_curves(
    summary: list[dict[str, Any]],
    output_path: Path,
    *,
    distribution: str,
    value_key: str,
    ylabel: str,
    include_ideal: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for model in ("LED baseline", "Direct CNN", "Invariant correction CNN"):
        rows = [
            row
            for row in summary
            if row["model"] == model and row["test_distribution"] == distribution
        ]
        rows.sort(key=lambda row: row["target_position_ps"])
        x = np.asarray([row["target_position_ps"] for row in rows])
        y = np.asarray([row[value_key] for row in rows])
        yerr_key = value_key.removesuffix("_mean") + "_std"
        yerr = np.asarray([row.get(yerr_key, 0.0) for row in rows])
        ax.plot(x, y, marker="o", markersize=3, linewidth=1.4, label=model)
        if np.any(yerr > 0):
            ax.fill_between(x, y - yerr, y + yerr, alpha=0.12)
    if include_ideal:
        all_x = np.asarray(sorted({row["target_position_ps"] for row in summary if row["test_distribution"] == distribution}))
        ax.plot(all_x, all_x, linestyle="--", linewidth=1.2, label="Ideal")
    ax.set_xlabel("True augmented TOF [ps]")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} on {distribution} test positions")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_representative_spectra(
    predictions: dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray]],
    output_path: Path,
    *,
    distribution: str,
) -> None:
    available = [value for key, value in predictions.items() if key[1] == distribution]
    if not available:
        return
    truth = available[0][0]
    positions = np.unique(truth)
    chosen = [positions[0], positions[len(positions) // 2], positions[-1]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for axis, position in zip(axes, chosen, strict=True):
        for model in ("LED baseline", "Direct CNN", "Invariant correction CNN"):
            candidates = [
                value
                for (name, test_name, _seed), value in predictions.items()
                if name == model and test_name == distribution
            ]
            if not candidates:
                continue
            values = np.concatenate([pred[truth_values == position] for truth_values, pred in candidates])
            axis.hist(values, bins=60, histtype="step", density=True, linewidth=1.3, label=model)
        axis.axvline(position, linestyle="--", linewidth=1.0)
        axis.set_title(f"True TOF {position:.0f} ps")
        axis.set_xlabel("Prediction [ps]")
    axes[0].set_ylabel("Density")
    axes[-1].legend(fontsize=8)
    fig.suptitle(f"Prediction spectra at representative {distribution} positions")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def evaluate_experiment(
    *,
    dataset_path: Path,
    training_results: list[dict[str, Any]],
    analysis_config: dict[str, Any],
    model_config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = load_cnn_dataset_cache(dataset_path)
    batch_size = int(model_config["training"].get("evaluation_batch_size", model_config["training"]["batch_size"]))
    num_workers = int(model_config["training"].get("evaluation_num_workers", 0))
    evaluation_device = resolve_device(str(model_config["parallel"].get("evaluation_device", "auto")))
    fit_config = analysis_config["fit"]
    metric_rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    prediction_store: dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray]] = {}

    direct_results = [item for item in training_results if item["model_type"] == "direct"]
    correction_results = [item for item in training_results if item["model_type"] == "correction"]
    if not direct_results or not correction_results:
        raise RuntimeError("both direct and correction models are required for evaluation")

    # Baseline calibration is the training mean of the original LED time difference.
    led_center = float(np.mean(cache["correction_train_led_delta_ps"]))
    test_led = cache["correction_test_led_delta_ps"].astype(np.float64)
    discrete_positions = cache["discrete_positions_ps"].astype(np.float64)
    uniform_positions = cache["uniform_positions_ps"].astype(np.float64)

    def expanded_baseline(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        truth = np.repeat(positions, test_led.size)
        prediction = np.concatenate([test_led + position - led_center for position in positions])
        return truth, prediction

    for distribution, positions in (("discrete", discrete_positions), ("uniform", uniform_positions)):
        truth, prediction = expanded_baseline(positions)
        seed = -1
        metrics = regression_metrics(truth, prediction)
        per_position = per_position_metrics(
            truth,
            prediction,
            method="LED baseline",
            seed=seed,
            test_distribution=distribution,
            fit_config=fit_config,
        )
        metrics.update(
            {
                "model": "LED baseline",
                "seed": seed,
                "test_distribution": distribution,
                "average_position_ctr_ps": _safe_nanmean([row["gaussian_ctr_ps"] for row in per_position]),
                "average_absolute_bias_ps": float(np.mean([row["absolute_bias_ps"] for row in per_position])),
            }
        )
        metric_rows.append(metrics)
        position_rows.extend(per_position)
        prediction_store[("LED baseline", distribution, seed)] = (truth, prediction)

    for result in direct_results:
        seed = int(result["seed"])
        model, checkpoint = load_trained_model(result["checkpoint"], device=evaluation_device)
        for distribution, prefix in (
            ("discrete", "direct_test_discrete"),
            ("uniform", "direct_test_uniform"),
        ):
            truth = cache[f"{prefix}_y"].astype(np.float64)
            prediction = predict_array(
                model,
                cache[f"{prefix}_x"],
                mean=np.asarray(checkpoint["normalization_mean"]),
                std=np.asarray(checkpoint["normalization_std"]),
                device=evaluation_device,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            metrics = regression_metrics(truth, prediction)
            per_position = per_position_metrics(
                truth,
                prediction,
                method="Direct CNN",
                seed=seed,
                test_distribution=distribution,
                fit_config=fit_config,
            )
            metrics.update(
                {
                    "model": "Direct CNN",
                    "seed": seed,
                    "test_distribution": distribution,
                    "average_position_ctr_ps": _safe_nanmean([row["gaussian_ctr_ps"] for row in per_position]),
                    "average_absolute_bias_ps": float(np.mean([row["absolute_bias_ps"] for row in per_position])),
                }
            )
            metric_rows.append(metrics)
            position_rows.extend(per_position)
            prediction_store[("Direct CNN", distribution, seed)] = (truth, prediction)
        del model
        if evaluation_device.type == "cuda":
            torch.cuda.empty_cache()

    correction_x = cache["correction_test_x"]
    for result in correction_results:
        seed = int(result["seed"])
        model, checkpoint = load_trained_model(result["checkpoint"], device=evaluation_device)
        centered_correction = predict_array(
            model,
            correction_x,
            mean=np.asarray(checkpoint["normalization_mean"]),
            std=np.asarray(checkpoint["normalization_std"]),
            device=evaluation_device,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        absolute_correction = float(checkpoint["correction_center_ps"]) + centered_correction
        for distribution, positions in (("discrete", discrete_positions), ("uniform", uniform_positions)):
            truth = np.repeat(positions, test_led.size)
            prediction = np.concatenate(
                [test_led + position - absolute_correction for position in positions]
            )
            metrics = regression_metrics(truth, prediction)
            per_position = per_position_metrics(
                truth,
                prediction,
                method="Invariant correction CNN",
                seed=seed,
                test_distribution=distribution,
                fit_config=fit_config,
            )
            metrics.update(
                {
                    "model": "Invariant correction CNN",
                    "seed": seed,
                    "test_distribution": distribution,
                    "average_position_ctr_ps": _safe_nanmean([row["gaussian_ctr_ps"] for row in per_position]),
                    "average_absolute_bias_ps": float(np.mean([row["absolute_bias_ps"] for row in per_position])),
                }
            )
            metric_rows.append(metrics)
            position_rows.extend(per_position)
            prediction_store[("Invariant correction CNN", distribution, seed)] = (truth, prediction)
        del model
        if evaluation_device.type == "cuda":
            torch.cuda.empty_cache()

    aggregate_metrics = _aggregate_metric_rows(metric_rows)
    aggregate_positions = _position_summary(position_rows)
    _write_csv(output_dir / "metrics_by_realization.csv", metric_rows)
    _write_csv(output_dir / "metrics_summary.csv", aggregate_metrics)
    _write_csv(output_dir / "per_position_metrics_by_realization.csv", position_rows)
    _write_csv(output_dir / "per_position_metrics_summary.csv", aggregate_positions)

    _plot_metric_comparison(aggregate_metrics, output_dir / "rmse_comparison.png", "rmse_ps", "RMSE [ps]")
    _plot_metric_comparison(
        aggregate_metrics,
        output_dir / "ctr_comparison.png",
        "average_position_ctr_ps",
        "Average fitted CTR [ps]",
    )
    _plot_metric_comparison(
        aggregate_metrics,
        output_dir / "bias_comparison.png",
        "average_absolute_bias_ps",
        "Average absolute bias [ps]",
    )
    for distribution in ("discrete", "uniform"):
        _plot_position_curves(
            aggregate_positions,
            output_dir / f"mean_response_{distribution}.png",
            distribution=distribution,
            value_key="prediction_mean_ps_mean",
            ylabel="Mean predicted TOF [ps]",
            include_ideal=True,
        )
        _plot_position_curves(
            aggregate_positions,
            output_dir / f"ctr_vs_position_{distribution}.png",
            distribution=distribution,
            value_key="gaussian_ctr_ps_mean",
            ylabel="Fitted CTR [ps]",
        )
        _plot_position_curves(
            aggregate_positions,
            output_dir / f"bias_vs_position_{distribution}.png",
            distribution=distribution,
            value_key="bias_ps_mean",
            ylabel="Bias [ps]",
        )
        _plot_representative_spectra(
            prediction_store,
            output_dir / f"representative_spectra_{distribution}.png",
            distribution=distribution,
        )

    prediction_arrays: dict[str, np.ndarray] = {}
    for (model_name, distribution, seed), (truth, prediction) in prediction_store.items():
        safe_model = model_name.lower().replace(" ", "_")
        prediction_arrays.setdefault(f"truth_{distribution}", truth.astype(np.float32))
        prediction_arrays[f"prediction_{safe_model}_{distribution}_seed_{seed}"] = prediction.astype(np.float32)
    np.savez_compressed(output_dir / "predictions.npz", **prediction_arrays)
    _plot_training_histories(training_results, output_dir / "training_curves.png")

    summary = {
        "dataset_path": str(dataset_path),
        "training_realizations": training_results,
        "metrics_summary": aggregate_metrics,
    }
    with (output_dir / "experiment_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    return summary
