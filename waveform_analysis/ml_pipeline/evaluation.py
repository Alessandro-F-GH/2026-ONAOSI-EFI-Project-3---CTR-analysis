from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.fit import FitResult, fit_delta_times_integer_fs

from .common import atomic_json, json_safe
from .data import EnergyCache, SplitData
from .model import (
    build_correction_model,
    model_label,
    model_output_path,
    model_type,
)
from .plots import plot_method_fit, plot_metric_comparison
from .torch_data import CorrectionDataset, Normalization
from .training import _loader_kwargs, _resolve_device, predict_loader


def _delta_ps(times_fs: np.ndarray) -> np.ndarray:
    values = np.asarray(times_fs, dtype=np.int64)
    return (values[:, 0] - values[:, 1]).astype(np.float64) / 1000.0


def _fit(values_ps: np.ndarray, method: str, fit_config: dict[str, Any]) -> FitResult:
    values_fs = np.rint(np.asarray(values_ps, dtype=np.float64) * 1000.0).astype(np.int64)
    return fit_delta_times_integer_fs(
        values_fs,
        method=method,
        parameter=0.0,
        n_total=int(values_fs.size),
        n_selected=int(values_fs.size),
        config=fit_config,
    )


def _calibrate_from_training_fit(
    train_values_ps: np.ndarray,
    *,
    true_tof_ps: float,
    method: str,
    fit_config: dict[str, Any],
) -> tuple[float, FitResult]:
    result = _fit(train_values_ps, method + " calibration", fit_config)
    if not result.success:
        raise RuntimeError(f"Calibration Gaussian fit failed for {method}: {result.message}")
    offset = float(result.mean_ps - true_tof_ps)
    return offset, result


def _metric_row(
    method: str,
    fit: FitResult,
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


@torch.no_grad()
def _swapped_corrections(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> np.ndarray:
    model.eval()
    values: list[np.ndarray] = []
    for waveforms, _, _ in loader:
        swapped = waveforms[:, [1, 0], :].to(device, non_blocking=True)
        correction = model(swapped)
        values.append(correction.cpu().numpy().astype(np.float64))
    return np.concatenate(values)


def evaluate_final_test(
    cache: EnergyCache,
    splits: SplitData,
    pipeline_config: dict[str, Any],
    model_config: dict[str, Any],
    *,
    checkpoint_path: Path | None,
    logger: Any,
) -> dict[str, Any]:
    if model_type(model_config) == "catch22_random_forest":
        from .catch22_random_forest import evaluate_catch22_random_forest

        return evaluate_catch22_random_forest(
            cache,
            splits,
            pipeline_config,
            model_config,
            checkpoint_path=checkpoint_path,
            logger=logger,
        )

    checkpoint_dir = model_output_path(pipeline_config, "checkpoint_dir", model_config)
    checkpoint_path = checkpoint_path or checkpoint_dir / "best_validation.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    device = _resolve_device(model_config["training"].get("device", "auto"))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    context = checkpoint["context"]
    expected = {
        "pipeline_config_hash": pipeline_config["_config_hash"],
        "model_config_hash": model_config["_config_hash"],
        "dataset_fingerprint": cache.manifest["fingerprint"],
        "split_fingerprint": splits.manifest["fingerprint"],
    }
    for key, value in expected.items():
        if context.get(key) != value:
            raise RuntimeError(f"Checkpoint mismatch for {key}")

    normalization = Normalization(
        mean_mV=float(context["normalization"]["mean_mV"]),
        std_mV=float(context["normalization"]["std_mV"]),
    )
    led_center_ps = float(context["led_center_ps"])
    batch_size = int(model_config["training"]["batch_size"])
    train_dataset = CorrectionDataset(
        cache, splits.train, normalization, led_center_ps
    )
    test_dataset = CorrectionDataset(cache, splits.test, normalization, led_center_ps)
    common = _loader_kwargs(pipeline_config, device)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, drop_last=False, **common
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, **common
    )

    model = build_correction_model(
        model_config, input_length=int(cache.windows_mV.shape[2])
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    train_prediction = predict_loader(model, train_loader, device)
    test_prediction = predict_loader(model, test_loader, device)

    train_led = _delta_ps(cache.led_time_fs[splits.train])
    test_led = _delta_ps(cache.led_time_fs[splits.test])
    train_cfd = _delta_ps(cache.cfd_time_fs[splits.train])
    test_cfd = _delta_ps(cache.cfd_time_fs[splits.test])
    train_corrected = train_led - np.asarray(
        train_prediction["correction_ps"], dtype=np.float64
    )
    test_correction = np.asarray(test_prediction["correction_ps"], dtype=np.float64)
    test_corrected = test_led - test_correction

    true_tof_ps = float(pipeline_config["data"]["true_tof_ps"])
    fit_config = pipeline_config["fit"]
    led_offset, led_calibration_fit = _calibrate_from_training_fit(
        train_led,
        true_tof_ps=true_tof_ps,
        method="Energy LED standard",
        fit_config=fit_config,
    )
    cfd_offset, cfd_calibration_fit = _calibrate_from_training_fit(
        train_cfd,
        true_tof_ps=true_tof_ps,
        method="Energy CFD standard",
        fit_config=fit_config,
    )
    # The corrected estimator is TOF_LED - C_LED - y_theta.  Reuse the frozen
    # LED calibration offset so a non-zero-mean learned correction remains visible
    # as bias instead of being silently recalibrated away.
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
    fits: dict[str, FitResult] = {}
    calibration_fits: dict[str, FitResult] = {
        "Energy LED standard": led_calibration_fit,
        "Energy CFD standard": cfd_calibration_fit,
    }
    plot_dir = model_output_path(pipeline_config, "plot_dir", model_config) / "final_evaluation"
    dpi = int(pipeline_config["plotting"].get("dpi", 180))

    for method, testing_values, offset, calibration_source in methods:
        calibrated = testing_values - offset
        final_fit = _fit(calibrated, method, fit_config)
        row = _metric_row(
            method, final_fit, true_tof_ps, offset, calibration_source
        )
        metrics.append(row)
        fits[method] = final_fit
        filename = (
            method.lower()
            .replace(" ", "_")
            .replace("+", "plus")
            .replace("-", "_")
        )
        plot_method_fit(
            final_fit,
            plot_dir / f"{filename}_gaussian_fit.png",
            true_tof_ps=true_tof_ps,
            bias_ps=float(row["bias_ps"]),
            dpi=dpi,
        )
        logger.info(
            "%s | bias %.3f ps | CTR %.3f ± %.3f ps",
            method,
            row["bias_ps"],
            row["ctr_ps"],
            row["ctr_error_ps"],
        )

    plot_metric_comparison(metrics, plot_dir / "method_comparison.png", dpi)

    swapped_correction = _swapped_corrections(model, test_loader, device)
    swapped_raw_led = -test_led
    swapped_corrected = swapped_raw_led - swapped_correction
    corrected_offset = led_offset
    swapped_calibrated = swapped_corrected + corrected_offset
    swapped_true_tof = -true_tof_ps
    swapped_fit = _fit(
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
    metrics_path = output_dir / "final_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
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
        "metrics": metrics,
        "calibration_fits": {
            key: value.as_dict() for key, value in calibration_fits.items()
        },
        "final_fits": {key: value.as_dict() for key, value in fits.items()},
        "swap_test": swap_test,
    }
    atomic_json(output_dir / "final_evaluation.json", summary)
    logger.info("Final evaluation written to %s", output_dir)
    return summary
