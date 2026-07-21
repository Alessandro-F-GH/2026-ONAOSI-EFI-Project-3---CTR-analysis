from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from scipy.stats import norm

from .tabular import read_table, write_table
from .models import FitResult

FIT_FIELDS = [
    "success",
    "status",
    "area_events",
    "area_error_events",
    "mean_lsb",
    "mean_error_lsb",
    "sigma_lsb",
    "sigma_error_lsb",
    "fit_low_lsb",
    "fit_high_lsb",
    "bin_width_lsb",
    "histogram_start_lsb",
    "histogram_counts",
    "chi_square",
    "ndof",
    "reduced_chi_square",
]
FIT_DEBUG_FIELDS = [
    "success",
    "status",
    "area_events",
    "area_error_events",
    "mean_lsb",
    "mean_error_lsb",
    "sigma_lsb",
    "sigma_error_lsb",
    "fit_low_lsb",
    "fit_high_lsb",
    "bin_width_lsb",
    "histogram_edges_lsb",
    "histogram_counts",
    "expected_counts",
    "chi_square",
    "ndof",
    "reduced_chi_square",
]


def _robust_center_scale(values: np.ndarray, minimum_scale: float) -> tuple[float, float]:
    center = float(np.median(values))
    scale = 1.4826 * float(np.median(np.abs(values.astype(float) - center)))
    if not math.isfinite(scale) or scale <= 0:
        scale = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    return center, max(scale, minimum_scale)


def quantized_edges(values_lsb: np.ndarray, cfg: dict, forced_width: int | None = None) -> tuple[np.ndarray, int]:
    values = np.asarray(values_lsb, dtype=np.int64)
    if values.size == 0:
        raise RuntimeError("Cannot create a histogram from no timing values")
    minimum = int(values.min())
    maximum = int(values.max())
    if forced_width is None:
        desired_bins = int(math.ceil(values.size / float(cfg["target_events_per_bin"])))
        desired_bins = max(int(cfg["min_histogram_bins"]), min(int(cfg["max_histogram_bins"]), desired_bins))
        width = max(int(cfg["min_bin_width_lsb"]), int(math.ceil((maximum - minimum + 1) / desired_bins)))
    else:
        width = max(1, int(forced_width))
    start = math.floor(minimum / width) * width
    bins = max(1, int(math.ceil((maximum - start + 1) / width)))
    if bins > int(cfg["max_histogram_bins"]):
        width *= int(math.ceil(bins / int(cfg["max_histogram_bins"])))
        start = math.floor(minimum / width) * width
        bins = max(1, int(math.ceil((maximum - start + 1) / width)))
    edges = start - 0.5 + np.arange(bins + 1, dtype=float) * width
    return edges, width


def _seed_from_histogram(edges: np.ndarray, counts: np.ndarray, smooth_sigma: float) -> tuple[float, float]:
    smooth = gaussian_filter1d(counts.astype(float), smooth_sigma, mode="nearest")
    peak_index = int(np.argmax(smooth))
    centers = 0.5 * (edges[:-1] + edges[1:])
    peak = float(centers[peak_index])
    half = 0.5 * float(smooth[peak_index])
    left = peak_index
    while left > 0 and smooth[left] > half:
        left -= 1
    right = peak_index
    while right < smooth.size - 1 and smooth[right] > half:
        right += 1
    width = float(centers[right] - centers[left]) if right > left else float(np.median(np.diff(edges)))
    return peak, max(width / 2.355, 0.5 * float(np.median(np.diff(edges))), 1e-9)


def _reject_outliers(values_lsb: np.ndarray, cfg: dict) -> np.ndarray:
    values = np.asarray(values_lsb, dtype=np.int64)
    if values.size < int(cfg["min_events"]):
        return values
    coarse_width = max(int(cfg["min_bin_width_lsb"]), int(cfg.get("outlier_bin_width_lsb", 8)))
    edges, _ = quantized_edges(values, cfg, forced_width=coarse_width)
    counts, _ = np.histogram(values, bins=edges)
    peak, _ = _seed_from_histogram(edges, counts, float(cfg["smooth_sigma_bins"]))
    half_width = max(coarse_width, int(cfg.get("outlier_seed_half_width_bins", 3)) * coarse_width)
    core = values[np.abs(values.astype(float) - peak) <= half_width]
    if core.size < int(cfg.get("outlier_minimum_core_events", 20)):
        order = np.argsort(np.abs(values.astype(float) - peak))
        core = values[order[: min(values.size, int(cfg.get("outlier_minimum_core_events", 20)))]]
    center, scale = _robust_center_scale(core, float(cfg["outlier_minimum_scale_lsb"]))
    keep = np.abs(values.astype(float) - center) <= float(cfg["outlier_z_threshold"]) * scale
    filtered = values[keep]
    return filtered if filtered.size >= int(cfg["min_events"]) else values


def _expected(edges: np.ndarray, area: float, mean: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-12)
    return area * np.diff(norm.cdf(edges, loc=mean, scale=sigma))


def _fit_quality(
    edges: np.ndarray,
    counts: np.ndarray,
    expected: np.ndarray,
    fit_low: float,
    fit_high: float,
) -> tuple[float, int, float]:
    centers = 0.5 * (edges[:-1] + edges[1:])
    mask = (centers >= fit_low) & (centers <= fit_high)
    observed = counts[mask].astype(float)
    predicted = expected[mask].astype(float)
    valid = np.isfinite(predicted) & (predicted > 0.0)
    if not np.any(valid):
        return math.nan, 0, math.nan
    chi_square = float(np.sum((observed[valid] - predicted[valid]) ** 2 / predicted[valid]))
    ndof = max(0, int(np.count_nonzero(valid)) - 3)
    reduced = chi_square / ndof if ndof > 0 else math.nan
    return chi_square, ndof, reduced


def _numerical_hessian(function, point: np.ndarray) -> np.ndarray:
    point = np.asarray(point, dtype=float)
    steps = np.asarray([
        1e-4 * max(1.0, abs(point[0])),
        1e-5 * max(1.0, abs(point[1])),
        1e-4 * max(1.0, abs(point[2])),
    ])
    size = point.size
    result = np.zeros((size, size), dtype=float)
    base = float(function(point))
    for i in range(size):
        ei = np.zeros(size)
        ei[i] = steps[i]
        result[i, i] = (function(point + ei) - 2.0 * base + function(point - ei)) / (steps[i] ** 2)
        for j in range(i + 1, size):
            ej = np.zeros(size)
            ej[j] = steps[j]
            value = (
                function(point + ei + ej)
                - function(point + ei - ej)
                - function(point - ei + ej)
                + function(point - ei - ej)
            ) / (4.0 * steps[i] * steps[j])
            result[i, j] = value
            result[j, i] = value
    return result


def _fit_range(
    edges: np.ndarray,
    counts: np.ndarray,
    low: float,
    high: float,
    area_seed: float,
    mean_seed: float,
    sigma_seed: float,
    cfg: dict,
) -> dict[str, Any]:
    centers = 0.5 * (edges[:-1] + edges[1:])
    mask = (centers >= low) & (centers <= high)
    if np.count_nonzero(mask) < int(cfg["minimum_fit_bins"]):
        raise RuntimeError("Fit range contains too few bins")
    local_edges = np.concatenate(([edges[:-1][mask][0]], edges[1:][mask]))
    observed = counts[mask].astype(float)
    if observed.sum() < int(cfg["minimum_fit_events"]):
        raise RuntimeError("Fit range contains too few events")
    bin_width = float(np.median(np.diff(edges)))
    minimum_sigma = max(float(cfg.get("minimum_sigma_bin_fraction", 0.25)) * bin_width, 1e-9)
    maximum_sigma = max(minimum_sigma * 2.0, float(cfg.get("maximum_sigma_range_fraction", 0.5)) * (local_edges[-1] - local_edges[0]))
    sigma_seed = min(max(sigma_seed, minimum_sigma), maximum_sigma)
    mean_seed = min(max(mean_seed, local_edges[0]), local_edges[-1])
    area_seed = max(area_seed, observed.sum(), 1.0)

    def objective(parameters: np.ndarray) -> float:
        area = math.exp(float(parameters[0]))
        mean = float(parameters[1])
        sigma = math.exp(float(parameters[2]))
        expected = np.clip(_expected(local_edges, area, mean, sigma), 1e-12, None)
        return float(np.sum(expected - observed * np.log(expected)))

    initial = np.asarray([math.log(area_seed), mean_seed, math.log(sigma_seed)], dtype=float)
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=[
            (math.log(max(1e-6, observed.sum() * 0.1)), math.log(max(10.0, observed.sum() * 100.0))),
            (float(local_edges[0]), float(local_edges[-1])),
            (math.log(minimum_sigma), math.log(maximum_sigma)),
        ],
        options={"maxiter": int(cfg["optimizer_maxiter"]), "ftol": 1e-12},
    )
    transformed = np.asarray(result.x, dtype=float)
    area = math.exp(float(transformed[0]))
    mean = float(transformed[1])
    sigma = math.exp(float(transformed[2]))
    area_error = mean_error = sigma_error = math.nan
    try:
        covariance = np.linalg.inv(_numerical_hessian(objective, transformed))
        area_error = area * math.sqrt(max(0.0, float(covariance[0, 0])))
        mean_error = math.sqrt(max(0.0, float(covariance[1, 1])))
        sigma_error = sigma * math.sqrt(max(0.0, float(covariance[2, 2])))
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        pass
    return {
        "success": bool(result.success),
        "status": str(result.message),
        "mask": mask,
        "area": area,
        "area_error": area_error,
        "mean": mean,
        "mean_error": mean_error,
        "sigma": sigma,
        "sigma_error": sigma_error,
        "low": float(local_edges[0]),
        "high": float(local_edges[-1]),
    }


def fit_timing(values_lsb: np.ndarray, cfg: dict) -> FitResult:
    values = np.asarray(values_lsb, dtype=np.int64)
    if values.size < int(cfg["min_events"]):
        raise RuntimeError(f"Only {values.size} timing values available for fit")
    fit_values = _reject_outliers(values, cfg)
    edges, bin_width = quantized_edges(fit_values, cfg)
    counts, _ = np.histogram(fit_values, bins=edges)
    mean_seed, sigma_seed = _seed_from_histogram(edges, counts, float(cfg["smooth_sigma_bins"]))
    low = mean_seed - float(cfg["initial_half_width_sigma"]) * sigma_seed
    high = mean_seed + float(cfg["initial_half_width_sigma"]) * sigma_seed
    minimum_half_width = 0.5 * int(cfg["minimum_fit_bins"]) * bin_width
    low = min(low, mean_seed - minimum_half_width)
    high = max(high, mean_seed + minimum_half_width)
    area_seed = float(fit_values.size)
    final: dict[str, Any] | None = None
    for _ in range(int(cfg["iterations"])):
        final = _fit_range(edges, counts, low, high, area_seed, mean_seed, sigma_seed, cfg)
        area_seed = final["area"]
        mean_seed = final["mean"]
        sigma_seed = final["sigma"]
        next_low = mean_seed - float(cfg["refit_half_width_sigma"]) * sigma_seed
        next_high = mean_seed + float(cfg["refit_half_width_sigma"]) * sigma_seed
        next_low = min(next_low, mean_seed - minimum_half_width)
        next_high = max(next_high, mean_seed + minimum_half_width)
        if abs(next_low - low) <= 0.05 * bin_width and abs(next_high - high) <= 0.05 * bin_width:
            break
        low, high = next_low, next_high
    if final is None:
        raise RuntimeError("Gaussian fit did not run")
    expected = _expected(edges, final["area"], final["mean"], final["sigma"])
    chi_square, ndof, reduced_chi_square = _fit_quality(
        edges,
        counts,
        expected,
        final["low"],
        final["high"],
    )
    return FitResult(
        success=final["success"],
        status=final["status"],
        area_events=final["area"],
        area_error_events=final["area_error"],
        mean_lsb=final["mean"],
        mean_error_lsb=final["mean_error"],
        sigma_lsb=final["sigma"],
        sigma_error_lsb=final["sigma_error"],
        fit_low_lsb=final["low"],
        fit_high_lsb=final["high"],
        bin_width_lsb=bin_width,
        histogram_edges_lsb=edges,
        histogram_counts=counts.astype(np.int64),
        expected_counts=expected,
        chi_square=chi_square,
        ndof=ndof,
        reduced_chi_square=reduced_chi_square,
    )


def fit_to_row(fit: FitResult, diagnostic_mode: str = "compact") -> dict[str, Any]:
    row = {
        "success": int(fit.success),
        "status": fit.status,
        "area_events": fit.area_events,
        "area_error_events": fit.area_error_events,
        "mean_lsb": fit.mean_lsb,
        "mean_error_lsb": fit.mean_error_lsb,
        "sigma_lsb": fit.sigma_lsb,
        "sigma_error_lsb": fit.sigma_error_lsb,
        "fit_low_lsb": fit.fit_low_lsb,
        "fit_high_lsb": fit.fit_high_lsb,
        "bin_width_lsb": fit.bin_width_lsb,
        "histogram_counts": json.dumps(
            fit.histogram_counts.tolist(), separators=(",", ":")
        ),
        "chi_square": fit.chi_square,
        "ndof": fit.ndof,
        "reduced_chi_square": fit.reduced_chi_square,
    }
    if str(diagnostic_mode).lower() == "debug":
        row["histogram_edges_lsb"] = json.dumps(
            fit.histogram_edges_lsb.tolist(), separators=(",", ":")
        )
        row["expected_counts"] = json.dumps(
            fit.expected_counts.tolist(), separators=(",", ":")
        )
    else:
        row["histogram_start_lsb"] = float(fit.histogram_edges_lsb[0])
    return row


def write_fit_csv(
    path: str | Path,
    fit: FitResult,
    diagnostic_mode: str = "compact",
) -> None:
    debug = str(diagnostic_mode).lower() == "debug"
    write_table(
        path,
        FIT_DEBUG_FIELDS if debug else FIT_FIELDS,
        [fit_to_row(fit, diagnostic_mode)],
    )


def load_fit_csv(path: str | Path) -> FitResult:
    rows = read_table(path)
    if len(rows) != 1:
        raise RuntimeError(f"Expected one fit row in {path}")
    row = rows[0]
    counts = np.asarray(json.loads(row["histogram_counts"]), dtype=np.int64)
    bin_width = int(row["bin_width_lsb"])
    if row.get("histogram_edges_lsb", "") != "":
        edges = np.asarray(json.loads(row["histogram_edges_lsb"]), dtype=float)
    else:
        start = float(row["histogram_start_lsb"])
        edges = start + np.arange(counts.size + 1, dtype=float) * bin_width
    fit_low = float(row["fit_low_lsb"])
    fit_high = float(row["fit_high_lsb"])
    if row.get("expected_counts", "") != "":
        expected = np.asarray(json.loads(row["expected_counts"]), dtype=float)
    else:
        expected = _expected(
            edges,
            float(row["area_events"]),
            float(row["mean_lsb"]),
            float(row["sigma_lsb"]),
        )
    if row.get("chi_square", "") != "" and row.get("ndof", "") != "":
        chi_square = float(row["chi_square"])
        ndof = int(row["ndof"])
        reduced = float(row["reduced_chi_square"])
    else:
        chi_square, ndof, reduced = _fit_quality(
            edges, counts, expected, fit_low, fit_high
        )
    return FitResult(
        success=bool(int(row["success"])),
        status=str(row["status"]),
        area_events=float(row["area_events"]),
        area_error_events=float(row["area_error_events"]),
        mean_lsb=float(row["mean_lsb"]),
        mean_error_lsb=float(row["mean_error_lsb"]),
        sigma_lsb=float(row["sigma_lsb"]),
        sigma_error_lsb=float(row["sigma_error_lsb"]),
        fit_low_lsb=fit_low,
        fit_high_lsb=fit_high,
        bin_width_lsb=bin_width,
        histogram_edges_lsb=edges,
        histogram_counts=counts,
        expected_counts=expected,
        chi_square=chi_square,
        ndof=ndof,
        reduced_chi_square=reduced,
    )
