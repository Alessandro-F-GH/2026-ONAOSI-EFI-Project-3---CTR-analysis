from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr

from .signal import INVALID_TIME_FS

FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))
FS_PER_PS = 1000.0


@dataclass
class FitResult:
    method: str
    parameter: float
    success: bool
    n_total: int
    n_selected: int
    n_valid: int
    n_fit: int
    crossing_efficiency: float
    mean_ps: float
    mean_error_ps: float
    sigma_ps: float
    sigma_error_ps: float
    ctr_ps: float
    ctr_error_ps: float
    chi2: float
    ndof: int
    fit_low_ps: float
    fit_high_ps: float
    iterations: int
    message: str = ""
    bin_width_ps: float = np.nan
    bin_phase_ps: float = np.nan
    phase_ctr_std_ps: float = np.nan
    edges_ps: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    counts: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    expected: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)

    @property
    def chi2_ndof(self) -> float:
        return self.chi2 / self.ndof if self.ndof > 0 else np.nan

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "parameter": self.parameter,
            "success": self.success,
            "n_total": self.n_total,
            "n_selected": self.n_selected,
            "n_rejected": self.n_total - self.n_selected,
            "n_valid": self.n_valid,
            "n_fit": self.n_fit,
            "crossing_efficiency": self.crossing_efficiency,
            "mean_ps": self.mean_ps,
            "mean_error_ps": self.mean_error_ps,
            "sigma_ps": self.sigma_ps,
            "sigma_error_ps": self.sigma_error_ps,
            "ctr_ps": self.ctr_ps,
            "ctr_error_ps": self.ctr_error_ps,
            "chi2": self.chi2,
            "ndof": self.ndof,
            "chi2_ndof": self.chi2_ndof,
            "fit_low_ps": self.fit_low_ps,
            "fit_high_ps": self.fit_high_ps,
            "iterations": self.iterations,
            "bin_width_ps": self.bin_width_ps,
            "bin_phase_ps": self.bin_phase_ps,
            "phase_ctr_std_ps": self.phase_ctr_std_ps,
            "message": self.message,
        }


def _failure(*, method: str, parameter: float, n_total: int, n_selected: int,
             n_valid: int, message: str) -> FitResult:
    return FitResult(
        method=method, parameter=float(parameter), success=False,
        n_total=int(n_total), n_selected=int(n_selected), n_valid=int(n_valid),
        n_fit=0, crossing_efficiency=n_valid / n_selected if n_selected else 0.0,
        mean_ps=np.nan, mean_error_ps=np.nan, sigma_ps=np.nan,
        sigma_error_ps=np.nan, ctr_ps=np.nan, ctr_error_ps=np.nan,
        chi2=np.nan, ndof=0, fit_low_ps=np.nan, fit_high_ps=np.nan,
        iterations=0, message=message,
    )


def _robust_location_scale(values_fs: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values_fs, dtype=np.float64)
    center = float(np.median(values))
    q16, q84 = np.quantile(values, [0.1586552539, 0.8413447461])
    sigma = 0.5 * float(q84 - q16)
    if not np.isfinite(sigma) or sigma <= 0.0:
        mad = float(np.median(np.abs(values - center)))
        sigma = 1.4826022185 * mad
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = float(np.std(values, ddof=0))
    return center, max(sigma, 1.0)


def _bin_probabilities(edges_fs: np.ndarray, mean_fs: float, sigma_fs: float) -> np.ndarray:
    z = (np.asarray(edges_fs, dtype=np.float64) - float(mean_fs)) / float(sigma_fs)
    cdf = ndtr(z)
    probabilities = np.diff(cdf)
    total = float(np.sum(probabilities))
    if not np.isfinite(total) or total <= 0.0:
        return np.full(probabilities.shape, np.nan, dtype=np.float64)
    probabilities = probabilities / total
    return np.clip(probabilities, np.finfo(np.float64).tiny, 1.0)


def _poisson_deviance(counts: np.ndarray, expected: np.ndarray) -> float:
    observed = np.asarray(counts, dtype=np.float64)
    expected = np.clip(np.asarray(expected, dtype=np.float64), np.finfo(np.float64).tiny, None)
    term = expected - observed
    positive = observed > 0.0
    term[positive] += observed[positive] * np.log(observed[positive] / expected[positive])
    return float(2.0 * np.sum(term))


def _fit_histogram_all_events(
    values_fs: np.ndarray,
    *,
    bin_width_fs: int,
    phase_fraction: float,
    initial_mean_fs: float,
    initial_sigma_fs: float,
) -> dict[str, Any] | None:
    values = np.asarray(values_fs, dtype=np.int64)
    if values.size < 3:
        return None
    width = int(bin_width_fs)
    phase_fs = float(phase_fraction) * width
    vmin = int(np.min(values))
    vmax = int(np.max(values))
    # Add one complete bin of margin so the histogram contains every event even
    # when the bin origin is shifted. No event-level rejection occurs here.
    low = np.floor((vmin - phase_fs) / width) * width + phase_fs - width
    high = np.ceil((vmax - phase_fs) / width) * width + phase_fs + width
    n_bins = max(5, int(np.ceil((high - low) / width)))
    edges = low + np.arange(n_bins + 1, dtype=np.float64) * width
    if edges[-1] <= vmax:
        edges = np.append(edges, edges[-1] + width)
    counts, edges = np.histogram(values, bins=edges)
    if int(np.sum(counts)) != int(values.size):
        return None

    minimum_sigma = max(width * 0.20, 1.0)
    maximum_sigma = max(float(vmax - vmin) * 2.0, initial_sigma_fs * 10.0, minimum_sigma * 10.0)

    def objective(theta: np.ndarray) -> float:
        mean = float(theta[0])
        sigma = float(np.exp(theta[1]))
        probabilities = _bin_probabilities(edges, mean, sigma)
        if np.any(~np.isfinite(probabilities)):
            return float("inf")
        return float(-np.sum(counts * np.log(probabilities)))

    x0 = np.asarray([initial_mean_fs, np.log(max(initial_sigma_fs, minimum_sigma))], dtype=np.float64)
    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=[(float(vmin) - width, float(vmax) + width),
                (np.log(minimum_sigma), np.log(maximum_sigma))],
        options={"maxiter": 2000, "ftol": 1e-12},
    )

    # L-BFGS-B can report an abnormal line-search termination for very broad or
    # strongly non-Gaussian candidates even when its final parameters are finite
    # and useful.  Evaluation must not crash simply because a candidate is poor.
    # Use the optimizer result whenever it is numerically valid; otherwise fall
    # back to the all-event Gaussian MLE (arithmetic mean/std).  No event is
    # removed in either path.
    optimizer_usable = (
        np.asarray(getattr(result, "x", []), dtype=np.float64).shape == (2,)
        and np.all(np.isfinite(result.x))
        and np.isfinite(objective(np.asarray(result.x, dtype=np.float64)))
    )
    if optimizer_usable:
        mean_fs = float(result.x[0])
        sigma_fs = float(np.exp(result.x[1]))
        used_fallback = False
    else:
        mean_fs = float(np.mean(values, dtype=np.float64))
        sigma_fs = float(np.std(values, dtype=np.float64, ddof=0))
        sigma_fs = float(np.clip(sigma_fs, minimum_sigma, maximum_sigma))
        used_fallback = True

    probabilities = _bin_probabilities(edges, mean_fs, sigma_fs)
    if np.any(~np.isfinite(probabilities)):
        return None
    expected = values.size * probabilities
    deviance = _poisson_deviance(counts, expected)
    ndof = max(0, int(counts.size) - 2)

    mean_error_fs = np.nan
    sigma_error_fs = np.nan
    if not used_fallback:
        try:
            inverse = result.hess_inv.todense() if hasattr(result.hess_inv, "todense") else np.asarray(result.hess_inv)
            inverse = np.asarray(inverse, dtype=np.float64)
            if inverse.shape == (2, 2):
                mean_error_fs = float(np.sqrt(max(inverse[0, 0], 0.0)))
                log_sigma_error = float(np.sqrt(max(inverse[1, 1], 0.0)))
                sigma_error_fs = sigma_fs * log_sigma_error
        except Exception:
            pass
    else:
        # Standard large-sample errors for the direct all-event Gaussian MLE.
        mean_error_fs = sigma_fs / np.sqrt(max(values.size, 1))
        sigma_error_fs = sigma_fs / np.sqrt(max(2 * values.size, 1))

    return {
        "mean_fs": mean_fs,
        "sigma_fs": sigma_fs,
        "mean_error_fs": mean_error_fs,
        "sigma_error_fs": sigma_error_fs,
        "deviance": deviance,
        "ndof": ndof,
        "edges_fs": edges,
        "counts": counts,
        "expected": expected,
        "phase_fs": phase_fs,
        "iterations": int(getattr(result, "nit", 0)),
        "optimizer_fallback": bool(used_fallback),
    }


def fit_delta_times_integer_fs(
    delta_fs: np.ndarray,
    *,
    method: str,
    parameter: float,
    n_total: int,
    n_selected: int,
    config: dict[str, Any],
) -> FitResult:
    """Fit one Gaussian timing distribution using every prepared event.

    A single fixed histogram bin width is used for every evaluation. Several
    bin-origin alignments are tried over exactly one bin width; the alignment
    with the smallest reduced Poisson deviance is retained. Thus LED, CFD, ML
    and multithreshold results differ only because of their data, not because
    the fitter changes its bin size.

    No event is rejected by this function. Dataset-level selection must happen
    before evaluation.
    """
    raw = np.asarray(delta_fs)
    if raw.ndim != 1:
        raw = raw.reshape(-1)
    if not np.issubdtype(raw.dtype, np.integer):
        raise TypeError("delta_fs must be an integer array")
    values = raw.astype(np.int64, copy=False)
    n_valid = int(values.size)
    min_events = int(config.get("min_events", 10))
    if n_valid < min_events:
        return _failure(method=method, parameter=parameter, n_total=n_total,
                        n_selected=n_selected, n_valid=n_valid,
                        message=f"Only {n_valid} valid events; need {min_events}")

    center_fs, sigma0_fs = _robust_location_scale(values)
    width_ps = float(config.get("histogram_bin_ps", 10.0))
    if not np.isfinite(width_ps) or width_ps <= 0.0:
        raise ValueError("fit.histogram_bin_ps must be positive")
    bin_width_fs = max(1, int(np.rint(width_ps * FS_PER_PS)))
    phase_count = max(1, int(config.get("bin_phase_count", 10)))

    fits: list[dict[str, Any]] = []
    for phase_index in range(phase_count):
        phase_fraction = phase_index / phase_count
        fitted = _fit_histogram_all_events(
            values,
            bin_width_fs=bin_width_fs,
            phase_fraction=phase_fraction,
            initial_mean_fs=center_fs,
            initial_sigma_fs=sigma0_fs,
        )
        if fitted is not None:
            fits.append(fitted)
    if not fits:
        return _failure(method=method, parameter=parameter, n_total=n_total,
                        n_selected=n_selected, n_valid=n_valid,
                        message="Gaussian fit failed for every bin alignment")

    def quality(item: dict[str, Any]) -> tuple[float, float]:
        reduced = item["deviance"] / item["ndof"] if item["ndof"] > 0 else float("inf")
        # Bin width is fixed during the phase scan, so the phase with minimum
        # reduced Poisson deviance is the direct analogue of the requested
        # minimum-chi-square bin-origin search.
        return (float(reduced), float(item["deviance"]))

    best = min(fits, key=quality)
    phase_ctrs = np.asarray([FWHM_FACTOR * item["sigma_fs"] / FS_PER_PS for item in fits])
    sigma_fs = float(best["sigma_fs"])
    sigma_error_fs = float(best["sigma_error_fs"])
    return FitResult(
        method=method,
        parameter=float(parameter),
        success=True,
        n_total=int(n_total),
        n_selected=int(n_selected),
        n_valid=n_valid,
        n_fit=n_valid,
        crossing_efficiency=n_valid / n_selected if n_selected else 0.0,
        mean_ps=float(best["mean_fs"]) / FS_PER_PS,
        mean_error_ps=float(best["mean_error_fs"]) / FS_PER_PS,
        sigma_ps=sigma_fs / FS_PER_PS,
        sigma_error_ps=sigma_error_fs / FS_PER_PS,
        ctr_ps=FWHM_FACTOR * sigma_fs / FS_PER_PS,
        ctr_error_ps=FWHM_FACTOR * sigma_error_fs / FS_PER_PS,
        chi2=float(best["deviance"]),
        ndof=int(best["ndof"]),
        fit_low_ps=float(best["edges_fs"][0]) / FS_PER_PS,
        fit_high_ps=float(best["edges_fs"][-1]) / FS_PER_PS,
        iterations=int(best["iterations"]),
        message="",
        bin_width_ps=bin_width_fs / FS_PER_PS,
        bin_phase_ps=float(best["phase_fs"]) / FS_PER_PS,
        phase_ctr_std_ps=float(np.std(phase_ctrs, ddof=0)) if phase_ctrs.size > 1 else 0.0,
        edges_ps=np.asarray(best["edges_fs"], dtype=np.float64) / FS_PER_PS,
        counts=np.asarray(best["counts"], dtype=np.int64),
        expected=np.asarray(best["expected"], dtype=np.float64),
    )


# Compatibility helpers used by traditional timing scans.
def scan_timing_grid(
    times_a_fs: np.ndarray,
    times_b_fs: np.ndarray,
    selected: np.ndarray,
    parameters: np.ndarray,
    *,
    method: str,
    config: dict[str, Any],
) -> list[FitResult]:
    a_grid = np.asarray(times_a_fs)
    b_grid = np.asarray(times_b_fs)
    selected_mask = np.asarray(selected, dtype=bool)
    parameters = np.asarray(parameters, dtype=np.float64)
    if a_grid.shape != b_grid.shape:
        raise ValueError(f"{method} channel timing grids have different shapes")
    if a_grid.ndim != 2 or a_grid.shape[1] != parameters.size:
        raise ValueError(f"{method} timing-grid shape does not match parameter grid")
    if not np.issubdtype(a_grid.dtype, np.integer) or not np.issubdtype(b_grid.dtype, np.integer):
        raise TypeError(f"{method} timestamp arrays must contain integer fs ticks")
    if selected_mask.shape != (a_grid.shape[0],):
        raise ValueError("selection mask shape does not match timing arrays")

    n_total = int(selected_mask.size)
    n_selected = int(np.count_nonzero(selected_mask))
    results: list[FitResult] = []
    for index, parameter in enumerate(parameters):
        a = a_grid[selected_mask, index].astype(np.int64, copy=False)
        b = b_grid[selected_mask, index].astype(np.int64, copy=False)
        valid = (a != INVALID_TIME_FS) & (b != INVALID_TIME_FS)
        results.append(
            fit_delta_times_integer_fs(
                a[valid] - b[valid], method=method, parameter=float(parameter),
                n_total=n_total, n_selected=n_selected, config=config,
            )
        )
    return results


def choose_best(results: list[FitResult]) -> FitResult | None:
    successful = [item for item in results if item.success and np.isfinite(item.ctr_ps)]
    return min(successful, key=lambda item: item.ctr_ps) if successful else None
