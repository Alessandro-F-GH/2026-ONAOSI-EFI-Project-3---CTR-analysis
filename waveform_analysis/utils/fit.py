from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import curve_fit

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
            "message": self.message,
        }


def _gaussian(x_fs: np.ndarray, amplitude: float, mean_fs: float, sigma_fs: float) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((x_fs - mean_fs) / sigma_fs) ** 2)


def _failure(
    *, method: str, parameter: float, n_total: int, n_selected: int, n_valid: int, message: str
) -> FitResult:
    return FitResult(
        method=method,
        parameter=float(parameter),
        success=False,
        n_total=int(n_total),
        n_selected=int(n_selected),
        n_valid=int(n_valid),
        n_fit=0,
        crossing_efficiency=n_valid / n_selected if n_selected > 0 else 0.0,
        mean_ps=np.nan,
        mean_error_ps=np.nan,
        sigma_ps=np.nan,
        sigma_error_ps=np.nan,
        ctr_ps=np.nan,
        ctr_error_ps=np.nan,
        chi2=np.nan,
        ndof=0,
        fit_low_ps=np.nan,
        fit_high_ps=np.nan,
        iterations=0,
        message=message,
    )


def fit_delta_times_integer_fs(
    delta_fs: np.ndarray,
    *,
    method: str,
    parameter: float,
    n_total: int,
    n_selected: int,
    config: dict[str, Any],
) -> FitResult:
    """Iteratively fit a Gaussian to integer-femtosecond time differences.

    Raw event coordinates, histogram limits, and histogram edges remain int64 fs.
    The continuous Gaussian parameters are necessarily floating point but retain
    femtosecond units until the public result and plots are produced in ps.
    """
    raw = np.asarray(delta_fs)
    if raw.ndim != 1:
        raw = raw.reshape(-1)
    if not np.issubdtype(raw.dtype, np.integer):
        raise TypeError("delta_fs must be an integer array")
    values = raw.astype(np.int64, copy=False)
    n_valid = int(values.size)
    min_events = int(config["min_events"])
    if n_valid < min_events:
        return _failure(
            method=method,
            parameter=parameter,
            n_total=n_total,
            n_selected=n_selected,
            n_valid=n_valid,
            message=f"Only {n_valid} valid events; need {min_events}",
        )

    range_ps = config.get("histogram_range_ps")
    bin_width_fs = int(np.rint(float(config["histogram_bin_ps"]) * FS_PER_PS))
    if bin_width_fs <= 0:
        raise ValueError("fit.histogram_bin_ps must produce a positive integer-fs width")

    if range_ps is None:
        # Automatic range for mixed operating regimes.  Use a robust symmetric
        # interval whose bin grid remains anchored at zero, so changing tails does
        # not shift histogram bin centers between methods or data splits.
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            return _failure(
                method=method,
                parameter=parameter,
                n_total=n_total,
                n_selected=n_selected,
                n_valid=n_valid,
                message="No finite events available for automatic histogram range",
            )
        robust_abs_fs = float(np.quantile(np.abs(finite_values.astype(np.float64)), 0.999))
        minimum_half_fs = 2.0 * float(config["initial_half_width_ps"]) * FS_PER_PS
        requested_half_fs = max(robust_abs_fs * 1.10, minimum_half_fs, float(bin_width_fs))
        half_bins = max(1, int(np.ceil(requested_half_fs / bin_width_fs)))
        half_range_fs = half_bins * bin_width_fs
        low_fs = np.int64(-half_range_fs)
        high_fs = np.int64(half_range_fs)
    else:
        low_fs = np.int64(np.rint(float(range_ps[0]) * FS_PER_PS))
        high_fs = np.int64(np.rint(float(range_ps[1]) * FS_PER_PS))

    if high_fs <= low_fs:
        raise ValueError("invalid integer histogram settings")
    edges_fs = np.arange(
        int(low_fs),
        int(high_fs) + bin_width_fs,
        bin_width_fs,
        dtype=np.int64,
    )
    if edges_fs[-1] < high_fs:
        edges_fs = np.append(edges_fs, high_fs)
    counts, edges_fs = np.histogram(values, bins=edges_fs)
    centers_fs = ((edges_fs[:-1] + edges_fs[1:]) // 2).astype(np.int64)
    if counts.size == 0 or np.max(counts) <= 0:
        return _failure(
            method=method,
            parameter=parameter,
            n_total=n_total,
            n_selected=n_selected,
            n_valid=n_valid,
            message="No events inside configured histogram range",
        )

    peak_index = int(np.argmax(counts))
    mean_fs = float(centers_fs[peak_index])
    initial_half_fs = float(config["initial_half_width_ps"]) * FS_PER_PS
    sigma_fs = max(float(bin_width_fs), initial_half_fs / 4.0)
    fit_low_fs = int(np.rint(mean_fs - initial_half_fs))
    fit_high_fs = int(np.rint(mean_fs + initial_half_fs))
    iteration_sigma = float(config["iteration_sigma"])
    max_iterations = int(config["max_iterations"])
    tolerance_fs = float(config["convergence_tolerance_ps"]) * FS_PER_PS
    minimum_fit_bins = int(config["minimum_fit_bins"])
    min_sigma_fs = float(config["minimum_sigma_bins"]) * bin_width_fs

    covariance = None
    amplitude = float(np.max(counts))
    iterations = 0
    success = False
    message = ""

    for iteration in range(max_iterations):
        mask = (centers_fs >= fit_low_fs) & (centers_fs <= fit_high_fs)
        x_int = centers_fs[mask]
        y_int = counts[mask]
        if x_int.size < minimum_fit_bins or np.count_nonzero(y_int) < 4:
            message = "Too few populated bins in iterative fit interval"
            break
        x = x_int.astype(np.float64)
        y = y_int.astype(np.float64)
        uncertainty = np.sqrt(np.maximum(y, 1.0))
        p0 = [max(float(np.max(y)), 1.0), mean_fs, max(sigma_fs, min_sigma_fs)]
        try:
            parameters, covariance = curve_fit(
                _gaussian,
                x,
                y,
                p0=p0,
                sigma=uncertainty,
                absolute_sigma=True,
                bounds=(
                    [0.0, fit_low_fs, min_sigma_fs],
                    [np.inf, fit_high_fs, max(fit_high_fs - fit_low_fs, min_sigma_fs)],
                ),
                maxfev=30_000,
            )
        except Exception as exc:
            message = f"Gaussian fit failed: {exc}"
            break
        amplitude, new_mean_fs, new_sigma_fs = [float(item) for item in parameters]
        if not np.all(np.isfinite(parameters)) or new_sigma_fs <= 0:
            message = "Gaussian fit returned invalid parameters"
            break
        iterations = iteration + 1
        converged = (
            abs(new_mean_fs - mean_fs) <= tolerance_fs
            and abs(new_sigma_fs - sigma_fs) <= tolerance_fs
        )
        mean_fs, sigma_fs = new_mean_fs, new_sigma_fs
        fit_low_fs = int(np.rint(mean_fs - iteration_sigma * sigma_fs))
        fit_high_fs = int(np.rint(mean_fs + iteration_sigma * sigma_fs))
        success = True
        if converged:
            break

    if not success:
        return _failure(
            method=method,
            parameter=parameter,
            n_total=n_total,
            n_selected=n_selected,
            n_valid=n_valid,
            message=message or "Fit did not converge",
        )

    # Final fit on the exact interval that is reported and plotted.
    final_mask = (centers_fs >= fit_low_fs) & (centers_fs <= fit_high_fs)
    final_x_int = centers_fs[final_mask]
    final_y_int = counts[final_mask]
    if final_x_int.size < minimum_fit_bins:
        return _failure(
            method=method,
            parameter=parameter,
            n_total=n_total,
            n_selected=n_selected,
            n_valid=n_valid,
            message="Final fit interval contains too few bins",
        )
    final_x = final_x_int.astype(np.float64)
    final_y = final_y_int.astype(np.float64)
    uncertainty = np.sqrt(np.maximum(final_y, 1.0))
    try:
        parameters, covariance = curve_fit(
            _gaussian,
            final_x,
            final_y,
            p0=[amplitude, mean_fs, sigma_fs],
            sigma=uncertainty,
            absolute_sigma=True,
            bounds=(
                [0.0, fit_low_fs, min_sigma_fs],
                [np.inf, fit_high_fs, max(fit_high_fs - fit_low_fs, min_sigma_fs)],
            ),
            maxfev=30_000,
        )
        amplitude, mean_fs, sigma_fs = [float(item) for item in parameters]
    except Exception as exc:
        return _failure(
            method=method,
            parameter=parameter,
            n_total=n_total,
            n_selected=n_selected,
            n_valid=n_valid,
            message=f"Final Gaussian fit failed: {exc}",
        )

    expected_full = _gaussian(centers_fs.astype(np.float64), amplitude, mean_fs, sigma_fs)
    expected_fit = _gaussian(final_x, amplitude, mean_fs, sigma_fs)
    variance = np.maximum(expected_fit, 1.0)
    chi2 = float(np.sum((final_y - expected_fit) ** 2 / variance))
    ndof = max(0, int(final_x.size) - 3)
    n_fit = int(np.count_nonzero((values >= fit_low_fs) & (values <= fit_high_fs)))

    errors = np.full(3, np.nan)
    if covariance is not None and covariance.shape == (3, 3):
        diagonal = np.diag(covariance)
        errors = np.sqrt(np.where(diagonal >= 0, diagonal, np.nan))
    mean_error_fs = float(errors[1])
    sigma_error_fs = float(errors[2])

    return FitResult(
        method=method,
        parameter=float(parameter),
        success=True,
        n_total=int(n_total),
        n_selected=int(n_selected),
        n_valid=n_valid,
        n_fit=n_fit,
        crossing_efficiency=n_valid / n_selected if n_selected > 0 else 0.0,
        mean_ps=mean_fs / FS_PER_PS,
        mean_error_ps=mean_error_fs / FS_PER_PS,
        sigma_ps=sigma_fs / FS_PER_PS,
        sigma_error_ps=sigma_error_fs / FS_PER_PS,
        ctr_ps=FWHM_FACTOR * sigma_fs / FS_PER_PS,
        ctr_error_ps=FWHM_FACTOR * sigma_error_fs / FS_PER_PS,
        chi2=chi2,
        ndof=ndof,
        fit_low_ps=fit_low_fs / FS_PER_PS,
        fit_high_ps=fit_high_fs / FS_PER_PS,
        iterations=iterations,
        message="",
        edges_ps=edges_fs.astype(np.float64) / FS_PER_PS,
        counts=counts.astype(np.int64),
        expected=expected_full,
    )

#----------------
## OTHER UTILS ##
#----------------

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
        delta_fs = a[valid] - b[valid]
        results.append(
            fit_delta_times_integer_fs(
                delta_fs,
                method=method,
                parameter=float(parameter),
                n_total=n_total,
                n_selected=n_selected,
                config=config,
            )
        )
    return results


def choose_best(results: list[FitResult]) -> FitResult | None:
    successful = [item for item in results if item.success and np.isfinite(item.ctr_ps)]
    return min(successful, key=lambda item: item.ctr_ps) if successful else None
