from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit


@dataclass
class PhotopeakResult:
    channel: int
    success: bool
    mean_mV: float
    sigma_mV: float
    mean_error_mV: float
    sigma_error_mV: float
    selection_low_mV: float
    selection_high_mV: float
    chi2: float
    ndof: int
    iterations: int
    message: str = ""
    edges_mV: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    counts: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    expected: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    fit_low_mV: float = np.nan
    fit_high_mV: float = np.nan

    @property
    def chi2_ndof(self) -> float:
        return self.chi2 / self.ndof if self.ndof > 0 else np.nan

    def as_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "success": self.success,
            "mean_mV": self.mean_mV,
            "sigma_mV": self.sigma_mV,
            "mean_error_mV": self.mean_error_mV,
            "sigma_error_mV": self.sigma_error_mV,
            "selection_low_mV": self.selection_low_mV,
            "selection_high_mV": self.selection_high_mV,
            "chi2": self.chi2,
            "ndof": self.ndof,
            "chi2_ndof": self.chi2_ndof,
            "iterations": self.iterations,
            "fit_low_mV": self.fit_low_mV,
            "fit_high_mV": self.fit_high_mV,
            "message": self.message,
        }


def _gaussian(x: np.ndarray, amplitude: float, mean: float, sigma: float) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def fit_photopeak(
    amplitudes_mV: np.ndarray,
    *,
    channel: int,
    config: dict[str, Any],
) -> PhotopeakResult:
    values = np.asarray(amplitudes_mV, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 20:
        return PhotopeakResult(
            channel=channel,
            success=False,
            mean_mV=np.nan,
            sigma_mV=np.nan,
            mean_error_mV=np.nan,
            sigma_error_mV=np.nan,
            selection_low_mV=np.nan,
            selection_high_mV=np.nan,
            chi2=np.nan,
            ndof=0,
            iterations=0,
            message="Too few finite amplitudes",
        )

    bin_width = float(config["histogram_bin_mV"])
    low_edge = np.floor(np.min(values) / bin_width) * bin_width
    high_edge = np.ceil(np.max(values) / bin_width) * bin_width
    if high_edge <= low_edge:
        high_edge = low_edge + bin_width
    edges = np.arange(low_edge, high_edge + 1.01 * bin_width, bin_width, dtype=np.float64)
    counts, edges = np.histogram(values, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])

    quantile_cut = float(np.quantile(values, float(config["search_quantile_min"])))
    search = centers >= quantile_cut
    if not np.any(search):
        search = np.ones_like(centers, dtype=bool)
    smooth = gaussian_filter1d(
        counts.astype(np.float64),
        sigma=float(config["smoothing_sigma_bins"]),
        mode="nearest",
    )
    candidate_indices = np.flatnonzero(search)
    peak_index = int(candidate_indices[np.argmax(smooth[candidate_indices])])
    mean = float(centers[peak_index])
    sigma = max(bin_width, float(config["initial_half_width_mV"]) / 3.0)
    fit_low = mean - float(config["initial_half_width_mV"])
    fit_high = mean + float(config["initial_half_width_mV"])
    max_iterations = int(config["max_iterations"])
    iteration_sigma = float(config["iteration_sigma"])
    tolerance = float(config["convergence_tolerance_mV"])

    covariance = None
    fitted_counts = np.empty(0)
    fitted_centers = np.empty(0)
    iterations = 0
    message = ""
    success = False

    for iteration in range(max_iterations):
        mask = (centers >= fit_low) & (centers <= fit_high)
        x = centers[mask]
        y = counts[mask].astype(np.float64)
        if x.size < 7 or np.count_nonzero(y) < 4:
            message = "Too few populated bins in photopeak fit interval"
            break
        amplitude0 = max(float(np.max(y)), 1.0)
        p0 = [amplitude0, mean, max(sigma, 0.5 * bin_width)]
        uncertainty = np.sqrt(np.maximum(y, 1.0))
        try:
            parameters, covariance = curve_fit(
                _gaussian,
                x,
                y,
                p0=p0,
                sigma=uncertainty,
                absolute_sigma=True,
                bounds=(
                    [0.0, fit_low, 0.1 * bin_width],
                    [np.inf, fit_high, max(high_edge - low_edge, bin_width)],
                ),
                maxfev=20_000,
            )
        except Exception as exc:  # scipy exposes several optimizer exception types
            message = f"Photopeak fit failed: {exc}"
            break
        amplitude, new_mean, new_sigma = [float(item) for item in parameters]
        if not np.all(np.isfinite(parameters)) or new_sigma <= 0:
            message = "Photopeak fit returned invalid parameters"
            break
        iterations = iteration + 1
        fitted_centers = x
        fitted_counts = _gaussian(x, amplitude, new_mean, new_sigma)
        converged = abs(new_mean - mean) <= tolerance and abs(new_sigma - sigma) <= tolerance
        mean, sigma = new_mean, new_sigma
        fit_low = mean - iteration_sigma * sigma
        fit_high = mean + iteration_sigma * sigma
        success = True
        if converged:
            break

    if not success:
        return PhotopeakResult(
            channel=channel,
            success=False,
            mean_mV=np.nan,
            sigma_mV=np.nan,
            mean_error_mV=np.nan,
            sigma_error_mV=np.nan,
            selection_low_mV=np.nan,
            selection_high_mV=np.nan,
            chi2=np.nan,
            ndof=0,
            iterations=iterations,
            message=message or "Photopeak fit did not converge",
            edges_mV=edges,
            counts=counts,
        )

    final_mask = (centers >= fit_low) & (centers <= fit_high)
    final_x = centers[final_mask]
    final_y = counts[final_mask].astype(np.float64)
    amplitude = float(np.max(fitted_counts)) if fitted_counts.size else float(np.max(final_y))
    # Refit once on the final interval so the stored curve and chi-square correspond
    # exactly to the displayed range.
    uncertainty = np.sqrt(np.maximum(final_y, 1.0))
    try:
        final_parameters, covariance = curve_fit(
            _gaussian,
            final_x,
            final_y,
            p0=[amplitude, mean, sigma],
            sigma=uncertainty,
            absolute_sigma=True,
            bounds=(
                [0.0, fit_low, 0.1 * bin_width],
                [np.inf, fit_high, max(high_edge - low_edge, bin_width)],
            ),
            maxfev=20_000,
        )
        amplitude, mean, sigma = [float(item) for item in final_parameters]
    except Exception:
        final_parameters = np.array([amplitude, mean, sigma], dtype=np.float64)
    expected = _gaussian(centers, amplitude, mean, sigma)
    expected_fit = _gaussian(final_x, amplitude, mean, sigma)
    variance = np.maximum(expected_fit, 1.0)
    chi2 = float(np.sum((final_y - expected_fit) ** 2 / variance))
    ndof = max(0, int(final_x.size) - 3)

    errors = np.full(3, np.nan)
    if covariance is not None and covariance.shape == (3, 3):
        diagonal = np.diag(covariance)
        errors = np.sqrt(np.where(diagonal >= 0, diagonal, np.nan))

    selection_low = mean + float(config["selection_sigma_low"]) * sigma
    selection_high = mean + float(config["selection_sigma_high"]) * sigma
    return PhotopeakResult(
        channel=channel,
        success=True,
        mean_mV=mean,
        sigma_mV=sigma,
        mean_error_mV=float(errors[1]),
        sigma_error_mV=float(errors[2]),
        selection_low_mV=selection_low,
        selection_high_mV=selection_high,
        chi2=chi2,
        ndof=ndof,
        iterations=iterations,
        message="",
        edges_mV=edges,
        counts=counts,
        expected=expected,
        fit_low_mV=fit_low,
        fit_high_mV=fit_high,
    )


def photopeak_mask(values_mV: np.ndarray, result: PhotopeakResult) -> np.ndarray:
    values = np.asarray(values_mV, dtype=np.float64)
    if not result.success:
        return np.zeros(values.shape, dtype=bool)
    return (
        np.isfinite(values)
        & (values >= result.selection_low_mV)
        & (values <= result.selection_high_mV)
    )
