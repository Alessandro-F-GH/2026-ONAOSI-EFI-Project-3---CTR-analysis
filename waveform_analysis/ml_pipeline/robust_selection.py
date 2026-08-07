from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RobustLocationScale:
    center_ps: float
    scale_ps: float
    method: str
    training_event_count: int


def fit_median_mad_z(values_ps: np.ndarray) -> RobustLocationScale:
    """Fit a robust center and Gaussian-consistent scale.

    The primary scale is ``1.4826 * MAD``.  If MAD is exactly zero, the
    Gaussian-consistent IQR scale is used.  If both are zero, the distribution
    is constant and the scale is reported as infinity so every finite value at
    the center is retained without introducing an arbitrary epsilon.
    """

    values = np.asarray(values_ps, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size < 3:
        raise ValueError("At least three finite training values are required for robust z selection")

    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    scale = 1.4826 * mad
    method = "median_mad"
    if not np.isfinite(scale) or scale <= 0.0:
        q25, q75 = np.percentile(values, [25.0, 75.0])
        iqr = float(q75 - q25)
        scale = iqr / 1.349 if iqr > 0.0 else float("inf")
        method = "median_iqr_fallback" if np.isfinite(scale) else "constant_distribution"
    return RobustLocationScale(
        center_ps=center,
        scale_ps=float(scale),
        method=method,
        training_event_count=int(values.size),
    )


def robust_z_mask(
    values_ps: np.ndarray,
    fitted: RobustLocationScale,
    z_threshold: float,
) -> np.ndarray:
    threshold = float(z_threshold)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("z_threshold must be finite and positive")
    values = np.asarray(values_ps, dtype=np.float64)
    finite = np.isfinite(values)
    if np.isinf(fitted.scale_ps):
        return finite & np.isclose(values, fitted.center_ps, rtol=0.0, atol=0.0)
    if not np.isfinite(fitted.scale_ps) or fitted.scale_ps <= 0.0:
        raise ValueError("Robust scale must be positive, finite, or infinity")
    z = np.abs(values - fitted.center_ps) / fitted.scale_ps
    return finite & (z <= threshold)
