from __future__ import annotations

from typing import Any

import numpy as np

from utils.fit import FitResult, fit_delta_times_integer_fs


def fit_times_ps(values_ps: np.ndarray, method: str, fit_config: dict[str, Any]) -> FitResult:
    values = np.asarray(values_ps, dtype=np.float64).reshape(-1)
    values_fs = np.rint(values * 1000.0).astype(np.int64)
    return fit_delta_times_integer_fs(
        values_fs,
        method=method,
        parameter=0.0,
        n_total=int(values_fs.size),
        n_selected=int(values_fs.size),
        config=fit_config,
    )


def distribution_metrics(
    values_ps: np.ndarray,
    *,
    true_value_ps: float,
    fit: FitResult,
) -> dict[str, Any]:
    values = np.asarray(values_ps, dtype=np.float64).reshape(-1)
    return {
        "event_count": int(values.size),
        "true_value_ps": float(true_value_ps),
        "ctr_ps": float(fit.ctr_ps) if fit.success else float("nan"),
        "ctr_error_ps": float(fit.ctr_error_ps) if fit.success else float("nan"),
        "gaussian_mean_ps": float(fit.mean_ps) if fit.success else float("nan"),
        "gaussian_mean_error_ps": float(fit.mean_error_ps) if fit.success else float("nan"),
        "gaussian_bias_ps": float(fit.mean_ps - true_value_ps) if fit.success else float("nan"),
        "arithmetic_mean_ps": float(np.mean(values)),
        "arithmetic_bias_ps": float(np.mean(values) - true_value_ps),
        "standard_deviation_ps": float(np.std(values, ddof=0)),
        "chi2_ndof": float(fit.chi2_ndof) if fit.success else float("nan"),
        "fit_success": bool(fit.success),
        "fit_message": str(fit.message),
    }


FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


def residual_metrics(values_ps: np.ndarray) -> dict[str, Any]:
    """All-event CTR metrics used by the study/reporting layer.

    This deliberately does not fit or clip the distribution: CTR is
    ``2*sqrt(2*ln(2))*sample_std`` over every finite event in the requested
    evaluation split.  Model training itself is unchanged and continues to use
    the model-specific losses/selection logic from the working pipeline.
    """
    values = np.asarray(values_ps, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n == 0:
        return {
            "n": 0,
            "mean_ps": float("nan"),
            "std_ps": float("nan"),
            "ctr_ps": float("nan"),
            "rmse_ps": float("nan"),
            "bias_ps": float("nan"),
        }
    mean = float(np.mean(values))
    rmse = float(np.sqrt(np.mean(values * values)))
    if n < 2:
        std = float("nan")
        ctr = float("nan")
    else:
        std = float(np.std(values, ddof=1))
        ctr = float(FWHM_PER_SIGMA * std)
    return {
        "n": n,
        "mean_ps": mean,
        "std_ps": std,
        "ctr_ps": ctr,
        "rmse_ps": rmse,
        "bias_ps": mean,
    }


def ctr_bootstrap_uncertainty(
    values_ps: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 12345,
) -> float:
    """Statistical uncertainty of fixed-model CTR from event resampling only."""
    values = np.asarray(values_ps, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size < 3 or int(n_bootstrap) <= 1:
        return float("nan")
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(n_bootstrap), dtype=np.float64)
    for index in range(draws.size):
        sample = rng.choice(values, size=values.size, replace=True)
        draws[index] = FWHM_PER_SIGMA * np.std(sample, ddof=1)
    return float(np.std(draws, ddof=1))
