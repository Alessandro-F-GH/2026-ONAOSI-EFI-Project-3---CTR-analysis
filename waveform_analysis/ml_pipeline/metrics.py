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
