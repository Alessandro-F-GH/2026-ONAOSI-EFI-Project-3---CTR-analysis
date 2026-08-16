from __future__ import annotations

import copy
from typing import Any


class MLConfigError(ValueError):
    """Raised when a study/model configuration violates the supported protocol."""


def resolve_fit_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Return the single shared CTR-fit configuration used by every evaluator."""

    fit = copy.deepcopy(config or {})
    fit.setdefault("histogram_bin_ps", 10.0)  # fallback when adaptive binning is disabled
    fit.setdefault("min_events", 10)
    adaptive = fit.setdefault("adaptive_binning", {})
    if not isinstance(adaptive, dict):
        raise MLConfigError("fit.adaptive_binning must be an object")
    adaptive.setdefault("enabled", True)
    adaptive.setdefault("bins_per_fwhm", 10.0)
    adaptive.setdefault("min_bin_ps", 1.0)
    adaptive.setdefault("max_bin_ps", 25.0)
    adaptive.setdefault("phase_count", 8)

    if int(fit["min_events"]) < 3:
        raise MLConfigError("fit.min_events must be >= 3")
    if float(fit["histogram_bin_ps"]) <= 0:
        raise MLConfigError("fit.histogram_bin_ps must be positive")
    if bool(adaptive["enabled"]):
        if float(adaptive["bins_per_fwhm"]) <= 0:
            raise MLConfigError("fit.adaptive_binning.bins_per_fwhm must be positive")
        if float(adaptive["min_bin_ps"]) <= 0 or float(adaptive["max_bin_ps"]) <= 0:
            raise MLConfigError("adaptive bin limits must be positive")
        if float(adaptive["min_bin_ps"]) > float(adaptive["max_bin_ps"]):
            raise MLConfigError("adaptive min_bin_ps cannot exceed max_bin_ps")
        if int(adaptive["phase_count"]) < 1:
            raise MLConfigError("fit.adaptive_binning.phase_count must be >= 1")
    return fit
