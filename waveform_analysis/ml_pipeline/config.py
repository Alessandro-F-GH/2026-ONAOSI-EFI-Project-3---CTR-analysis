from __future__ import annotations

import copy
from typing import Any


class MLConfigError(ValueError):
    """Raised when a study/model configuration violates the supported protocol."""


def resolve_fit_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Return the single shared CTR-fit configuration used by every evaluator.

    Histogram width is fixed globally.  The only histogram degree of freedom is
    the bin-origin alignment scanned over ``bin_phase_count`` equally spaced
    phases in one bin width.
    """

    fit = copy.deepcopy(config or {})
    if "adaptive_binning" in fit:
        raise MLConfigError(
            "fit.adaptive_binning is obsolete; use fixed fit.histogram_bin_ps "
            "and fit.bin_phase_count"
        )
    fit.setdefault("histogram_bin_ps", 10.0)
    fit.setdefault("bin_phase_count", 10)
    fit.setdefault("min_events", 10)

    if int(fit["min_events"]) < 3:
        raise MLConfigError("fit.min_events must be >= 3")
    if float(fit["histogram_bin_ps"]) <= 0:
        raise MLConfigError("fit.histogram_bin_ps must be positive")
    if int(fit["bin_phase_count"]) < 1:
        raise MLConfigError("fit.bin_phase_count must be >= 1")
    return fit
