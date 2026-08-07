from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_matched_effective_ridge.py"
SPEC = importlib.util.spec_from_file_location("run_matched_effective_ridge", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_effective_df_is_monotone_in_alpha() -> None:
    singular = np.array([5.0, 2.0, 0.5])
    low = MODULE.effective_df_from_singular_values(singular, 0.1)
    high = MODULE.effective_df_from_singular_values(singular, 10.0)
    assert low > high
    assert 0.0 < high < 3.0


@pytest.mark.parametrize("target", [0.25, 1.0, 1.75, 2.5])
def test_alpha_matches_requested_effective_df(target: float) -> None:
    singular = np.array([7.0, 3.0, 1.0])
    alpha, achieved, rank = MODULE.alpha_for_target_effective_df(
        singular,
        target,
        matrix_shape=(100, 3),
        absolute_tolerance=1.0e-10,
    )
    assert alpha > 0.0
    assert rank == 3
    assert achieved == pytest.approx(target, abs=1.0e-8)
    assert MODULE.effective_df_from_singular_values(singular, alpha) == pytest.approx(
        target, abs=1.0e-8
    )


def test_rank_target_is_ols_limit() -> None:
    singular = np.array([4.0, 2.0, 1.0])
    alpha, achieved, rank = MODULE.alpha_for_target_effective_df(
        singular,
        3.0,
        matrix_shape=(50, 3),
    )
    assert alpha == 0.0
    assert achieved == 3.0
    assert rank == 3


def test_infeasible_target_is_rejected() -> None:
    singular = np.array([4.0, 2.0, 0.0])
    with pytest.raises(ValueError, match="exceeds training design rank"):
        MODULE.alpha_for_target_effective_df(
            singular,
            2.5,
            matrix_shape=(50, 3),
        )


def test_ridge_bias_calibration_zeroes_training_mean_residual() -> None:
    rng = np.random.default_rng(17)
    x = rng.normal(size=(200, 8))
    coefficient_true = rng.normal(size=8)
    y = x @ coefficient_true + 3.5 + rng.normal(scale=0.1, size=200)
    coefficient, pair_bias = MODULE._fit_ridge(
        x,
        y,
        2.0,
        solver="svd",
        tolerance=1.0e-10,
        max_iterations=10000,
    )
    residual = y - (x @ coefficient + pair_bias)
    assert float(np.mean(residual)) == pytest.approx(0.0, abs=1.0e-12)
