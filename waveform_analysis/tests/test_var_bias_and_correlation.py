from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ml_pipeline.evaluation import _pairwise_model_output_correlation
from ml_pipeline.losses import var_bias_loss, var_bias_value_from_metrics
from ml_pipeline.plots import plot_model_output_correlation


def test_var_bias_uses_population_variance_plus_existing_bias_penalty() -> None:
    prediction = torch.tensor([1.0, 3.0, 5.0], dtype=torch.float64)
    target = torch.zeros(3, dtype=torch.float64)

    loss, penalty = var_bias_loss(
        prediction,
        target,
        bias_weight=0.5,
        target_scale=2.0,
    )

    expected_variance = 8.0 / 3.0
    expected_penalty = 0.5 * (3.0 / 2.0) ** 2
    assert torch.isclose(
        loss,
        torch.tensor(expected_variance + expected_penalty, dtype=torch.float64),
    )
    assert torch.isclose(
        penalty,
        torch.tensor(expected_penalty, dtype=torch.float64),
    )


def test_var_bias_base_term_is_invariant_to_constant_residual_shift() -> None:
    target = torch.zeros(4, dtype=torch.float64)
    first = torch.tensor([-2.0, -1.0, 1.0, 2.0], dtype=torch.float64)
    shifted = first + 100.0

    first_loss, _ = var_bias_loss(first, target, bias_weight=0.0)
    shifted_loss, _ = var_bias_loss(shifted, target, bias_weight=0.0)

    assert torch.isclose(first_loss, shifted_loss)


def test_metric_form_matches_torch_var_bias_objective() -> None:
    residual = np.asarray([-4.0, -1.0, 2.0, 7.0], dtype=np.float64)
    rmse = float(np.sqrt(np.mean(residual**2)))
    bias = float(np.mean(residual))
    value, variance, penalty = var_bias_value_from_metrics(
        rmse_ps=rmse,
        bias_ps=bias,
        bias_weight=3.0,
        target_scale=5.0,
    )
    expected_variance = float(np.var(residual, ddof=0))
    expected_penalty = 3.0 * (bias / 5.0) ** 2
    assert np.isclose(variance, expected_variance)
    assert np.isclose(penalty, expected_penalty)
    assert np.isclose(value, expected_variance + expected_penalty)


def test_pairwise_model_output_correlation_and_plot(tmp_path: Path) -> None:
    outputs = [
        ("model_a", np.asarray([1.0, 2.0, 3.0, 4.0])),
        ("model_b", np.asarray([2.0, 4.0, 6.0, 8.0])),
        ("model_c", np.asarray([-1.0, -2.0, -3.0, -4.0])),
        ("constant", np.asarray([5.0, 5.0, 5.0, 5.0])),
    ]

    labels, matrix, counts = _pairwise_model_output_correlation(outputs)
    assert labels == ["model_a", "model_b", "model_c", "constant"]
    np.testing.assert_allclose(matrix[:3, :3], [[1, 1, -1], [1, 1, -1], [-1, -1, 1]])
    assert np.isnan(matrix[0, 3])
    assert matrix[3, 3] == 1.0
    assert np.all(counts == 4)

    output = tmp_path / "correlation.png"
    plot_model_output_correlation(matrix, labels, output, dpi=72, annotate=True)
    assert output.is_file()
    assert output.stat().st_size > 0
