from __future__ import annotations

import torch


def mse_residual_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    residual = prediction - target
    return torch.mean(residual.square())


def var_bias_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    bias_weight: float,
    target_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Population residual variance plus the existing squared-bias penalty.

    The residual variance is evaluated with ``correction=0`` semantics:

    ``mean((residual - mean(residual))**2)``.

    The bias penalty intentionally preserves the previous normalization and
    weighting convention used by ``mse_bias``.
    """

    if float(bias_weight) < 0.0:
        raise ValueError("bias_weight must be non-negative")
    residual = prediction - target
    residual_mean = torch.mean(residual)
    variance = torch.mean((residual - residual_mean).square())
    scale = max(float(target_scale), 1e-12)
    penalty = float(bias_weight) * (residual_mean / scale).square()
    return variance + penalty, penalty


def var_bias_value_from_metrics(
    *,
    rmse_ps: float,
    bias_ps: float,
    bias_weight: float,
    target_scale: float = 1.0,
) -> tuple[float, float, float]:
    """Return ``(objective, variance, bias_penalty)`` from global metrics."""

    if float(bias_weight) < 0.0:
        raise ValueError("bias_weight must be non-negative")
    variance = max(float(rmse_ps) ** 2 - float(bias_ps) ** 2, 0.0)
    scale = max(float(target_scale), 1e-12)
    penalty = float(bias_weight) * (float(bias_ps) / scale) ** 2
    return variance + penalty, variance, penalty


def residual_rmse_loss_ps(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    residual = prediction - target
    return torch.sqrt(torch.mean(residual.square()) + float(epsilon))
