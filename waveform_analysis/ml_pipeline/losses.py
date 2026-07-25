from __future__ import annotations

import torch


def residual_std_loss_ps(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Calibration-invariant loss: sqrt of residual variance, in ps.

    The residual is centered inside each batch, so a constant offset cannot
    improve the objective. This matches the proposed variance-reduction
    criterion while keeping the optimized and logged quantity in physical units.
    """

    residual = prediction - target
    centered = residual - torch.mean(residual)
    return torch.sqrt(torch.mean(centered * centered) + float(epsilon))
