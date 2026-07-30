from __future__ import annotations
import torch

def mse_residual_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    residual = prediction - target
    return torch.mean(residual.square())

def mse_bias_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    bias_weight: float,
    target_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if float(bias_weight) < 0.0:
        raise ValueError("bias_weight must be non-negative")
    residual = prediction - target
    mse = torch.mean(residual.square())
    scale = max(float(target_scale), 1e-12)
    penalty = float(bias_weight) * (torch.mean(residual) / scale).square()
    return mse + penalty, penalty

def residual_rmse_loss_ps(prediction: torch.Tensor, target: torch.Tensor, *, epsilon: float = 1e-12) -> torch.Tensor:
    residual = prediction - target
    return torch.sqrt(torch.mean(residual.square()) + float(epsilon))
