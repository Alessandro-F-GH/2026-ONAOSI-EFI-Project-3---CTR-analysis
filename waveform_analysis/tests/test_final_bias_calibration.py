from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ml_pipeline.models.mlp_regressor import (
    AntisymmetricMLPRegressor,
    _apply_zero_bias_constraint,
)
from ml_pipeline.training_utils import predict_loader


def test_residual_mean_calibration_zeroes_arithmetic_train_bias() -> None:
    waveforms = torch.zeros((6, 2, 5), dtype=torch.float32)
    target = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    led = target.clone()
    cfd = target.clone()
    true_tof = torch.zeros_like(target)
    model = AntisymmetricMLPRegressor(
        {
            "hidden_units": [],
            "activation": "identity",
            "dropout": 0.0,
            "batch_norm": False,
        },
        input_length=5,
    )
    loader = DataLoader(
        TensorDataset(waveforms, target, led, cfd, true_tof),
        batch_size=3,
        shuffle=False,
    )

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    adjustment, _ = _apply_zero_bias_constraint(
        model, loader, torch.device("cpu"), mode="residual_mean"
    )
    prediction = predict_loader(model, loader, torch.device("cpu"))

    assert np.isclose(adjustment, float(torch.mean(target)), atol=1.0e-7)
    assert abs(float(np.mean(prediction["residual_ps"]))) < 1.0e-7
