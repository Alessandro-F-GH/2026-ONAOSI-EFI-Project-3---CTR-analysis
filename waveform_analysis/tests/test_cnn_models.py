from __future__ import annotations

import torch

from utils.cnn_models import DirectTOFCNN, TranslationInvariantCorrectionCNN


DIRECT_CONFIG = {
    "conv_channels": [8, 8, 16, 16],
    "kernel_sizes": [11, 7, 5, 5],
    "pool_after": [1, 3],
    "pool_size": 2,
    "adaptive_pool_bins": 2,
    "fc_units": 16,
    "dropout": 0.1,
    "activation": "relu",
}

CORRECTION_CONFIG = {
    "conv_channels": [8, 8, 16, 16],
    "kernel_sizes": [11, 7, 5, 5],
    "pool_after": [1, 3],
    "pool_size": 2,
    "adaptive_pool_bins": 2,
    "embedding_units": 16,
    "head_units": 8,
    "dropout": 0.1,
    "activation": "relu",
}


def test_direct_model_output_shape() -> None:
    model = DirectTOFCNN(DIRECT_CONFIG)
    output = model(torch.randn(5, 2, 35))
    assert output.shape == (5,)


def test_correction_model_is_antisymmetric() -> None:
    model = TranslationInvariantCorrectionCNN(CORRECTION_CONFIG)
    model.eval()
    values = torch.randn(6, 2, 35)
    with torch.no_grad():
        forward = model(values)
        swapped = model(values[:, [1, 0], :])
    assert torch.allclose(forward, -swapped, atol=1e-6)
