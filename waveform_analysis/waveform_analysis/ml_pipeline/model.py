from __future__ import annotations

from typing import Any

import torch
from torch import nn


def _activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


class SingleChannelCNN(nn.Module):
    """Shared single-channel map g_theta used by the antisymmetric estimator."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        architecture = config["architecture"]
        conv_channels = [int(value) for value in architecture["conv_channels"]]
        kernels = [int(value) for value in architecture["kernel_sizes"]]
        pools = [int(value) for value in architecture["pool_sizes"]]
        use_batch_norm = bool(architecture.get("batch_norm", True))
        dropout = float(architecture.get("dropout", 0.0))
        activation_name = str(architecture.get("activation", "relu"))

        layers: list[nn.Module] = []
        in_channels = 1
        for out_channels, kernel, pool in zip(
            conv_channels, kernels, pools, strict=True
        ):
            layers.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel,
                    padding=kernel // 2,
                    bias=not use_batch_norm,
                )
            )
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_channels))
            layers.append(_activation(activation_name))
            if pool > 1:
                layers.append(nn.MaxPool1d(pool))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_channels = out_channels
        layers.append(nn.AdaptiveAvgPool1d(1))
        self.features = nn.Sequential(*layers)

        dense_units = [int(value) for value in architecture.get("dense_units", [])]
        dense: list[nn.Module] = []
        in_features = conv_channels[-1]
        for out_features in dense_units:
            dense.append(nn.Linear(in_features, out_features))
            dense.append(_activation(activation_name))
            if dropout > 0:
                dense.append(nn.Dropout(dropout))
            in_features = out_features
        dense.append(nn.Linear(in_features, 1))
        self.head = nn.Sequential(*dense)

        max_abs = architecture.get("max_abs_single_channel_output_ps")
        self.max_abs_output_ps = None if max_abs is None else float(max_abs)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)
        if waveform.ndim != 3 or waveform.shape[1] != 1:
            raise ValueError("SingleChannelCNN expects [batch, length] or [batch, 1, length]")
        features = self.features(waveform).squeeze(-1)
        output = self.head(features).squeeze(-1)
        if self.max_abs_output_ps is not None:
            output = self.max_abs_output_ps * torch.tanh(
                output / self.max_abs_output_ps
            )
        return output


class AntisymmetricCorrectionCNN(nn.Module):
    """y_theta(s1, s2) = g_theta(s1) - g_theta(s2)."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.shared = SingleChannelCNN(config)

    def forward(
        self, waveform_pair: torch.Tensor, *, return_single_outputs: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if waveform_pair.ndim != 3 or waveform_pair.shape[1] != 2:
            raise ValueError("Model expects waveform pairs with shape [batch, 2, length]")
        # Evaluate both channels in one shared batch. This is faster and makes
        # normalization layers use exactly the same batch statistics for s1 and s2.
        batch, _, length = waveform_pair.shape
        shared_output = self.shared(waveform_pair.reshape(batch * 2, length))
        shared_output = shared_output.reshape(batch, 2)
        g1 = shared_output[:, 0]
        g2 = shared_output[:, 1]
        correction = g1 - g2
        if return_single_outputs:
            return correction, g1, g2
        return correction
