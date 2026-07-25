from __future__ import annotations

from typing import Any

import torch
from torch import nn


class DirectTOFCNN(nn.Module):
    """Paper-inspired CNN that predicts TOF directly from a waveform pair.

    The first 2-D convolution spans both detector waveforms, as in Berg and
    Cherry (2018). Subsequent convolutions operate along the time axis.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        channels = [int(item) for item in config["conv_channels"]]
        kernels = [int(item) for item in config["kernel_sizes"]]
        pool_after = {int(item) for item in config.get("pool_after", [])}
        pool_size = int(config.get("pool_size", 2))
        activation_name = str(config.get("activation", "relu")).lower()
        activation: type[nn.Module]
        if activation_name == "gelu":
            activation = nn.GELU
        elif activation_name == "relu":
            activation = nn.ReLU
        else:
            raise ValueError(f"unsupported activation: {activation_name}")

        self.first = nn.Sequential(
            nn.Conv2d(
                1,
                channels[0],
                kernel_size=(2, kernels[0]),
                padding=(0, kernels[0] // 2),
            ),
            activation(),
        )
        layers: list[nn.Module] = []
        in_channels = channels[0]
        if 0 in pool_after:
            layers.append(nn.MaxPool1d(pool_size))
        for index in range(1, len(channels)):
            layers.extend(
                [
                    nn.Conv1d(
                        in_channels,
                        channels[index],
                        kernel_size=kernels[index],
                        padding=kernels[index] // 2,
                    ),
                    activation(),
                ]
            )
            in_channels = channels[index]
            if index in pool_after:
                layers.append(nn.MaxPool1d(pool_size))
        self.temporal = nn.Sequential(*layers)
        pool_bins = int(config.get("adaptive_pool_bins", 4))
        self.pool = nn.AdaptiveAvgPool1d(pool_bins)
        fc_units = int(config.get("fc_units", 256))
        dropout = float(config.get("dropout", 0.5))
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * pool_bins, fc_units),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(fc_units, 1),
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        if waveforms.ndim != 3 or waveforms.shape[1] != 2:
            raise ValueError("DirectTOFCNN expects input shape [batch, 2, samples]")
        values = self.first(waveforms.unsqueeze(1)).squeeze(2)
        values = self.temporal(values)
        values = self.pool(values)
        return self.regressor(values).squeeze(-1)


class SharedWaveformEncoder(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        channels = [int(item) for item in config["conv_channels"]]
        kernels = [int(item) for item in config["kernel_sizes"]]
        pool_after = {int(item) for item in config.get("pool_after", [])}
        pool_size = int(config.get("pool_size", 2))
        activation_name = str(config.get("activation", "relu")).lower()
        activation: type[nn.Module]
        if activation_name == "gelu":
            activation = nn.GELU
        elif activation_name == "relu":
            activation = nn.ReLU
        else:
            raise ValueError(f"unsupported activation: {activation_name}")
        layers: list[nn.Module] = []
        in_channels = 1
        for index, (out_channels, kernel) in enumerate(zip(channels, kernels, strict=True)):
            layers.extend(
                [
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel,
                        padding=kernel // 2,
                    ),
                    activation(),
                ]
            )
            in_channels = out_channels
            if index in pool_after:
                layers.append(nn.MaxPool1d(pool_size))
        self.layers = nn.Sequential(*layers)
        pool_bins = int(config.get("adaptive_pool_bins", 4))
        self.pool = nn.AdaptiveAvgPool1d(pool_bins)
        embedding_units = int(config.get("embedding_units", 128))
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * pool_bins, embedding_units),
            activation(),
        )
        self.output_units = embedding_units

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 2:
            raise ValueError("SharedWaveformEncoder expects shape [batch, samples]")
        values = self.layers(waveform.unsqueeze(1))
        values = self.pool(values)
        return self.projection(values)


class TranslationInvariantCorrectionCNN(nn.Module):
    """Siamese timing-correction model.

    Each timing waveform is cropped around its own threshold crossing. The same
    encoder and scalar timing head are used for both detector branches. Their
    difference predicts the centered LED timing error. Absolute source position
    is deliberately absent from the input.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.encoder = SharedWaveformEncoder(config)
        hidden = int(config.get("head_units", 64))
        dropout = float(config.get("dropout", 0.2))
        self.head = nn.Sequential(
            nn.Linear(self.encoder.output_units, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        if waveforms.ndim != 3 or waveforms.shape[1] != 2:
            raise ValueError(
                "TranslationInvariantCorrectionCNN expects [batch, 2, samples]"
            )
        correction3 = self.head(self.encoder(waveforms[:, 0, :])).squeeze(-1)
        correction4 = self.head(self.encoder(waveforms[:, 1, :])).squeeze(-1)
        return correction3 - correction4


def build_model(model_type: str, model_config: dict[str, Any]) -> nn.Module:
    if model_type == "direct":
        return DirectTOFCNN(model_config["direct_model"])
    if model_type == "correction":
        return TranslationInvariantCorrectionCNN(model_config["correction_model"])
    raise ValueError(f"unknown model type: {model_type}")
