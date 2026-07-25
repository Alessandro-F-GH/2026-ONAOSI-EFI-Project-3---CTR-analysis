from __future__ import annotations

from typing import Any

import torch
from torch import nn


SUPPORTED_MODEL_TYPES = ("cnn", "time_series_mlp", "catch22_random_forest")


def _activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


def model_type(config: dict[str, Any]) -> str:
    """Return the canonical model type, keeping old CNN configs compatible."""
    raw = str(config.get("model_type", "cnn")).strip().lower()
    aliases = {
        "cnn": "cnn",
        "conv1d": "cnn",
        "time_series_regressor": "time_series_mlp",
        "timeseries_regressor": "time_series_mlp",
        "time_series_mlp": "time_series_mlp",
        "mlp": "time_series_mlp",
        "catch22_random_forest": "catch22_random_forest",
        "catch22_rf": "catch22_random_forest",
        "random_forest": "catch22_random_forest",
        "rf": "catch22_random_forest",
    }
    if raw not in aliases:
        raise ValueError(
            f"Unsupported model_type {raw!r}. Supported values: {SUPPORTED_MODEL_TYPES}"
        )
    return aliases[raw]


def model_label(config: dict[str, Any]) -> str:
    kind = model_type(config)
    if kind == "cnn":
        return "CNN"
    if kind == "time_series_mlp":
        return "time-series MLP"
    if kind == "catch22_random_forest":
        catch24 = bool(config.get("features", {}).get("catch24", True))
        return "catch24 random forest" if catch24 else "catch22 random forest"
    raise AssertionError(kind)


def model_slug(config: dict[str, Any]) -> str:
    return model_type(config)


class _BoundedScalarOutput(nn.Module):
    """Apply an optional smooth symmetric bound to a scalar prediction."""

    def __init__(self, max_abs_output_ps: float | None) -> None:
        super().__init__()
        self.max_abs_output_ps = max_abs_output_ps

    def forward(self, output: torch.Tensor) -> torch.Tensor:
        if self.max_abs_output_ps is None:
            return output
        return self.max_abs_output_ps * torch.tanh(output / self.max_abs_output_ps)


class SingleChannelCNN(nn.Module):
    """Shared single-channel convolutional map g_theta."""

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
        self.output_bound = _BoundedScalarOutput(
            None if max_abs is None else float(max_abs)
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)
        if waveform.ndim != 3 or waveform.shape[1] != 1:
            raise ValueError("SingleChannelCNN expects [batch, length] or [batch, 1, length]")
        features = self.features(waveform).squeeze(-1)
        output = self.head(features).squeeze(-1)
        return self.output_bound(output)


class SingleChannelTimeSeriesMLP(nn.Module):
    """Standard fixed-window time-series regressor.

    The preprocessed waveform window is already fixed length.  This model therefore
    treats its samples as an ordered feature vector and applies a conventional MLP
    regressor.  It is deliberately simpler than the CNN: it has no convolutions,
    learned filters, or temporal pooling.
    """

    def __init__(self, config: dict[str, Any], input_length: int) -> None:
        super().__init__()
        if input_length <= 0:
            raise ValueError("input_length must be positive")
        architecture = config["architecture"]
        hidden_units = [int(value) for value in architecture.get("hidden_units", [])]
        use_batch_norm = bool(architecture.get("batch_norm", False))
        dropout = float(architecture.get("dropout", 0.0))
        activation_name = str(architecture.get("activation", "relu"))

        layers: list[nn.Module] = []
        in_features = int(input_length)
        for out_features in hidden_units:
            layers.append(nn.Linear(in_features, out_features, bias=not use_batch_norm))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_features))
            layers.append(_activation(activation_name))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = out_features
        layers.append(nn.Linear(in_features, 1))
        self.regressor = nn.Sequential(*layers)
        self.input_length = int(input_length)

        max_abs = architecture.get("max_abs_single_channel_output_ps")
        self.output_bound = _BoundedScalarOutput(
            None if max_abs is None else float(max_abs)
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 3 and waveform.shape[1] == 1:
            waveform = waveform[:, 0, :]
        if waveform.ndim != 2:
            raise ValueError(
                "SingleChannelTimeSeriesMLP expects [batch, length] or [batch, 1, length]"
            )
        if waveform.shape[1] != self.input_length:
            raise ValueError(
                f"Expected waveform length {self.input_length}, got {waveform.shape[1]}"
            )
        output = self.regressor(waveform).squeeze(-1)
        return self.output_bound(output)


class AntisymmetricCorrectionModel(nn.Module):
    """Generic y_theta(s1, s2) = g_theta(s1) - g_theta(s2) wrapper."""

    def __init__(self, shared: nn.Module) -> None:
        super().__init__()
        self.shared = shared

    def forward(
        self, waveform_pair: torch.Tensor, *, return_single_outputs: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if waveform_pair.ndim != 3 or waveform_pair.shape[1] != 2:
            raise ValueError("Model expects waveform pairs with shape [batch, 2, length]")
        # Both channels are evaluated in one shared batch.  This guarantees the same
        # parameters and, when present, the same normalization statistics for s1/s2.
        batch, _, length = waveform_pair.shape
        shared_output = self.shared(waveform_pair.reshape(batch * 2, length))
        shared_output = shared_output.reshape(batch, 2)
        g1 = shared_output[:, 0]
        g2 = shared_output[:, 1]
        correction = g1 - g2
        if return_single_outputs:
            return correction, g1, g2
        return correction


class AntisymmetricCorrectionCNN(AntisymmetricCorrectionModel):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(SingleChannelCNN(config))


class AntisymmetricCorrectionTimeSeriesMLP(AntisymmetricCorrectionModel):
    def __init__(self, config: dict[str, Any], input_length: int) -> None:
        super().__init__(SingleChannelTimeSeriesMLP(config, input_length))


def build_correction_model(
    config: dict[str, Any], *, input_length: int
) -> AntisymmetricCorrectionModel:
    kind = model_type(config)
    if kind == "cnn":
        return AntisymmetricCorrectionCNN(config)
    if kind == "time_series_mlp":
        return AntisymmetricCorrectionTimeSeriesMLP(config, input_length)
    if kind == "catch22_random_forest":
        raise ValueError(
            "catch22_random_forest is a scikit-learn model; use the dedicated "
            "training/evaluation dispatch instead of build_correction_model"
        )
    raise AssertionError(kind)


def model_output_path(
    pipeline_config: dict[str, Any], key: str, model_config: dict[str, Any]
) -> "Path":
    """Return a model-specific output path while leaving shared caches unscoped."""
    from pathlib import Path

    base = Path(pipeline_config["paths"][key])
    separate = bool(pipeline_config["paths"].get("separate_model_outputs", True))
    # Keep the historical CNN paths unchanged so existing CNN caches/checkpoints
    # remain reusable. Alternative models receive their own subdirectory.
    if separate and model_type(model_config) != "cnn":
        return base / model_slug(model_config)
    return base
