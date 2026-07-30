from __future__ import annotations

import copy
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import BSpline

from ..dataset import PreparedDataset, load_prepared_dataset
from .led import led_delta_ps

_ALLOWED_AMPLITUDE_NORMALIZATION = {"none", "max_abs"}


@dataclass(frozen=True)
class LinearSplineArtifact:
    config: dict[str, Any]
    normalization_mean: float
    normalization_std: float
    feature_mean: np.ndarray
    feature_std: np.ndarray
    coefficients: np.ndarray
    input_length: int
    relative_time_ps: np.ndarray


def validate_linear_spline_config(config: dict[str, Any]) -> None:
    forbidden = {
        "loss",
        "optimizer",
        "selection_metric",
        "bias_weight",
        "bias_normalization",
    }
    overlap = sorted(forbidden.intersection(config))
    if overlap:
        raise ValueError(f"Linear spline does not accept ML parameters: {overlap}")
    n_basis = int(config.get("n_basis", 16))
    degree = int(config.get("spline_degree", 3))
    if degree < 0:
        raise ValueError("spline_degree must be non-negative")
    if n_basis <= degree:
        raise ValueError("n_basis must be greater than spline_degree")
    normalization = str(
        config.get("amplitude_normalization", config.get("normalize_amplitude", "none"))
    )
    if normalization not in _ALLOWED_AMPLITUDE_NORMALIZATION:
        raise ValueError("Invalid amplitude_normalization")
    if float(config.get("amplitude_epsilon", 1e-6)) <= 0.0:
        raise ValueError("amplitude_epsilon must be positive")
    if float(config.get("smoothness_penalty", 1e-3)) < 0.0:
        raise ValueError("smoothness_penalty must be non-negative")
    if float(config.get("ridge_penalty", 1e-10)) < 0.0:
        raise ValueError("ridge_penalty must be non-negative")


def _resolve_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    if "normalize_amplitude" in cfg and "amplitude_normalization" not in cfg:
        cfg["amplitude_normalization"] = (
            "max_abs" if bool(cfg.pop("normalize_amplitude")) else "none"
        )
    cfg.setdefault("n_basis", 16)
    cfg.setdefault("spline_degree", 3)
    cfg.setdefault("amplitude_normalization", "none")
    cfg.setdefault("amplitude_epsilon", 1e-6)
    cfg.setdefault("smoothness_penalty", 1e-3)
    cfg.setdefault("ridge_penalty", 1e-10)
    validate_linear_spline_config(cfg)
    return cfg


def build_bspline_basis(input_length: int, n_basis: int, degree: int) -> np.ndarray:
    input_length = int(input_length)
    n_basis = int(n_basis)
    degree = int(degree)
    if input_length <= 1:
        raise ValueError("input_length must be greater than one")
    if n_basis <= degree:
        raise ValueError("n_basis must be greater than degree")
    internal_count = n_basis - degree - 1
    internal = (
        np.linspace(0.0, 1.0, internal_count + 2, dtype=np.float64)[1:-1]
        if internal_count > 0
        else np.empty(0, dtype=np.float64)
    )
    knots = np.concatenate(
        [
            np.zeros(degree + 1, dtype=np.float64),
            internal,
            np.ones(degree + 1, dtype=np.float64),
        ]
    )
    time = np.linspace(0.0, 1.0, input_length, dtype=np.float64)
    basis = BSpline.design_matrix(time, knots, degree, extrapolate=False).toarray()
    expected = (input_length, n_basis)
    if basis.shape != expected:
        raise RuntimeError(f"Unexpected B-spline basis shape {basis.shape}; expected {expected}")
    return np.asarray(basis, dtype=np.float64)


def _compute_normalization(
    dataset: PreparedDataset, indices: np.ndarray
) -> tuple[float, float]:
    values = np.asarray(
        dataset.windows_mV[np.asarray(indices, dtype=np.int64)], dtype=np.float64
    )
    mean = float(np.mean(values))
    std = float(max(np.std(values), 1e-12))
    return mean, std


def _features(
    dataset: PreparedDataset,
    indices: np.ndarray,
    cfg: dict[str, Any],
    mean: float,
    std: float,
) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64)
    windows = np.asarray(dataset.windows_mV[idx], dtype=np.float64)
    windows = (windows - mean) / std
    if cfg["amplitude_normalization"] == "max_abs":
        scale = np.max(np.abs(windows), axis=2, keepdims=True)
        windows = windows / np.maximum(scale, float(cfg["amplitude_epsilon"]))
    basis = build_bspline_basis(
        dataset.input_length,
        int(cfg["n_basis"]),
        int(cfg["spline_degree"]),
    )
    return np.einsum("ncl,lk->nck", windows, basis, optimize=True) / float(
        dataset.input_length
    )


def _difference_matrix(n_features: int) -> np.ndarray:
    if n_features <= 1:
        return np.eye(n_features, dtype=np.float64)
    return np.diff(np.eye(n_features, dtype=np.float64), axis=0)


def fit_linear_spline(
    dataset: PreparedDataset | str | Path,
    train_indices: np.ndarray | None = None,
    config: dict[str, Any] | None = None,
) -> LinearSplineArtifact:
    if not isinstance(dataset, PreparedDataset):
        dataset = load_prepared_dataset(dataset)
    cfg = _resolve_config(config or {})
    indices = np.asarray(
        dataset.train if train_indices is None else train_indices, dtype=np.int64
    )
    if indices.size == 0:
        raise ValueError("Cannot fit linear spline on an empty training set")
    mean, std = _compute_normalization(dataset, indices)
    features = _features(dataset, indices, cfg, mean, std)
    design = features[:, 0, :] - features[:, 1, :]
    target = led_delta_ps(dataset, indices) - dataset.true_tof_ps
    feature_mean = np.mean(design, axis=0)
    feature_std = np.std(design, axis=0)
    feature_std = np.where(feature_std > 1e-12, feature_std, 1.0)
    normalized = (design - feature_mean) / feature_std
    smoothness = float(cfg["smoothness_penalty"])
    ridge = float(cfg["ridge_penalty"])
    penalty = _difference_matrix(normalized.shape[1])
    system = (
        (normalized.T @ normalized) / normalized.shape[0]
        + smoothness * (penalty.T @ penalty)
        + ridge * np.eye(normalized.shape[1])
    )
    rhs = (normalized.T @ target) / normalized.shape[0]
    try:
        coefficients = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(system, rhs, rcond=None)[0]
    return LinearSplineArtifact(
        config=cfg,
        normalization_mean=mean,
        normalization_std=std,
        feature_mean=np.asarray(feature_mean, dtype=np.float64),
        feature_std=np.asarray(feature_std, dtype=np.float64),
        coefficients=np.asarray(coefficients, dtype=np.float64),
        input_length=dataset.input_length,
        relative_time_ps=np.asarray(dataset.relative_time_ps, dtype=np.float64),
    )


def predict_linear_spline(
    artifact: LinearSplineArtifact,
    dataset: PreparedDataset,
    indices: np.ndarray,
) -> np.ndarray:
    if dataset.input_length != artifact.input_length:
        raise ValueError("Dataset input length does not match linear spline artifact")
    current_grid = np.asarray(dataset.relative_time_ps, dtype=np.float64)
    if current_grid.shape != artifact.relative_time_ps.shape or not np.allclose(
        current_grid, artifact.relative_time_ps, rtol=0.0, atol=1e-9
    ):
        raise ValueError("Dataset time grid does not match linear spline artifact")
    features = _features(
        dataset,
        indices,
        artifact.config,
        artifact.normalization_mean,
        artifact.normalization_std,
    )
    design = features[:, 0, :] - features[:, 1, :]
    normalized = (design - artifact.feature_mean) / artifact.feature_std
    return np.asarray(normalized @ artifact.coefficients, dtype=np.float64)


def _write_config_csv(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["parameter", "value"])
        for key in sorted(config):
            writer.writerow([key, config[key]])


def _read_config_csv(path: Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    with path.open("r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = row["parameter"]
            text = row["value"]
            if key in {"n_basis", "spline_degree"}:
                values[key] = int(text)
            elif key in {
                "amplitude_epsilon",
                "smoothness_penalty",
                "ridge_penalty",
            }:
                values[key] = float(text)
            else:
                values[key] = text
    return values


def save_linear_spline_artifact(
    artifact: LinearSplineArtifact, output_dir: str | Path
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "fitted_parameters.npz",
        feature_mean=artifact.feature_mean,
        feature_std=artifact.feature_std,
        coefficients=artifact.coefficients,
        normalization_mean=np.asarray([artifact.normalization_mean]),
        normalization_std=np.asarray([artifact.normalization_std]),
        input_length=np.asarray([artifact.input_length], dtype=np.int64),
        relative_time_ps=artifact.relative_time_ps,
    )
    _write_config_csv(output / "configuration.csv", artifact.config)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["method", "input_length"])
        writer.writeheader()
        writer.writerow({"method": "linear_spline", "input_length": artifact.input_length})


def load_linear_spline_artifact(path: str | Path) -> LinearSplineArtifact:
    path = Path(path)
    with np.load(path / "fitted_parameters.npz", allow_pickle=False) as params:
        config_path = path / "configuration.csv"
        config = _read_config_csv(config_path) if config_path.is_file() else {}
        return LinearSplineArtifact(
            config=_resolve_config(config),
            normalization_mean=float(params["normalization_mean"][0]),
            normalization_std=float(params["normalization_std"][0]),
            feature_mean=np.asarray(params["feature_mean"], dtype=np.float64),
            feature_std=np.asarray(params["feature_std"], dtype=np.float64),
            coefficients=np.asarray(params["coefficients"], dtype=np.float64),
            input_length=int(params["input_length"][0]),
            relative_time_ps=np.asarray(params["relative_time_ps"], dtype=np.float64),
        )
