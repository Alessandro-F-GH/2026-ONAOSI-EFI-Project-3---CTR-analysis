from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.fit import FitResult
from utils.plots import plot_best_fit

from .common import atomic_json, write_csv_rows
from .dataset import (
    PreparedDataset,
    load_prepared_dataset,
    prepared_dataset_view,
    window_slice_indices,
)
from .metrics import distribution_metrics, fit_times_ps
from .models import build_model
from .standard_methods import cfd_delta_ps, led_delta_ps, load_linear_spline_artifact, predict_linear_spline
from .plots import plot_metric_bars
from .torch_data import CorrectionDataset, Normalization
from .training_utils import predict_loader, resolve_device


@dataclass(frozen=True)
class TrainedModel:
    model_name: str
    model_type: str
    checkpoint: Path
    validation_rmse_ps: float
    train_dir: Path


def _read_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Invalid training summary: {path}")
    return value


def _model_from_summary(summary_path: Path) -> TrainedModel | None:
    summary = _read_summary(summary_path)
    if "best_checkpoint" not in summary or "model_type" not in summary:
        return None
    checkpoint = Path(summary["best_checkpoint"]).resolve()
    if not checkpoint.is_file():
        return None
    return TrainedModel(
        model_name=str(summary["model_name"]),
        model_type=str(summary["model_type"]),
        checkpoint=checkpoint,
        validation_rmse_ps=float(summary.get("best_validation_rmse_ps", float("nan"))),
        train_dir=summary_path.parent.resolve(),
    )


def _model_from_checkpoint(checkpoint: Path) -> TrainedModel:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    context = payload.get("context", {})
    model_type = str(context.get("model_type", "")).strip()
    if not model_type:
        raise ValueError(f"Checkpoint does not contain model_type metadata: {checkpoint}")
    return TrainedModel(
        model_name=str(context.get("model_name", checkpoint.stem)),
        model_type=model_type,
        checkpoint=checkpoint.resolve(),
        validation_rmse_ps=float("nan"),
        train_dir=checkpoint.parent.parent.resolve(),
    )


def discover_models(config: dict[str, Any], logger: Any) -> list[TrainedModel]:
    explicit = [Path(value) for value in config.get("models", [])]
    candidates: list[TrainedModel] = []
    if explicit:
        for path in explicit:
            if path.is_dir():
                summary_path = path / "training_summary.json"
                if not summary_path.is_file():
                    raise FileNotFoundError(f"Training summary not found in explicit model directory: {path}")
                model = _model_from_summary(summary_path)
                if model is not None:
                    candidates.append(model)
            elif path.name == "training_summary.json":
                model = _model_from_summary(path)
                if model is not None:
                    candidates.append(model)
            elif path.suffix == ".pt":
                candidates.append(_model_from_checkpoint(path))
            else:
                raise ValueError(f"Unsupported model path: {path}")
        return candidates

    search_dir = Path(config["model_search_dir"])
    summaries = sorted(search_dir.rglob("training_summary.json")) if search_dir.is_dir() else []
    if not summaries:
        return []
    grouped: dict[str, TrainedModel] = {}
    for summary_path in summaries:
        candidate = _model_from_summary(summary_path)
        if candidate is None:
            continue
        previous = grouped.get(candidate.model_name)
        if previous is None or candidate.validation_rmse_ps < previous.validation_rmse_ps:
            grouped[candidate.model_name] = candidate
    models = sorted(grouped.values(), key=lambda item: item.model_name)
    logger.info("Automatically selected %d best model checkpoint(s) from %s", len(models), search_dir)
    return models


def _filename(value: str) -> str:
    text = "".join(character.lower() if character.isalnum() else "_" for character in value)
    return "_".join(part for part in text.split("_") if part)


def _metric_row(
    *,
    blind_name: str,
    method: str,
    values_ps: np.ndarray,
    true_tof_ps: float,
    fit: FitResult,
    model: TrainedModel | None,
) -> dict[str, Any]:
    row = {
        "blind_test": blind_name,
        "method": method,
        "model_name": "" if model is None else model.model_name,
        "model_type": "" if model is None else model.model_type,
        "checkpoint": "" if model is None else str(model.checkpoint),
        "true_tof_ps": float(true_tof_ps),
    }
    metrics = distribution_metrics(values_ps, true_value_ps=true_tof_ps, fit=fit)
    metrics["true_tof_ps"] = metrics.pop("true_value_ps")
    row.update(metrics)
    return row


def _make_loader(dataset: PreparedDataset, normalization: Normalization, config: dict[str, Any], device: torch.device) -> DataLoader:
    evaluation_dataset = CorrectionDataset(dataset, dataset.evaluation, normalization)
    workers = int(config.get("num_workers", 0))
    kwargs: dict[str, Any] = {
        "batch_size": int(config.get("batch_size", 512)),
        "shuffle": False,
        "drop_last": False,
        "num_workers": workers,
        "pin_memory": bool(config.get("pin_memory", device.type == "cuda")),
    }
    if workers > 0:
        kwargs["persistent_workers"] = bool(config.get("persistent_workers", False))
        kwargs["prefetch_factor"] = int(config.get("prefetch_factor", 2))
    return DataLoader(evaluation_dataset, **kwargs)


def _evaluate_model(
    trained: TrainedModel,
    dataset: PreparedDataset,
    config: dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    payload = torch.load(trained.checkpoint, map_location=device, weights_only=False)
    context = payload.get("context", {})
    checkpoint_contract = context.get("dataset_contract")
    if isinstance(checkpoint_contract, dict):
        for field in (
            "led_timestamp_source",
            "cfd_timestamp_source",
            "ml_window_alignment_source",
            "timing_channel_waveforms_saved",
        ):
            expected = checkpoint_contract.get(field)
            actual = dataset.manifest.get(field)
            if expected != actual:
                raise ValueError(
                    f"Model {trained.model_name} was trained with {field}={expected!r}, "
                    f"but evaluation dataset {dataset.directory} has {field}={actual!r}. "
                    "Do not mix energy-LED and timing-LED preprocessing products."
                )
    input_length = int(context["input_length"])
    data_view = dict(context.get("data_view", {}))
    if "window_before_ns" in data_view and "window_after_ns" in data_view:
        start, stop = window_slice_indices(
            dataset,
            float(data_view["window_before_ns"]),
            float(data_view["window_after_ns"]),
        )
        dataset = prepared_dataset_view(
            dataset,
            window_start=start,
            window_stop=stop,
        )
    if input_length != dataset.input_length:
        raise ValueError(
            f"Model {trained.model_name} expects {input_length} samples, "
            f"but the resolved blind-data window has {dataset.input_length}"
        )
    normalization = Normalization(
        mean_mV=float(context["normalization"]["mean_mV"]),
        std_mV=float(context["normalization"]["std_mV"]),
    )
    model = build_model(
        str(context["model_type"]),
        dict(context["model_config"]),
        input_length,
    ).to(device)
    model.load_state_dict(payload["model_state"])
    loader = _make_loader(dataset, normalization, config, device)
    prediction = predict_loader(model, loader, device)
    return np.asarray(prediction["corrected_ps"], dtype=np.float64)



def _view_for_time_grid(dataset: PreparedDataset, expected: np.ndarray) -> PreparedDataset:
    expected = np.asarray(expected, dtype=np.float64)
    current = np.asarray(dataset.relative_time_ps, dtype=np.float64)
    if current.shape == expected.shape and np.allclose(current, expected, rtol=0.0, atol=1e-9):
        return dataset
    if expected.size > current.size:
        raise ValueError("Standard-method artifact uses a longer time grid than the blind dataset")
    for start in range(0, current.size - expected.size + 1):
        stop = start + expected.size
        if np.allclose(current[start:stop], expected, rtol=0.0, atol=1e-9):
            return prepared_dataset_view(dataset, window_start=start, window_stop=stop)
    raise ValueError("Standard-method artifact time grid is not present in the blind dataset")

def evaluate_models(config: dict[str, Any], *, logger: Any) -> dict[str, Any]:
    output_root = Path(config["output"]["evaluation_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    models = discover_models(config, logger)
    device = resolve_device(config.get("device", "auto"))
    dpi = int(config["plotting"].get("dpi", 180))
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for blind in config["blind_tests"]:
        blind_name = str(blind["name"])
        dataset = load_prepared_dataset(blind["dataset"])
        if dataset.evaluation.size == 0:
            raise RuntimeError(f"Blind dataset has no evaluation events: {dataset.directory}")
        output_dir = output_root / _filename(blind_name)
        plot_dir = output_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        indices = dataset.evaluation
        led_values = led_delta_ps(dataset, indices)
        cfd_values = cfd_delta_ps(dataset, indices)
        method_values: list[tuple[str, np.ndarray, TrainedModel | None]] = []
        std_cfg = config.get("standard_methods", {})
        if std_cfg.get("led", True):
            method_values.append(("LED", led_values, None))
        if std_cfg.get("cfd", True):
            method_values.append(("CFD", cfd_values, None))
        spline_cfg = std_cfg.get("linear_spline", {})
        if spline_cfg.get("enabled", False):
            artifact = load_linear_spline_artifact(Path(spline_cfg["artifact"]))
            spline_dataset = _view_for_time_grid(dataset, artifact.relative_time_ps)
            correction = predict_linear_spline(artifact, spline_dataset, indices)
            method_values.append(("LED + linear_spline correction", led_values - correction, None))

        for trained in models:
            corrected = _evaluate_model(trained, dataset, config, device)
            method_values.append((f"LED + {trained.model_name} correction", corrected, trained))

        rows: list[dict[str, Any]] = []
        fit_details: dict[str, Any] = {}
        for method, values, trained in method_values:
            fit = fit_times_ps(values, method, config["fit"])
            row = _metric_row(
                blind_name=blind_name,
                method=method,
                values_ps=values,
                true_tof_ps=dataset.true_tof_ps,
                fit=fit,
                model=trained,
            )
            rows.append(row)
            fit_details[method] = fit.as_dict()
            plot_best_fit(fit, plot_dir / f"gaussian_fit_{_filename(method)}.png", dpi=dpi)
            logger.info(
                "%s | %s | CTR %.3f ps | Gaussian bias %.3f ps",
                blind_name, method, row["ctr_ps"], row["gaussian_bias_ps"],
            )

        write_csv_rows(output_dir / "metrics.csv", rows)
        plot_metric_bars(rows, plot_dir, dpi)
        summary = {
            "blind_test": blind_name,
            "dataset": str(dataset.directory),
            "dataset_fingerprint": dataset.manifest["fingerprint"],
            "models": [
                {
                    "model_name": model.model_name,
                    "model_type": model.model_type,
                    "checkpoint": str(model.checkpoint),
                    "selected_validation_rmse_ps": model.validation_rmse_ps,
                }
                for model in models
            ],
            "metrics": rows,
            "fit_details": fit_details,
        }
        atomic_json(output_dir / "evaluation_summary.json", summary)
        summaries.append(summary)
        all_rows.extend(rows)

    write_csv_rows(output_root / "all_metrics.csv", all_rows)
    result = {"evaluation_dir": str(output_root), "blind_tests": summaries}
    atomic_json(output_root / "evaluation_index.json", result)
    return result
