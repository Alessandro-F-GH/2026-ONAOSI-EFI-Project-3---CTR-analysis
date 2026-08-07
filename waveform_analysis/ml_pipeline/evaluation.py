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
from .correction_analysis import analyze_right_corrections, save_correction_analysis
from .dataset import (
    PreparedDataset,
    load_prepared_dataset,
    load_prepared_dataset_spec,
    prepared_dataset_view,
    window_slice_indices,
)
from .input_transform import (
    materialize_training_input_cache,
    normalize_input_transform,
    normalize_subsampling_factor,
    transformed_subsampled_dataset_input_length,
)
from .metrics import distribution_metrics, fit_times_ps
from .models import build_model
from .prediction import (
    normalize_input_waveforms,
    normalize_prediction_target,
    prediction_dataset_view,
    prediction_window_dataset_view,
)
from .standard_methods import cfd_delta_ps, led_delta_ps, load_linear_spline_artifact, predict_linear_spline
from .plots import plot_metric_bars, plot_model_output_correlation
from .torch_data import CorrectionDataset, Normalization
from .training_utils import predict_loader, resolve_device


@dataclass(frozen=True)
class TrainedModel:
    model_name: str
    model_type: str
    checkpoint: Path
    validation_rmse_ps: float
    train_dir: Path
    input_transform: str
    input_waveform_source: str = "energy"
    prediction_target: str = "prepared_led"


@dataclass(frozen=True)
class ModelPrediction:
    corrected_ps: np.ndarray
    predicted_correction_ps: np.ndarray
    raw_ps: np.ndarray
    true_tof_ps: np.ndarray
    dataset_view: PreparedDataset
    input_transform: str
    input_waveform_source: str
    prediction_target: str


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
        input_transform=normalize_input_transform(summary.get("input_transform", "none")),
        input_waveform_source=normalize_input_waveforms(
            summary.get("input_waveform_source", "energy")
        ),
        prediction_target=normalize_prediction_target(
            summary.get("prediction_target", "prepared_led")
        ),
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
        input_transform=normalize_input_transform(context.get("input_transform", "none")),
        input_waveform_source=normalize_input_waveforms(
            context.get("input_waveform_source", "energy")
        ),
        prediction_target=normalize_prediction_target(
            context.get("prediction_target", "prepared_led")
        ),
    )


def load_trained_model(path: str | Path) -> TrainedModel:
    """Load one evaluator-compatible model from a run directory, summary, or checkpoint."""

    resolved = Path(path).resolve()
    if resolved.is_dir():
        summary_path = resolved / "training_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(
                f"Training summary not found in model directory: {resolved}"
            )
        model = _model_from_summary(summary_path)
        if model is None:
            raise ValueError(f"Invalid training summary: {summary_path}")
        return model
    if resolved.name == "training_summary.json":
        model = _model_from_summary(resolved)
        if model is None:
            raise ValueError(f"Invalid training summary: {resolved}")
        return model
    if resolved.suffix == ".pt":
        return _model_from_checkpoint(resolved)
    raise ValueError(f"Unsupported model path: {resolved}")


def discover_models(config: dict[str, Any], logger: Any) -> list[TrainedModel]:
    explicit = [Path(value) for value in config.get("models", [])]
    candidates: list[TrainedModel] = []
    if explicit:
        for path in explicit:
            candidates.append(load_trained_model(path))
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

    allowed_types = {str(value).strip() for value in config.get("model_types", []) if str(value).strip()}
    allowed_names = {str(value).strip() for value in config.get("model_names", []) if str(value).strip()}
    if allowed_types:
        models = [model for model in models if model.model_type in allowed_types]
    if allowed_names:
        models = [model for model in models if model.model_name in allowed_names]

    logger.info(
        "Automatically selected %d best model checkpoint(s) from %s%s%s",
        len(models),
        search_dir,
        f" | model_types={sorted(allowed_types)}" if allowed_types else "",
        f" | model_names={sorted(allowed_names)}" if allowed_names else "",
    )
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
        "input_transform": "" if model is None else model.input_transform,
        "input_waveform_source": (
            "" if model is None else model.input_waveform_source
        ),
        "prediction_target": "" if model is None else model.prediction_target,
        "true_tof_ps": float(true_tof_ps),
        "top_right_correction_ps": "",
        "top_right_correction_event_id": "",
        "top_right_correction_dataset_index": "",
        "right_correction_count": "",
        "wrong_correction_count": "",
        "right_correction_fraction": "",
    }
    metrics = distribution_metrics(values_ps, true_value_ps=true_tof_ps, fit=fit)
    metrics["true_tof_ps"] = metrics.pop("true_value_ps")
    row.update(metrics)
    return row


def _make_loader(
    dataset: PreparedDataset,
    normalization: Normalization,
    config: dict[str, Any],
    device: torch.device,
    input_transform: str,
    subsampling_factor: int = 1,
) -> DataLoader:
    evaluation_dataset = CorrectionDataset(
        dataset,
        dataset.evaluation,
        normalization,
        input_transform=input_transform,
        subsampling_factor=subsampling_factor,
    )
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


def evaluate_trained_model(
    trained: TrainedModel,
    dataset: PreparedDataset,
    config: dict[str, Any],
    device: torch.device,
) -> ModelPrediction:
    payload = torch.load(trained.checkpoint, map_location=device, weights_only=False)
    context = payload.get("context", {})
    input_waveform_source = normalize_input_waveforms(
        context.get("input_waveform_source", trained.input_waveform_source)
    )
    prediction_target = normalize_prediction_target(
        context.get("prediction_target", trained.prediction_target)
    )
    data_view = dict(context.get("data_view", {}))
    if "window_before_ns" in data_view and "window_after_ns" in data_view:
        dataset = prediction_window_dataset_view(
            dataset,
            input_waveforms=input_waveform_source,
            target=prediction_target,
            before_ns=float(data_view["window_before_ns"]),
            after_ns=float(data_view["window_after_ns"]),
        )
    else:
        dataset = prediction_dataset_view(
            dataset,
            input_waveforms=input_waveform_source,
            target=prediction_target,
        )
    checkpoint_contract = context.get("dataset_contract")
    if isinstance(checkpoint_contract, dict):
        for field in (
            "timing_channel_waveforms_saved",
            "waveform_grid",
            "window_anchor_timestamps_saved",
            "correction_target_reference",
            "window_anchor_shift_factored",
            "factorization_anchor_source",
            "factorization_anchor_component",
        ):
            if field not in checkpoint_contract:
                continue
            expected = checkpoint_contract.get(field)
            actual = dataset.manifest.get(field)
            if expected != actual:
                raise ValueError(
                    f"Model {trained.model_name} was trained with {field}={expected!r}, "
                    f"but evaluation dataset {dataset.directory} has {field}={actual!r}. "
                    "Rebuild or select a compatible canonical prepared dataset."
                )
    input_length = int(context["input_length"])
    input_transform = normalize_input_transform(
        context.get("input_transform", trained.input_transform)
    )
    subsampling_factor = normalize_subsampling_factor(
        context.get(
            "subsampling_factor",
            context.get("preprocessing", {}).get("subsampling_factor", 1),
        )
    )
    resolved_input_length = transformed_subsampled_dataset_input_length(
        dataset, input_transform, subsampling_factor
    )
    if input_length != resolved_input_length:
        raise ValueError(
            f"Model {trained.model_name} expects {input_length} samples after "
            f"input_transform={input_transform!r}, but the resolved blind-data "
            f"window produces {resolved_input_length}"
        )
    normalization = Normalization.from_dict(context["normalization"])
    model_type = str(context["model_type"])
    model = build_model(
        model_type,
        dict(context["model_config"]),
        input_length,
    ).to(device)
    model.load_state_dict(payload["model_state"])
    # Replay the training representation exactly. Training materializes transformed
    # waveforms once and subsequently passes ``input_transform="none"`` to the
    # PyTorch dataset. Evaluation now follows the same path instead of applying
    # ``np.diff`` item-by-item, which avoids dtype/shape inconsistencies and makes
    # differentiated MLP and linear-SVR checkpoints use identical input data.
    raw_dataset_view = dataset
    output_config = config.get("output", {})
    default_evaluation_dir = (
        output_config.get("evaluation_dir")
        if isinstance(output_config, dict)
        else None
    )
    evaluation_root = Path(
        config.get(
            "input_transform_cache_dir",
            Path(default_evaluation_dir or trained.train_dir) / ".input_cache",
        )
    )
    model_dataset, _ = materialize_training_input_cache(
        raw_dataset_view,
        input_transform,
        evaluation_root / _filename(trained.model_name),
        chunk_size=int(config.get("input_transform_chunk_size", 2048)),
        rebuild=bool(config.get("rebuild_input_transform_cache", False)),
    )
    resolved_materialized_length = transformed_subsampled_dataset_input_length(
        model_dataset, "none", subsampling_factor
    )
    if resolved_materialized_length != input_length:
        raise ValueError(
            f"Model {trained.model_name} expects {input_length} samples, but the "
            "materialized and subsampled evaluation representation has "
            f"{resolved_materialized_length}"
        )
    loader = _make_loader(
        model_dataset,
        normalization,
        config,
        device,
        "none",
        subsampling_factor=subsampling_factor,
    )
    prediction = predict_loader(model, loader, device)
    return ModelPrediction(
        corrected_ps=np.asarray(prediction["corrected_ps"], dtype=np.float64),
        predicted_correction_ps=np.asarray(
            prediction.get("total_led_correction_ps", prediction["prediction_ps"]),
            dtype=np.float64,
        ),
        raw_ps=np.asarray(prediction["led_ps"], dtype=np.float64),
        true_tof_ps=np.asarray(prediction["true_tof_ps"], dtype=np.float64),
        dataset_view=raw_dataset_view,
        input_transform=input_transform,
        input_waveform_source=input_waveform_source,
        prediction_target=prediction_target,
    )


def _evaluate_model(
    trained: TrainedModel,
    dataset: PreparedDataset,
    config: dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    """Backward-compatible wrapper returning only corrected event values."""

    return evaluate_trained_model(trained, dataset, config, device).corrected_ps



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

def _pairwise_model_output_correlation(
    outputs: list[tuple[str, np.ndarray]],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Pearson correlation of per-event predicted corrections.

    Correlations are computed pairwise using only events that are finite for
    both models. Constant off-diagonal outputs are reported as NaN because
    Pearson correlation is undefined in that case.
    """

    labels = [str(label) for label, _values in outputs]
    count = len(outputs)
    matrix = np.full((count, count), np.nan, dtype=np.float64)
    event_counts = np.zeros((count, count), dtype=np.int64)
    arrays = [np.asarray(values, dtype=np.float64).reshape(-1) for _label, values in outputs]
    if arrays:
        expected = arrays[0].size
        for label, values in zip(labels, arrays):
            if values.size != expected:
                raise ValueError(
                    f"Model output {label!r} has {values.size} events; expected {expected}"
                )

    for row in range(count):
        for column in range(row, count):
            finite = np.isfinite(arrays[row]) & np.isfinite(arrays[column])
            n_events = int(np.count_nonzero(finite))
            event_counts[row, column] = event_counts[column, row] = n_events
            if n_events < 2:
                continue
            left = arrays[row][finite]
            right = arrays[column][finite]
            if row == column:
                value = 1.0
            elif float(np.std(left, ddof=0)) == 0.0 or float(np.std(right, ddof=0)) == 0.0:
                value = float("nan")
            else:
                value = float(np.corrcoef(left, right)[0, 1])
            matrix[row, column] = matrix[column, row] = value
    return labels, matrix, event_counts


def _write_correlation_matrix_csv(
    path: Path,
    labels: list[str],
    matrix: np.ndarray,
) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", *labels])
        for label, row in zip(labels, np.asarray(matrix, dtype=np.float64)):
            writer.writerow(
                [
                    label,
                    *[
                        "" if not np.isfinite(value) else f"{float(value):.12g}"
                        for value in row
                    ],
                ]
            )


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
        dataset = load_prepared_dataset_spec(blind)
        if dataset.evaluation.size == 0:
            raise RuntimeError(f"Blind dataset has no evaluation events: {dataset.directory}")
        output_dir = output_root / _filename(blind_name)
        plot_dir = output_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        indices = dataset.evaluation
        led_values = led_delta_ps(dataset, indices)
        cfd_values = cfd_delta_ps(dataset, indices)
        method_values: list[
            tuple[str, np.ndarray, TrainedModel | None, ModelPrediction | None]
        ] = []
        std_cfg = config.get("standard_methods", {})
        if std_cfg.get("led", True):
            added_explicit_led = False
            if dataset.energy_led_time_fs is not None:
                energy_led_values = (
                    np.asarray(dataset.energy_led_time_fs[indices, 0], dtype=np.float64)
                    - np.asarray(dataset.energy_led_time_fs[indices, 1], dtype=np.float64)
                ) / 1000.0
                method_values.append(("Energy LED", energy_led_values, None, None))
                added_explicit_led = True
            if dataset.timing_led_time_fs is not None:
                timing_led_values = (
                    np.asarray(dataset.timing_led_time_fs[indices, 0], dtype=np.float64)
                    - np.asarray(dataset.timing_led_time_fs[indices, 1], dtype=np.float64)
                ) / 1000.0
                method_values.append(("Timing LED", timing_led_values, None, None))
                added_explicit_led = True
            if not added_explicit_led:
                method_values.append(("LED", led_values, None, None))
        if std_cfg.get("cfd", True):
            method_values.append(("CFD", cfd_values, None, None))
        spline_cfg = std_cfg.get("linear_spline", {})
        if spline_cfg.get("enabled", False):
            artifact = load_linear_spline_artifact(Path(spline_cfg["artifact"]))
            spline_dataset = _view_for_time_grid(dataset, artifact.relative_time_ps)
            correction = predict_linear_spline(artifact, spline_dataset, indices)
            method_values.append(
                ("LED + linear_spline correction", led_values - correction, None, None)
            )

        skipped_models: list[dict[str, str]] = []
        skip_incompatible = bool(config.get("skip_incompatible_models", True))
        for trained in models:
            try:
                prediction = evaluate_trained_model(trained, dataset, config, device)
            except (ValueError, KeyError, RuntimeError) as exc:
                if not skip_incompatible:
                    raise
                reason = str(exc)
                logger.warning(
                    "Skipping incompatible model %s (%s) on blind dataset %s: %s",
                    trained.model_name,
                    trained.model_type,
                    blind_name,
                    reason,
                )
                skipped_models.append(
                    {
                        "model_name": trained.model_name,
                        "model_type": trained.model_type,
                        "checkpoint": str(trained.checkpoint),
                        "reason": reason,
                    }
                )
                continue
            method_values.append(
                (
                    f"{trained.prediction_target} + {trained.model_name} correction",
                    prediction.corrected_ps,
                    trained,
                    prediction,
                )
            )

        rows: list[dict[str, Any]] = []
        fit_details: dict[str, Any] = {}
        correction_details: dict[str, Any] = {}
        model_output_values = [
            (trained.model_name, prediction.predicted_correction_ps)
            for _method, _values, trained, prediction in method_values
            if trained is not None and prediction is not None
        ]
        correction_cfg = config.get("correction_analysis", {})
        correction_enabled = bool(correction_cfg.get("enabled", False))
        for method, values, trained, prediction in method_values:
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

            if correction_enabled and trained is not None and prediction is not None:
                analysis = analyze_right_corrections(
                    prediction.dataset_view,
                    prediction.dataset_view.evaluation,
                    raw_ps=prediction.raw_ps,
                    corrected_ps=prediction.corrected_ps,
                    predicted_correction_ps=prediction.predicted_correction_ps,
                    true_tof_ps=prediction.true_tof_ps,
                    top_n=int(correction_cfg.get("top_n", 10)),
                    minimum_improvement_ps=float(
                        correction_cfg.get("minimum_improvement_ps", 0.0)
                    ),
                )
                analysis_dir = (
                    output_dir / "top_corrections" / _filename(method)
                )
                payload = save_correction_analysis(
                    analysis,
                    prediction.dataset_view,
                    output_dir=analysis_dir,
                    input_transform=prediction.input_transform,
                    input_waveform_source=prediction.input_waveform_source,
                    prediction_target=prediction.prediction_target,
                    model_name=trained.model_name,
                    dpi=dpi,
                    save_waveform_plots=bool(
                        correction_cfg.get("save_waveform_plots", True)
                    ),
                )
                correction_details[method] = payload
                row.update(
                    {
                        "top_right_correction_ps": analysis.summary[
                            "top_right_correction_ps"
                        ],
                        "top_right_correction_event_id": analysis.summary[
                            "top_right_correction_event_id"
                        ],
                        "top_right_correction_dataset_index": analysis.summary[
                            "top_right_correction_dataset_index"
                        ],
                        "right_correction_count": analysis.summary[
                            "right_correction_count"
                        ],
                        "wrong_correction_count": analysis.summary[
                            "wrong_correction_count"
                        ],
                        "right_correction_fraction": analysis.summary[
                            "right_correction_fraction"
                        ],
                    }
                )
                top_value = analysis.summary["top_right_correction_ps"]
                if top_value is None:
                    logger.info(
                        "%s | %s | Top right correction: none above %.3f ps",
                        blind_name,
                        method,
                        float(correction_cfg.get("minimum_improvement_ps", 0.0)),
                    )
                else:
                    logger.info(
                        "%s | %s | Top right correction: %.3f ps | event_id=%s | row=%s",
                        blind_name,
                        method,
                        float(top_value),
                        analysis.summary["top_right_correction_event_id"],
                        analysis.summary["top_right_correction_dataset_index"],
                    )
            logger.info(
                "%s | %s | CTR %.3f ps | Gaussian bias %.3f ps",
                blind_name, method, row["ctr_ps"], row["gaussian_bias_ps"],
            )

        correlation_summary: dict[str, Any] = {
            "enabled": False,
            "quantity": "predicted_correction_ps",
            "model_count": len(model_output_values),
        }
        correlation_cfg = config.get("model_output_correlation", {})
        correlation_enabled = bool(correlation_cfg.get("enabled", True))
        if correlation_enabled and len(model_output_values) >= 2:
            correlation_labels, correlation_matrix, correlation_counts = (
                _pairwise_model_output_correlation(model_output_values)
            )
            correlation_csv = output_dir / "model_output_correlation.csv"
            correlation_plot = plot_dir / "model_output_correlation.png"
            _write_correlation_matrix_csv(
                correlation_csv, correlation_labels, correlation_matrix
            )
            plot_model_output_correlation(
                correlation_matrix,
                correlation_labels,
                correlation_plot,
                dpi=dpi,
                annotate=bool(correlation_cfg.get("annotate", True)),
            )
            correlation_summary = {
                "enabled": True,
                "quantity": "predicted_correction_ps",
                "model_count": len(correlation_labels),
                "labels": correlation_labels,
                "matrix": correlation_matrix.tolist(),
                "pairwise_event_counts": correlation_counts.tolist(),
                "csv": str(correlation_csv.resolve()),
                "plot": str(correlation_plot.resolve()),
            }
            logger.info(
                "%s | model correction-output correlation matrix saved for %d models",
                blind_name,
                len(correlation_labels),
            )
        elif correlation_enabled:
            logger.info(
                "%s | correlation matrix skipped: at least two compatible models are required",
                blind_name,
            )

        write_csv_rows(output_dir / "metrics.csv", rows)
        plot_metric_bars(rows, plot_dir, dpi)
        summary = {
            "blind_test": blind_name,
            "skipped_models": skipped_models,
            "dataset": str(dataset.directory),
            "dataset_fingerprint": dataset.manifest["fingerprint"],
            "models": [
                {
                    "model_name": model.model_name,
                    "model_type": model.model_type,
                    "checkpoint": str(model.checkpoint),
                    "selected_validation_rmse_ps": model.validation_rmse_ps,
                    "input_transform": model.input_transform,
                    "input_waveform_source": model.input_waveform_source,
                    "prediction_target": model.prediction_target,
                }
                for model in models
            ],
            "metrics": rows,
            "fit_details": fit_details,
            "model_output_correlation": correlation_summary,
            "correction_analysis": correction_details,
        }
        atomic_json(output_dir / "evaluation_summary.json", summary)
        summaries.append(summary)
        all_rows.extend(rows)

    write_csv_rows(output_root / "all_metrics.csv", all_rows)
    result = {"evaluation_dir": str(output_root), "blind_tests": summaries}
    atomic_json(output_root / "evaluation_index.json", result)
    return result
