from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .common import atomic_json, write_csv_rows
from .dataset import PreparedDataset
from .plots import plot_top_correction_event


@dataclass(frozen=True)
class CorrectionAnalysisResult:
    """Ranked event-level effect of an ML timing correction.

    ``improvement_ps`` is the decrease in absolute distance from the known TOF:

        |raw - true| - |corrected - true|

    Positive values are useful ("right") corrections; negative values move the
    event farther from the known mean.
    """

    summary: dict[str, Any]
    top_events: list[dict[str, Any]]


def _python_scalar(value: Any) -> Any:
    scalar = np.asarray(value).reshape(-1)[0]
    if isinstance(scalar, np.generic):
        scalar = scalar.item()
    if isinstance(scalar, bytes):
        return scalar.decode("utf-8", errors="replace")
    return scalar


def _pair_value(array: np.ndarray, index: int, channel: int) -> float | None:
    values = np.asarray(array[index]).reshape(-1)
    if channel >= values.size:
        return None
    return float(values[channel])


def _filename_component(value: Any) -> str:
    text = str(value)
    safe = "".join(
        character if character.isalnum() or character in ("-", "_") else "_"
        for character in text
    )
    return safe.strip("_") or "unknown"


def analyze_right_corrections(
    dataset: PreparedDataset,
    evaluation_indices: np.ndarray,
    *,
    raw_ps: np.ndarray,
    corrected_ps: np.ndarray,
    predicted_correction_ps: np.ndarray,
    true_tof_ps: np.ndarray | float,
    top_n: int,
    minimum_improvement_ps: float = 0.0,
) -> CorrectionAnalysisResult:
    """Find events whose model correction most reduces timing error.

    Ranking by raw prediction magnitude would reward large but physically wrong
    corrections. This function instead ranks by the actual reduction in absolute
    residual with respect to the known TOF.
    """

    indices = np.asarray(evaluation_indices, dtype=np.int64)
    raw = np.asarray(raw_ps, dtype=np.float64).reshape(-1)
    corrected = np.asarray(corrected_ps, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted_correction_ps, dtype=np.float64).reshape(-1)
    true = np.asarray(true_tof_ps, dtype=np.float64)
    if true.ndim == 0:
        true = np.full(raw.shape, float(true), dtype=np.float64)
    else:
        true = true.reshape(-1)

    size = raw.size
    for name, values in (
        ("evaluation_indices", indices),
        ("corrected_ps", corrected),
        ("predicted_correction_ps", predicted),
        ("true_tof_ps", true),
    ):
        if values.size != size:
            raise ValueError(
                f"Correction analysis length mismatch: raw_ps has {size} values, "
                f"but {name} has {values.size}"
            )
    if int(top_n) <= 0:
        raise ValueError("top_n must be positive")
    minimum = float(minimum_improvement_ps)
    if minimum < 0.0:
        raise ValueError("minimum_improvement_ps must be non-negative")

    raw_residual = raw - true
    corrected_residual = corrected - true
    raw_abs_error = np.abs(raw_residual)
    corrected_abs_error = np.abs(corrected_residual)
    improvement = raw_abs_error - corrected_abs_error

    right_mask = improvement > minimum
    wrong_mask = improvement < -minimum
    neutral_mask = ~(right_mask | wrong_mask)
    right_positions = np.flatnonzero(right_mask)
    ranked_positions = right_positions[
        np.argsort(-improvement[right_positions], kind="stable")
    ][: int(top_n)]

    top_events: list[dict[str, Any]] = []
    for rank, position in enumerate(ranked_positions, start=1):
        row_index = int(indices[position])
        original_error = float(raw_abs_error[position])
        fraction_removed = (
            float(improvement[position] / original_error)
            if original_error > 0.0
            else 0.0
        )
        record = {
            "rank": int(rank),
            "evaluation_position": int(position),
            "dataset_index": row_index,
            "event_id": _python_scalar(dataset.event_id[row_index]),
            "event_index": _python_scalar(dataset.event_index[row_index]),
            "source_file_id": _python_scalar(dataset.source_file_id[row_index]),
            "source_run_index": _python_scalar(dataset.source_run_index[row_index]),
            "true_tof_ps": float(true[position]),
            "raw_ps": float(raw[position]),
            "predicted_correction_ps": float(predicted[position]),
            "corrected_ps": float(corrected[position]),
            "raw_residual_ps": float(raw_residual[position]),
            "corrected_residual_ps": float(corrected_residual[position]),
            "raw_abs_error_ps": original_error,
            "corrected_abs_error_ps": float(corrected_abs_error[position]),
            "improvement_ps": float(improvement[position]),
            "fraction_of_error_removed": fraction_removed,
            "overshot_true_tof": bool(
                raw_residual[position] != 0.0
                and corrected_residual[position] != 0.0
                and np.sign(raw_residual[position]) != np.sign(corrected_residual[position])
            ),
            "amplitude_ch1_mV": _pair_value(dataset.amplitude_mV, row_index, 0),
            "amplitude_ch2_mV": _pair_value(dataset.amplitude_mV, row_index, 1),
            "noise_rms_ch1_mV": _pair_value(dataset.noise_rms_mV, row_index, 0),
            "noise_rms_ch2_mV": _pair_value(dataset.noise_rms_mV, row_index, 1),
        }
        top_events.append(record)

    positive_values = improvement[right_mask]
    top_record = top_events[0] if top_events else None
    summary = {
        "definition": "abs(raw_ps - true_tof_ps) - abs(corrected_ps - true_tof_ps)",
        "event_count": int(size),
        "minimum_improvement_ps": minimum,
        "right_correction_count": int(np.count_nonzero(right_mask)),
        "wrong_correction_count": int(np.count_nonzero(wrong_mask)),
        "neutral_correction_count": int(np.count_nonzero(neutral_mask)),
        "right_correction_fraction": (
            float(np.mean(right_mask)) if size else 0.0
        ),
        "mean_right_correction_ps": (
            float(np.mean(positive_values)) if positive_values.size else None
        ),
        "median_right_correction_ps": (
            float(np.median(positive_values)) if positive_values.size else None
        ),
        "top_right_correction_ps": (
            None if top_record is None else float(top_record["improvement_ps"])
        ),
        "top_right_correction_event_id": (
            None if top_record is None else top_record["event_id"]
        ),
        "top_right_correction_dataset_index": (
            None if top_record is None else int(top_record["dataset_index"])
        ),
        "saved_top_event_count": int(len(top_events)),
    }
    return CorrectionAnalysisResult(summary=summary, top_events=top_events)


def save_correction_analysis(
    result: CorrectionAnalysisResult,
    dataset: PreparedDataset,
    *,
    output_dir: Path,
    input_transform: str,
    input_waveform_source: str,
    prediction_target: str,
    model_name: str,
    dpi: int,
    save_waveform_plots: bool = True,
) -> dict[str, Any]:
    """Save ranked metadata and one diagnostic waveform plot per top event."""

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(output_dir / "top_right_corrections.csv", result.top_events)

    plot_paths: list[str] = []
    if save_waveform_plots:
        plot_dir = output_dir / "waveforms"
        for record in result.top_events:
            event_component = _filename_component(record["event_id"])
            filename = (
                f"rank_{int(record['rank']):03d}_"
                f"event_{event_component}_row_{int(record['dataset_index'])}.png"
            )
            path = plot_dir / filename
            plot_top_correction_event(
                dataset,
                record,
                path,
                input_transform=input_transform,
                input_waveform_source=input_waveform_source,
                prediction_target=prediction_target,
                model_name=model_name,
                dpi=dpi,
            )
            plot_paths.append(str(path))

    payload = {
        "model_name": model_name,
        "input_transform": input_transform,
        "input_waveform_source": input_waveform_source,
        "prediction_target": prediction_target,
        "summary": result.summary,
        "top_events": result.top_events,
        "waveform_plots": plot_paths,
    }
    atomic_json(output_dir / "correction_analysis.json", payload)
    return payload
