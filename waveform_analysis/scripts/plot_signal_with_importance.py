from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_pipeline.dataset import load_prepared_dataset
from ml_pipeline.prediction import prediction_dataset_view
from ml_pipeline.input_transform import (
    apply_input_transform,
    normalize_input_transform,
    transform_relative_time_ps,
)


def _load_importance(
    weights_npz: Path,
    reduction: str,
    run_index: int | None,
) -> tuple[np.ndarray, np.ndarray, str, str, str]:
    with np.load(weights_npz, allow_pickle=False) as data:
        if "normalized_path_importance" not in data:
            raise KeyError(
                f"{weights_npz} does not contain 'normalized_path_importance'. "
                "Use the output of analyze_mlp_initialization_weights_v3.py."
            )
        importance_all = np.log(np.asarray(data["normalized_path_importance"], dtype=np.float64))
        relative_time_ps = np.asarray(data["relative_time_ps"], dtype=np.float64)
        input_transform = normalize_input_transform(
            str(np.asarray(data["input_transform"]).reshape(-1)[0])
            if "input_transform" in data
            else "none"
        )
        input_waveform_source = (
            str(np.asarray(data["input_waveform_source"]).reshape(-1)[0])
            if "input_waveform_source" in data else "energy"
        )
        prediction_target = (
            str(np.asarray(data["prediction_target"]).reshape(-1)[0])
            if "prediction_target" in data else "prepared_led"
        )

    if importance_all.ndim != 2:
        raise ValueError(
            f"Expected normalized_path_importance with shape (runs, samples), got {importance_all.shape}"
        )

    if run_index is not None:
        if run_index < 0 or run_index >= importance_all.shape[0]:
            raise IndexError(
                f"--run-index={run_index} is out of range for {importance_all.shape[0]} runs"
            )
        importance = importance_all[run_index]
    else:
        if reduction == "mean":
            importance = np.mean(importance_all, axis=0)
        elif reduction == "median":
            importance = np.median(importance_all, axis=0)
        else:
            raise ValueError(f"Unsupported reduction: {reduction}")

    return (
        relative_time_ps, importance, input_transform,
        input_waveform_source, prediction_target,
    )


def _find_event_position(
    dataset,
    *,
    event_id: int | None,
    row_index: int | None,
) -> int:
    if event_id is None and row_index is None:
        raise ValueError("Provide either --event-id or --row-index")

    if event_id is not None and row_index is not None:
        raise ValueError("Use only one of --event-id or --row-index")

    if row_index is not None:
        if row_index < 0 or row_index >= dataset.windows_mV.shape[0]:
            raise IndexError(
                f"--row-index={row_index} is out of range for {dataset.windows_mV.shape[0]} events"
            )
        return int(row_index)

    matches = np.flatnonzero(np.asarray(dataset.event_id) == int(event_id))
    if matches.size == 0:
        raise KeyError(f"Event id {event_id} not found in dataset")
    if matches.size > 1:
        raise ValueError(
            f"Event id {event_id} appears {matches.size} times; use --row-index instead"
        )
    return int(matches[0])


def _build_colored_line(
    x: np.ndarray,
    y: np.ndarray,
    c: np.ndarray,
    *,
    cmap: str,
    linewidth: float,
) -> LineCollection:
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    segment_colors = 0.5 * (c[:-1] + c[1:])
    norm = Normalize(vmin=float(np.min(segment_colors)), vmax=float(np.max(segment_colors)))
    if np.isclose(norm.vmin, norm.vmax):
        norm = Normalize(vmin=float(norm.vmin), vmax=float(norm.vmin + 1.0))

    collection = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        linewidth=linewidth,
    )
    collection.set_array(segment_colors)
    return collection


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot one waveform using weight importance as the line color."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Prepared dataset directory",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="weights.npz produced by analyze_mlp_initialization_weights_v3.py",
    )
    parser.add_argument(
        "--channel",
        type=int,
        required=True,
        help="Channel index inside windows_mV (usually 0 or 1)",
    )
    parser.add_argument(
        "--event-id",
        type=int,
        default=None,
        help="Value from event_id.npy",
    )
    parser.add_argument(
        "--row-index",
        type=int,
        default=None,
        help="Direct row index in the prepared dataset",
    )
    parser.add_argument(
        "--run-index",
        type=int,
        default=None,
        help="Use importance from a specific run instead of averaging all runs",
    )
    parser.add_argument(
        "--reduction",
        choices=("mean", "median"),
        default="mean",
        help="How to combine importance across runs when --run-index is not given",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Matplotlib colormap",
    )
    parser.add_argument(
        "--linewidth",
        type=float,
        default=2.5,
        help="Waveform line width",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional custom title",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output image path; if omitted, show interactively",
    )
    args = parser.parse_args()

    dataset = load_prepared_dataset(args.dataset.resolve())

    time_ps_model, importance, input_transform, input_waveform_source, prediction_target = _load_importance(
        args.weights.resolve(),
        reduction=args.reduction,
        run_index=args.run_index,
    )
    dataset = prediction_dataset_view(
        dataset,
        input_waveforms=input_waveform_source,
        target=prediction_target,
    )
    row = _find_event_position(dataset, event_id=args.event_id, row_index=args.row_index)

    if args.channel < 0 or args.channel >= dataset.windows_mV.shape[1]:
        raise IndexError(
            f"--channel={args.channel} is out of range for {dataset.windows_mV.shape[1]} channels"
        )

    signal = np.asarray(dataset.windows_mV[row, args.channel], dtype=np.float64)
    time_ps_signal = np.asarray(dataset.relative_time_ps, dtype=np.float64)
    signal = np.asarray(apply_input_transform(signal, input_transform), dtype=np.float64)
    time_ps_signal = transform_relative_time_ps(time_ps_signal, input_transform)

    if signal.shape[0] != importance.shape[0]:
        raise ValueError(
            "Signal length and importance length do not match: "
            f"{signal.shape[0]} vs {importance.shape[0]}. "
            "The model-input transform stored in the weights file could not be "
            "reproduced from this canonical dataset."
        )

    if time_ps_signal.shape[0] != time_ps_model.shape[0]:
        raise ValueError(
            "Dataset relative_time_ps and weights relative_time_ps do not match in length: "
            f"{time_ps_signal.shape[0]} vs {time_ps_model.shape[0]}"
        )

    fig, ax = plt.subplots(figsize=(10, 5.5))
    line = _build_colored_line(
        time_ps_signal,
        signal,
        importance,
        cmap=args.cmap,
        linewidth=args.linewidth,
    )
    ax.add_collection(line)
    ax.autoscale()
    ax.set_xlabel("Relative time [ps]")
    ax.set_ylabel("Signal [mV]")

    event_id_value = int(np.asarray(dataset.event_id)[row])
    default_title = (
        f"Signal colored by weight importance | row={row} | event_id={event_id_value} | "
        f"channel={args.channel} | transform={input_transform}"
    )
    ax.set_title(args.title or default_title)

    cbar = fig.colorbar(line, ax=ax)
    cbar.set_label("Weight importance")

    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=200, bbox_inches="tight")
        print(f"Saved plot to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()