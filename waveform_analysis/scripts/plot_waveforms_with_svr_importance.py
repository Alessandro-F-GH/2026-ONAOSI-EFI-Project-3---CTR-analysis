from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, TwoSlopeNorm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_pipeline.dataset import load_prepared_dataset
from ml_pipeline.evaluation import load_trained_model
from ml_pipeline.input_transform import (
    apply_input_transform,
    normalize_input_transform,
    transform_relative_time_ps,
)
from ml_pipeline.prediction import prediction_dataset_view


def _find_event_position(dataset, *, event_id: int | None, row_index: int | None) -> int:
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


def _load_linear_svr(model_path: Path) -> tuple[object, np.ndarray, str]:
    trained = load_trained_model(model_path)
    if trained.model_type != "linear_svr":
        raise ValueError(
            f"This script supports only linear_svr checkpoints, got {trained.model_type!r}"
        )
    payload = torch.load(trained.checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state", {})
    if "weight" not in state:
        raise KeyError(f"Checkpoint does not contain linear_svr weight: {trained.checkpoint}")
    weight = np.asarray(state["weight"], dtype=np.float64).reshape(-1)
    context = payload.get("context", {})
    input_transform = normalize_input_transform(context.get("input_transform", trained.input_transform))
    return trained, weight, input_transform


def _differentiate_weights_to_raw(diff_weight: np.ndarray) -> np.ndarray:
    diff_weight = np.asarray(diff_weight, dtype=np.float64).reshape(-1)
    raw = np.empty(diff_weight.size + 1, dtype=np.float64)
    raw[0] = -diff_weight[0]
    if diff_weight.size > 1:
        raw[1:-1] = diff_weight[:-1] - diff_weight[1:]
    raw[-1] = diff_weight[-1]
    return raw


def _resolve_plot_representation(
    signal_a: np.ndarray,
    signal_b: np.ndarray,
    time_ps: np.ndarray,
    weight: np.ndarray,
    input_transform: str,
    plot_space: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    signal_a = np.asarray(signal_a, dtype=np.float64)
    signal_b = np.asarray(signal_b, dtype=np.float64)
    time_ps = np.asarray(time_ps, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)

    if plot_space == "model":
        plot_a = np.asarray(apply_input_transform(signal_a, input_transform), dtype=np.float64)
        plot_b = np.asarray(apply_input_transform(signal_b, input_transform), dtype=np.float64)
        plot_t = transform_relative_time_ps(time_ps, input_transform)
        plot_w = weight
    elif plot_space == "raw":
        plot_a = signal_a
        plot_b = signal_b
        plot_t = time_ps
        plot_w = _differentiate_weights_to_raw(weight) if input_transform == "differentiate" else weight
    else:
        raise ValueError(f"Unsupported plot space: {plot_space}")

    if plot_a.shape[0] != plot_w.shape[0]:
        raise ValueError(
            "Signal length and importance length do not match after resolving the plot space: "
            f"{plot_a.shape[0]} vs {plot_w.shape[0]}"
        )
    return plot_a, plot_b, plot_t, plot_w


def _importance_values(
    plot_a: np.ndarray,
    plot_b: np.ndarray,
    weight: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, str]:
    difference = np.asarray(plot_a - plot_b, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    if mode == "signed_weight":
        return weight, "Signed SVR weight"
    if mode == "abs_weight":
        return np.abs(weight), "Absolute SVR weight"
    if mode == "pair_contribution":
        return np.abs(weight * difference), "Absolute pair contribution |w_j (s1_j - s2_j)|"
    raise ValueError(f"Unsupported importance mode: {mode}")


def _make_norm(values: np.ndarray, signed: bool):
    values = np.asarray(values, dtype=np.float64)
    if signed:
        vmax = float(np.max(np.abs(values)))
        if np.isclose(vmax, 0.0):
            vmax = 1.0
        return TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    return Normalize(vmin=vmin, vmax=vmax)


def _add_colored_line(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    c: np.ndarray,
    *,
    cmap: str,
    norm,
    linewidth: float,
    label: str,
) -> LineCollection:
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    segment_colors = 0.5 * (c[:-1] + c[1:])
    collection = LineCollection(segments, cmap=cmap, norm=norm, linewidth=linewidth)
    collection.set_array(segment_colors)
    ax.add_collection(collection)
    ax.autoscale()
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.25)
    return collection


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot waveform pairs from a prepared dataset, using linear-SVR feature importance "
            "as the line color."
        )
    )
    parser.add_argument("--dataset", type=Path, required=True, help="Prepared dataset directory")
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Linear-SVR run directory, training_summary.json, or checkpoint .pt",
    )
    parser.add_argument("--event-id", type=int, default=None, help="Value from event_id.npy")
    parser.add_argument("--row-index", type=int, default=None, help="Direct row index")
    parser.add_argument(
        "--plot-space",
        choices=("raw", "model"),
        default="raw",
        help=(
            "'model': plot exactly the features seen by the SVR; 'raw': plot original waveforms. "
            "If the SVR uses differentiation, raw mode converts the differentiated weights into "
            "equivalent raw-sample coefficients."
        ),
    )
    parser.add_argument(
        "--importance",
        choices=("abs_weight", "signed_weight", "pair_contribution"),
        default="abs_weight",
        help=(
            "Importance measure: absolute coefficient, signed coefficient, or event-specific "
            "absolute pair contribution."
        ),
    )
    parser.add_argument("--cmap", type=str, default="viridis", help="Matplotlib colormap")
    parser.add_argument("--linewidth", type=float, default=2.5, help="Line width")
    parser.add_argument(
        "--hide-difference",
        action="store_true",
        help="Hide the pair-difference subplot",
    )
    parser.add_argument("--title", type=str, default=None, help="Optional custom title")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output image path; if omitted, show interactively",
    )
    args = parser.parse_args()

    trained, weight, input_transform = _load_linear_svr(args.model.resolve())

    dataset = load_prepared_dataset(args.dataset.resolve())
    dataset = prediction_dataset_view(
        dataset,
        input_waveforms=trained.input_waveform_source,
        target=trained.prediction_target,
    )
    row = _find_event_position(dataset, event_id=args.event_id, row_index=args.row_index)

    raw_pair = np.asarray(dataset.windows_mV[row], dtype=np.float64)
    if raw_pair.shape[0] != 2:
        raise ValueError(f"Expected waveform pairs with exactly 2 channels, got shape {raw_pair.shape}")
    signal_a, signal_b = raw_pair[0], raw_pair[1]
    time_ps = np.asarray(dataset.relative_time_ps, dtype=np.float64)

    plot_a, plot_b, plot_t, plot_w = _resolve_plot_representation(
        signal_a,
        signal_b,
        time_ps,
        weight,
        input_transform,
        args.plot_space,
    )
    importance, colorbar_label = _importance_values(plot_a, plot_b, plot_w, args.importance)
    signed = args.importance == "signed_weight"
    norm = _make_norm(importance, signed=signed)

    nrows = 2 if args.hide_difference else 3
    fig, axes = plt.subplots(nrows=nrows, figsize=(11, 3.3 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]
    axes = np.atleast_1d(axes)

    collection = _add_colored_line(
        axes[0], plot_t, plot_a, importance,
        cmap=args.cmap, norm=norm, linewidth=args.linewidth, label="Detector 1 [mV]"
    )
    _add_colored_line(
        axes[1], plot_t, plot_b, importance,
        cmap=args.cmap, norm=norm, linewidth=args.linewidth, label="Detector 2 [mV]"
    )
    if not args.hide_difference:
        _add_colored_line(
            axes[2], plot_t, plot_a - plot_b, importance,
            cmap=args.cmap, norm=norm, linewidth=args.linewidth, label="Pair difference [mV]"
        )

    axes[-1].set_xlabel("Relative time [ps]")

    event_id_value = int(np.asarray(dataset.event_id)[row])
    default_title = (
        f"Linear SVR waveform importance | model={trained.model_name} | row={row} | "
        f"event_id={event_id_value} | input_transform={input_transform} | "
        f"plot_space={args.plot_space} | importance={args.importance}"
    )
    fig.suptitle(args.title or default_title, y=0.995, fontsize=11)

    cbar = fig.colorbar(collection, ax=list(axes), fraction=0.03, pad=0.02)
    cbar.set_label(colorbar_label)

    fig.tight_layout(rect=(0.0, 0.0, 0.96, 0.98))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=200, bbox_inches="tight")
        print(f"Saved plot to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
