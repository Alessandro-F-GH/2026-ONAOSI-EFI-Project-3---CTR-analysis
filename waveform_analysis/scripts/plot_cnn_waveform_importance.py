from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_pipeline.dataset import (
    load_prepared_dataset,
    prepared_dataset_view,
    window_slice_indices,
)
from ml_pipeline.evaluation import load_trained_model
from ml_pipeline.input_transform import (
    apply_input_transform,
    normalize_input_transform,
    transform_relative_time_ps,
)
from ml_pipeline.models import build_model
from ml_pipeline.prediction import prediction_dataset_view
from ml_pipeline.training_utils import resolve_device


def _find_row(dataset: Any, event_id: int | None, row_index: int | None) -> int:
    if event_id is not None and row_index is not None:
        raise ValueError("Use only one of --event-id or --row-index")
    if row_index is None and event_id is None:
        return 0
    if row_index is not None:
        if not 0 <= row_index < dataset.windows_mV.shape[0]:
            raise IndexError(f"row index {row_index} is outside the dataset")
        return int(row_index)
    matches = np.flatnonzero(np.asarray(dataset.event_id) == int(event_id))
    if matches.size == 0:
        raise KeyError(f"event_id={event_id} was not found")
    if matches.size > 1:
        raise ValueError(f"event_id={event_id} appears more than once; use --row-index")
    return int(matches[0])


def _load_model(model_path: Path, dataset: Any, device: torch.device):
    trained = load_trained_model(model_path)
    if trained.model_type != "cnn_regressor":
        raise ValueError(
            f"Expected a cnn_regressor checkpoint, found {trained.model_type!r}"
        )

    payload = torch.load(trained.checkpoint, map_location=device, weights_only=False)
    context = payload.get("context", {})

    dataset = prediction_dataset_view(
        dataset,
        input_waveforms=context.get(
            "input_waveform_source", trained.input_waveform_source
        ),
        target=context.get("prediction_target", trained.prediction_target),
    )

    data_view = dict(context.get("data_view", {}))
    if "window_before_ns" in data_view and "window_after_ns" in data_view:
        start, stop = window_slice_indices(
            dataset,
            float(data_view["window_before_ns"]),
            float(data_view["window_after_ns"]),
        )
        dataset = prepared_dataset_view(dataset, window_start=start, window_stop=stop)

    input_transform = normalize_input_transform(context.get("input_transform", "none"))
    input_length = int(context["input_length"])
    model = build_model(
        str(context["model_type"]),
        dict(context["model_config"]),
        input_length,
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    mean_mV = float(context["normalization"]["mean_mV"])
    std_mV = float(context["normalization"]["std_mV"])
    if not np.isfinite(std_mV) or std_mV <= 0.0:
        raise ValueError("Invalid checkpoint normalization standard deviation")

    return trained, model, dataset, input_transform, mean_mV, std_mV


def _model_input(
    pair_mV: np.ndarray,
    *,
    input_transform: str,
    mean_mV: float,
    std_mV: float,
) -> np.ndarray:
    transformed = np.asarray(
        apply_input_transform(pair_mV, input_transform), dtype=np.float32
    )
    return (transformed - np.float32(mean_mV)) / np.float32(std_mV)


def _integrated_gradients(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    *,
    steps: int,
    internal_batch_size: int,
) -> torch.Tensor:
    """Integrated gradients for the scalar pair correction.

    The baseline is zero in normalized model space, i.e. the checkpoint's mean
    waveform level. A trapezoidal approximation is used along the straight path
    from the baseline to each input.
    """
    if steps < 2:
        raise ValueError("--ig-steps must be at least 2")
    baseline = torch.zeros_like(inputs)
    delta = inputs - baseline
    alphas = torch.linspace(0.0, 1.0, steps, device=inputs.device)
    gradient_sum = torch.zeros_like(inputs)

    for start in range(0, steps, internal_batch_size):
        alpha_batch = alphas[start : start + internal_batch_size]
        scaled = baseline.unsqueeze(0) + alpha_batch[:, None, None, None] * delta.unsqueeze(0)
        flat = scaled.reshape(-1, *inputs.shape[1:]).requires_grad_(True)
        output = model(flat)
        gradient = torch.autograd.grad(
            outputs=output.sum(),
            inputs=flat,
            retain_graph=False,
            create_graph=False,
        )[0]
        gradient = gradient.reshape(alpha_batch.shape[0], *inputs.shape)

        weights = torch.ones(alpha_batch.shape[0], device=inputs.device)
        if start == 0:
            weights[0] = 0.5
        if start + alpha_batch.shape[0] == steps:
            weights[-1] = 0.5
        gradient_sum += torch.sum(
            gradient * weights[:, None, None, None], dim=0
        )

    average_gradient = gradient_sum / float(steps - 1)
    return delta * average_gradient


def _colored_line(
    ax: Any,
    x: np.ndarray,
    y: np.ndarray,
    importance: np.ndarray,
    *,
    cmap: str,
    norm: Normalize,
    linewidth: float,
) -> LineCollection:
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    segment_values = 0.5 * (importance[:-1] + importance[1:])
    collection = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        linewidth=linewidth,
    )
    collection.set_array(segment_values)
    ax.add_collection(collection)
    ax.autoscale()
    return collection


def _write_csv(path: Path, time_ps: np.ndarray, importance: np.ndarray) -> None:
    total = float(np.sum(importance))
    normalized = importance / total if total > 0.0 else np.zeros_like(importance)
    order = np.argsort(importance)[::-1]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "rank",
                "sample_index",
                "relative_time_ps",
                "mean_absolute_integrated_gradient",
                "normalized_importance",
            ),
        )
        writer.writeheader()
        for rank, index in enumerate(order, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "sample_index": int(index),
                    "relative_time_ps": float(time_ps[index]),
                    "mean_absolute_integrated_gradient": float(importance[index]),
                    "normalized_importance": float(normalized[index]),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot CNN temporal feature importance using Integrated Gradients. "
            "Importance is computed for the complete pair correction g(s1)-g(s2)."
        )
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="CNN run directory, training_summary.json, or best.pt",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--events",
        type=int,
        default=64,
        help="Number of evaluation events used for global importance",
    )
    parser.add_argument(
        "--event-batch-size",
        type=int,
        default=4,
        help="Events processed together; reduce this when memory is limited",
    )
    parser.add_argument(
        "--ig-steps",
        type=int,
        default=16,
        help="Integration steps; 16 is a low-compute starting point",
    )
    parser.add_argument(
        "--ig-internal-batch-size",
        type=int,
        default=4,
        help="Number of interpolation points evaluated together",
    )
    parser.add_argument(
        "--subset",
        choices=("evaluation", "validation", "train", "all"),
        default="evaluation",
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--row-index", type=int, default=None)
    parser.add_argument("--event-id", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--linewidth", type=float, default=2.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.events <= 0 or args.event_batch_size <= 0:
        raise ValueError("--events and --event-batch-size must be positive")

    device = resolve_device(args.device)
    dataset = load_prepared_dataset(args.dataset.resolve())
    trained, model, dataset, input_transform, mean_mV, std_mV = _load_model(
        args.model.resolve(), dataset, device
    )

    if args.subset == "all":
        candidate_indices = np.arange(dataset.windows_mV.shape[0], dtype=np.int64)
    else:
        candidate_indices = np.asarray(getattr(dataset, args.subset), dtype=np.int64)
    if candidate_indices.size == 0:
        raise ValueError(f"Dataset subset {args.subset!r} is empty")

    rng = np.random.default_rng(args.seed)
    count = min(int(args.events), int(candidate_indices.size))
    selected = np.sort(rng.choice(candidate_indices, size=count, replace=False))

    importance_sum: np.ndarray | None = None
    signed_sum: np.ndarray | None = None
    processed = 0

    for start in range(0, selected.size, args.event_batch_size):
        indices = selected[start : start + args.event_batch_size]
        raw_pair = np.asarray(dataset.windows_mV[indices], dtype=np.float32)
        model_array = _model_input(
            raw_pair,
            input_transform=input_transform,
            mean_mV=mean_mV,
            std_mV=std_mV,
        )
        inputs = torch.from_numpy(model_array).to(device)
        attribution = _integrated_gradients(
            model,
            inputs,
            steps=args.ig_steps,
            internal_batch_size=args.ig_internal_batch_size,
        )
        attribution_np = attribution.detach().cpu().numpy().astype(np.float64)

        # Sum detector contributions at each temporal location. Absolute values are
        # aggregated only after attribution, preserving the pair model's signs.
        temporal_abs = np.sum(np.abs(attribution_np), axis=1)
        temporal_signed = np.sum(attribution_np, axis=1)
        block_abs = np.sum(temporal_abs, axis=0)
        block_signed = np.sum(temporal_signed, axis=0)
        importance_sum = block_abs if importance_sum is None else importance_sum + block_abs
        signed_sum = block_signed if signed_sum is None else signed_sum + block_signed
        processed += int(indices.size)
        print(f"Attributed {processed}/{selected.size} events", flush=True)

    assert importance_sum is not None and signed_sum is not None
    importance = importance_sum / float(processed)
    signed_attribution = signed_sum / float(processed)
    time_ps = transform_relative_time_ps(dataset.relative_time_ps, input_transform)
    if time_ps.shape[0] != importance.shape[0]:
        raise RuntimeError("Attribution and time-grid lengths do not match")

    display_row = _find_row(dataset, args.event_id, args.row_index)
    display_raw = np.asarray(dataset.windows_mV[display_row], dtype=np.float64)
    display_pair = np.asarray(
        apply_input_transform(display_raw, input_transform), dtype=np.float64
    )

    # The same global temporal importance colors both detector waveforms.
    vmax = float(np.max(importance))
    norm = Normalize(vmin=0.0, vmax=vmax if vmax > 0.0 else 1.0)

    figure, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=False)

    axes[0].plot(time_ps, signed_attribution)
    axes[0].axhline(0.0, linewidth=0.8)
    axes[0].set_title("Mean signed Integrated-Gradients attribution")
    axes[0].set_ylabel("Attribution")
    axes[0].grid(True, alpha=0.25)

    normalized_importance = (
        importance / np.sum(importance)
        if np.sum(importance) > 0.0
        else np.zeros_like(importance)
    )
    axes[1].plot(time_ps, normalized_importance)
    axes[1].set_title("Global temporal importance distribution")
    axes[1].set_ylabel("Normalized importance")
    axes[1].grid(True, alpha=0.25)

    line = _colored_line(
        axes[2],
        time_ps,
        display_pair[0],
        importance,
        cmap=args.cmap,
        norm=norm,
        linewidth=args.linewidth,
    )
    _colored_line(
        axes[2],
        time_ps,
        display_pair[1],
        importance,
        cmap=args.cmap,
        norm=norm,
        linewidth=args.linewidth,
    )
    axes[2].set_title(
        f"Representative waveform pair colored by global importance | row={display_row}"
    )
    axes[2].set_ylabel("Signal feature [mV]")
    axes[2].set_xlabel("Relative time [ps]")
    axes[2].grid(True, alpha=0.25)
    colorbar = figure.colorbar(line, ax=axes[2], pad=0.01)
    colorbar.set_label("Mean absolute Integrated Gradients")

    top_k = min(max(int(args.top_k), 1), importance.size)
    top_indices = np.argsort(importance)[-top_k:][::-1]
    labels = [f"{time_ps[index] / 1000.0:.3f}" for index in top_indices]
    axes[3].bar(np.arange(top_k), normalized_importance[top_indices])
    axes[3].set_xticks(np.arange(top_k))
    axes[3].set_xticklabels(labels, rotation=70, ha="right")
    axes[3].set_title(f"Top {top_k} temporal features")
    axes[3].set_xlabel("Relative time [ns]")
    axes[3].set_ylabel("Normalized importance")
    axes[3].grid(True, axis="y", alpha=0.25)

    event_id_value = int(np.asarray(dataset.event_id)[display_row])
    figure.suptitle(
        f"CNN waveform importance | model={trained.model_name} | "
        f"transform={input_transform} | events={processed} | "
        f"IG steps={args.ig_steps} | display event_id={event_id_value}",
        fontsize=12,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)

    csv_path = output.with_name(output.stem + "_importance.csv")
    _write_csv(csv_path, time_ps, importance)
    print(f"Saved figure: {output}")
    print(f"Saved ranked importance: {csv_path}")


if __name__ == "__main__":
    main()
