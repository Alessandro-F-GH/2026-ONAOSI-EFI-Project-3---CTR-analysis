from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_pipeline.dataset import load_prepared_dataset
from ml_pipeline.evaluation import load_trained_model
from ml_pipeline.input_transform import (
    normalize_input_transform,
    transform_relative_time_ps,
)
from ml_pipeline.prediction import prediction_dataset_view


def _load_linear_svr(model_path: Path):
    trained = load_trained_model(model_path)
    if trained.model_type != "linear_svr":
        raise ValueError(
            f"Expected a linear_svr model, got {trained.model_type!r} from {model_path}"
        )

    payload = torch.load(trained.checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state", {})
    if "weight" not in state:
        raise KeyError(f"Checkpoint has no linear SVR weight: {trained.checkpoint}")

    weight = np.asarray(state["weight"], dtype=np.float64).reshape(-1)
    context = payload.get("context", {})
    input_transform = normalize_input_transform(
        context.get("input_transform", trained.input_transform)
    )
    return trained, weight, input_transform


def _differentiated_to_raw_weights(weight: np.ndarray) -> np.ndarray:
    """Convert weights acting on dx[j] = x[j+1] - x[j] to raw-sample weights."""
    weight = np.asarray(weight, dtype=np.float64).reshape(-1)
    if weight.size == 0:
        raise ValueError("Cannot convert an empty weight vector")

    raw = np.empty(weight.size + 1, dtype=np.float64)
    raw[0] = -weight[0]
    if weight.size > 1:
        raw[1:-1] = weight[:-1] - weight[1:]
    raw[-1] = weight[-1]
    return raw


def _feature_scale(
    dataset,
    indices: np.ndarray,
    input_transform: str,
    chunk_size: int,
) -> np.ndarray:
    """Compute std of pair-difference features without loading the full matrix."""
    count = 0
    mean = None
    m2 = None

    for start in range(0, indices.size, chunk_size):
        selected = indices[start : start + chunk_size]
        pair = np.asarray(dataset.windows_mV[selected], dtype=np.float64)
        features = pair[:, 0, :] - pair[:, 1, :]
        if input_transform == "differentiate":
            features = np.diff(features, axis=1)

        block_count = features.shape[0]
        block_mean = np.mean(features, axis=0)
        block_m2 = np.sum((features - block_mean) ** 2, axis=0)

        if mean is None:
            count = block_count
            mean = block_mean
            m2 = block_m2
            continue

        delta = block_mean - mean
        new_count = count + block_count
        mean = mean + delta * block_count / new_count
        m2 = m2 + block_m2 + delta**2 * count * block_count / new_count
        count = new_count

    if count < 2 or mean is None or m2 is None:
        raise ValueError("At least two events are required to compute feature scale")
    return np.sqrt(m2 / count)


def _save_top_csv(
    path: Path,
    time_ps: np.ndarray,
    weight: np.ndarray,
    importance: np.ndarray,
    top_indices: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "rank",
                "sample_index",
                "relative_time_ps",
                "weight",
                "absolute_weight",
                "importance",
                "normalized_importance",
            ],
        )
        writer.writeheader()
        total = float(np.sum(importance))
        for rank, index in enumerate(top_indices, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "sample_index": int(index),
                    "relative_time_ps": float(time_ps[index]),
                    "weight": float(weight[index]),
                    "absolute_weight": float(abs(weight[index])),
                    "importance": float(importance[index]),
                    "normalized_importance": (
                        float(importance[index] / total) if total > 0.0 else 0.0
                    ),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot linear-SVR weights, their importance distribution in time, "
            "and the top-k most important time samples."
        )
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Linear-SVR run directory, training_summary.json, or best.pt checkpoint",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help=(
            "Prepared dataset. Required only for --importance scale_adjusted. "
            "The model waveform source is selected automatically."
        ),
    )
    parser.add_argument(
        "--importance",
        choices=("absolute_weight", "scale_adjusted"),
        default="absolute_weight",
        help=(
            "absolute_weight uses |w_j|; scale_adjusted uses |w_j| times the "
            "standard deviation of the pair-difference feature."
        ),
    )
    parser.add_argument(
        "--space",
        choices=("model", "raw"),
        default="model",
        help=(
            "model plots coefficients in the exact SVR input representation; raw converts "
            "differentiated-input coefficients to equivalent raw-waveform coefficients."
        ),
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--histogram-bins", type=int, default=80)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output PNG path",
    )
    parser.add_argument(
        "--top-csv",
        type=Path,
        default=None,
        help="Optional CSV path for top features",
    )
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    trained, model_weight, input_transform = _load_linear_svr(args.model.resolve())

    dataset = None
    if args.dataset is not None:
        dataset = load_prepared_dataset(args.dataset.resolve())
        dataset = prediction_dataset_view(
            dataset,
            input_waveforms=trained.input_waveform_source,
            target=trained.prediction_target,
        )
        model_time_ps = transform_relative_time_ps(
            np.asarray(dataset.relative_time_ps, dtype=np.float64),
            input_transform,
        )
    else:
        summary_path = trained.train_dir / "training_summary.json"
        raise ValueError(
            "--dataset is required to recover the physical relative-time axis. "
            f"Use a prepared dataset compatible with {summary_path}."
        )

    if model_time_ps.shape[0] != model_weight.shape[0]:
        raise ValueError(
            "Dataset/model length mismatch: model has "
            f"{model_weight.shape[0]} features, but the resolved dataset gives "
            f"{model_time_ps.shape[0]}."
        )

    if args.importance == "absolute_weight":
        model_importance = np.abs(model_weight)
        importance_label = r"Absolute coefficient $|w_j|$"
    else:
        indices = np.asarray(dataset.train, dtype=np.int64)
        if indices.size == 0:
            indices = np.asarray(dataset.evaluation, dtype=np.int64)
        scale = _feature_scale(dataset, indices, input_transform, args.chunk_size)
        if scale.shape != model_weight.shape:
            raise ValueError(
                f"Feature scale shape {scale.shape} does not match weight shape {model_weight.shape}"
            )
        model_importance = np.abs(model_weight) * scale
        importance_label = r"Scale-adjusted importance $|w_j|\sigma_j$"

    if args.space == "raw" and input_transform == "differentiate":
        weight = _differentiated_to_raw_weights(model_weight)
        time_ps = np.asarray(dataset.relative_time_ps, dtype=np.float64)
        if args.importance == "scale_adjusted":
            raw_indices = np.asarray(dataset.train, dtype=np.int64)
            if raw_indices.size == 0:
                raw_indices = np.asarray(dataset.evaluation, dtype=np.int64)
            raw_scale = _feature_scale(dataset, raw_indices, "none", args.chunk_size)
            importance = np.abs(weight) * raw_scale
        else:
            importance = np.abs(weight)
    else:
        weight = model_weight
        time_ps = model_time_ps
        importance = model_importance

    total_importance = float(np.sum(importance))
    normalized = (
        importance / total_importance
        if total_importance > 0.0
        else np.zeros_like(importance)
    )
    cumulative = np.cumsum(normalized)

    top_k = min(args.top_k, importance.size)
    top_indices = np.argsort(importance)[-top_k:][::-1]
    top_order_time = top_indices[np.argsort(time_ps[top_indices])]

    fig = plt.figure(figsize=(13, 12))
    grid = fig.add_gridspec(4, 1, height_ratios=(1.2, 1.2, 1.0, 1.4))

    ax_weight = fig.add_subplot(grid[0])
    ax_weight.plot(time_ps, weight, linewidth=1.2)
    ax_weight.axhline(0.0, linewidth=0.8)
    ax_weight.scatter(time_ps[top_indices], weight[top_indices], s=22, zorder=3)
    ax_weight.set_ylabel("SVR weight")
    ax_weight.set_title("Signed linear-SVR coefficient profile")
    ax_weight.grid(True, alpha=0.25)

    ax_importance = fig.add_subplot(grid[1], sharex=ax_weight)
    ax_importance.plot(time_ps, normalized, linewidth=1.2)
    ax_importance.fill_between(time_ps, normalized, alpha=0.25)
    ax_importance.scatter(
        time_ps[top_indices], normalized[top_indices], s=22, zorder=3
    )
    ax_importance.set_ylabel("Normalized importance")
    ax_importance.set_title(importance_label)
    ax_importance.grid(True, alpha=0.25)

    ax_hist = fig.add_subplot(grid[2])
    ax_hist.hist(weight, bins=args.histogram_bins)
    ax_hist.set_xlabel("SVR weight")
    ax_hist.set_ylabel("Feature count")
    ax_hist.set_title("Weight-value distribution")
    ax_hist.grid(True, alpha=0.2)

    ax_top = fig.add_subplot(grid[3])
    labels = [f"{time_ps[index]:.1f}" for index in top_order_time]
    ax_top.bar(labels, normalized[top_order_time])
    ax_top.set_xlabel("Relative time [ps]")
    ax_top.set_ylabel("Normalized importance")
    ax_top.set_title(f"Top {top_k} most important time samples")
    ax_top.tick_params(axis="x", rotation=60)
    ax_top.grid(True, axis="y", alpha=0.25)

    title = (
        f"Linear SVR feature importance | model={trained.model_name} | "
        f"transform={input_transform} | space={args.space} | metric={args.importance}"
    )
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    csv_path = args.top_csv or args.output.with_name(args.output.stem + "_top_features.csv")
    _save_top_csv(csv_path, time_ps, weight, importance, top_indices)

    print(f"Saved figure: {args.output}")
    print(f"Saved top-feature table: {csv_path}")
    print("Top features:")
    for rank, index in enumerate(top_indices, start=1):
        print(
            f"  {rank:2d}. sample={index:5d} | time={time_ps[index]:10.3f} ps | "
            f"weight={weight[index]: .6e} | normalized importance={normalized[index]:.6e}"
        )


if __name__ == "__main__":
    main()
