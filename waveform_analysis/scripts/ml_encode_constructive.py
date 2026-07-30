from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.lib.format import open_memmap

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from ml_pipeline.common import atomic_json, write_csv_rows
from ml_pipeline.dataset import (
    PreparedDataset,
    load_prepared_dataset,
    prepared_dataset_view,
    window_slice_indices,
)
from ml_pipeline.models import build_model
from ml_pipeline.models.constructive_mlp_encoder import (
    AntisymmetricConstructiveMLPEncoder,
)
from ml_pipeline.training_utils import resolve_device


def _checkpoint_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_dataset_view(
    dataset: PreparedDataset,
    checkpoint_context: dict[str, Any],
) -> PreparedDataset:
    data_view = dict(checkpoint_context.get("data_view", {}))
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
    expected = int(checkpoint_context["input_length"])
    if dataset.input_length != expected:
        raise ValueError(
            f"Checkpoint expects {expected} waveform samples, but dataset view has "
            f"{dataset.input_length}"
        )
    return dataset


def _write_split_file(dataset: PreparedDataset, output: Path) -> None:
    with (output / "splits.npz.tmp").open("wb") as stream:
        np.savez_compressed(
            stream,
            train=np.asarray(dataset.train, dtype=np.int64),
            validation=np.asarray(dataset.validation, dtype=np.int64),
            test=np.asarray(dataset.test, dtype=np.int64),
            evaluation=np.asarray(dataset.evaluation, dtype=np.int64),
        )
    (output / "splits.npz.tmp").replace(output / "splits.npz")


def encode_dataset(
    *,
    checkpoint: Path,
    dataset_path: Path,
    output: Path,
    batch_size: int,
    device: torch.device,
    overwrite: bool,
) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    context = dict(payload.get("context", {}))
    model_type = str(context.get("model_type", ""))
    if model_type != "constructive_mlp_encoder":
        raise ValueError(
            "ml_encode_constructive.py requires a constructive_mlp_encoder checkpoint"
        )
    dataset = _resolve_dataset_view(load_prepared_dataset(dataset_path), context)
    model = build_model(
        model_type,
        dict(context["model_config"]),
        int(context["input_length"]),
    ).to(device)
    if not isinstance(model, AntisymmetricConstructiveMLPEncoder):
        raise TypeError("Loaded checkpoint did not build a constructive encoder")
    model.load_state_dict(payload["model_state"])
    model.eval()
    unit_count = int(model.unit_count)
    if unit_count <= 0:
        raise ValueError("Checkpoint contains no trained constructive units")

    if output.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {output}. Use --overwrite to replace it."
            )
        shutil.rmtree(output)
    temporary = output.with_name(output.name + ".building")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, exist_ok=True)

    event_count = int(dataset.windows_mV.shape[0])
    encoded_channels = open_memmap(
        temporary / "encoded_channels.npy",
        mode="w+",
        dtype=np.float32,
        shape=(event_count, 2, unit_count),
    )
    encoded_difference = open_memmap(
        temporary / "encoded_pair_difference.npy",
        mode="w+",
        dtype=np.float32,
        shape=(event_count, unit_count),
    )
    prediction = open_memmap(
        temporary / "predicted_led_correction_ps.npy",
        mode="w+",
        dtype=np.float32,
        shape=(event_count,),
    )
    target = open_memmap(
        temporary / "target_led_correction_ps.npy",
        mode="w+",
        dtype=np.float32,
        shape=(event_count,),
    )

    normalization = dict(context["normalization"])
    mean = np.float32(normalization["mean_mV"])
    std = np.float32(normalization["std_mV"])
    if not np.isfinite(std) or std <= 0.0:
        raise ValueError("Checkpoint normalization standard deviation is invalid")

    with torch.no_grad():
        for start in range(0, event_count, batch_size):
            stop = min(start + batch_size, event_count)
            waveforms = np.asarray(
                dataset.windows_mV[start:stop], dtype=np.float32
            ).copy()
            waveforms = (waveforms - mean) / std
            tensor = torch.from_numpy(waveforms).to(device)
            hidden = model.encode_pair(tensor)
            correction = model(tensor)
            hidden_np = hidden.detach().cpu().numpy().astype(np.float32)
            encoded_channels[start:stop] = hidden_np
            encoded_difference[start:stop] = hidden_np[:, 0] - hidden_np[:, 1]
            prediction[start:stop] = correction.detach().cpu().numpy().astype(np.float32)

            led_delta = (
                np.asarray(dataset.led_time_fs[start:stop, 0], dtype=np.float64)
                - np.asarray(dataset.led_time_fs[start:stop, 1], dtype=np.float64)
            ) / 1000.0
            target[start:stop] = (led_delta - float(dataset.true_tof_ps)).astype(
                np.float32
            )

    for array in (encoded_channels, encoded_difference, prediction, target):
        array.flush()
        mmap = getattr(array, "_mmap", None)
        if mmap is not None:
            mmap.close()

    for name in (
        "event_id",
        "event_index",
        "source_file_id",
        "source_run_index",
        "bias_voltage_V",
        "led_time_fs",
        "cfd_time_fs",
    ):
        np.save(temporary / f"{name}.npy", np.asarray(getattr(dataset, name)))
    _write_split_file(dataset, temporary)
    write_csv_rows(
        temporary / "encoded_features.csv",
        [
            {
                "feature_index": index,
                "feature_name": f"constructive_unit_{index:03d}",
                "activation": "identity",
            }
            for index in range(unit_count)
        ],
    )

    manifest = {
        "format_version": 1,
        "artifact_type": "constructive_mlp_encoded_dataset",
        "source_dataset": str(dataset_path.resolve()),
        "source_dataset_fingerprint": dataset.manifest.get("fingerprint", ""),
        "source_event_count": event_count,
        "source_input_length": int(dataset.input_length),
        "encoded_dimension": unit_count,
        "encoded_channels_shape": [event_count, 2, unit_count],
        "encoded_pair_difference_shape": [event_count, unit_count],
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _checkpoint_hash(checkpoint),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "model_type": model_type,
        "model_name": str(context.get("model_name", checkpoint.stem)),
        "normalization": normalization,
        "true_tof_ps": float(dataset.true_tof_ps),
        "split_counts": {
            "train": int(dataset.train.size),
            "validation": int(dataset.validation.size),
            "test": int(dataset.test.size),
            "evaluation": int(dataset.evaluation.size),
        },
        "feature_semantics": (
            "encoded_channels[event, channel, unit] stores the frozen identity-unit "
            "activation. encoded_pair_difference stores channel_1 minus channel_2."
        ),
        "linearity_note": (
            "The encoder is affine in the globally normalized waveform because all "
            "hidden activations are identity."
        ),
    }
    atomic_json(temporary / "manifest.json", manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Transform prepared waveform datasets with a frozen constructive MLP "
            "encoder and save the low-dimensional hidden-unit representation."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        action="append",
        required=True,
        help="Prepared dataset directory. Repeat for multiple datasets.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    device = resolve_device(args.device)
    output_root = args.output_root.resolve()
    summaries = []
    for dataset in args.dataset:
        dataset = dataset.resolve()
        output = output_root / dataset.name
        summary = encode_dataset(
            checkpoint=checkpoint,
            dataset_path=dataset,
            output=output,
            batch_size=int(args.batch_size),
            device=device,
            overwrite=bool(args.overwrite),
        )
        summaries.append(summary)
        print(
            f"Encoded {dataset} -> {output} | dimension "
            f"{summary['encoded_dimension']}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(output_root / "encoding_summary.json", {"datasets": summaries})


if __name__ == "__main__":
    main()
