from __future__ import annotations

import copy
import shutil
from pathlib import Path
from typing import Any

from .common import atomic_json, set_global_seed
from .config import resolve_fit_config
from .dataset import PreparedDataset, load_prepared_dataset
from .models import train_registered_model
from .torch_data import compute_normalization
from .training_context import TrainingContext



def _validate_dataset_contract(
    datasets: list[PreparedDataset],
    contract: dict[str, Any] | None,
    logger: Any,
) -> None:
    fields = (
        "led_timestamp_source",
        "cfd_timestamp_source",
        "ml_window_alignment_source",
        "timing_channel_waveforms_saved",
    )
    observed: dict[str, set[Any]] = {field: set() for field in fields}
    for dataset in datasets:
        for field in fields:
            observed[field].add(dataset.manifest.get(field))

    inconsistent = {field: values for field, values in observed.items() if len(values) > 1}
    if inconsistent:
        raise ValueError(
            "Training datasets have incompatible preprocessing contracts: "
            + "; ".join(
                f"{field}={sorted(map(str, values))}"
                for field, values in inconsistent.items()
            )
        )

    resolved = {field: next(iter(values)) for field, values in observed.items()}
    logger.info(
        "Prepared-data contract | LED %s | CFD %s | ML alignment %s | "
        "timing waveforms saved=%s",
        resolved["led_timestamp_source"],
        resolved["cfd_timestamp_source"],
        resolved["ml_window_alignment_source"],
        resolved["timing_channel_waveforms_saved"],
    )

    if contract is None:
        return
    if not isinstance(contract, dict):
        raise ValueError("data_contract must be an object")
    aliases = {
        "expected_led_timestamp_source": "led_timestamp_source",
        "expected_cfd_timestamp_source": "cfd_timestamp_source",
        "expected_ml_window_alignment_source": "ml_window_alignment_source",
        "expected_timing_channel_waveforms_saved": "timing_channel_waveforms_saved",
    }
    for configured_name, manifest_name in aliases.items():
        if configured_name not in contract:
            continue
        expected = contract[configured_name]
        actual = resolved[manifest_name]
        if actual != expected:
            raise ValueError(
                f"Prepared-data contract mismatch for {manifest_name}: "
                f"expected {expected!r}, found {actual!r}. "
                "Check the dataset path and rebuild preprocessing if needed."
            )

def train_model(
    config: dict[str, Any],
    *,
    restart: bool,
    logger: Any,
    prepared_datasets: list[PreparedDataset] | None = None,
    data_view: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Unified entry point for every automatically discovered model.

    The experiment engine may inject zero-copy prepared-dataset views. Standalone
    training continues to load dataset paths from the configuration.
    """

    config = copy.deepcopy(config)
    config["fit"] = resolve_fit_config(config.get("fit"))
    output_dir = Path(config["output"]["train_dir"])
    if restart and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    datasets = (
        list(prepared_datasets)
        if prepared_datasets is not None
        else [load_prepared_dataset(path) for path in config["datasets"]]
    )
    _validate_dataset_contract(datasets, config.get("data_contract"), logger)
    for dataset in datasets:
        if dataset.train.size == 0 or dataset.validation.size == 0:
            raise RuntimeError(
                f"Dataset has no training/validation split: {dataset.directory}"
            )
    input_lengths = {dataset.input_length for dataset in datasets}
    if len(input_lengths) != 1:
        raise ValueError(
            "Training datasets have incompatible waveform lengths: "
            f"{sorted(input_lengths)}"
        )
    input_length = int(input_lengths.pop())

    training_config = config["training"]
    data_seed = int(training_config.get("data_seed", training_config.get("seed", 12345)))
    set_global_seed(data_seed)
    normalization = compute_normalization(
        [(dataset, dataset.train) for dataset in datasets],
        chunk_size=int(config["training"].get("normalization_chunk_size", 2048)),
    )

    model_config = dict(config["model"])
    model_type = str(model_config.pop("type"))
    model_name = str(model_config.pop("name"))
    artifacts = dict(config.get("artifacts", {}))
    if bool(artifacts.get("save_config", True)):
        atomic_json(output_dir / "train_config_used.json", config)

    context = TrainingContext(
        config=config,
        model_type=model_type,
        model_name=model_name,
        model_config=model_config,
        datasets=datasets,
        input_length=input_length,
        normalization=normalization,
        output_dir=output_dir,
        plot_dir=plot_dir,
        checkpoint_dir=checkpoint_dir,
        logger=logger,
        data_view=dict(data_view or {}),
    )
    return train_registered_model(model_type, context)
