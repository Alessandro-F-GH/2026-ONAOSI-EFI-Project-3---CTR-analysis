from __future__ import annotations

import copy
import shutil
from pathlib import Path
from typing import Any

from .common import atomic_json, canonical_hash, set_global_seed
from .config import resolve_fit_config
from .dataset import PreparedDataset, load_prepared_dataset_spec
from .input_transform import (
    materialize_training_input_cache,
    normalize_subsampling_factor,
    resolve_input_transform,
    subsampled_dataset_input_length,
)
from .models import train_registered_model
from .prediction import prediction_dataset_view, resolve_prediction_config
from .torch_data import Normalization, compute_normalization
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
        "waveform_grid",
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
        "timing waveforms saved=%s | waveform grid=%s",
        resolved["led_timestamp_source"],
        resolved["cfd_timestamp_source"],
        resolved["ml_window_alignment_source"],
        resolved["timing_channel_waveforms_saved"],
        resolved["waveform_grid"],
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
    normalization_override: Normalization | None = None,
) -> dict[str, Any]:
    """Unified entry point for every automatically discovered model.

    The experiment engine may inject zero-copy prepared-dataset views. Standalone
    training continues to load dataset paths from the configuration.
    """

    config = copy.deepcopy(config)
    config["fit"] = resolve_fit_config(config.get("fit"))
    input_transform = resolve_input_transform(config)
    config["input_transform"] = input_transform
    preprocessing_config = config.setdefault("preprocessing", {})
    model_section = config.get("model") if isinstance(config.get("model"), dict) else {}
    legacy_model_factor = model_section.pop("subsampling_factor", None)
    configured_factor = preprocessing_config.get("subsampling_factor")
    if configured_factor is not None and legacy_model_factor is not None:
        if normalize_subsampling_factor(configured_factor) != normalize_subsampling_factor(
            legacy_model_factor
        ):
            raise ValueError(
                "Conflicting subsampling factors: preprocessing.subsampling_factor "
                "and legacy model.subsampling_factor"
            )
    subsampling_factor = normalize_subsampling_factor(
        configured_factor if configured_factor is not None else legacy_model_factor
    )
    preprocessing_config["subsampling_factor"] = subsampling_factor
    prediction = resolve_prediction_config(config)
    config["prediction"] = prediction
    if isinstance(config.get("model"), dict):
        config["model"].pop("input_transform", None)
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
        else [load_prepared_dataset_spec(spec) for spec in config["datasets"]]
    )
    _validate_dataset_contract(datasets, config.get("data_contract"), logger)
    for dataset in datasets:
        if dataset.train.size == 0 or dataset.validation.size == 0:
            raise RuntimeError(
                f"Dataset has no training/validation split: {dataset.directory}"
            )

    datasets = [
        prediction_dataset_view(
            dataset,
            input_waveforms=prediction["input_waveforms"],
            target=prediction["target"],
        )
        for dataset in datasets
    ]
    logger.info(
        "Prediction task | input waveforms=%s | target=%s",
        prediction["input_waveforms"],
        prediction["target"],
    )

    transformed_datasets: list[PreparedDataset] = []
    input_cache_dirs: list[Path] = []
    configured_cache_root = config.get("training", {}).get("input_transform_cache_dir")
    cache_root = (
        Path(configured_cache_root).resolve()
        if configured_cache_root
        else output_dir / "input_cache"
    )
    cache_chunk_size = int(
        config["training"].get(
            "input_transform_chunk_size",
            config["training"].get("normalization_chunk_size", 2048),
        )
    )
    for dataset in datasets:
        transformed, cache_dir = materialize_training_input_cache(
            dataset,
            input_transform,
            cache_root,
            chunk_size=cache_chunk_size,
            rebuild=False,
            logger=logger,
        )
        transformed_datasets.append(transformed)
        if cache_dir is not None:
            input_cache_dirs.append(cache_dir)
    datasets = transformed_datasets
    logger.info(
        "Model input preprocessing | transform=%s | subsampling_factor=%d",
        input_transform,
        subsampling_factor,
    )

    input_lengths = {
        subsampled_dataset_input_length(dataset, subsampling_factor)
        for dataset in datasets
    }
    if len(input_lengths) != 1:
        raise ValueError(
            "Training datasets have incompatible waveform lengths: "
            f"{sorted(input_lengths)}"
        )
    input_length = int(input_lengths.pop())

    training_config = config["training"]
    data_seed = int(training_config.get("data_seed", training_config.get("seed", 12345)))
    set_global_seed(data_seed)

    model_config = dict(config["model"])
    model_type = str(model_config.pop("type"))
    model_name = str(model_config.pop("name"))

    # All retained waveform models consume the same normalized prepared input.
    # The experiment engine may reuse the tiny train-only normalization statistics
    # across hyperparameter candidates that share exactly the same fit subset.
    normalization = normalization_override
    if normalization is None:
        normalization = compute_normalization(
            [(dataset, dataset.train) for dataset in datasets],
            chunk_size=int(config["training"].get("normalization_chunk_size", 2048)),
            featurewise=False,
            subsampling_factor=subsampling_factor,
        )
    artifacts = dict(config.get("artifacts", {}))
    if bool(artifacts.get("save_config", True)):
        atomic_json(output_dir / "train_config_used.json", config)

    representation_descriptor = {
        "input_transform": input_transform,
        "subsampling_factor": int(subsampling_factor),
        "input_length": int(input_length),
        "input_waveforms": prediction["input_waveforms"],
        "prediction_target": prediction["target"],
        "data_view": dict(data_view or {}),
        "datasets": [
            {
                "fingerprint": dataset.manifest.get("fingerprint"),
                "prediction_view": dataset.manifest.get("prediction_view", {}),
                "input_component_lengths": dataset.manifest.get("input_component_lengths"),
                "window_before_ns": dataset.manifest.get("window_before_ns"),
                "window_after_ns": dataset.manifest.get("window_after_ns"),
            }
            for dataset in datasets
        ],
    }
    config["representation_fingerprint"] = canonical_hash(representation_descriptor)

    context = TrainingContext(
        config=config,
        model_type=model_type,
        model_name=model_name,
        model_config=model_config,
        datasets=datasets,
        input_length=input_length,
        normalization=normalization,
        input_transform=input_transform,
        subsampling_factor=subsampling_factor,
        input_waveform_source=prediction["input_waveforms"],
        prediction_target=prediction["target"],
        input_cache_dirs=tuple(input_cache_dirs),
        output_dir=output_dir,
        plot_dir=plot_dir,
        checkpoint_dir=checkpoint_dir,
        logger=logger,
        data_view=dict(data_view or {}),
    )
    return train_registered_model(model_type, context)
