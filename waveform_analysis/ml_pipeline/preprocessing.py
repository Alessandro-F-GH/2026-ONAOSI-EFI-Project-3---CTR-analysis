from __future__ import annotations

from pathlib import Path
from typing import Any

from .data import prepare_energy_cache, prepare_splits
from .dataset import (
    PreparedDataset,
    materialize_prepared_dataset,
    materialize_training_and_blind_datasets,
)


def preprocess_dataset(config: dict[str, Any], *, rebuild: bool, logger: Any) -> PreparedDataset:
    cache = prepare_energy_cache(
        Path(config["data"]["input_root"]),
        Path(config["cache"]["raw_cache_dir"]),
        config,
        rebuild=rebuild or not bool(config["cache"].get("reuse", True)),
        logger=logger,
    )
    splits = prepare_splits(
        cache,
        Path(config["cache"]["selection_cache_dir"]),
        config,
        rebuild=rebuild or not bool(config["cache"].get("reuse", True)),
        logger=logger,
    )

    if "blind_test" in config["dataset"]:
        training_dataset, blind_dataset = materialize_training_and_blind_datasets(
            cache,
            splits,
            config,
            rebuild=rebuild,
            logger=logger,
        )
        first_label = (
            "development"
            if bool(config.get("split", {}).get("development_blind", False))
            else "train/validation"
        )
        logger.info(
            "Frozen split saved separately | %s: %s | blind test: %s",
            first_label,
            training_dataset.directory,
            blind_dataset.directory,
        )
        return training_dataset

    return materialize_prepared_dataset(cache, splits, config, rebuild=rebuild, logger=logger)
