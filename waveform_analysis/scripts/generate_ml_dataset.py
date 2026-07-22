from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from utils.config import config_copy, load_config
from utils.ml_dataset import (
    extract_selection_features,
    finalize_and_write_dataset,
    generate_dataset_rows,
    load_selection_features,
    save_selection_features,
    select_events,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a CSV dataset for ML timing correction after the same "
            "photopeak/noise selection used by the CTR pipeline"
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Converted ROOT run")
    parser.add_argument("--config", required=True, type=Path, help="Analysis JSON config")
    parser.add_argument("--output", required=True, type=Path, help="Output directory")
    parser.add_argument(
        "--reuse-selection-features",
        action="store_true",
        help="Reuse the lightweight selection cache when compatible",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Optional test override; 0 means all events",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override ml_dataset.parallel.workers; 1 disables multiprocessing",
    )
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def configure_logging(output_directory: Path, level_name: str) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, level_name.upper(), logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = logging.FileHandler(
        output_directory / "dataset_generation.log",
        mode="w",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


def main() -> None:
    args = parse_args()
    config = config_copy(load_config(args.config))
    if "ml_dataset" not in config:
        raise ValueError("config must contain an ml_dataset section")
    if args.max_events is not None:
        config["io"]["max_events"] = int(args.max_events)

    args.output.mkdir(parents=True, exist_ok=True)
    configure_logging(
        args.output,
        str(config["ml_dataset"].get("logging_level", "INFO")),
    )
    logger = logging.getLogger(__name__)
    logger.info("Input ROOT: %s", args.input)
    logger.info("Output directory: %s", args.output)

    with (args.output / "config_used.json").open("w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2)

    cache_filename = str(
        config["ml_dataset"].get("selection_cache_filename", "selection_features.npz")
    )
    cache_path = args.output / cache_filename
    reuse_cache = args.reuse_selection_features or bool(
        config["ml_dataset"].get("reuse_selection_cache", False)
    )
    if reuse_cache and cache_path.is_file():
        logger.info("Loading selection cache: %s", cache_path)
        selection_features = load_selection_features(cache_path, config, args.input)
    else:
        selection_features = extract_selection_features(args.input, config)
        save_selection_features(cache_path, selection_features)
        logger.info("Selection cache written: %s", cache_path)

    selection = select_events(selection_features, config)
    logger.info(
        "Selected events before dataset-specific validity checks: %d/%d",
        selection.cutflow["selected"],
        selection.cutflow["total"],
    )

    rows, dataset_rejections = generate_dataset_rows(
        args.input,
        selection.selected,
        config,
        workers_override=args.workers,
    )
    dataset_filename = str(config["ml_dataset"].get("filename", "ml_dataset.csv"))
    dataset_path = args.output / dataset_filename
    dataset_summary = finalize_and_write_dataset(rows, dataset_path, config)

    cutflow = dict(selection.cutflow)
    cutflow["dataset_waveform_valid"] = len(rows)
    cutflow["dataset_waveform_rejected"] = int(sum(dataset_rejections.values()))
    cutflow["dataset_rejection_reasons"] = dict(sorted(dataset_rejections.items()))
    with (args.output / "dataset_cutflow.json").open("w", encoding="utf-8") as stream:
        json.dump(cutflow, stream, indent=2)

    summary = {
        "input": str(args.input),
        "selection_cache": str(cache_path),
        "cutflow": cutflow,
        "photopeak": [item.as_dict() for item in selection.photopeak_results],
        "dataset": dataset_summary,
    }
    with (args.output / "dataset_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(_json_safe(summary), stream, indent=2, allow_nan=False)

    logger.info("Final dataset: %s", dataset_path)
    logger.info("Accepted rows: %d", len(rows))


if __name__ == "__main__":
    main()
