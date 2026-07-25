from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from ml_pipeline.common import atomic_json, setup_logging
from ml_pipeline.config import load_pipeline_config
from ml_pipeline.data import prepare_energy_cache, prepare_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the energy-channel-only ML dataset and frozen 80/10/10 split"
    )
    parser.add_argument(
        "--pipeline-config",
        type=Path,
        default=PROJECT / "config" / "ml_pipeline_config.json",
    )
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--rebuild-split", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_pipeline_config(args.pipeline_config, PROJECT)
    logger = setup_logging(
        Path(config["paths"]["log_dir"]) / "prepare.log",
        config["logging"].get("level", "INFO"),
    )
    input_path = Path(config["data"]["input_root"])
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input ROOT file not found: {input_path}. Edit data.input_root in the pipeline config."
        )
    work_dir = Path(config["paths"]["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(work_dir / "ml_pipeline_config_used.json", config)
    cache = prepare_energy_cache(
        input_path,
        Path(config["paths"]["dataset_cache_dir"]),
        config,
        rebuild=args.rebuild_cache,
        logger=logger,
    )
    splits = prepare_splits(
        cache,
        Path(config["paths"]["split_dir"]),
        config,
        rebuild=args.rebuild_split,
        logger=logger,
    )
    logger.info(
        "Preparation complete. Frozen selected split: train=%d, validation=%d, test=%d",
        splits.train.size,
        splits.validation.size,
        splits.test.size,
    )


if __name__ == "__main__":
    main()
