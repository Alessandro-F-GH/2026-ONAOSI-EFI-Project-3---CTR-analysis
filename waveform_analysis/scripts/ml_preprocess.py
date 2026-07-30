from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from ml_pipeline.common import setup_logging
from ml_pipeline.config import load_preprocess_config
from ml_pipeline.preprocessing import preprocess_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create post-selection ML datasets from ROOT waveforms. When "
            "dataset.blind_test is configured, the frozen test partition is saved "
            "as a separate blind dataset before model training."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    config = load_preprocess_config(args.config, PROJECT)
    output = Path(config["dataset"]["output_dir"])
    logger = setup_logging(
        output.parent / f"{output.name}.preprocess.log",
        config["logging"].get("level", "INFO"),
    )
    preprocess_dataset(config, rebuild=args.rebuild, logger=logger)


if __name__ == "__main__":
    main()
