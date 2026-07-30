from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from ml_pipeline.common import setup_logging
from ml_pipeline.config import load_train_config
from ml_pipeline.training import train_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train any registered ML correction model from prepared datasets. "
            "The model module is discovered automatically from ml_pipeline/models/."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()

    config = load_train_config(args.config, PROJECT)
    output = Path(config["output"]["train_dir"])
    logger = setup_logging(
        output.parent / f"{output.name}.train.log",
        config["logging"].get("level", "INFO"),
    )
    train_model(config, restart=args.restart, logger=logger)


if __name__ == "__main__":
    main()
