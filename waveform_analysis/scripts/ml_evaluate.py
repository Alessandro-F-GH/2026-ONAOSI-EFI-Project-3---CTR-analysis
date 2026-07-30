from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from ml_pipeline.common import setup_logging
from ml_pipeline.config import load_evaluate_config
from ml_pipeline.evaluation import evaluate_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained models on prepared blind datasets.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_evaluate_config(args.config, PROJECT)
    output = Path(config["output"]["evaluation_dir"])
    logger = setup_logging(output / "evaluation.log", config["logging"].get("level", "INFO"))
    evaluate_models(config, logger=logger)


if __name__ == "__main__":
    main()
