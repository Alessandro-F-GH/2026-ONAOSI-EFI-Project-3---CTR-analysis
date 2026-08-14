from __future__ import annotations

import argparse
import copy
from pathlib import Path
import shutil
import sys

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from ml_pipeline.common import setup_logging
from ml_pipeline.study import run_study
from ml_pipeline.study_config import load_study_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the raw-waveform multithreshold SVR comparison using the same prepared "
            "datasets, random CV protocol and global Gaussian fit as the main ML experiment."
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="Main experiment JSON")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--rebuild-preprocessing", action="store_true")
    args = parser.parse_args()

    config = load_study_config(args.config, PROJECT)
    if not bool(config.get("multithreshold", {}).get("enabled", False)):
        raise ValueError("multithreshold.enabled must be true")
    config = copy.deepcopy(config)
    config["models"] = []
    config["_model_spaces"] = []
    original = Path(config["experiment"]["output_dir"])
    config["experiment"]["name"] += "_multithreshold_only"
    config["experiment"]["output_dir"] = str(original.with_name(original.name + "_multithreshold_only"))
    output = Path(config["experiment"]["output_dir"])
    if args.restart and output.exists():
        shutil.rmtree(output)
    logger = setup_logging(output / "study.log", config.get("logging", {}).get("level", "INFO"))
    run_study(
        config,
        dry_run=False,
        resume=False,
        restart=False,
        rebuild_preprocessing=args.rebuild_preprocessing,
        logger=logger,
    )


if __name__ == "__main__":
    main()
