from __future__ import annotations

import argparse
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
            "Run the compact CTR study: permanent dataset preparation, random "
            "development/blind split, fold-wise CV model selection, early stopping "
            "on a train-only subset, one blind evaluation, standards, XAI and "
            "raw-only multithreshold SVR."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", "--check", dest="dry_run", action="store_true",
                        help="Validate the config/model registry and list discovered input files without training")
    parser.add_argument("--resume", action="store_true",
                        help="Resume a partial compatible run: completed CV candidates/files are skipped")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--rebuild-preprocessing", action="store_true")
    args = parser.parse_args()

    config = load_study_config(args.config, PROJECT)
    output = Path(config["experiment"]["output_dir"])
    if args.restart and output.exists():
        shutil.rmtree(output)

    logger = setup_logging(output / "study.log", config.get("logging", {}).get("level", "INFO"))
    result = run_study(
        config,
        dry_run=args.dry_run,
        resume=args.resume,
        restart=False,
        rebuild_preprocessing=args.rebuild_preprocessing,
        logger=logger,
    )
    logger.info("Study complete | %s", result)


if __name__ == "__main__":
    main()
