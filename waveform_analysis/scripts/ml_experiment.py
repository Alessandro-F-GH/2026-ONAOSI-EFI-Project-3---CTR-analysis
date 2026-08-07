from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from ml_pipeline.common import restrict_to_study_progress, setup_logging
from ml_pipeline.study import run_study
from ml_pipeline.study_config import load_study_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the folder-driven multi-file ML study: preprocessing, channel modes, "
            "windows, transforms, common losses, model-specific hyperparameter spaces, "
            "cross-validation, and blind validation-quality audit."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--rebuild-preprocessing", action="store_true")
    args = parser.parse_args()

    config = load_study_config(args.config, PROJECT)
    output = Path(config["experiment"]["output_dir"])

    # On Windows, a FileHandler keeps study.log locked. Restart cleanup must
    # therefore happen before setup_logging opens the new log file.
    if args.restart and output.exists():
        shutil.rmtree(output)

    logger = restrict_to_study_progress(
        setup_logging(
            output / "study.log", config.get("logging", {}).get("level", "INFO")
        )
    )
    result = run_study(
        config,
        dry_run=args.dry_run,
        resume=args.resume,
        restart=False,
        rebuild_preprocessing=args.rebuild_preprocessing,
        logger=logger,
    )
    logger.info(
        "Study complete | output=%s | rows=%s",
        result.get("output_dir"),
        result.get("row_count"),
        extra={"study_progress": True},
    )


if __name__ == "__main__":
    main()
