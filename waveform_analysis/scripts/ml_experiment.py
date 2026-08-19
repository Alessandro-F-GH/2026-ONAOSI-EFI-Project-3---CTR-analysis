from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from ml_pipeline.common import setup_logging
from ml_pipeline.experiment_config import load_experiment_config, public_resolved_config
from ml_pipeline.study import run_study


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the compact CTR study with modular configuration, dictionary "
            "modes and additive artifact-aware resume."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--dry-run", "--check", dest="dry_run", action="store_true",
        help="Validate resolved configuration/model availability without training",
    )
    parser.add_argument(
        "--print-resolved-config", action="store_true",
        help="Print the fully merged/effective JSON configuration and exit",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help=(
            "Reuse compatible scientific results and create only missing/additional "
            "report artifacts such as XAI, TOP/WORST, distributions or CFD views"
        ),
    )
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--rebuild-preprocessing", action="store_true")
    args = parser.parse_args()

    if args.resume and args.restart:
        parser.error("--resume and --restart are mutually exclusive")

    config = load_experiment_config(args.config, PROJECT)
    if args.print_resolved_config:
        print(json.dumps(public_resolved_config(config), indent=2))
        return

    output = Path(config["experiment"]["output_dir"])
    if args.restart and output.exists():
        shutil.rmtree(output)

    logger = setup_logging(
        output / "study.log",
        config.get("logging", {}).get("level", "INFO"),
    )
    for source in config.get("_config_sources", []):
        logger.info(
            "Config module | role=%s | %s | hash=%s",
            source.get("role"),
            source.get("path"),
            str(source.get("content_hash", ""))[:12],
        )

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
