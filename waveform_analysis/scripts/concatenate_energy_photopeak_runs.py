from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ml_pipeline.concatenate_energy_runs import (  # noqa: E402
    concatenate_energy_photopeak_runs,
    load_concatenation_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Iterate over processed ROOT runs, fit the two energy photopeaks "
            "independently in each run, and concatenate only events passing both cuts."
        )
    )
    parser.add_argument(
        "--config",
        default="config/concatenate_energy_photopeak_config.json",
        help="Path to the concatenation JSON configuration.",
    )
    parser.add_argument(
        "--input-folder",
        default=None,
        help="Optional override for input.folder.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional override for output.root_file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output ROOT file and manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_concatenation_config(args.config)
    level_name = str(config.get("logging", {}).get("level", "INFO")).upper()
    logging.basicConfig(
        level=getattr(logging, level_name, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("concatenate_energy_photopeak_runs")
    output_root, manifest = concatenate_energy_photopeak_runs(
        config,
        input_folder_override=args.input_folder,
        output_root_override=args.output_root,
        overwrite_override=True if args.overwrite else None,
        logger=logger,
    )
    logger.info("Done: %s", output_root)
    logger.info("Manifest: %s", manifest)


if __name__ == "__main__":
    main()
