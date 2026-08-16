from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from ml_pipeline.common import setup_logging
from ml_pipeline.prepared_data import plot_prepared_signal_examples, prepare_file_dataset
from ml_pipeline.study import run_study
from ml_pipeline.study_config import discover_root_files, load_study_config


def _prepare_only(config, *, rebuild: bool, logger) -> dict:
    files = discover_root_files(config)
    if not files:
        raise FileNotFoundError(
            f"No ROOT files match {config['data']['root_glob']} in {config['data']['root_folder']}"
        )
    examples = Path(config["experiment"]["output_dir"]) / "preprocessing_examples"
    for index, root_file in enumerate(files, start=1):
        logger.info("Prepare %d/%d | %s", index, len(files), root_file.name)
        dataset = prepare_file_dataset(config, root_file, rebuild=rebuild, logger=logger)
        plot_prepared_signal_examples(
            dataset, examples / f"{root_file.stem}.png",
            dpi=int(config["reporting"]["dpi"]),
        )
    return {"prepared_files": len(files), "prepared_dir": config["preprocessing"]["prepared_dir"]}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the unified CTR experiment using the retained working ML/model pipeline. "
            "Validation can be holdout, CV, or nested (inner holdout/CV)."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", "--check", dest="dry_run", action="store_true")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Prepare/reuse permanent selected datasets and exit without model training")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--rebuild-preprocessing", action="store_true")
    args = parser.parse_args()

    config = load_study_config(args.config, PROJECT)
    output = Path(config["experiment"]["output_dir"])
    if args.restart and output.exists():
        shutil.rmtree(output)
    logger = setup_logging(output / "study.log", config.get("logging", {}).get("level", "INFO"))

    if args.prepare_only:
        result = _prepare_only(config, rebuild=args.rebuild_preprocessing, logger=logger)
    else:
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
