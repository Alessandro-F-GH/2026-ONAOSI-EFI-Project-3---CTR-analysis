from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from ml_pipeline.common import setup_logging
from ml_pipeline.experiment_config import load_experiment_config
from ml_pipeline.prepared_data import plot_prepared_signal_examples, prepare_file_dataset
from ml_pipeline.study_config import discover_root_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize permanent post-selection ML datasets")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    config = load_experiment_config(args.config, PROJECT)
    output = Path(config["experiment"]["output_dir"])
    logger = setup_logging(output / "preprocess.log", config.get("logging", {}).get("level", "INFO"))
    files = discover_root_files(config)
    if not files:
        raise FileNotFoundError(
            f"No ROOT files match {config['data']['root_glob']} in {config['data']['root_folder']}"
        )
    examples = output / "preprocessing_examples"
    for index, root_file in enumerate(files, start=1):
        logger.info("Preprocess %d/%d | %s", index, len(files), root_file.name)
        dataset = prepare_file_dataset(config, root_file, rebuild=args.rebuild, logger=logger)
        plot_prepared_signal_examples(dataset, examples / f"{root_file.stem}.png", dpi=int(config["reporting"]["dpi"]))
    logger.info("Prepared %d permanent datasets in %s", len(files), config["preprocessing"]["prepared_dir"])


if __name__ == "__main__":
    main()
