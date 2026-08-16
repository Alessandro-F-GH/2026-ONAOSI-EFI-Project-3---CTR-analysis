from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from ml_pipeline.common import setup_logging
from ml_pipeline.prepared_data import plot_prepared_signal_examples, prepare_file_dataset
from ml_pipeline.study_config import discover_root_files, load_study_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the permanent post-selection ML dataset for every ROOT file. "
            "The output contains raw native-grid waveforms and, when requested, a "
            "separate denoised waveform representation; no train/CV/blind split is stored."
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="Experiment JSON")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    config = load_study_config(args.config, PROJECT)
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
        plot_prepared_signal_examples(
            dataset,
            examples / f"{root_file.stem}.png",
            dpi=int(config["reporting"]["dpi"]),
        )
    logger.info("Prepared %d permanent datasets in %s", len(files), config["preprocessing"]["prepared_dir"])


if __name__ == "__main__":
    main()
