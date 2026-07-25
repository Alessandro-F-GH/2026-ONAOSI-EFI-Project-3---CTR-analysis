from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from ml_pipeline.common import setup_logging
from ml_pipeline.config import load_model_config, load_pipeline_config
from ml_pipeline.data import prepare_energy_cache, prepare_splits
from ml_pipeline.evaluation import evaluate_final_test
from ml_pipeline.model import model_output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run final blind-test LED vs CFD vs model-corrected LED comparison"
    )
    parser.add_argument(
        "--pipeline-config",
        type=Path,
        default=PROJECT / "config" / "ml_pipeline_config.json",
    )
    parser.add_argument(
        "--model-config",
        "--cnn-config",
        dest="model_config",
        type=Path,
        default=PROJECT / "config" / "cnn_config.json",
        help="Model JSON. --cnn-config remains as a backward-compatible alias.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = load_pipeline_config(args.pipeline_config, PROJECT)
    model = load_model_config(args.model_config)
    logger = setup_logging(
        model_output_path(pipeline, "log_dir", model) / "evaluation.log",
        pipeline["logging"].get("level", "INFO"),
    )
    input_path = Path(pipeline["data"]["input_root"])
    cache = prepare_energy_cache(
        input_path,
        Path(pipeline["paths"]["dataset_cache_dir"]),
        pipeline,
        rebuild=False,
        logger=logger,
    )
    splits = prepare_splits(
        cache,
        Path(pipeline["paths"]["split_dir"]),
        pipeline,
        rebuild=False,
        logger=logger,
    )
    evaluate_final_test(
        cache,
        splits,
        pipeline,
        model,
        checkpoint_path=args.checkpoint,
        logger=logger,
    )


if __name__ == "__main__":
    main()
