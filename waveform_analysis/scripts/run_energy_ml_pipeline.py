from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from ml_pipeline.common import atomic_json, setup_logging
from ml_pipeline.config import load_model_config, load_pipeline_config
from ml_pipeline.data import prepare_energy_cache, prepare_splits
from ml_pipeline.evaluation import evaluate_final_test
from ml_pipeline.model import model_output_path
from ml_pipeline.training import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare, train, and evaluate the separated energy-only ML pipeline"
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
        help="CNN, time-series MLP, or Catch22 random-forest JSON configuration",
    )
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--rebuild-split", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--restart", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = load_pipeline_config(args.pipeline_config, PROJECT)
    model = load_model_config(args.model_config)
    logger = setup_logging(
        model_output_path(pipeline, "log_dir", model) / "pipeline.log",
        pipeline["logging"].get("level", "INFO"),
    )
    input_path = Path(pipeline["data"]["input_root"])
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input ROOT file not found: {input_path}. Edit data.input_root in the pipeline config."
        )
    work_dir = model_output_path(pipeline, "work_dir", model)
    work_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(work_dir / "ml_pipeline_config_used.json", pipeline)
    atomic_json(work_dir / "model_config_used.json", model)
    cache = prepare_energy_cache(
        input_path,
        Path(pipeline["paths"]["dataset_cache_dir"]),
        pipeline,
        rebuild=args.rebuild_cache,
        logger=logger,
    )
    splits = prepare_splits(
        cache,
        Path(pipeline["paths"]["split_dir"]),
        pipeline,
        rebuild=args.rebuild_split,
        logger=logger,
    )
    train_model(
        cache,
        splits,
        pipeline,
        model,
        resume=args.resume,
        restart=args.restart,
        logger=logger,
    )
    evaluate_final_test(
        cache,
        splits,
        pipeline,
        model,
        checkpoint_path=None,
        logger=logger,
    )


if __name__ == "__main__":
    main()
