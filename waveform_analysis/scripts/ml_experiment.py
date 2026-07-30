from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from ml_pipeline.common import setup_logging
from ml_pipeline.experiments import load_experiment_config, run_experiment

def main() -> None:
    parser = argparse.ArgumentParser(description="Run model-independent cross-validation and hyperparameter experiments.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    config = load_experiment_config(args.config, PROJECT)
    logger = setup_logging(Path(config["output_dir"]) / "experiment.log", config.get("logging", {}).get("level", "INFO"))
    run_experiment(config, dry_run=args.dry_run, resume=args.resume, restart=args.restart, logger=logger)

if __name__ == "__main__":
    main()
