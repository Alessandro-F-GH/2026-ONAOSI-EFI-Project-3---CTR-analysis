from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import logging
from pathlib import Path
import shutil
import sys
from typing import Any

import torch

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from utils.cnn_config import (
    load_experiment_config,
    load_model_config,
    load_preprocessing_config,
)
from utils.cnn_data import build_cnn_dataset_cache
from utils.cnn_evaluation import evaluate_experiment
from utils.cnn_training import train_model_run
from utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the PET TOF CNN experiment: discrete translation augmentation, "
            "same-position versus uniform-position tests, and an invariant correction CNN"
        )
    )
    parser.add_argument(
        "--experiment-config",
        required=True,
        type=Path,
        help="JSON file containing input/output paths and references to preprocessing/model configs",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Rebuild the augmented waveform cache even if it already exists",
    )
    parser.add_argument(
        "--training-only",
        action="store_true",
        help="Train models but skip evaluation",
    )
    parser.add_argument(
        "--reuse-trained",
        action="store_true",
        help="Reuse checkpoints listed in output/training_results.json instead of retraining",
    )
    return parser.parse_args()


def configure_logging(output_dir: Path, level: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [
        logging.StreamHandler(),
        logging.FileHandler(output_dir / "cnn_experiment.log", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def _resolved_device_pool(model_config: dict[str, Any]) -> list[str]:
    requested = [str(item) for item in model_config["parallel"].get("device_pool", ["auto"])]
    if "auto" not in requested:
        return requested
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        available = [f"cuda:{index}" for index in range(torch.cuda.device_count())]
        resolved: list[str] = []
        auto_index = 0
        for item in requested:
            if item == "auto":
                resolved.append(available[auto_index % len(available)])
                auto_index += 1
            else:
                resolved.append(item)
        return resolved
    return ["cpu" if item == "auto" else item for item in requested]


def _effective_parallel_runs(model_config: dict[str, Any]) -> int:
    requested = int(model_config["parallel"].get("max_parallel_runs", 1))
    devices = _resolved_device_pool(model_config)
    cuda_devices = sorted({item for item in devices if item.startswith("cuda")})
    if cuda_devices and requested > len(cuda_devices):
        logging.warning(
            "Configured %d concurrent runs but only %d distinct CUDA device(s); reducing concurrency to %d",
            requested,
            len(cuda_devices),
            len(cuda_devices),
        )
        return max(1, len(cuda_devices))
    return requested


def _device_for_task(index: int, model_config: dict[str, Any]) -> str:
    devices = _resolved_device_pool(model_config)
    return devices[index % len(devices)]

def _copy_configs(
    output_dir: Path,
    experiment_config: dict[str, Any],
    preprocessing_config: dict[str, Any],
    model_config: dict[str, Any],
) -> None:
    used_dir = output_dir / "configs_used"
    used_dir.mkdir(parents=True, exist_ok=True)
    for name, config in (
        ("experiment.json", experiment_config),
        ("preprocessing.json", preprocessing_config),
        ("cnn_model.json", model_config),
    ):
        serializable = {key: value for key, value in config.items() if not key.startswith("_")}
        with (used_dir / name).open("w", encoding="utf-8") as stream:
            json.dump(serializable, stream, indent=2)
    analysis_path = Path(experiment_config["analysis_config"])
    shutil.copy2(analysis_path, used_dir / "analysis.json")


def main() -> None:
    args = parse_args()
    experiment = load_experiment_config(args.experiment_config)
    preprocessing = load_preprocessing_config(experiment["preprocessing_config"])
    model_config = load_model_config(experiment["model_config"])
    input_path = Path(experiment["input_root"])
    output_dir = Path(experiment["output_dir"])
    cache_dir = Path(experiment["cache_dir"])
    configure_logging(output_dir, str(experiment.get("logging_level", "INFO")))
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    _copy_configs(output_dir, experiment, preprocessing, model_config)

    dataset_path = cache_dir / str(preprocessing["cache"]["filename"])
    reuse_cache = bool(experiment.get("reuse_preprocessed", True)) and not args.rebuild_cache
    if reuse_cache and dataset_path.is_file():
        logging.info("Using augmented waveform cache: %s", dataset_path)
    else:
        dataset_path, summary = build_cnn_dataset_cache(
            input_path=input_path,
            analysis_config_path=Path(experiment["analysis_config"]),
            preprocessing_config=preprocessing,
            cache_dir=cache_dir,
            max_events=int(experiment.get("max_events", 0)),
            reuse_selection_cache=bool(preprocessing["cache"].get("reuse_selection_cache", True)),
        )
        logging.info("Dataset summary: %s", summary.as_dict())

    training_results_path = output_dir / "training_results.json"
    if args.reuse_trained:
        if not training_results_path.is_file():
            raise FileNotFoundError(
                f"--reuse-trained requested but {training_results_path} does not exist"
            )
        with training_results_path.open("r", encoding="utf-8") as stream:
            training_results = json.load(stream)
        missing = [item["checkpoint"] for item in training_results if not Path(item["checkpoint"]).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing trained checkpoints: {missing}")
        logging.info("Reusing %d trained CNN realizations", len(training_results))
    else:
        seeds = [int(item) for item in model_config["training"]["seeds"]]
        tasks: list[dict[str, Any]] = []
        index = 0
        for model_type in ("direct", "correction"):
            for seed in seeds:
                run_dir = output_dir / "training" / model_type / f"seed_{seed}"
                tasks.append(
                    {
                        "model_type": model_type,
                        "seed": seed,
                        "dataset_path": str(dataset_path),
                        "run_dir": str(run_dir),
                        "model_config": model_config,
                        "preprocessing_config": preprocessing,
                        "device": _device_for_task(index, model_config),
                    }
                )
                index += 1

        max_parallel_runs = _effective_parallel_runs(model_config)
        for run_number, task in enumerate(tasks, start=1):
            task["run_number"] = run_number
            task["total_runs"] = len(tasks)
            task["parallel_runs"] = max_parallel_runs

        training_results: list[dict[str, Any]] = []
        if max_parallel_runs == 1:
            logging.info(
                "Training %d CNN runs sequentially (safe default); device pool=%s",
                len(tasks),
                _resolved_device_pool(model_config),
            )
            for task in tasks:
                training_results.append(train_model_run(task))
        else:
            logging.info("Training %d CNN runs with up to %d concurrent processes", len(tasks), max_parallel_runs)
            with ProcessPoolExecutor(max_workers=max_parallel_runs) as executor:
                futures = {executor.submit(train_model_run, task): task for task in tasks}
                for future in as_completed(futures):
                    task = futures[future]
                    result = future.result()
                    training_results.append(result)
                    logging.info(
                        "Completed %s seed=%s: validation loss %.4f",
                        task["model_type"],
                        task["seed"],
                        result["best_validation_loss"],
                    )
        training_results.sort(key=lambda item: (item["model_type"], int(item["seed"])))
        with training_results_path.open("w", encoding="utf-8") as stream:
            json.dump(training_results, stream, indent=2)

    if not args.training_only:
        analysis_config = load_config(experiment["analysis_config"])
        evaluate_experiment(
            dataset_path=dataset_path,
            training_results=training_results,
            analysis_config=analysis_config,
            model_config=model_config,
            output_dir=output_dir / "evaluation",
        )
    logging.info("CNN TOF experiment complete: %s", output_dir)


if __name__ == "__main__":
    main()
