from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from utils.config import config_copy, load_config
from utils.ml_dataset import (
    extract_selection_features,
    filter_rows_by_led_mad,
    load_selection_features,
    save_selection_features,
    select_events,
    plot_event_waveforms,
)
from utils.tof_shift_experiment import (
    assign_artificial_shifts,
    finalize_shift_datasets,
    generate_shift_dataset_rows,
    matched_integer_uniform_half_width,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build paired discrete-position and continuous-position TOF datasets "
            "from the same selected oscilloscope events"
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Converted ROOT run")
    parser.add_argument("--config", required=True, type=Path, help="Analysis JSON config")
    parser.add_argument("--output", required=True, type=Path, help="Output directory")
    parser.add_argument(
        "--reuse-selection-features",
        action="store_true",
        help="Reuse a compatible lightweight selection cache",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Optional event limit for testing; 0 means all",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override tof_shift_experiment.parallel.workers",
    )
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def configure_logging(output_directory: Path, level_name: str) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, level_name.upper(), logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = logging.FileHandler(
        output_directory / "tof_shift_dataset_generation.log",
        mode="w",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


def main() -> None:
    args = parse_args()
    config = config_copy(load_config(args.config))
    if "tof_shift_experiment" not in config:
        raise ValueError("config must contain a tof_shift_experiment section")
    if args.max_events is not None:
        config["io"]["max_events"] = int(args.max_events)

    experiment = config["tof_shift_experiment"]
    args.output.mkdir(parents=True, exist_ok=True)
    configure_logging(
        args.output,
        str(experiment.get("logging_level", "INFO")),
    )
    logger = logging.getLogger(__name__)
    logger.info("Input ROOT: %s", args.input)
    logger.info("Output directory: %s", args.output)

    with (args.output / "config_used.json").open("w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2)

    cache_filename = str(
        experiment.get("selection_cache_filename", "selection_features.npz")
    )
    cache_path = args.output / cache_filename
    reuse_cache = args.reuse_selection_features or bool(
        experiment.get("reuse_selection_cache", False)
    )
    if reuse_cache and cache_path.is_file():
        logger.info("Loading selection cache: %s", cache_path)
        selection_features = load_selection_features(cache_path, config, args.input)
    else:
        selection_features = extract_selection_features(args.input, config)
        save_selection_features(cache_path, selection_features)
        logger.info("Selection cache written: %s", cache_path)

    selection = select_events(selection_features, config)
    logger.info(
        "Selected events before shift-specific validity checks: %d/%d",
        selection.cutflow["selected"],
        selection.cutflow["total"],
    )

    discrete_values = [
        int(item) for item in experiment.get("discrete_shifts_ps", [-80, 0, 80])
    ]
    configured_continuous_width = experiment.get("continuous_max_abs_ps")
    continuous_width = (
        None
        if configured_continuous_width is None
        else int(configured_continuous_width)
    )
    discrete_shifts, discrete_groups, continuous_shifts = assign_artificial_shifts(
        selection.selected,
        discrete_shifts_ps=discrete_values,
        continuous_max_abs_ps=continuous_width,
        random_seed=int(experiment.get("random_seed", 42)),
    )
    variance_match = matched_integer_uniform_half_width(max(discrete_values))
    effective_continuous_width = (
        int(variance_match["continuous_half_width_ps"])
        if continuous_width is None
        else continuous_width
    )
    logger.info(
        "Variance match: discrete Var=%.3f ps^2; exact b=%.6f ps; "
        "integer-uniform b=%d ps gives Var=%.3f ps^2",
        float(variance_match["discrete_variance_ps2"]),
        float(variance_match["exact_uniform_half_width_ps"]),
        effective_continuous_width,
        float(effective_continuous_width * (effective_continuous_width + 1)) / 3.0,
    )
    selected_indices = np.flatnonzero(selection.selected)
    unique_discrete, counts_discrete = np.unique(
        discrete_shifts[selected_indices], return_counts=True
    )
    logger.info(
        "Discrete shift allocation: %s",
        ", ".join(
            f"{int(shift):+d} ps -> {int(count)}"
            for shift, count in zip(unique_discrete, counts_discrete, strict=True)
        ),
    )
    logger.info(
        "Continuous shift range observed: [%d, %d] ps; configured support=[-%d, +%d] ps",
        int(np.min(continuous_shifts[selected_indices])),
        int(np.max(continuous_shifts[selected_indices])),
        effective_continuous_width,
        effective_continuous_width,
    )

    discrete_rows, continuous_rows, rejections = generate_shift_dataset_rows(
        args.input,
        selection.selected,
        discrete_shifts,
        discrete_groups,
        continuous_shifts,
        config,
        workers_override=args.workers,
    )

    mad_config = config.get("led_mad_filter", {})
    mad_threshold = float(mad_config.get("threshold", 5.0))
    discrete_rows, mad_summary, worst_event_index = filter_rows_by_led_mad(
        discrete_rows, threshold=mad_threshold
    )
    retained_event_indices = {int(row["meta_event_index"]) for row in discrete_rows}
    continuous_rows = [
        row for row in continuous_rows
        if int(row["meta_event_index"]) in retained_event_indices
    ]
    if len(discrete_rows) != len(continuous_rows):
        raise RuntimeError("MAD filtering broke discrete/continuous row alignment")
    logger.info(
        "LED MAD filter: retained %d/%d paired events; rejected=%d; threshold=%.3g",
        mad_summary["events_after"],
        mad_summary["events_before"],
        mad_summary["events_rejected"],
        mad_threshold,
    )
    if worst_event_index is not None:
        outlier_plot_path = args.output / str(
            mad_config.get("largest_outlier_plot", "largest_led_mad_outlier_waveforms.png")
        )
        plot_event_waveforms(
            args.input,
            worst_event_index,
            config,
            outlier_plot_path,
            title=(
                "Largest LED MAD outlier | "
                f"event={worst_event_index} | "
                f"distance={mad_summary['worst_mad_distance']:.3f} | "
                f"TOF={mad_summary['worst_led_tof_ps']:.3f} ps"
            ),
        )
        mad_summary["largest_outlier_plot"] = str(outlier_plot_path)
        logger.info("Largest LED MAD outlier plot: %s", outlier_plot_path)

    dataset_summary = finalize_shift_datasets(
        discrete_rows,
        continuous_rows,
        args.output,
        config,
    )

    cutflow = dict(selection.cutflow)
    cutflow["led_mad_filter"] = mad_summary
    cutflow["paired_shift_dataset_valid"] = len(discrete_rows)
    cutflow["paired_shift_dataset_rejected"] = int(sum(rejections.values()))
    cutflow["shift_dataset_rejection_reasons"] = dict(sorted(rejections.items()))
    with (args.output / "tof_shift_cutflow.json").open("w", encoding="utf-8") as stream:
        json.dump(cutflow, stream, indent=2)

    summary = {
        "input": str(args.input),
        "selection_cache": str(cache_path),
        "cutflow": cutflow,
        "photopeak": [item.as_dict() for item in selection.photopeak_results],
        "led_mad_filter": mad_summary,
        "dataset": dataset_summary,
    }
    with (args.output / "tof_shift_dataset_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(_json_safe(summary), stream, indent=2, allow_nan=False)

    logger.info("Paired accepted rows per dataset: %d", len(discrete_rows))
    logger.info("Dataset generation complete")


if __name__ == "__main__":
    main()
