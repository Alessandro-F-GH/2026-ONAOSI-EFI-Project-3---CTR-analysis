#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from utils.config import load_config
from utils.group_analysis import (
    GROUP_SUMMARY_FILENAME,
    run_all_compatible_group_analyses,
    run_compatible_group_analysis,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Pool candidate-preprocessed Janus runs by compatibility group, calibrate "
            "one average delay per side, concatenate matched binaries, run selection "
            "plus timing fit, and maintain one consolidated group-results summary."
        )
    )
    parser.add_argument(
        "main_run_output",
        nargs="?",
        help=(
            "Path to a representative run output directory, for example "
            "outputs/07-10/analysis/Run4700. Omit it when using --all-groups."
        ),
    )
    parser.add_argument("-c", "--config", default="config/janus_pipeline.json")
    parser.add_argument(
        "--all-groups",
        action="store_true",
        help=(
            "Analyze every compatibility group in the configured dataset and write "
            f"one {GROUP_SUMMARY_FILENAME} table with one row per group."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute group-level stages even when their cache is valid.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip runs whose candidate-preprocessing outputs are missing.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Optional grouped-analysis directory. By default results are written "
            "under <dataset output>/grouped_analysis/."
        ),
    )
    args = parser.parse_args()

    if args.all_groups and args.main_run_output:
        parser.error("main_run_output must be omitted when --all-groups is used")
    if not args.all_groups and not args.main_run_output:
        parser.error("provide main_run_output or use --all-groups")

    root = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    cfg = load_config(config_path, root)

    if args.all_groups:
        summary_path = run_all_compatible_group_analyses(
            cfg,
            overwrite=args.overwrite,
            skip_missing=args.skip_missing,
            output_root=args.output_root,
        )
    else:
        group_dir = run_compatible_group_analysis(
            args.main_run_output,
            cfg,
            overwrite=args.overwrite,
            skip_missing=args.skip_missing,
            output_root=args.output_root,
        )
        summary_path = group_dir.parent / GROUP_SUMMARY_FILENAME

    print(f"Grouped results summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
