from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


METHOD_SPECS = (
    ("best_led", "LED", "threshold_mV", "mV"),
    ("best_cfd", "CFD", "fraction", ""),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect the best LED and CFD fits from all analysis result folders "
            "into one summary CSV."
        )
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results"),
        help="Root directory containing run result folders (default: results)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output CSV path. Default: <results>/ctr_summary.csv"
        ),
    )
    parser.add_argument(
        "--overall-only",
        action="store_true",
        help=(
            "Write only the method with the smallest successful CTR for each run. "
            "By default, one LED row and one CFD row are written."
        ),
    )
    return parser.parse_args()


def finite_or_blank(value: Any) -> Any:
    """Return an empty field for None, NaN, or infinite values."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, str)):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return number if math.isfinite(number) else ""


def run_sort_key(name: str) -> tuple[float, str]:
    """Sort folders such as 44V-370mV, 45V-400mV, ... numerically."""
    match = re.search(r"(\d+(?:\.\d+)?)\s*V", name, flags=re.IGNORECASE)
    voltage = float(match.group(1)) if match else math.inf
    return voltage, name.lower()


def discover_summaries(results_root: Path) -> list[Path]:
    if not results_root.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_root}")
    return sorted(
        results_root.rglob("summary.json"),
        key=lambda path: run_sort_key(str(path.parent.relative_to(results_root))),
    )


def parameter_fields(method: str, parameter: Any) -> tuple[str, Any, str]:
    if method == "LED":
        return "threshold_mV", finite_or_blank(parameter), "mV"
    return "fraction", finite_or_blank(parameter), ""


def make_row(
    *,
    results_root: Path,
    summary_path: Path,
    summary: dict[str, Any],
    result_key: str,
    method: str,
) -> dict[str, Any] | None:
    fit = summary.get(result_key)
    if not isinstance(fit, dict):
        return None

    run_directory = summary_path.parent
    relative_run = str(run_directory.relative_to(results_root))
    parameter_name, parameter_value, parameter_unit = parameter_fields(
        method, fit.get("parameter")
    )

    n_total = fit.get("n_total", summary.get("events_processed"))
    n_selected = fit.get("n_selected")
    n_rejected = fit.get("n_rejected")
    if n_rejected is None and isinstance(n_total, int) and isinstance(n_selected, int):
        n_rejected = n_total - n_selected

    return {
        "run": relative_run,
        "input_file": summary.get("input", ""),
        "method": method,
        "parameter_name": parameter_name,
        "parameter_value": parameter_value,
        "parameter_unit": parameter_unit,
        "success": bool(fit.get("success", False)),
        "ctr_ps": finite_or_blank(fit.get("ctr_ps")),
        "ctr_error_ps": finite_or_blank(fit.get("ctr_error_ps")),
        "mean_ps": finite_or_blank(fit.get("mean_ps")),
        "mean_error_ps": finite_or_blank(fit.get("mean_error_ps")),
        "sigma_ps": finite_or_blank(fit.get("sigma_ps")),
        "sigma_error_ps": finite_or_blank(fit.get("sigma_error_ps")),
        "chi2": finite_or_blank(fit.get("chi2")),
        "ndof": finite_or_blank(fit.get("ndof")),
        "chi2_ndof": finite_or_blank(fit.get("chi2_ndof")),
        "fit_low_ps": finite_or_blank(fit.get("fit_low_ps")),
        "fit_high_ps": finite_or_blank(fit.get("fit_high_ps")),
        "iterations": finite_or_blank(fit.get("iterations")),
        "n_total": finite_or_blank(n_total),
        "n_selected": finite_or_blank(n_selected),
        "n_rejected": finite_or_blank(n_rejected),
        "n_valid": finite_or_blank(fit.get("n_valid")),
        "n_fit": finite_or_blank(fit.get("n_fit")),
        "crossing_efficiency": finite_or_blank(
            fit.get("crossing_efficiency")
        ),
        "message": fit.get("message", ""),
        "summary_file": str(summary_path),
    }


def read_rows(
    results_root: Path,
    summary_paths: Iterable[Path],
    *,
    overall_only: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    for summary_path in summary_paths:
        try:
            with summary_path.open("r", encoding="utf-8") as stream:
                summary = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"{summary_path}: {exc}")
            continue

        run_rows: list[dict[str, Any]] = []
        for result_key, method, _, _ in METHOD_SPECS:
            row = make_row(
                results_root=results_root,
                summary_path=summary_path,
                summary=summary,
                result_key=result_key,
                method=method,
            )
            if row is not None:
                run_rows.append(row)

        if overall_only and run_rows:
            successful = [
                row
                for row in run_rows
                if row["success"] and row["ctr_ps"] != ""
            ]
            if successful:
                run_rows = [min(successful, key=lambda row: float(row["ctr_ps"]))]
            else:
                run_rows = run_rows[:1]

        rows.extend(run_rows)

    method_order = {"LED": 0, "CFD": 1}
    rows.sort(
        key=lambda row: (
            run_sort_key(str(row["run"])),
            method_order.get(str(row["method"]), 99),
        )
    )
    return rows, warnings


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run",
        "input_file",
        "method",
        "parameter_name",
        "parameter_value",
        "parameter_unit",
        "success",
        "ctr_ps",
        "ctr_error_ps",
        "mean_ps",
        "mean_error_ps",
        "sigma_ps",
        "sigma_error_ps",
        "chi2",
        "ndof",
        "chi2_ndof",
        "fit_low_ps",
        "fit_high_ps",
        "iterations",
        "n_total",
        "n_selected",
        "n_rejected",
        "n_valid",
        "n_fit",
        "crossing_efficiency",
        "message",
        "summary_file",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    results_root = args.results.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else results_root / "ctr_summary.csv"
    )

    summary_paths = discover_summaries(results_root)
    if not summary_paths:
        raise SystemExit(
            f"No summary.json files found below: {results_root}"
        )

    rows, warnings = read_rows(
        results_root,
        summary_paths,
        overall_only=args.overall_only,
    )
    if not rows:
        raise SystemExit("No best LED/CFD fit entries were found.")

    write_csv(output, rows)

    print(f"Found run summaries: {len(summary_paths)}")
    print(f"Rows written:        {len(rows)}")
    print(f"Summary CSV:         {output}")
    if warnings:
        print(f"Warnings:            {len(warnings)}")
        for warning in warnings:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()
