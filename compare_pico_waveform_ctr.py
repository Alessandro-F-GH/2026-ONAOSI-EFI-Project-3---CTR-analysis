#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
WAVEFORM_ROOT = REPO_ROOT / "waveform_analysis"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(WAVEFORM_ROOT) not in sys.path:
    sys.path.insert(0, str(WAVEFORM_ROOT))

from utils.config import config_copy, load_config
from utils.pipeline import build_selection, extract_features, load_features, save_features
from utils_fit import choose_best, fit_delta_times_integer_fs, scan_timing_grid
from utils_fit.plotting import plot_ctr_comparison, plot_gaussian_fit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Pico-TDC timing-channel CTR with oscilloscope adaptive LED"
    )
    parser.add_argument("--pico-summary", required=True, type=Path)
    parser.add_argument("--scope-root-folder", required=True, type=Path)
    parser.add_argument("--scope-config", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("ctr_pico_vs_scope"))
    parser.add_argument("--root-pattern", default="*.root")
    parser.add_argument(
        "--threshold-selection-stage", choices=("blind", "validation"), default="blind",
        help="blind=oracle; validation=choose on development holdout then evaluate on blind",
    )
    parser.add_argument("--blind-fraction", type=float, default=0.2)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--voltage-pattern", default=r"(?P<voltage>\d+(?:\.\d+)?)V")
    parser.add_argument("--reuse-features", action="store_true")
    parser.add_argument("--pico-acquisition-mode", default=None)
    parser.add_argument("--allow-legacy-pico-summary", action="store_true")
    return parser.parse_args()


def _stable_seed(base: int, text: str) -> int:
    payload = f"{base}|{text}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _voltage(path: Path, pattern: str) -> float:
    match = re.search(pattern, path.name)
    if not match:
        raise ValueError(f"Cannot infer voltage from {path.name!r} using {pattern!r}")
    try:
        return float(match.group("voltage"))
    except (IndexError, KeyError):
        return float(match.group(1))


def _split_selected(
    selected: np.ndarray,
    *,
    blind_fraction: float,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.flatnonzero(np.asarray(selected, dtype=bool))
    if indices.size < 5:
        raise RuntimeError("Too few selected events to build comparison split")
    if not 0.0 < blind_fraction < 0.5:
        raise ValueError("--blind-fraction must be in (0, 0.5)")
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("--validation-fraction must be in (0, 0.5)")

    order = np.random.default_rng(seed).permutation(indices)
    n_blind = max(1, int(round(order.size * blind_fraction)))
    n_blind = min(n_blind, order.size - 2)
    blind = np.sort(order[:n_blind])
    development = order[n_blind:]
    n_validation = max(1, int(round(development.size * validation_fraction)))
    n_validation = min(n_validation, development.size - 1)
    validation = np.sort(development[:n_validation])
    train = np.sort(development[n_validation:])
    return train, validation, blind


def _mask(size: int, indices: np.ndarray) -> np.ndarray:
    result = np.zeros(size, dtype=bool)
    result[np.asarray(indices, dtype=np.int64)] = True
    return result


def _evaluate_fixed_led_threshold(
    features: dict[str, np.ndarray],
    selected_indices: np.ndarray,
    threshold_index: int,
    threshold_mV: float,
    fit_config: dict[str, Any],
    *,
    method: str,
):
    a = np.asarray(features["t_led_a_fs"])[selected_indices, threshold_index]
    b = np.asarray(features["t_led_b_fs"])[selected_indices, threshold_index]
    invalid = np.iinfo(np.int64).min
    valid = (a != invalid) & (b != invalid)
    return fit_delta_times_integer_fs(
        a[valid].astype(np.int64) - b[valid].astype(np.int64),
        method=method,
        parameter=float(threshold_mV),
        n_total=int(features["event_id"].size),
        n_selected=int(selected_indices.size),
        config=fit_config,
    )


def _scope_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    cfg = config_copy(load_config(args.scope_config))
    roots = sorted(args.scope_root_folder.glob(args.root_pattern))
    if not roots:
        raise RuntimeError(f"No ROOT files match {args.root_pattern!r} in {args.scope_root_folder}")

    cache_root = args.output / "scope_feature_cache"
    fit_plot_root = args.output / "scope_led_fits"
    cache_root.mkdir(parents=True, exist_ok=True)
    fit_plot_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for root_file in roots:
        voltage = _voltage(root_file, args.voltage_pattern)
        cache = cache_root / f"{root_file.stem}.npz"
        if args.reuse_features and cache.is_file():
            features = load_features(cache, cfg, root_file)
        else:
            features = extract_features(root_file, cfg)
            save_features(cache, features)

        selection = build_selection(features, cfg)
        _, validation, blind = _split_selected(
            selection.selected,
            blind_fraction=args.blind_fraction,
            validation_fraction=args.validation_fraction,
            seed=_stable_seed(args.seed, root_file.name),
        )

        thresholds = np.asarray(features["led_thresholds_mV"], dtype=np.float64)
        if args.threshold_selection_stage == "blind":
            score_indices = blind
            series_label = "Oscilloscope adaptive LED · oracle blind threshold"
        else:
            score_indices = validation
            series_label = "Oscilloscope adaptive LED · validation-selected threshold"

        score_results = scan_timing_grid(
            features["t_led_a_fs"],
            features["t_led_b_fs"],
            _mask(features["event_id"].size, score_indices),
            thresholds,
            method=f"LED threshold selection ({args.threshold_selection_stage})",
            config=cfg["fit"],
        )
        chosen = choose_best(score_results)
        if chosen is None:
            raise RuntimeError(f"No successful LED threshold fit for {root_file.name}")

        threshold_index = int(np.argmin(np.abs(thresholds - chosen.parameter)))
        if args.threshold_selection_stage == "blind":
            final_fit = chosen
        else:
            final_fit = _evaluate_fixed_led_threshold(
                features, blind, threshold_index, chosen.parameter, cfg["fit"],
                method="Oscilloscope LED blind",
            )
            if not final_fit.success:
                raise RuntimeError(f"Blind LED fit failed for {root_file.name}: {final_fit.message}")

        plot_gaussian_fit(
            final_fit,
            fit_plot_root / f"{root_file.stem}.png",
            dpi=int(cfg.get("plot", {}).get("dpi", 180)),
            title=f"{root_file.stem} · adaptive LED · {chosen.parameter:g} mV · blind",
        )
        rows.append({
            "source_file": root_file.name,
            "voltage_V": voltage,
            "threshold_selection_stage": args.threshold_selection_stage,
            "threshold_mV": chosen.parameter,
            "selection_ctr_ps": chosen.ctr_ps,
            "selection_ctr_error_ps": chosen.ctr_error_ps,
            "ctr_ps": final_fit.ctr_ps,
            "ctr_error_ps": final_fit.ctr_error_ps,
            "n_blind": int(blind.size),
            "n_validation": int(validation.size),
            "series_label": series_label,
            "fit_metric": "common_bin_integrated_gaussian_all_events",
        })
    return rows


def _read_pico_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    with args.pico_summary.open("r", encoding="utf-8", newline="") as stream:
        raw_rows = list(csv.DictReader(stream))
    if not raw_rows:
        raise RuntimeError(f"No rows in {args.pico_summary}")

    if not args.allow_legacy_pico_summary:
        metrics = {str(row.get("fit_metric", "")) for row in raw_rows}
        if metrics != {"common_bin_integrated_gaussian_all_events"}:
            raise RuntimeError(
                "Pico summary was not generated with the unified fitter. "
                "Rerun Janus after applying the patch."
            )

    rows: list[dict[str, Any]] = []
    for row in raw_rows:
        if args.pico_acquisition_mode is not None and row.get("AcquisitionMode") != args.pico_acquisition_mode:
            continue
        try:
            voltage = float(row["Voltage"])
            ctr = float(row["CTR_ps"])
        except (KeyError, TypeError, ValueError):
            continue
        try:
            error = float(row["CTR_error_ps"])
        except (KeyError, TypeError, ValueError):
            error = float("nan")
        try:
            threshold = float(row["T_th"])
        except (KeyError, TypeError, ValueError):
            threshold = float("nan")
        rows.append({
            "run_id": row.get("run_id", ""),
            "voltage_V": voltage,
            "timing_threshold_mV": threshold,
            "ctr_ps": ctr,
            "ctr_error_ps": error,
            "acquisition_mode": row.get("AcquisitionMode", ""),
            "fit_metric": row.get("fit_metric", ""),
        })
    if not rows:
        raise RuntimeError("No valid Pico-TDC rows remain after filtering")
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    scope = _scope_rows(args)
    pico = _read_pico_rows(args)
    _write_csv(args.output / "oscilloscope_adaptive_led.csv", scope)
    _write_csv(args.output / "pico_tdc.csv", pico)

    scope_by_voltage = {float(row["voltage_V"]): row for row in scope}
    paired: list[dict[str, Any]] = []
    for p in pico:
        s = scope_by_voltage.get(float(p["voltage_V"]))
        if s is None:
            continue
        paired.append({
            "voltage_V": p["voltage_V"],
            "pico_run_id": p["run_id"],
            "pico_timing_threshold_mV": p["timing_threshold_mV"],
            "pico_ctr_ps": p["ctr_ps"],
            "pico_ctr_error_ps": p["ctr_error_ps"],
            "scope_led_threshold_mV": s["threshold_mV"],
            "scope_threshold_selection_stage": s["threshold_selection_stage"],
            "scope_ctr_ps": s["ctr_ps"],
            "scope_ctr_error_ps": s["ctr_error_ps"],
            "pico_minus_scope_ps": float(p["ctr_ps"]) - float(s["ctr_ps"]),
            "fit_metric": "common_bin_integrated_gaussian_all_events",
        })
    _write_csv(args.output / "paired_comparison.csv", paired)

    title = "Pico-TDC vs oscilloscope adaptive LED"
    if args.threshold_selection_stage == "blind":
        title += " · oscilloscope threshold optimized on blind (oracle)"
    else:
        title += " · threshold selected on validation"
    plot_ctr_comparison(pico, scope, args.output / "ctr_vs_voltage.png", title=title)

    metadata = {
        "fit_metric": "common_bin_integrated_gaussian_all_events",
        "threshold_selection_stage": args.threshold_selection_stage,
        "blind_fraction": args.blind_fraction,
        "validation_fraction": args.validation_fraction,
        "seed": args.seed,
        "warning": (
            "Threshold selected on blind: oracle/optimistic estimate."
            if args.threshold_selection_stage == "blind" else ""
        ),
    }
    with (args.output / "comparison_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    print(f"Wrote comparison to {args.output}")
    if args.threshold_selection_stage == "blind":
        print("WARNING: threshold selected on blind; use validation mode for unbiased final result.")


if __name__ == "__main__":
    main()
