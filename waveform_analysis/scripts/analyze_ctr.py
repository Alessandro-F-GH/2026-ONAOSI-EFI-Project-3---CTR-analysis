from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from waveform_analysis.config import config_copy, load_config
from waveform_analysis.fit import FitResult, choose_best, scan_timing_grid
from waveform_analysis.io import event_count, read_metadata
from waveform_analysis.pipeline import (
    build_selection,
    extract_features,
    load_features,
    save_features,
)
from waveform_analysis.plots import (
    plot_best_fit,
    plot_energy_correlation,
    plot_energy_photopeaks,
    plot_noise_distributions,
    plot_scan,
    plot_toa_for_parameter,
    plot_trigger_toa,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="C-compatible CTR analysis from one converted waveform ROOT file"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--reuse-features",
        action="store_true",
        help="Reuse the feature cache when waveform-processing settings match",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Optional test override; 0 means all events",
    )
    return parser.parse_args()


def write_scan(path: Path, results: list[FitResult]) -> None:
    rows = [item.as_dict() for item in results]
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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


def _index_of_parameter(parameters: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(np.asarray(parameters, dtype=np.float64) - float(value))))


def main() -> None:
    args = parse_args()
    config = config_copy(load_config(args.config))
    if args.max_events is not None:
        config["io"]["max_events"] = int(args.max_events)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "config_used.json").open("w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2)

    metadata = read_metadata(args.input)
    cache_path = args.output / str(config["cache"]["filename"])
    reuse = args.reuse_features or bool(config["cache"].get("reuse", False))
    if reuse and cache_path.is_file():
        print(f"Loading feature cache: {cache_path}")
        features = load_features(cache_path, config, args.input)
    else:
        features = extract_features(args.input, config)
        save_features(cache_path, features)
        print(f"Feature cache: {cache_path}")

    selection = build_selection(features, config)
    selected = selection.selected
    print(
        f"Selected events: {selection.cutflow['selected']}/"
        f"{selection.cutflow['total']}"
    )

    led_parameters = features["led_thresholds_mV"]
    cfd_parameters = features["cfd_fractions"]
    led_results = scan_timing_grid(
        features["t_led_a_fs"],
        features["t_led_b_fs"],
        selected,
        led_parameters,
        method="LED",
        config=config["fit"],
    )
    cfd_results = scan_timing_grid(
        features["t_cfd_a_fs"],
        features["t_cfd_b_fs"],
        selected,
        cfd_parameters,
        method="CFD",
        config=config["fit"],
    )
    best_led = choose_best(led_results)
    best_cfd = choose_best(cfd_results)

    write_scan(args.output / "led_scan.csv", led_results)
    write_scan(args.output / "cfd_scan.csv", cfd_results)
    with (args.output / "cutflow.json").open("w", encoding="utf-8") as stream:
        json.dump(selection.cutflow, stream, indent=2)

    plot_config = config["plot"]
    dpi = int(plot_config["dpi"])
    energy_channels = features["energy_channels_zero_based"].astype(np.int64)
    timing_channels = features["timing_channels_zero_based"].astype(np.int64)
    plot_energy_photopeaks(
        features["amplitude_mV"],
        energy_channels,
        list(selection.photopeak_results),
        selected,
        args.output / "energy_photopeak_selection.png",
        dpi=dpi,
        bins=int(plot_config["energy_bins"]),
    )
    plot_energy_correlation(
        features["amplitude_mV"],
        energy_channels,
        selected,
        args.output / "energy_correlation.png",
        dpi=dpi,
    )
    plot_noise_distributions(
        features["noise_rms_mV"],
        timing_channels,
        selected,
        args.output / "timing_noise.png",
        dpi=dpi,
        noise_limit_mV=config["selection"].get("timing_noise_max_mV"),
    )
    plot_trigger_toa(
        features["trigger_time_fs"],
        timing_channels,
        selected,
        args.output / "timing_trigger_toa.png",
        dpi=dpi,
        bins=int(plot_config["toa_bins"]),
    )
    plot_scan(
        led_results,
        best_led,
        "LED threshold [mV]",
        args.output / "ctr_vs_led_threshold.png",
        dpi=dpi,
        errorbars=bool(plot_config["scan_errorbars"]),
    )
    plot_scan(
        cfd_results,
        best_cfd,
        "CFD fraction",
        args.output / "ctr_vs_cfd_fraction.png",
        dpi=dpi,
        errorbars=bool(plot_config["scan_errorbars"]),
    )

    timing_channel_numbers = [int(item) + 1 for item in timing_channels]
    if best_led is not None:
        plot_best_fit(best_led, args.output / "best_led_fit.png", dpi=dpi)
        led_index = _index_of_parameter(led_parameters, best_led.parameter)
        plot_toa_for_parameter(
            features["t_led_a_fs"],
            features["t_led_b_fs"],
            selected,
            led_index,
            timing_channel_numbers,
            args.output / "best_led_toa.png",
            title=f"Best LED · {best_led.parameter:g} mV",
            dpi=dpi,
            bins=int(plot_config["toa_bins"]),
        )
    if best_cfd is not None:
        plot_best_fit(best_cfd, args.output / "best_cfd_fit.png", dpi=dpi)
        cfd_index = _index_of_parameter(cfd_parameters, best_cfd.parameter)
        plot_toa_for_parameter(
            features["t_cfd_a_fs"],
            features["t_cfd_b_fs"],
            selected,
            cfd_index,
            timing_channel_numbers,
            args.output / "best_cfd_toa.png",
            title=f"Best CFD · {best_cfd.parameter:g}",
            dpi=dpi,
            bins=int(plot_config["toa_bins"]),
        )

    summary = {
        "input": str(args.input),
        "source_metadata": metadata,
        "events_in_root": event_count(args.input),
        "events_processed": int(features["event_id"].size),
        "cutflow": selection.cutflow,
        "photopeak": [item.as_dict() for item in selection.photopeak_results],
        "best_selection_rule": "minimum CTR among mathematically successful fits",
        "timing_storage_unit": "int64 femtoseconds",
        "photopeak_window": "mean - 2 sigma to mean + 4 sigma",
        "best_led": None if best_led is None else best_led.as_dict(),
        "best_cfd": None if best_cfd is None else best_cfd.as_dict(),
    }
    with (args.output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(_json_safe(summary), stream, indent=2, allow_nan=False)

    if best_led is None:
        print("Best LED: no successful fit")
    else:
        print(
            f"Best LED: threshold={best_led.parameter:g} mV, "
            f"CTR={best_led.ctr_ps:.1f} ps, χ²/ndf={best_led.chi2_ndof:.3g}"
        )
    if best_cfd is None:
        print("Best CFD: no successful fit")
    else:
        print(
            f"Best CFD: fraction={best_cfd.parameter:g}, "
            f"CTR={best_cfd.ctr_ps:.1f} ps, χ²/ndf={best_cfd.chi2_ndof:.3g}"
        )
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
