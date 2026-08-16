from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
import sys
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from ml_pipeline.dataset import PreparedDataset, load_prepared_dataset
from utils.signal import INVALID_TIME_FS


MODE_ENERGY = "energy_to_energy"
MODE_TIMING = "timing_to_timing"
REFERENCE_LED = "led"
REFERENCE_CFD = "cfd"


def make_logger() -> logging.Logger:
    logger = logging.getLogger("timing_error_signal_difference")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    return logger


def _dataset_directories(
    datasets: list[Path] | None,
    prepared_root: Path | None,
) -> list[Path]:
    found: list[Path] = []

    if datasets:
        for path in datasets:
            path = path.resolve()
            if not (path / "manifest.json").is_file():
                raise FileNotFoundError(f"Not a prepared dataset: {path}")
            found.append(path)

    if prepared_root is not None:
        root = prepared_root.resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Prepared-data root does not exist: {root}")

        if (root / "manifest.json").is_file():
            found.append(root)
        else:
            for child in sorted(root.iterdir()):
                if child.is_dir() and (child / "manifest.json").is_file():
                    found.append(child)

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in found:
        if path not in seen:
            unique.append(path)
            seen.add(path)

    if not unique:
        raise FileNotFoundError(
            "No prepared datasets found. Use --dataset PATH or "
            "--prepared-root processed_data/ml_prepared."
        )
    return unique


def _valid_timing_mask(times_fs: np.ndarray) -> np.ndarray:
    values = np.asarray(times_fs)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(
            f"Timing timestamps must have shape [event,2], got {values.shape}"
        )
    return np.all(values != INVALID_TIME_FS, axis=1)


def _waveforms_for_mode(
    dataset: PreparedDataset,
    mode: str,
    *,
    energy_variant: str,
    timing_variant: str,
) -> tuple[np.ndarray, np.ndarray, str, tuple[str, str]]:
    if mode == MODE_ENERGY:
        variant = energy_variant
        if variant == "auto":
            variant = "denoised" if dataset.denoised_windows_mV is not None else "raw"

        if variant == "denoised":
            if dataset.denoised_windows_mV is None:
                raise ValueError(
                    f"{dataset.directory.name}: denoised energy waveforms unavailable"
                )
            waveforms = dataset.denoised_windows_mV
        elif variant == "raw":
            waveforms = dataset.windows_mV
        else:
            raise ValueError(f"Unsupported energy variant: {variant}")

        channels = dataset.manifest.get("energy_channels_one_based", [1, 2])
        if not isinstance(channels, (list, tuple)) or len(channels) < 2:
            channels = [1, 2]

        return (
            waveforms,
            np.asarray(dataset.relative_time_ps),
            variant,
            (f"energy ch{channels[0]}", f"energy ch{channels[1]}"),
        )

    if mode == MODE_TIMING:
        if dataset.timing_windows_mV is None or dataset.timing_relative_time_ps is None:
            raise ValueError(
                f"{dataset.directory.name}: timing waveforms unavailable"
            )

        variant = timing_variant
        if variant == "auto":
            variant = "raw"

        if variant == "denoised":
            if dataset.denoised_timing_windows_mV is None:
                raise ValueError(
                    f"{dataset.directory.name}: denoised timing waveforms unavailable"
                )
            waveforms = dataset.denoised_timing_windows_mV
        elif variant == "raw":
            waveforms = dataset.timing_windows_mV
        else:
            raise ValueError(f"Unsupported timing variant: {variant}")

        channels = dataset.manifest.get("timing_channels_one_based", [3, 4])
        if not isinstance(channels, (list, tuple)) or len(channels) < 2:
            channels = [3, 4]

        return (
            waveforms,
            np.asarray(dataset.timing_relative_time_ps),
            variant,
            (f"timing ch{channels[0]}", f"timing ch{channels[1]}"),
        )

    raise ValueError(f"Unsupported mode: {mode}")


def _reference_times(
    dataset: PreparedDataset,
    mode: str,
    reference: str,
) -> np.ndarray:
    if mode == MODE_ENERGY:
        if reference == REFERENCE_LED:
            values = (
                dataset.energy_led_time_fs
                if dataset.energy_led_time_fs is not None
                else dataset.led_time_fs
            )
        elif reference == REFERENCE_CFD:
            values = (
                dataset.energy_cfd_time_fs
                if dataset.energy_cfd_time_fs is not None
                else dataset.cfd_time_fs
            )
        else:
            raise ValueError(reference)

    elif mode == MODE_TIMING:
        if reference == REFERENCE_LED:
            values = dataset.timing_led_time_fs
        elif reference == REFERENCE_CFD:
            values = dataset.timing_cfd_time_fs
        else:
            raise ValueError(reference)

        if values is None:
            raise ValueError(
                f"{dataset.directory.name}: {reference.upper()} timestamps "
                f"not available for {mode}"
            )
    else:
        raise ValueError(mode)

    if values is None:
        raise ValueError(
            f"{dataset.directory.name}: {reference.upper()} timestamps "
            f"not available for {mode}"
        )
    return np.asarray(values)


def _error_groups(
    times_fs: np.ndarray,
    *,
    true_tof_ps: float,
    side_center_sigma: float,
    side_half_width_sigma: float,
    center_half_width_sigma: float,
) -> tuple[float, float, list[dict[str, object]]]:
    valid = _valid_timing_mask(times_fs)

    delta_ps = (
        np.asarray(times_fs[:, 0], dtype=np.float64)
        - np.asarray(times_fs[:, 1], dtype=np.float64)
    ) / 1000.0

    residual_ps = delta_ps - float(true_tof_ps)
    valid &= np.isfinite(residual_ps)

    if np.count_nonzero(valid) < 3:
        raise RuntimeError("Need at least three valid timing pairs")

    # Remove only the global calibration offset. The remaining quantity is the
    # event-by-event timing error used to define the three populations.
    calibration_mean_ps = float(np.mean(residual_ps[valid]))
    centered_error_ps = residual_ps - calibration_mean_ps
    sigma_ps = float(np.std(centered_error_ps[valid], ddof=1))

    if not np.isfinite(sigma_ps) or sigma_ps <= 0.0:
        raise RuntimeError(f"Invalid timing-error sigma: {sigma_ps}")

    z = centered_error_ps / sigma_ps

    definitions = [
        {
            "key": "minus2sigma",
            "label": "−2σ",
            "low": -float(side_center_sigma) - float(side_half_width_sigma),
            "high": -float(side_center_sigma) + float(side_half_width_sigma),
        },
        {
            "key": "center",
            "label": "0σ",
            "low": -float(center_half_width_sigma),
            "high": +float(center_half_width_sigma),
        },
        {
            "key": "plus2sigma",
            "label": "+2σ",
            "low": +float(side_center_sigma) - float(side_half_width_sigma),
            "high": +float(side_center_sigma) + float(side_half_width_sigma),
        },
    ]

    groups: list[dict[str, object]] = []
    for definition in definitions:
        low = float(definition["low"])
        high = float(definition["high"])
        indices = np.flatnonzero(valid & (z >= low) & (z <= high)).astype(np.int64)

        groups.append(
            {
                **definition,
                "indices": indices,
                "n": int(indices.size),
                "mean_z": float(np.mean(z[indices])) if indices.size else np.nan,
                "mean_error_ps": (
                    float(np.mean(centered_error_ps[indices]))
                    if indices.size
                    else np.nan
                ),
                "std_error_ps": (
                    float(np.std(centered_error_ps[indices], ddof=1))
                    if indices.size > 1
                    else np.nan
                ),
            }
        )

    return calibration_mean_ps, sigma_ps, groups


def _mean_signal_difference(
    waveforms: np.ndarray,
    indices: np.ndarray,
    *,
    chunk_size: int,
) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    n_samples = int(waveforms.shape[2])

    if indices.size == 0:
        return np.full(n_samples, np.nan, dtype=np.float64)

    sums = np.zeros(n_samples, dtype=np.float64)
    counts = np.zeros(n_samples, dtype=np.int64)

    for start in range(0, indices.size, chunk_size):
        block_idx = indices[start : start + chunk_size]
        block = np.asarray(waveforms[block_idx], dtype=np.float64)

        # Difference first, average second.
        difference = block[:, 0, :] - block[:, 1, :]
        finite = np.isfinite(difference)

        sums += np.where(finite, difference, 0.0).sum(axis=0)
        counts += finite.sum(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        mean = sums / counts
    mean[counts == 0] = np.nan
    return mean


def _plot_reference(
    *,
    dataset: PreparedDataset,
    mode: str,
    reference: str,
    waveforms: np.ndarray,
    time_ps: np.ndarray,
    variant: str,
    channel_labels: tuple[str, str],
    groups: list[dict[str, object]],
    calibration_mean_ps: float,
    sigma_ps: float,
    output: Path,
    chunk_size: int,
    dpi: int,
) -> None:
    means: dict[str, np.ndarray] = {}
    for group in groups:
        means[str(group["key"])] = _mean_signal_difference(
            waveforms,
            np.asarray(group["indices"], dtype=np.int64),
            chunk_size=chunk_size,
        )

    time_ns = np.asarray(time_ps, dtype=np.float64) / 1000.0

    fig, axis = plt.subplots(figsize=(11.0, 5.8), constrained_layout=True)

    for group in groups:
        key = str(group["key"])
        low = float(group["low"])
        high = float(group["high"])
        n = int(group["n"])

        axis.plot(
            time_ns,
            means[key],
            linewidth=1.8,
            label=f"{group['label']} [{low:+.1f},{high:+.1f}]σ · n={n}",
        )

    axis.axhline(0.0, linestyle="--", linewidth=1.0, alpha=0.6)
    axis.axvline(0.0, linestyle="--", linewidth=1.0, alpha=0.6)
    axis.grid(True, alpha=0.25)

    axis.set_xlabel("Time relative to LED-aligned native anchor [ns]")
    axis.set_ylabel(
        f"Mean difference {channel_labels[0]} − {channel_labels[1]} [mV]"
    )
    axis.legend(loc="best", fontsize=9)

    source_name = Path(
        dataset.manifest.get("source_root", dataset.directory.name)
    ).name
    mode_label = "energy → energy" if mode == MODE_ENERGY else "timing → timing"
    ref_label = reference.upper()

    fig.suptitle(
        f"{source_name} · {mode_label} · {variant} · grouped by {ref_label} error\n"
        f"{ref_label} calibration mean={calibration_mean_ps:+.1f} ps · "
        f"σ(error)={sigma_ps:.1f} ps",
        fontsize=13,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _summary_rows(
    *,
    dataset: PreparedDataset,
    mode: str,
    reference: str,
    variant: str,
    true_tof_ps: float,
    calibration_mean_ps: float,
    sigma_ps: float,
    groups: list[dict[str, object]],
) -> list[dict[str, object]]:
    source_name = Path(
        dataset.manifest.get("source_root", dataset.directory.name)
    ).name

    rows: list[dict[str, object]] = []
    for group in groups:
        rows.append(
            {
                "dataset": dataset.directory.name,
                "source_file": source_name,
                "mode": mode,
                "timing_reference": reference,
                "waveform_variant": variant,
                "true_tof_ps": true_tof_ps,
                "calibration_mean_ps": calibration_mean_ps,
                "error_sigma_ps": sigma_ps,
                "group": group["key"],
                "group_label": group["label"],
                "z_low": group["low"],
                "z_high": group["high"],
                "n": group["n"],
                "mean_z": group["mean_z"],
                "mean_centered_timing_error_ps": group["mean_error_ps"],
                "std_centered_timing_error_ps": group["std_error_ps"],
            }
        )
    return rows


def _write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze_dataset(
    dataset_path: Path,
    *,
    modes: Iterable[str],
    references: Iterable[str],
    output_dir: Path,
    true_tof_override_ps: float | None,
    energy_variant: str,
    timing_variant: str,
    side_center_sigma: float,
    side_half_width_sigma: float,
    center_half_width_sigma: float,
    chunk_size: int,
    dpi: int,
    logger: logging.Logger,
) -> list[dict[str, object]]:
    dataset = load_prepared_dataset(dataset_path)

    true_tof_ps = (
        float(true_tof_override_ps)
        if true_tof_override_ps is not None
        else float(dataset.true_tof_ps)
    )

    rows: list[dict[str, object]] = []

    for mode in modes:
        try:
            waveforms, time_ps, variant, channel_labels = _waveforms_for_mode(
                dataset,
                mode,
                energy_variant=energy_variant,
                timing_variant=timing_variant,
            )
        except ValueError as exc:
            logger.warning("%s | %s | skipped: %s", dataset.directory.name, mode, exc)
            continue

        for reference in references:
            try:
                timing_fs = _reference_times(dataset, mode, reference)
                calibration_mean_ps, sigma_ps, groups = _error_groups(
                    timing_fs,
                    true_tof_ps=true_tof_ps,
                    side_center_sigma=side_center_sigma,
                    side_half_width_sigma=side_half_width_sigma,
                    center_half_width_sigma=center_half_width_sigma,
                )
            except (ValueError, RuntimeError) as exc:
                logger.warning(
                    "%s | %s | %s | skipped: %s",
                    dataset.directory.name,
                    mode,
                    reference.upper(),
                    exc,
                )
                continue

            counts = ", ".join(
                f"{group['label']} n={group['n']}" for group in groups
            )
            logger.info(
                "%s | %s | %s error | calibration %+.1f ps | sigma %.1f ps | %s",
                dataset.directory.name,
                mode,
                reference.upper(),
                calibration_mean_ps,
                sigma_ps,
                counts,
            )

            reference_dir = output_dir / f"{reference}_error"
            filename = (
                f"{dataset.directory.name}__{mode}"
                f"__mean_signal_difference_by_{reference}_error.png"
            )

            _plot_reference(
                dataset=dataset,
                mode=mode,
                reference=reference,
                waveforms=waveforms,
                time_ps=time_ps,
                variant=variant,
                channel_labels=channel_labels,
                groups=groups,
                calibration_mean_ps=calibration_mean_ps,
                sigma_ps=sigma_ps,
                output=reference_dir / filename,
                chunk_size=chunk_size,
                dpi=dpi,
            )

            rows.extend(
                _summary_rows(
                    dataset=dataset,
                    mode=mode,
                    reference=reference,
                    variant=variant,
                    true_tof_ps=true_tof_ps,
                    calibration_mean_ps=calibration_mean_ps,
                    sigma_ps=sigma_ps,
                    groups=groups,
                )
            )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot mean event-wise waveform difference s1(t)-s2(t) for events "
            "grouped by calibrated LED and/or CFD timing error."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        action="append",
        default=None,
        help="Prepared dataset directory. May be specified multiple times.",
    )
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=None,
        help="Root containing prepared dataset directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/diagnostics/timing_error_signal_difference"),
    )
    parser.add_argument(
        "--mode",
        choices=["both", MODE_ENERGY, MODE_TIMING],
        default="both",
    )
    parser.add_argument(
        "--reference",
        choices=["both", REFERENCE_LED, REFERENCE_CFD],
        default="both",
        help="Timing estimator used to classify events by error.",
    )
    parser.add_argument(
        "--true-tof-ps",
        type=float,
        default=None,
        help="Override true TOF stored in prepared-data manifest.",
    )
    parser.add_argument(
        "--energy-variant",
        choices=["auto", "raw", "denoised"],
        default="auto",
    )
    parser.add_argument(
        "--timing-variant",
        choices=["auto", "raw", "denoised"],
        default="raw",
    )
    parser.add_argument("--side-center-sigma", type=float, default=2.0)
    parser.add_argument("--side-half-width-sigma", type=float, default=0.5)
    parser.add_argument("--center-half-width-sigma", type=float, default=0.5)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--dpi", type=int, default=180)

    args = parser.parse_args()

    if args.dataset is None and args.prepared_root is None:
        args.prepared_root = Path("processed_data/ml_prepared")

    modes = (
        [MODE_ENERGY, MODE_TIMING]
        if args.mode == "both"
        else [args.mode]
    )
    references = (
        [REFERENCE_LED, REFERENCE_CFD]
        if args.reference == "both"
        else [args.reference]
    )

    logger = make_logger()
    datasets = _dataset_directories(args.dataset, args.prepared_root)
    logger.info("Prepared datasets found: %d", len(datasets))

    all_rows: list[dict[str, object]] = []

    for dataset_path in datasets:
        all_rows.extend(
            analyze_dataset(
                dataset_path,
                modes=modes,
                references=references,
                output_dir=args.output_dir,
                true_tof_override_ps=args.true_tof_ps,
                energy_variant=args.energy_variant,
                timing_variant=args.timing_variant,
                side_center_sigma=args.side_center_sigma,
                side_half_width_sigma=args.side_half_width_sigma,
                center_half_width_sigma=args.center_half_width_sigma,
                chunk_size=max(1, int(args.chunk_size)),
                dpi=int(args.dpi),
                logger=logger,
            )
        )

    summary_path = args.output_dir / "timing_error_signal_difference_groups.csv"
    _write_summary(summary_path, all_rows)

    logger.info(
        "Done | LED plots=%s | CFD plots=%s | summary=%s",
        args.output_dir / "led_error",
        args.output_dir / "cfd_error",
        summary_path,
    )


if __name__ == "__main__":
    main()
