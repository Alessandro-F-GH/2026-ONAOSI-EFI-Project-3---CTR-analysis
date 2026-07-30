from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from ml_pipeline.data import prepare_energy_cache
from utils.photopeak import fit_photopeak, photopeak_mask


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def make_logger() -> logging.Logger:
    logger = logging.getLogger("led_correlation")
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


def configure_cache(
    base_config: dict,
    cache_dir: Path,
    *,
    timing_led_enabled: bool,
) -> dict:
    config = copy.deepcopy(base_config)

    config.setdefault("waveform", {})
    config["waveform"].setdefault("timing_channel_led", {})
    config["waveform"]["timing_channel_led"]["enabled"] = timing_led_enabled

    config.setdefault("cache", {})
    config["cache"]["raw_cache_dir"] = str(cache_dir)
    config["cache"]["reuse"] = True

    return config


def finite_led_mask(led_time_fs: np.ndarray) -> np.ndarray:
    # INVALID_TIME_FS is an extreme negative int64 value. Requiring reasonably
    # finite converted timestamps also excludes it safely.
    led = np.asarray(led_time_fs, dtype=np.int64)

    return (
        np.all(led > np.iinfo(np.int64).min // 2, axis=1)
        & np.all(led < np.iinfo(np.int64).max // 2, axis=1)
    )


def make_photopeak_mask(
    amplitudes_mV: np.ndarray,
    valid_mask: np.ndarray,
    config: dict,
    logger: logging.Logger,
) -> np.ndarray:
    amplitudes = np.asarray(amplitudes_mV, dtype=np.float64)

    if not bool(config["photopeak"].get("enabled", True)):
        logger.info("Photopeak selection disabled in configuration")
        return valid_mask.copy()

    selected = valid_mask.copy()

    for channel_position, channel_number in enumerate(
        config["channels"]["energy"]
    ):
        values = amplitudes[
            valid_mask,
            channel_position,
        ]

        result = fit_photopeak(
            values,
            channel=int(channel_number),
            config=config["photopeak"],
        )

        if not result.success:
            raise RuntimeError(
                f"Photopeak fit failed for energy channel {channel_number}: "
                f"{result.message}"
            )

        channel_mask = photopeak_mask(
            amplitudes[:, channel_position],
            result,
        )
        selected &= channel_mask

        logger.info(
            "Energy channel %d photopeak | mean %.3f mV | sigma %.3f mV | "
            "accepted %d/%d",
            channel_number,
            float(result.mean_mV),
            float(result.sigma_mV),
            int(np.count_nonzero(channel_mask & valid_mask)),
            int(np.count_nonzero(valid_mask)),
        )

    return selected


def centered_limits(
    x: np.ndarray,
    y: np.ndarray,
    percentile: float,
) -> tuple[float, float]:
    joined = np.concatenate([x, y])

    low_percentile = (100.0 - percentile) / 2.0
    high_percentile = 100.0 - low_percentile

    low, high = np.percentile(
        joined,
        [low_percentile, high_percentile],
    )

    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        low = float(np.min(joined))
        high = float(np.max(joined))

    margin = 0.05 * max(high - low, 1.0)
    return float(low - margin), float(high + margin)


def plot_scatter(
    energy_led_ps: np.ndarray,
    timing_led_ps: np.ndarray,
    output_path: Path,
    *,
    pearson_r: float,
    spearman_rho: float,
    max_points: int,
    seed: int,
    display_percentile: float,
) -> None:
    number = energy_led_ps.size

    if max_points > 0 and number > max_points:
        rng = np.random.default_rng(seed)
        plot_indices = rng.choice(
            number,
            size=max_points,
            replace=False,
        )
    else:
        plot_indices = np.arange(number)

    x_plot = energy_led_ps[plot_indices]
    y_plot = timing_led_ps[plot_indices]

    figure, axis = plt.subplots(figsize=(8.5, 7.0))

    axis.scatter(
        x_plot,
        y_plot,
        s=8,
        alpha=0.3,
        linewidths=0,
    )

    low, high = centered_limits(
        x_plot,
        y_plot,
        percentile=display_percentile,
    )

    axis.plot(
        [low, high],
        [low, high],
        linestyle="--",
        linewidth=1.2,
        label="y = x",
    )

    axis.set_xlim(low, high)
    axis.set_ylim(low, high)

    axis.set_xlabel(
        "Energy-channel LED difference "
        r"$t_{\mathrm{LED},1}-t_{\mathrm{LED},2}$ [ps]"
    )
    axis.set_ylabel(
        "Timing-channel LED difference "
        r"$t_{\mathrm{LED},3}-t_{\mathrm{LED},4}$ [ps]"
    )
    axis.set_title(
        "Energy LED vs timing LED\n"
        "Energy-photopeak events only"
    )
    axis.grid(True, alpha=0.25)
    axis.legend()

    statistics = (
        f"Selected events: {number:,}\n"
        f"Displayed events: {plot_indices.size:,}\n"
        f"Pearson r: {pearson_r:.4f}\n"
        f"Spearman ρ: {spearman_rho:.4f}"
    )

    axis.text(
        0.03,
        0.97,
        statistics,
        transform=axis.transAxes,
        va="top",
        ha="left",
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.85,
        },
    )

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_difference_histogram(
    energy_led_ps: np.ndarray,
    timing_led_ps: np.ndarray,
    output_path: Path,
    display_percentile: float,
) -> None:
    difference_ps = timing_led_ps - energy_led_ps

    lower_tail = (100.0 - display_percentile) / 2.0
    upper_tail = 100.0 - lower_tail
    low, high = np.percentile(
        difference_ps,
        [lower_tail, upper_tail],
    )

    visible = difference_ps[
        (difference_ps >= low)
        & (difference_ps <= high)
    ]

    figure, axis = plt.subplots(figsize=(8.5, 5.5))

    axis.hist(
        visible,
        bins=150,
        histtype="step",
        linewidth=1.3,
    )

    axis.axvline(
        np.median(difference_ps),
        linestyle="--",
        linewidth=1.2,
        label="Median",
    )

    axis.set_xlabel(
        r"$\Delta t_{\mathrm{LED,timing}}"
        r"-\Delta t_{\mathrm{LED,energy}}$ [ps]"
    )
    axis.set_ylabel("Events")
    axis.set_title(
        "Difference between timing-channel and energy-channel LED"
    )
    axis.grid(True, alpha=0.25)
    axis.legend()

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare energy-channel and timing-channel LED measurements "
            "on energy-photopeak events."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Timing-LED preprocessing JSON configuration.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/led_channel_correlation"),
        help="Output directory.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild both waveform caches.",
    )
    parser.add_argument(
        "--max-plot-points",
        type=int,
        default=100_000,
        help="Maximum number of points drawn in the scatterplot.",
    )
    parser.add_argument(
        "--display-percentile",
        type=float,
        default=99.5,
        help=(
            "Central percentile used only for plot limits. "
            "All selected events are retained in the CSV and correlations."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260729,
        help="Random seed used only for scatterplot downsampling.",
    )

    args = parser.parse_args()

    if not 0.0 < args.display_percentile <= 100.0:
        parser.error("--display-percentile must lie in (0, 100].")

    logger = make_logger()
    config_path = args.config.resolve()
    base_config = load_json(config_path)

    output_dir = args.output_dir.resolve()
    cache_root = output_dir / "cache"

    energy_config = configure_cache(
        base_config,
        cache_root / "energy_led",
        timing_led_enabled=False,
    )
    timing_config = configure_cache(
        base_config,
        cache_root / "timing_led",
        timing_led_enabled=True,
    )

    input_root = Path(base_config["data"]["input_root"])

    logger.info("Preparing energy-channel LED cache")
    energy_cache = prepare_energy_cache(
        input_root,
        Path(energy_config["cache"]["raw_cache_dir"]),
        energy_config,
        rebuild=args.rebuild,
        logger=logger,
    )

    logger.info("Preparing timing-channel LED cache")
    timing_cache = prepare_energy_cache(
        input_root,
        Path(timing_config["cache"]["raw_cache_dir"]),
        timing_config,
        rebuild=args.rebuild,
        logger=logger,
    )

    if energy_cache.event_id.shape != timing_cache.event_id.shape:
        raise RuntimeError(
            "Energy-LED and timing-LED caches have different event counts."
        )

    if not np.array_equal(
        np.asarray(energy_cache.event_id),
        np.asarray(timing_cache.event_id),
    ):
        raise RuntimeError(
            "Energy-LED and timing-LED caches do not contain events "
            "in the same order."
        )

    energy_led_fs = np.asarray(
        energy_cache.led_time_fs,
        dtype=np.int64,
    )
    timing_led_fs = np.asarray(
        timing_cache.led_time_fs,
        dtype=np.int64,
    )


    valid = (
        np.asarray(energy_cache.valid, dtype=bool)
        & np.asarray(timing_cache.valid, dtype=bool)
        & finite_led_mask(energy_led_fs)
        & finite_led_mask(timing_led_fs)
    )

    logger.info(
        "Events with valid energy and timing LED: %d/%d",
        int(np.count_nonzero(valid)),
        int(valid.size),
    )

    selected = make_photopeak_mask(
        np.asarray(energy_cache.amplitude_mV),
        valid,
        base_config,
        logger,
    )

    indices = np.flatnonzero(selected)

    if indices.size < 3:
        raise RuntimeError(
            f"Only {indices.size} events remain after photopeak selection."
        )

    energy_delta_ps = (
        energy_led_fs[indices, 0].astype(np.float64)
        - energy_led_fs[indices, 1].astype(np.float64)
    ) / 1000.0

    timing_delta_ps = (
        timing_led_fs[indices, 0].astype(np.float64)
        - timing_led_fs[indices, 1].astype(np.float64)
    ) / 1000.0

    finite = (
        np.isfinite(energy_delta_ps)
        & np.isfinite(timing_delta_ps)
    )

    indices = indices[finite]
    energy_delta_ps = energy_delta_ps[finite]
    timing_delta_ps = timing_delta_ps[finite]

    outlier_config = base_config["selection"].get(
    "led_outlier_rejection",
    {},
)

    if bool(outlier_config.get("enabled", False)):
        max_distance_ps = float(
            outlier_config["max_distance_ps"]
        )

        timing_median_ps = float(
            np.median(timing_delta_ps)
        )

        outlier_mask = (
            np.abs(timing_delta_ps - timing_median_ps)
            <= max_distance_ps
        )

        logger.info(
            "Timing LED outlier rejection | median %.3f ps | "
            "maximum distance %.3f ps | retained %d/%d",
            timing_median_ps,
            max_distance_ps,
            int(np.count_nonzero(outlier_mask)),
            int(outlier_mask.size),
        )

        indices = indices[outlier_mask]
        energy_delta_ps = energy_delta_ps[outlier_mask]
        timing_delta_ps = timing_delta_ps[outlier_mask]

    pearson_result = pearsonr(
        energy_delta_ps,
        timing_delta_ps,
    )
    spearman_result = spearmanr(
        energy_delta_ps,
        timing_delta_ps,
    )

    difference_ps = timing_delta_ps - energy_delta_ps

    logger.info(
        "Selected photopeak events: %d",
        int(indices.size),
    )
    logger.info(
        "Pearson correlation: r=%.6f, p=%.6g",
        float(pearson_result.statistic),
        float(pearson_result.pvalue),
    )
    logger.info(
        "Spearman correlation: rho=%.6f, p=%.6g",
        float(spearman_result.statistic),
        float(spearman_result.pvalue),
    )
    logger.info(
        "Timing minus energy LED | mean %.3f ps | median %.3f ps | "
        "standard deviation %.3f ps",
        float(np.mean(difference_ps)),
        float(np.median(difference_ps)),
        float(np.std(difference_ps, ddof=1)),
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    dataframe = pd.DataFrame(
        {
            "event_index": indices,
            "event_id": np.asarray(
                energy_cache.event_id[indices],
                dtype=np.int64,
            ),
            "energy_led_ch1_ps": (
                energy_led_fs[indices, 0].astype(np.float64)
                / 1000.0
            ),
            "energy_led_ch2_ps": (
                energy_led_fs[indices, 1].astype(np.float64)
                / 1000.0
            ),
            "timing_led_ch3_ps": (
                timing_led_fs[indices, 0].astype(np.float64)
                / 1000.0
            ),
            "timing_led_ch4_ps": (
                timing_led_fs[indices, 1].astype(np.float64)
                / 1000.0
            ),
            "energy_led_difference_ps": energy_delta_ps,
            "timing_led_difference_ps": timing_delta_ps,
            "timing_minus_energy_ps": difference_ps,
            "energy_amplitude_ch1_mV": np.asarray(
                energy_cache.amplitude_mV[indices, 0],
                dtype=np.float64,
            ),
            "energy_amplitude_ch2_mV": np.asarray(
                energy_cache.amplitude_mV[indices, 1],
                dtype=np.float64,
            ),
        }
    )

    csv_path = output_dir / "energy_timing_led_photopeak.csv"
    dataframe.to_csv(csv_path, index=False)

    scatter_path = output_dir / "energy_vs_timing_led_scatter.png"
    plot_scatter(
        energy_delta_ps,
        timing_delta_ps,
        scatter_path,
        pearson_r=float(pearson_result.statistic),
        spearman_rho=float(spearman_result.statistic),
        max_points=args.max_plot_points,
        seed=args.seed,
        display_percentile=args.display_percentile,
    )

    difference_path = output_dir / "timing_minus_energy_led_histogram.png"
    plot_difference_histogram(
        energy_delta_ps,
        timing_delta_ps,
        difference_path,
        display_percentile=args.display_percentile,
    )

    summary = {
        "input_root": str(input_root),
        "config": str(config_path),
        "photopeak_events": int(indices.size),
        "pearson_r": float(pearson_result.statistic),
        "pearson_pvalue": float(pearson_result.pvalue),
        "spearman_rho": float(spearman_result.statistic),
        "spearman_pvalue": float(spearman_result.pvalue),
        "timing_minus_energy_mean_ps": float(
            np.mean(difference_ps)
        ),
        "timing_minus_energy_median_ps": float(
            np.median(difference_ps)
        ),
        "timing_minus_energy_std_ps": float(
            np.std(difference_ps, ddof=1)
        ),
        "selection": (
            "valid LED on both channel families and energy photopeak only; "
            "no LED outlier rejection"
        ),
    }

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)

    logger.info("Scatterplot written to %s", scatter_path)
    logger.info("Difference histogram written to %s", difference_path)
    logger.info("Selected-event table written to %s", csv_path)
    logger.info("Summary written to %s", summary_path)


if __name__ == "__main__":
    main()