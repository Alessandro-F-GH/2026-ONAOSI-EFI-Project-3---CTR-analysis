from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
import re
import sys
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from ml_pipeline.dataset import PreparedDataset, load_prepared_dataset
from ml_pipeline.prediction import prediction_dataset_view
from ml_pipeline.torch_data import factored_correction_target_ps


MODE_CONFIG = {
    "energy_to_energy": ("energy", "energy_led"),
    "timing_to_timing": ("timing", "timing_led"),
}


def make_logger() -> logging.Logger:
    log = logging.getLogger("secondary_feature_correlation")
    log.setLevel(logging.INFO)
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        log.addHandler(handler)
    return log


def find_datasets(
    explicit: list[Path] | None,
    prepared_root: Path | None,
) -> list[Path]:
    found: list[Path] = []

    if explicit:
        for path in explicit:
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
            found.extend(
                child
                for child in sorted(root.iterdir())
                if child.is_dir() and (child / "manifest.json").is_file()
            )

    unique: list[Path] = []
    for path in found:
        if path not in unique:
            unique.append(path)

    if not unique:
        raise FileNotFoundError(
            "No prepared datasets found. Use --dataset or "
            "--prepared-root processed_data/ml_prepared."
        )
    return unique


def _window_indices(
    time_ps: np.ndarray,
    start_ns: float,
    end_ns: float,
) -> np.ndarray:
    start_ps = 1000.0 * float(start_ns)
    end_ps = 1000.0 * float(end_ns)

    if end_ps <= start_ps:
        raise ValueError("end_ns must be larger than start_ns")

    idx = np.flatnonzero((time_ps >= start_ps) & (time_ps <= end_ps))
    if idx.size < 3:
        raise ValueError(
            f"Window [{start_ns:g},{end_ns:g}] ns contains only {idx.size} samples"
        )
    return idx.astype(np.int64)


def _safe_trapz(y: np.ndarray, t_ns: np.ndarray) -> np.ndarray:
    # np.trapezoid is preferred in newer NumPy; fall back for older versions.
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return np.asarray(trapezoid(y, x=t_ns, axis=1), dtype=np.float64)


def _linear_slope(y: np.ndarray, t_ns: np.ndarray) -> np.ndarray:
    t0 = t_ns - float(np.mean(t_ns))
    denominator = float(np.sum(t0 * t0))
    if denominator <= 0.0:
        return np.full(y.shape[0], np.nan, dtype=np.float64)
    centered_y = y - np.mean(y, axis=1, keepdims=True)
    return np.sum(centered_y * t0[None, :], axis=1) / denominator


def _weighted_centroid(
    y: np.ndarray,
    t_ns: np.ndarray,
) -> np.ndarray:
    # Shift each waveform so the minimum in the ROI has zero weight.
    # This preserves a robust centroid even if the secondary structure sits
    # on a non-zero baseline or contains a shallow negative excursion.
    weights = y - np.min(y, axis=1, keepdims=True)
    denominator = np.sum(weights, axis=1)
    out = np.full(y.shape[0], np.nan, dtype=np.float64)
    good = denominator > 0.0
    out[good] = (
        np.sum(weights[good] * t_ns[None, :], axis=1)
        / denominator[good]
    )
    return out


def _roi_mean(
    y: np.ndarray,
    t_ns: np.ndarray,
    low_ns: float,
    high_ns: float,
) -> np.ndarray:
    mask = (t_ns >= float(low_ns)) & (t_ns <= float(high_ns))
    if np.count_nonzero(mask) == 0:
        return np.full(y.shape[0], np.nan, dtype=np.float64)
    return np.mean(y[:, mask], axis=1)


def per_channel_features(
    waveform: np.ndarray,
    t_ns: np.ndarray,
    *,
    early_roi: tuple[float, float],
    late_roi: tuple[float, float],
) -> dict[str, np.ndarray]:
    """Extract physically interpretable features from one detector channel."""

    y = np.asarray(waveform, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError(f"Expected [event,sample] waveform, got {y.shape}")

    max_index = np.argmax(y, axis=1)
    min_index = np.argmin(y, axis=1)
    event_index = np.arange(y.shape[0])

    maximum = y[event_index, max_index]
    minimum = y[event_index, min_index]

    early_mean = _roi_mean(y, t_ns, *early_roi)
    late_mean = _roi_mean(y, t_ns, *late_roi)

    return {
        "peak_amp_mV": maximum,
        "trough_amp_mV": minimum,
        "peak_to_peak_mV": maximum - minimum,
        "mean_mV": np.mean(y, axis=1),
        "std_mV": np.std(y, axis=1, ddof=1),
        "area_mV_ns": _safe_trapz(y, t_ns),
        "abs_area_mV_ns": _safe_trapz(np.abs(y), t_ns),
        "t_peak_ns": t_ns[max_index],
        "t_trough_ns": t_ns[min_index],
        "centroid_ns": _weighted_centroid(y, t_ns),
        "slope_mV_per_ns": _linear_slope(y, t_ns),
        "early_mean_mV": early_mean,
        "late_mean_mV": late_mean,
        "early_late_contrast_mV": early_mean - late_mean,
    }


def asymmetric_features(
    features_ch1: dict[str, np.ndarray],
    features_ch2: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if features_ch1.keys() != features_ch2.keys():
        raise ValueError("Channel feature sets differ")

    out: dict[str, np.ndarray] = {}

    for name in features_ch1:
        a = np.asarray(features_ch1[name], dtype=np.float64)
        b = np.asarray(features_ch2[name], dtype=np.float64)

        out[f"delta_{name}"] = a - b

        # Dimensionless asymmetry is useful for quantities whose scale may
        # change with SiPM voltage. Time-like features are intentionally left
        # as plain differences.
        if name not in {"t_peak_ns", "t_trough_ns", "centroid_ns", "slope_mV_per_ns"}:
            denominator = np.abs(a) + np.abs(b)
            normalized = np.full(a.shape, np.nan, dtype=np.float64)
            good = denominator > 1e-12
            normalized[good] = (a[good] - b[good]) / denominator[good]
            out[f"asym_{name}"] = normalized

    return out


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted p values."""
    p = np.asarray(p_values, dtype=np.float64)
    q = np.full(p.shape, np.nan, dtype=np.float64)
    finite = np.flatnonzero(np.isfinite(p))
    if finite.size == 0:
        return q

    ordered_local = np.argsort(p[finite])
    ordered = finite[ordered_local]
    m = float(ordered.size)

    adjusted = np.empty(ordered.size, dtype=np.float64)
    running = 1.0
    for reverse_rank in range(ordered.size - 1, -1, -1):
        rank = reverse_rank + 1
        value = p[ordered[reverse_rank]] * m / rank
        running = min(running, value)
        adjusted[reverse_rank] = min(1.0, running)

    q[ordered] = adjusted
    return q


def correlation_rows(
    features: dict[str, np.ndarray],
    target_ps: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for name, raw_values in features.items():
        x = np.asarray(raw_values, dtype=np.float64)
        y = np.asarray(target_ps, dtype=np.float64)
        finite = np.isfinite(x) & np.isfinite(y)

        if np.count_nonzero(finite) < 3:
            rows.append(
                {
                    "feature": name,
                    "n": int(np.count_nonzero(finite)),
                    "pearson_r": np.nan,
                    "pearson_p": np.nan,
                    "spearman_rho": np.nan,
                    "spearman_p": np.nan,
                    "linear_slope_ps_per_feature": np.nan,
                    "linear_intercept_ps": np.nan,
                    "r2_univariate": np.nan,
                }
            )
            continue

        xx = x[finite]
        yy = y[finite]

        if np.std(xx) <= 0.0 or np.std(yy) <= 0.0:
            pr = pp = sr = sp = np.nan
        else:
            pr, pp = pearsonr(xx, yy)
            sr, sp = spearmanr(xx, yy)

        design = np.column_stack([xx, np.ones_like(xx)])
        slope, intercept = np.linalg.lstsq(design, yy, rcond=None)[0]
        prediction = slope * xx + intercept
        ss_res = float(np.sum((yy - prediction) ** 2))
        ss_tot = float(np.sum((yy - np.mean(yy)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else np.nan

        rows.append(
            {
                "feature": name,
                "n": int(xx.size),
                "pearson_r": float(pr),
                "pearson_p": float(pp),
                "spearman_rho": float(sr),
                "spearman_p": float(sp),
                "linear_slope_ps_per_feature": float(slope),
                "linear_intercept_ps": float(intercept),
                "r2_univariate": float(r2),
            }
        )

    pearson_q = benjamini_hochberg(
        np.asarray([row["pearson_p"] for row in rows], dtype=np.float64)
    )
    spearman_q = benjamini_hochberg(
        np.asarray([row["spearman_p"] for row in rows], dtype=np.float64)
    )

    for row, pq, sq in zip(rows, pearson_q, spearman_q):
        row["pearson_q_fdr"] = float(pq)
        row["spearman_q_fdr"] = float(sq)

    rows.sort(
        key=lambda row: abs(float(row["pearson_r"]))
        if np.isfinite(float(row["pearson_r"]))
        else -np.inf,
        reverse=True,
    )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def plot_correlation_ranking(
    rows: list[dict[str, object]],
    *,
    title: str,
    output: Path,
    dpi: int,
) -> None:
    valid = [
        row
        for row in rows
        if np.isfinite(float(row["pearson_r"]))
    ]
    if not valid:
        return

    # Most informative first at the top.
    valid = valid[: min(20, len(valid))]
    names = [str(row["feature"]) for row in valid][::-1]
    pearson = np.asarray(
        [float(row["pearson_r"]) for row in valid][::-1]
    )

    fig, ax = plt.subplots(
        figsize=(10.0, max(5.0, 0.36 * len(valid) + 1.5)),
        constrained_layout=True,
    )
    y = np.arange(len(valid))
    ax.barh(y, pearson)
    ax.set_yticks(y, names)
    ax.axvline(0.0, linewidth=1.0, linestyle="--")
    ax.set_xlabel("Pearson r with factored correction target")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.22)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_feature_scatter(
    feature_name: str,
    values: np.ndarray,
    target_ps: np.ndarray,
    row: dict[str, object],
    *,
    title: str,
    output: Path,
    dpi: int,
) -> None:
    x = np.asarray(values, dtype=np.float64)
    y = np.asarray(target_ps, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 3:
        return

    xx = x[finite]
    yy = y[finite]
    slope = float(row["linear_slope_ps_per_feature"])
    intercept = float(row["linear_intercept_ps"])

    lo, hi = np.quantile(xx, [0.005, 0.995])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(xx)), float(np.max(xx))
    line_x = np.linspace(lo, hi, 200)
    line_y = slope * line_x + intercept

    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    ax.scatter(xx, yy, s=8, alpha=0.22)
    ax.plot(line_x, line_y, linewidth=2.0)

    ax.set_xlabel(feature_name)
    ax.set_ylabel("Factored correction target [ps]")
    ax.set_title(
        f"{title}\n"
        f"Pearson r={float(row['pearson_r']):+.3f} · "
        f"Spearman ρ={float(row['spearman_rho']):+.3f}"
    )
    ax.grid(alpha=0.2)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def analyze_dataset(
    dataset_path: Path,
    *,
    mode: str,
    start_ns: float,
    end_ns: float,
    early_roi: tuple[float, float],
    late_roi: tuple[float, float],
    output_root: Path,
    top_k: int,
    dpi: int,
    log: logging.Logger,
) -> None:
    dataset = load_prepared_dataset(dataset_path)

    input_waveforms, target_name = MODE_CONFIG[mode]
    view = prediction_dataset_view(
        dataset,
        input_waveforms=input_waveforms,
        target=target_name,
    )

    time_ps = np.asarray(view.relative_time_ps, dtype=np.float64)
    indices = _window_indices(time_ps, start_ns, end_ns)
    t_ns = time_ps[indices] / 1000.0

    pair = np.asarray(view.windows_mV[:, :, indices], dtype=np.float64)

    all_indices = np.arange(pair.shape[0], dtype=np.int64)
    target_ps = np.asarray(
        factored_correction_target_ps(view, all_indices),
        dtype=np.float64,
    )

    valid = (
        np.all(np.isfinite(pair), axis=(1, 2))
        & np.isfinite(target_ps)
    )
    if np.count_nonzero(valid) < 3:
        raise RuntimeError(
            f"{dataset_path.name}: fewer than 3 valid events"
        )

    pair = pair[valid]
    target_ps = target_ps[valid]
    event_ids = np.asarray(view.event_id)[valid]
    event_indices = np.asarray(view.event_index)[valid]

    ch1_features = per_channel_features(
        pair[:, 0, :],
        t_ns,
        early_roi=early_roi,
        late_roi=late_roi,
    )
    ch2_features = per_channel_features(
        pair[:, 1, :],
        t_ns,
        early_roi=early_roi,
        late_roi=late_roi,
    )
    pair_features = asymmetric_features(ch1_features, ch2_features)

    rows = correlation_rows(pair_features, target_ps)

    source_name = Path(
        dataset.manifest.get("source_root", dataset_path.name)
    ).name
    title = (
        f"{source_name} · {mode} · "
        f"secondary ROI [{start_ns:g},{end_ns:g}] ns"
    )

    out_dir = output_root / dataset_path.name / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    # Event-level features.
    feature_rows: list[dict[str, object]] = []
    for i in range(target_ps.size):
        row: dict[str, object] = {
            "event_id": event_ids[i],
            "event_index": int(event_indices[i]),
            "factored_correction_target_ps": float(target_ps[i]),
        }
        for feature_name, values in pair_features.items():
            row[feature_name] = float(values[i])
        feature_rows.append(row)

    write_csv(out_dir / "features.csv", feature_rows)
    write_csv(out_dir / "correlations.csv", rows)

    plot_correlation_ranking(
        rows,
        title=title,
        output=out_dir / "correlation_ranking.png",
        dpi=dpi,
    )

    scatter_dir = out_dir / "top_feature_scatter"
    for rank, row in enumerate(rows[: max(0, int(top_k))], start=1):
        feature = str(row["feature"])
        if not np.isfinite(float(row["pearson_r"])):
            continue
        plot_feature_scatter(
            feature,
            pair_features[feature],
            target_ps,
            row,
            title=title,
            output=scatter_dir
            / f"{rank:02d}_{safe_filename(feature)}.png",
            dpi=dpi,
        )

    best = rows[0] if rows else None
    if best is not None:
        log.info(
            "%s | %s | n=%d | target std=%.2f ps | best=%s | "
            "Pearson r=%+.3f | Spearman rho=%+.3f",
            dataset_path.name,
            mode,
            target_ps.size,
            float(np.std(target_ps, ddof=1)),
            best["feature"],
            float(best["pearson_r"]),
            float(best["spearman_rho"]),
        )

    # Compact metadata file, useful when comparing datasets.
    metadata = [
        {
            "dataset": dataset_path.name,
            "source_file": source_name,
            "mode": mode,
            "window_start_ns": start_ns,
            "window_end_ns": end_ns,
            "early_roi_start_ns": early_roi[0],
            "early_roi_end_ns": early_roi[1],
            "late_roi_start_ns": late_roi[0],
            "late_roi_end_ns": late_roi[1],
            "n_events": int(target_ps.size),
            "n_samples": int(indices.size),
            "target_mean_ps": float(np.mean(target_ps)),
            "target_std_ps": float(np.std(target_ps, ddof=1)),
            "best_feature": str(best["feature"]) if best else "",
            "best_pearson_r": float(best["pearson_r"]) if best else np.nan,
            "best_spearman_rho": float(best["spearman_rho"]) if best else np.nan,
        }
    ]
    write_csv(out_dir / "summary.csv", metadata)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract antisymmetric features from a fixed LED-relative waveform "
            "window and evaluate their correlation with the exact factored "
            "correction target used by the ML pipeline."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        action="append",
        default=None,
        help="Prepared dataset directory. May be repeated.",
    )
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=None,
        help="Root containing prepared dataset directories.",
    )
    parser.add_argument(
        "--mode",
        choices=sorted(MODE_CONFIG),
        default="timing_to_timing",
    )
    parser.add_argument(
        "--start-ns",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--end-ns",
        type=float,
        default=23.0,
    )

    # Attribution-inspired subregions from the previous waveform-SVR analysis.
    parser.add_argument(
        "--early-roi",
        type=float,
        nargs=2,
        metavar=("START_NS", "END_NS"),
        default=(20.2, 20.8),
    )
    parser.add_argument(
        "--late-roi",
        type=float,
        nargs=2,
        metavar=("START_NS", "END_NS"),
        default=(22.4, 22.9),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/diagnostics/secondary_feature_correlation"
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=6,
        help="Number of highest-|Pearson r| feature scatter plots.",
    )
    parser.add_argument("--dpi", type=int, default=180)

    args = parser.parse_args()

    if args.dataset is None and args.prepared_root is None:
        args.prepared_root = Path("processed_data/ml_prepared")

    early_roi = tuple(float(x) for x in args.early_roi)
    late_roi = tuple(float(x) for x in args.late_roi)

    if not (
        args.start_ns <= early_roi[0] < early_roi[1] <= args.end_ns
    ):
        raise ValueError("early ROI must lie inside the extraction window")
    if not (
        args.start_ns <= late_roi[0] < late_roi[1] <= args.end_ns
    ):
        raise ValueError("late ROI must lie inside the extraction window")

    log = make_logger()
    datasets = find_datasets(args.dataset, args.prepared_root)
    log.info("Prepared datasets found: %d", len(datasets))

    for dataset_path in datasets:
        try:
            analyze_dataset(
                dataset_path,
                mode=args.mode,
                start_ns=float(args.start_ns),
                end_ns=float(args.end_ns),
                early_roi=early_roi,
                late_roi=late_roi,
                output_root=args.output_dir,
                top_k=int(args.top_k),
                dpi=int(args.dpi),
                log=log,
            )
        except (ValueError, RuntimeError) as exc:
            log.warning("%s | skipped: %s", dataset_path.name, exc)

    log.info("Done | output=%s", args.output_dir)


if __name__ == "__main__":
    main()
