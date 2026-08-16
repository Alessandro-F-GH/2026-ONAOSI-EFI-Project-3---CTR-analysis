from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from ml_pipeline.dataset import PreparedDataset, load_prepared_dataset

ENERGY = "energy_to_energy"
TIMING = "timing_to_timing"


def make_logger() -> logging.Logger:
    log = logging.getLogger("matched_vs_random_fast")
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


def dataset_dirs(explicit: list[Path] | None, prepared_root: Path | None) -> list[Path]:
    found: list[Path] = []

    if explicit:
        found.extend(Path(path).resolve() for path in explicit)

    if prepared_root is not None:
        root = Path(prepared_root).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Prepared root does not exist: {root}")
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
        raise FileNotFoundError("No prepared datasets found")
    return unique


def mode_data(
    dataset: PreparedDataset,
    mode: str,
    energy_variant: str,
    timing_variant: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    if mode == ENERGY:
        variant = energy_variant
        if variant == "auto":
            variant = (
                "denoised"
                if dataset.denoised_windows_mV is not None
                else "raw"
            )

        if variant == "denoised":
            if dataset.denoised_windows_mV is None:
                raise ValueError("Denoised energy waveforms unavailable")
            waveforms = dataset.denoised_windows_mV
        else:
            waveforms = dataset.windows_mV

        return (
            waveforms,
            np.asarray(dataset.relative_time_ps, dtype=np.float64),
            variant,
        )

    if mode == TIMING:
        if (
            dataset.timing_windows_mV is None
            or dataset.timing_relative_time_ps is None
        ):
            raise ValueError("Timing waveforms unavailable")

        variant = timing_variant if timing_variant != "auto" else "raw"

        if variant == "denoised":
            if dataset.denoised_timing_windows_mV is None:
                raise ValueError("Denoised timing waveforms unavailable")
            waveforms = dataset.denoised_timing_windows_mV
        else:
            waveforms = dataset.timing_windows_mV

        return (
            waveforms,
            np.asarray(dataset.timing_relative_time_ps, dtype=np.float64),
            variant,
        )

    raise ValueError(mode)


def crop_and_subsample(
    waveforms: np.ndarray,
    time_ps: np.ndarray,
    start_ns: float | None,
    end_ns: float | None,
    factor: int,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.ones(time_ps.shape[0], dtype=bool)

    if start_ns is not None:
        mask &= time_ps >= 1000.0 * float(start_ns)
    if end_ns is not None:
        mask &= time_ps <= 1000.0 * float(end_ns)

    indices = np.flatnonzero(mask)
    if indices.size < 3:
        raise ValueError("Requested window contains fewer than 3 samples")

    factor = max(1, int(factor))
    indices = indices[::factor]

    if indices.size < 3:
        raise ValueError(
            f"Subsampling factor {factor} leaves fewer than 3 samples"
        )

    return waveforms[:, :, indices], time_ps[indices]


def derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    if n < 2:
        raise ValueError("Need at least two events")

    base = np.arange(n, dtype=np.int64)

    # A random cyclic shift is already a valid derangement and much cheaper
    # than repeatedly drawing full permutations until no fixed points remain.
    shift = int(rng.integers(1, n))
    return np.roll(base, shift)


def normalize_pearson(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise centered L2 normalization.

    Pearson(a_i,b_j) then becomes one dot product:
        normalized_a_i @ normalized_b_j
    """
    centered = a - np.mean(a, axis=1, keepdims=True)
    norm = np.linalg.norm(centered, axis=1)
    valid = np.isfinite(norm) & (norm > 0.0)

    out = np.zeros_like(centered, dtype=np.float32)
    out[valid] = (
        centered[valid] / norm[valid, None]
    ).astype(np.float32, copy=False)
    return out, valid


def normalize_cosine(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norm = np.linalg.norm(a, axis=1)
    valid = np.isfinite(norm) & (norm > 0.0)

    out = np.zeros_like(a, dtype=np.float32)
    out[valid] = (
        a[valid] / norm[valid, None]
    ).astype(np.float32, copy=False)
    return out, valid


def prepare_metric(
    ch1: np.ndarray,
    ch2: np.ndarray,
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    if metric == "pearson":
        a, va = normalize_pearson(ch1)
        b, vb = normalize_pearson(ch2)
    elif metric == "cosine":
        a, va = normalize_cosine(ch1)
        b, vb = normalize_cosine(ch2)
    else:
        raise ValueError(metric)

    valid = va & vb
    if np.count_nonzero(valid) < 3:
        raise RuntimeError(f"Too few valid events for {metric}")

    return (
        np.ascontiguousarray(a[valid], dtype=np.float32),
        np.ascontiguousarray(b[valid], dtype=np.float32),
    )


def paired_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Similarity of row-matched normalized waveforms."""
    return np.einsum("ij,ij->i", a, b, optimize=True)


def mean_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Mean similarity without materializing an event-wise vector."""
    total = np.einsum("ij,ij->", a, b, optimize=True)
    return float(total / a.shape[0])


def permutation_test(
    a: np.ndarray,
    b: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    matched = paired_similarity(a, b)
    matched_mean = float(np.mean(matched))

    null = np.empty(int(n_permutations), dtype=np.float64)

    for k in range(int(n_permutations)):
        perm = derangement(a.shape[0], rng)
        null[k] = mean_similarity(a, b[perm])

    random_mean = float(np.mean(null))
    gain = matched_mean - random_mean
    p = (
        1.0 + float(np.count_nonzero(null >= matched_mean))
    ) / (float(n_permutations) + 1.0)

    return matched, null, matched_mean, gain, p


def random_event_distribution(
    a: np.ndarray,
    b: np.ndarray,
    draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    values: list[np.ndarray] = []

    for _ in range(max(1, int(draws))):
        perm = derangement(a.shape[0], rng)
        values.append(paired_similarity(a, b[perm]))

    return np.concatenate(values)


def plot_distribution(
    matched: np.ndarray,
    random_values: np.ndarray,
    *,
    metric: str,
    title: str,
    path: Path,
    dpi: int,
) -> None:
    matched = matched[np.isfinite(matched)]
    random_values = random_values[np.isfinite(random_values)]

    combined = np.concatenate([matched, random_values])
    lo, hi = np.quantile(combined, [0.005, 0.995])
    if hi <= lo:
        lo, hi = float(np.min(combined)), float(np.max(combined))

    bins = np.linspace(lo, hi, 70)

    fig, ax = plt.subplots(figsize=(9.0, 5.2), constrained_layout=True)
    ax.hist(
        random_values,
        bins=bins,
        density=True,
        alpha=0.45,
        label=f"random · mean={np.mean(random_values):.4f}",
    )
    ax.hist(
        matched,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2.0,
        label=f"matched · mean={np.mean(matched):.4f}",
    )

    ax.set_xlabel(metric)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_null(
    null: np.ndarray,
    matched_mean: float,
    gain: float,
    p: float,
    *,
    metric: str,
    title: str,
    path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)

    ax.hist(null, bins=45, density=True, alpha=0.65)
    ax.axvline(
        matched_mean,
        linestyle="--",
        linewidth=2.0,
        label=f"matched mean={matched_mean:.5f}",
    )

    ax.set_xlabel(f"Mean {metric} after random pairing")
    ax.set_ylabel("Permutation density")
    ax.set_title(
        f"{title}\ngain={gain:.5f} · one-sided p={p:.4g}"
    )
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fast matched-vs-random waveform-pair similarity test. "
            "Waveforms are optionally temporally subsampled, normalized once, "
            "then permutation means are evaluated with vectorized dot products."
        )
    )

    parser.add_argument("--dataset", action="append", type=Path)
    parser.add_argument("--prepared-root", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/diagnostics/matched_vs_random_similarity"
        ),
    )

    parser.add_argument(
        "--mode",
        choices=["both", ENERGY, TIMING],
        default="both",
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

    parser.add_argument(
        "--metric",
        action="append",
        choices=["pearson", "cosine"],
        help="Repeat to request both. Default: pearson.",
    )

    parser.add_argument("--start-ns", type=float)
    parser.add_argument("--end-ns", type=float)

    parser.add_argument(
        "--subsampling-factor",
        type=int,
        default=8,
        help="Temporal waveform subsampling before similarity calculation.",
    )

    parser.add_argument(
        "--permutations",
        type=int,
        default=100,
        help="Permutation-test random pairings. Default reduced to 100.",
    )
    parser.add_argument(
        "--random-draws",
        type=int,
        default=1,
        help="Random pairings pooled for event-level histogram.",
    )

    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--dpi", type=int, default=180)

    args = parser.parse_args()

    if args.dataset is None and args.prepared_root is None:
        args.prepared_root = Path("processed_data/ml_prepared")

    modes = [ENERGY, TIMING] if args.mode == "both" else [args.mode]
    metrics = args.metric or ["pearson"]

    log = make_logger()
    rows: list[dict[str, object]] = []

    paths = dataset_dirs(args.dataset, args.prepared_root)
    log.info("Prepared datasets found: %d", len(paths))

    for path in paths:
        dataset = load_prepared_dataset(path)
        source = Path(
            dataset.manifest.get("source_root", path.name)
        ).name

        for mode in modes:
            try:
                waveforms, time_ps, variant = mode_data(
                    dataset,
                    mode,
                    args.energy_variant,
                    args.timing_variant,
                )

                waveforms, time_ps = crop_and_subsample(
                    waveforms,
                    time_ps,
                    args.start_ns,
                    args.end_ns,
                    args.subsampling_factor,
                )

                x = np.asarray(waveforms, dtype=np.float32)

                finite = np.all(np.isfinite(x), axis=(1, 2))
                x = x[finite]

                if x.shape[0] < 3:
                    raise RuntimeError("Need at least 3 finite waveform pairs")

                ch1 = np.ascontiguousarray(x[:, 0, :], dtype=np.float32)
                ch2 = np.ascontiguousarray(x[:, 1, :], dtype=np.float32)

                mode_label = (
                    "energy → energy"
                    if mode == ENERGY
                    else "timing → timing"
                )
                title = f"{source} · {mode_label} · {variant}"

                log.info(
                    "%s | %s | n=%d | samples=%d | subsampling=%d | "
                    "window=[%.3f, %.3f] ns",
                    path.name,
                    mode,
                    x.shape[0],
                    x.shape[2],
                    max(1, args.subsampling_factor),
                    time_ps[0] / 1000.0,
                    time_ps[-1] / 1000.0,
                )

                for metric_index, metric in enumerate(metrics):
                    a, b = prepare_metric(ch1, ch2, metric)

                    rng = np.random.default_rng(
                        int(args.seed) + 1009 * metric_index
                    )

                    matched, null, matched_mean, gain, p = permutation_test(
                        a,
                        b,
                        max(1, int(args.permutations)),
                        rng,
                    )

                    random_values = random_event_distribution(
                        a,
                        b,
                        max(1, int(args.random_draws)),
                        rng,
                    )

                    random_mean = float(np.mean(null))
                    random_std = (
                        float(np.std(null, ddof=1))
                        if null.size > 1
                        else np.nan
                    )

                    log.info(
                        "%s | matched %.5f | random %.5f ± %.5f | "
                        "gain %.5f | p=%.4g",
                        metric,
                        matched_mean,
                        random_mean,
                        random_std,
                        gain,
                        p,
                    )

                    root = args.output_dir / mode / metric

                    plot_distribution(
                        matched,
                        random_values,
                        metric=metric,
                        title=title,
                        path=root / f"{path.name}__distribution.png",
                        dpi=args.dpi,
                    )

                    plot_null(
                        null,
                        matched_mean,
                        gain,
                        p,
                        metric=metric,
                        title=title,
                        path=root / f"{path.name}__permutation_null.png",
                        dpi=args.dpi,
                    )

                    rows.append(
                        {
                            "dataset": path.name,
                            "source_file": source,
                            "mode": mode,
                            "variant": variant,
                            "window_start_ns": float(time_ps[0] / 1000.0),
                            "window_end_ns": float(time_ps[-1] / 1000.0),
                            "subsampling_factor": int(
                                max(1, args.subsampling_factor)
                            ),
                            "n_events": int(a.shape[0]),
                            "n_samples": int(a.shape[1]),
                            "metric": metric,
                            "matched_mean": matched_mean,
                            "matched_eventwise_std": float(
                                np.std(matched, ddof=1)
                            ),
                            "random_mean": random_mean,
                            "random_mean_std_permutation": random_std,
                            "similarity_gain": gain,
                            "p_one_sided": p,
                            "n_permutations": int(
                                max(1, args.permutations)
                            ),
                        }
                    )

            except (ValueError, RuntimeError) as exc:
                log.warning("%s | %s | skipped: %s", path.name, mode, exc)

    summary = args.output_dir / "matched_vs_random_similarity.csv"
    write_csv(summary, rows)
    log.info("Done | summary=%s", summary)


if __name__ == "__main__":
    main()
