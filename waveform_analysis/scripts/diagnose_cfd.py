from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from ml_pipeline.dataset import load_prepared_dataset


def _last_crossing_ps(time_ps: np.ndarray, signal: np.ndarray, fraction: float) -> float:
    y = np.asarray(signal, dtype=np.float64)
    t = np.asarray(time_ps, dtype=np.float64)
    peak = int(np.argmax(y))
    if peak <= 0:
        return np.nan
    amplitude = float(y[peak])
    threshold = float(fraction) * amplitude
    if not np.isfinite(threshold) or threshold <= 0.0:
        return np.nan
    y0, y1 = y[:peak], y[1 : peak + 1]
    valid = np.isfinite(y0) & np.isfinite(y1) & (y0 < threshold) & (y1 >= threshold)
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return np.nan
    j = int(idx[-1])
    if y1[j] == y0[j]:
        return np.nan
    alpha = (threshold - y0[j]) / (y1[j] - y0[j])
    return float(t[j] + alpha * (t[j + 1] - t[j]))


def _metrics(values: np.ndarray, true_tof_ps: float) -> dict[str, float]:
    residual = np.asarray(values, dtype=np.float64) - float(true_tof_ps)
    bias = float(np.mean(residual))
    variance = float(np.mean((residual - bias) ** 2))
    return {
        "bias_ps": bias,
        "std_ps": float(np.sqrt(variance)),
        "rmse_ps": float(np.sqrt(np.mean(residual**2))),
        "variance_ps2": variance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose CFD fraction and threshold behavior.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=[0.03, 0.045, 0.06, 0.1, 0.2, 0.3, 0.5],
    )
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results/cfd_diagnostic.csv"))
    args = parser.parse_args()

    dataset = load_prepared_dataset(args.dataset.resolve())
    indices = np.asarray(dataset.evaluation, dtype=np.int64)
    if args.max_events > 0:
        indices = indices[: args.max_events]
    if indices.size == 0:
        raise ValueError("No evaluation events in dataset")

    led = (
        np.asarray(dataset.led_time_fs[indices, 0], dtype=np.float64)
        - np.asarray(dataset.led_time_fs[indices, 1], dtype=np.float64)
    ) / 1000.0
    rows: list[dict[str, float | str | int]] = []
    led_metrics = _metrics(led, float(dataset.true_tof_ps))
    rows.append({"method": "stored_led", "fraction": "", "valid_events": int(led.size), **led_metrics})

    windows = np.asarray(dataset.windows_mV[indices], dtype=np.float64)
    time_ps = np.asarray(dataset.relative_time_ps, dtype=np.float64)
    amplitudes = np.max(windows, axis=2)

    for fraction in args.fractions:
        t0 = np.asarray([_last_crossing_ps(time_ps, w, fraction) for w in windows[:, 0]], dtype=np.float64)
        t1 = np.asarray([_last_crossing_ps(time_ps, w, fraction) for w in windows[:, 1]], dtype=np.float64)
        valid = np.isfinite(t0) & np.isfinite(t1)
        if not np.any(valid):
            continue
        delta = t0[valid] - t1[valid]
        threshold_values = amplitudes[valid] * float(fraction)
        rows.append(
            {
                "method": "recomputed_cfd",
                "fraction": float(fraction),
                "valid_events": int(np.count_nonzero(valid)),
                "threshold_median_mV": float(np.median(threshold_values)),
                "threshold_p05_mV": float(np.quantile(threshold_values, 0.05)),
                "threshold_p95_mV": float(np.quantile(threshold_values, 0.95)),
                **_metrics(delta, float(dataset.true_tof_ps)),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {args.output}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
