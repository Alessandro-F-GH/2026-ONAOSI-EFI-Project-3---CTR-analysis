from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .binary_io import (
    atomic_write_csv,
    discover_runs,
    file_signature,
    parse_run_info,
    read_meta,
)
from .cache import load_state, signature, stage_valid_any
from .config import stage_config
from .selection import collect_measurements, load_selection_csv, select_events
from .tabular import table_path

SUMMARY_FIELDS = [
    "AcquisitionMode",
    "Voltage",
    "E_th",
    "T_th",
    "run_count",
    "run_ids",
    "pair",
    "stage",
    "n_events",
    "mean_ns",
    "std_ns",
    "median_ns",
    "q1_ns",
    "q3_ns",
    "p1_ns",
    "p99_ns",
]


@dataclass(slots=True)
class GroupData:
    acquisition_mode: str
    voltage: int
    energy_threshold_mv: float
    timing_threshold_mv: float
    run_ids: list[str] = field(default_factory=list)
    pair_a_before_ns: list[np.ndarray] = field(default_factory=list)
    pair_a_after_ns: list[np.ndarray] = field(default_factory=list)
    pair_b_before_ns: list[np.ndarray] = field(default_factory=list)
    pair_b_after_ns: list[np.ndarray] = field(default_factory=list)


def _threshold_label(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _group_filename(group: GroupData) -> str:
    return (
        f"alignment_boxplot_{group.acquisition_mode}_V{group.voltage}_"
        f"E{_threshold_label(group.energy_threshold_mv)}_"
        f"T{_threshold_label(group.timing_threshold_mv)}.png"
    )


def _concat(parts: list[np.ndarray]) -> np.ndarray:
    valid = [np.asarray(part, dtype=float) for part in parts if np.asarray(part).size]
    return np.concatenate(valid) if valid else np.asarray([], dtype=float)


def _summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "n_events": 0,
            "mean_ns": "",
            "std_ns": "",
            "median_ns": "",
            "q1_ns": "",
            "q3_ns": "",
            "p1_ns": "",
            "p99_ns": "",
        }
    q1, median, q3, p1, p99 = np.percentile(values, [25.0, 50.0, 75.0, 1.0, 99.0])
    return {
        "n_events": int(values.size),
        "mean_ns": float(np.mean(values)),
        "std_ns": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "median_ns": float(median),
        "q1_ns": float(q1),
        "q3_ns": float(q3),
        "p1_ns": float(p1),
        "p99_ns": float(p99),
    }


def _load_or_select(
    run_dir: Path,
    preprocessed_path: Path,
    cfg: dict,
    recompute_selection: bool,
):
    selection_path = table_path(run_dir / "csv", "selection", cfg)
    state = load_state(run_dir / "state.json")
    selection_signature = signature(
        {"preprocessed": file_signature(preprocessed_path)},
        stage_config(cfg, "selection"),
    )
    cache_valid = stage_valid_any(
        state,
        "selection",
        [selection_signature],
        [selection_path],
    )
    if cache_valid and not recompute_selection:
        measurements, duration_mask, alignment_mask = load_selection_csv(selection_path)
        return measurements, duration_mask, duration_mask & alignment_mask, "cached selection"

    measurements, _ = collect_measurements(preprocessed_path, cfg)
    selection = select_events(measurements, cfg)
    return measurements, selection.duration_mask, selection.final_mask, "selection recomputed"


def _plot_group(
    path: Path,
    group: GroupData,
    dpi: int,
) -> None:
    pair_a_before = _concat(group.pair_a_before_ns)
    pair_a_after = _concat(group.pair_a_after_ns)
    pair_b_before = _concat(group.pair_b_before_ns)
    pair_b_after = _concat(group.pair_b_after_ns)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.8), sharey=True)
    definitions = (
        (axes[0], pair_a_before, pair_a_after, "ch3 − ch1"),
        (axes[1], pair_b_before, pair_b_after, "ch7 − ch5"),
    )
    rng = np.random.default_rng(12345)

    for axis, before, after, pair_label in definitions:
        datasets = [before, after]
        axis.boxplot(
            datasets,
            tick_labels=[f"Before\nN={before.size}", f"After\nN={after.size}"],
            showfliers=False,
            whis=1.5,
            showmeans=True,
            meanline=True,
            boxprops={"linewidth": 1.4},
            whiskerprops={"linewidth": 1.4},
            capprops={"linewidth": 1.4},
            medianprops={"linewidth": 1.8},
            meanprops={"linestyle": "--", "linewidth": 1.4},
        )

        for position, values in enumerate(datasets, start=1):
            values = np.asarray(values, dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue

            q1, q3 = np.percentile(values, [25.0, 75.0])
            iqr = q3 - q1
            lower_limit = q1 - 1.5 * iqr
            upper_limit = q3 + 1.5 * iqr
            outliers = values[(values < lower_limit) | (values > upper_limit)]
            if outliers.size == 0:
                continue

            jitter = rng.uniform(-0.07, 0.07, size=outliers.size)
            axis.scatter(
                position + jitter,
                outliers,
                s=18,
                marker="o",
                facecolors="none",
                edgecolors="black",
                linewidths=0.8,
                alpha=0.9,
                zorder=10,
            )

        axis.set_title(pair_label)
        axis.set_xlabel("Alignment filter")
        axis.grid(axis="y", alpha=0.22)

    axes[0].set_ylabel(r"$\Delta t = t_{time}-t_{energy}$ [ns]")
    ordered_runs = sorted(group.run_ids, key=lambda item: int(item.removeprefix("Run")))
    if len(ordered_runs) <= 4:
        run_text = ", ".join(ordered_runs)
    else:
        run_text = f"{len(ordered_runs)} runs ({ordered_runs[0]}–{ordered_runs[-1]})"

    fig.suptitle(
        f"{group.acquisition_mode} — V={group.voltage}, "
        f"E_th={group.energy_threshold_mv:g} mV, "
        f"T_th={group.timing_threshold_mv:g} mV\n"
        f"Runs: {run_text}"
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def run_alignment_study(
    cfg: dict,
    *,
    acquisition_mode: str = "STREAMING",
    output_dir: str | Path | None = None,
    recompute_selection: bool = False,
    overwrite: bool = False,
) -> None:
    requested_mode = acquisition_mode.strip().upper()
    if requested_mode not in {"STREAMING", "TRG_MATCHING", "ALL"}:
        raise ValueError("acquisition_mode must be STREAMING, TRG_MATCHING, or ALL")

    input_dir = Path(cfg["paths"]["input_dir"])
    pipeline_output = Path(cfg["paths"]["output_dir"])
    study_dir = Path(output_dir) if output_dir is not None else pipeline_output / "selection_alignment_study"
    plots_dir = study_dir / "plots"
    summary_path = study_dir / "alignment_distribution_summary.csv"

    runs = discover_runs(
        input_dir,
        cfg["files"]["data_pattern"],
        bool(cfg["files"]["recursive"]),
    )
    groups: dict[tuple[str, int, float, float], GroupData] = {}

    for run in runs:
        info = parse_run_info(run.info_path, cfg["thresholds"]["consistency"])
        if requested_mode != "ALL" and info.acquisition_mode != requested_mode:
            continue

        run_dir = pipeline_output / "analysis" / run.run_id
        preprocessed_path = run_dir / "preprocessed" / f"{run.run_id}_list.dat"
        if not preprocessed_path.exists():
            print(
                f"[{run.run_id}][alignment-study] SKIPPED — preprocessed binary not found",
                flush=True,
            )
            continue

        meta = read_meta(preprocessed_path, info.acquisition_mode)
        measurements, duration_mask, final_mask, source = _load_or_select(
            run_dir,
            preprocessed_path,
            cfg,
            recompute_selection,
        )
        if measurements.size == 0 or not np.any(duration_mask):
            print(
                f"[{run.run_id}][alignment-study] SKIPPED — no events pass the energy AND selection",
                flush=True,
            )
            continue

        ns_per_lsb = meta.toa_lsb_ps / 1000.0
        key = (
            info.acquisition_mode,
            run.voltage,
            info.energy_threshold_mv,
            info.timing_threshold_mv,
        )
        group = groups.setdefault(
            key,
            GroupData(
                acquisition_mode=info.acquisition_mode,
                voltage=run.voltage,
                energy_threshold_mv=info.energy_threshold_mv,
                timing_threshold_mv=info.timing_threshold_mv,
            ),
        )
        group.run_ids.append(run.run_id)
        group.pair_a_before_ns.append(measurements.alignment_a_lsb[duration_mask].astype(float) * ns_per_lsb)
        group.pair_a_after_ns.append(measurements.alignment_a_lsb[final_mask].astype(float) * ns_per_lsb)
        group.pair_b_before_ns.append(measurements.alignment_b_lsb[duration_mask].astype(float) * ns_per_lsb)
        group.pair_b_after_ns.append(measurements.alignment_b_lsb[final_mask].astype(float) * ns_per_lsb)
        print(
            f"[{run.run_id}][alignment-study] LOADED — {source}; "
            f"energy-selected={int(np.count_nonzero(duration_mask))}, "
            f"after-alignment={int(np.count_nonzero(final_mask))}",
            flush=True,
        )

    if not groups:
        raise RuntimeError("No eligible runs with existing preprocessed data were found")

    rows: list[dict[str, Any]] = []
    for group in sorted(
        groups.values(),
        key=lambda item: (
            item.acquisition_mode,
            item.voltage,
            item.energy_threshold_mv,
            item.timing_threshold_mv,
        ),
    ):
        plot_path = plots_dir / _group_filename(group)
        if overwrite or not plot_path.exists():
            _plot_group(
                plot_path,
                group,
                int(cfg["plots"]["dpi"]),
            )
            print(f"[alignment-study] COMPLETED — {plot_path}", flush=True)
        else:
            print(f"[alignment-study] SKIPPED — existing plot {plot_path.name}", flush=True)

        run_ids = ";".join(sorted(group.run_ids, key=lambda item: int(item.removeprefix("Run"))))
        definitions = (
            ("ch3-ch1", "before", _concat(group.pair_a_before_ns)),
            ("ch3-ch1", "after", _concat(group.pair_a_after_ns)),
            ("ch7-ch5", "before", _concat(group.pair_b_before_ns)),
            ("ch7-ch5", "after", _concat(group.pair_b_after_ns)),
        )
        for pair, stage, values in definitions:
            row = {
                "AcquisitionMode": group.acquisition_mode,
                "Voltage": group.voltage,
                "E_th": group.energy_threshold_mv,
                "T_th": group.timing_threshold_mv,
                "run_count": len(group.run_ids),
                "run_ids": run_ids,
                "pair": pair,
                "stage": stage,
            }
            row.update(_summary(values))
            rows.append(row)

    atomic_write_csv(summary_path, SUMMARY_FIELDS, rows)
    print(f"[alignment-study] SUMMARY — {summary_path}", flush=True)
