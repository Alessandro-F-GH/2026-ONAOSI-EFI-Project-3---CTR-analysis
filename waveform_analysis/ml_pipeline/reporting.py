from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .metrics import ctr_bootstrap_uncertainty, residual_metrics


def short_model_label(name: str) -> str:
    mapping = {
        "led": "LED",
        "cfd": "CFD",
        "linear_svr": "SVR",
        "constructive_mlp_encoder": "MLP",
        "constructive_mlp": "MLP",
        "cnn_regressor": "CNN",
        "cnn": "CNN",
        "multithreshold_svr": "MT-SVR",
    }
    key = str(name).strip().lower()
    return mapping.get(
        key,
        str(name).replace("_regressor", "").replace("_", " ").title(),
    )


def short_mode_label(mode: str) -> str:
    return (
        str(mode)
        .replace("energy_to_energy", "energy → energy")
        .replace("energy_to_timing", "energy → timing")
        .replace("timing_to_timing", "timing → timing")
    )


def format_ctr(ctr_ps: float, uncertainty_ps: float | None = None) -> str:
    if not np.isfinite(float(ctr_ps)):
        return "n/a"
    center = int(round(float(ctr_ps)))
    if uncertainty_ps is None or not np.isfinite(float(uncertainty_ps)):
        return f"{center} ps"
    return f"{center} ± {max(0, int(round(float(uncertainty_ps))))} ps"


def _save(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def _robust_bounds(groups: dict[str, np.ndarray]) -> tuple[float, float]:
    finite_groups = [
        np.asarray(values, dtype=float)[np.isfinite(values)]
        for values in groups.values()
        if np.asarray(values).size
    ]
    if not finite_groups:
        return -100.0, 100.0
    all_values = np.concatenate(finite_groups)
    if all_values.size < 2:
        return -100.0, 100.0
    median = float(np.median(all_values))
    mad = float(np.median(np.abs(all_values - median)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.std(all_values, ddof=1)) or 1.0
    return median - 7.0 * scale, median + 7.0 * scale


def eligible(methods: dict[str, np.ndarray], ratio_limit: float) -> set[str]:
    if "led" not in methods:
        return set(methods)
    led_ctr = residual_metrics(methods["led"])["ctr_ps"]
    if not np.isfinite(led_ctr) or led_ctr <= 0.0:
        return set(methods)
    output = {"led"}
    for name, values in methods.items():
        if name == "led":
            continue
        ctr = residual_metrics(values)["ctr_ps"]
        if np.isfinite(ctr) and ctr <= float(ratio_limit) * led_ctr:
            output.add(name)
    return output


def plot_result_distribution(
    path: Path,
    *,
    mode: str,
    methods: dict[str, np.ndarray],
    dpi: int,
    ratio_limit: float,
    bootstrap_samples: int,
    seed: int,
    split_label: str,
) -> dict[str, float]:
    """Plot train/development or blind residual distributions identically."""
    keep = eligible(methods, ratio_limit)
    visible = {
        key: np.asarray(value, dtype=float)
        for key, value in methods.items()
        if key in keep and np.sum(np.isfinite(value)) >= 2
    }
    if not visible:
        return {}

    low, high = _robust_bounds(visible)
    bins = np.linspace(low, high, 81)
    fig, axis = plt.subplots(figsize=(8.8, 5.0))
    uncertainties: dict[str, float] = {}

    for index, (name, values) in enumerate(visible.items()):
        values = values[np.isfinite(values)]
        metrics = residual_metrics(values)
        uncertainty = ctr_bootstrap_uncertainty(
            values,
            bootstrap_samples,
            seed + 137 * index,
        )
        uncertainties[name] = uncertainty
        axis.hist(
            values[(values >= low) & (values <= high)],
            bins=bins,
            histtype="step",
            density=True,
            linewidth=1.4,
            label=(
                f"{short_model_label(name)} — "
                f"CTR {format_ctr(metrics['ctr_ps'], uncertainty)}"
            ),
        )

    axis.set_title(f"{short_mode_label(mode)} · {split_label}")
    axis.set_xlabel("Residual timing error [ps]")
    axis.set_ylabel("Density")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, fontsize=9, loc="upper right")
    _save(fig, path, dpi)
    return uncertainties


def plot_blind_distribution(
    path: Path,
    *,
    mode: str,
    methods: dict[str, np.ndarray],
    dpi: int,
    ratio_limit: float,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, float]:
    """Compatibility wrapper for callers outside study.py."""
    return plot_result_distribution(
        path,
        mode=mode,
        methods=methods,
        dpi=dpi,
        ratio_limit=ratio_limit,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        split_label="Blind",
    )


def _series_rows(
    rows: list[dict[str, Any]],
    stage: str,
    selected_only: bool = True,
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("stage_name") == stage
        and (not selected_only or int(row.get("selected", 0)) == 1)
    ]


def plot_ctr_vs_voltage(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    stage: str,
    dpi: int,
    ratio_limit: float,
    title: str | None = None,
) -> None:
    selected = _series_rows(rows, stage, True)
    if not selected:
        return
    modes = sorted(set(str(row["mode"]) for row in selected))
    fig, axes = plt.subplots(
        len(modes),
        1,
        figsize=(9.0, 4.2 * len(modes)),
        squeeze=False,
    )
    for axis, mode in zip(axes[:, 0], modes):
        subset = [row for row in selected if row["mode"] == mode]
        for model in sorted(
            set(row["model"] for row in subset),
            key=lambda value: (value not in ("led", "cfd"), value),
        ):
            points = sorted(
                [
                    row
                    for row in subset
                    if row["model"] == model
                    and int(row.get("plot_included", 1)) == 1
                ],
                key=lambda row: float(row["voltage_V"]),
            )
            if not points:
                continue
            x_values = [float(row["voltage_V"]) for row in points]
            y_values = [float(row["ctr_ps"]) for row in points]
            errors = [
                float(
                    row.get(
                        "ctr_uncertainty_ps",
                        row.get("ctr_fold_std_ps", np.nan),
                    )
                )
                for row in points
            ]
            if np.all(np.isfinite(errors)):
                axis.errorbar(
                    x_values,
                    y_values,
                    yerr=errors,
                    marker="o",
                    linewidth=1.2,
                    capsize=2,
                    label=short_model_label(model),
                )
            else:
                axis.plot(
                    x_values,
                    y_values,
                    marker="o",
                    linewidth=1.2,
                    label=short_model_label(model),
                )
        axis.set_title(short_mode_label(mode))
        axis.set_xlabel("Bias voltage [V]")
        axis.set_ylabel("CTR [ps]")
        axis.grid(alpha=0.22)
        axis.legend(
            frameon=False,
            ncol=min(4, max(1, len(set(row["model"] for row in subset)))),
            fontsize=8,
        )
    if title:
        fig.suptitle(title, y=1.01)
    _save(fig, path, dpi)


def plot_window_scan_bars(
    output_dir: Path,
    *,
    candidate_rows: list[dict[str, Any]],
    report_rows: list[dict[str, Any]],
    codebooks: dict[str, dict[str, int]],
    windows: list[dict[str, Any]],
    dpi: int,
    ratio_limit: float,
) -> None:
    """Per-file validation plot for physical window-ablation studies."""
    if not candidate_rows or not windows:
        return
    reverse_file = {int(value): str(key) for key, value in codebooks["file"].items()}
    reverse_mode = {int(value): str(key) for key, value in codebooks["mode"].items()}
    reverse_model = {int(value): str(key) for key, value in codebooks["model"].items()}
    window_id_to_index = {
        str(window["id"]): int(codebooks["window"][str(window["id"])])
        for window in windows
    }

    def window_label(window: dict[str, Any]) -> str:
        start = float(window.get("start_ns", -float(window["before_ns"])))
        end = float(window.get("end_ns", float(window["after_ns"])))
        return f"[{start:g},{end:+g}]"

    output_dir.mkdir(parents=True, exist_ok=True)
    for file_id, file_name in sorted(reverse_file.items()):
        file_candidates = [
            row
            for row in candidate_rows
            if int(row.get("stage", -1)) == 0
            and int(row.get("file_id", -1)) == file_id
        ]
        if not file_candidates:
            continue
        mode_ids = sorted(
            {
                int(row["mode_id"])
                for row in file_candidates
                if int(row.get("mode_id", -1)) in reverse_mode
            }
        )
        if not mode_ids:
            continue
        fig, axes = plt.subplots(
            len(mode_ids),
            1,
            figsize=(10.5, 4.4 * len(mode_ids)),
            squeeze=False,
            constrained_layout=True,
        )
        for axis, mode_id in zip(axes[:, 0], mode_ids):
            mode = reverse_mode[mode_id]
            validation_rows = [
                row
                for row in report_rows
                if int(row.get("file_id", -1)) == file_id
                and str(row.get("mode", "")) == mode
                and str(row.get("stage_name", "")) == "validation"
            ]
            led_row = next(
                (row for row in validation_rows if str(row.get("model")) == "led"),
                None,
            )
            cfd_row = next(
                (row for row in validation_rows if str(row.get("model")) == "cfd"),
                None,
            )
            led_ctr = float(led_row["ctr_ps"]) if led_row is not None else float("nan")
            cfd_ctr = float(cfd_row["ctr_ps"]) if cfd_row is not None else float("nan")
            model_ids = sorted(
                {
                    int(row["model_id"])
                    for row in file_candidates
                    if int(row.get("mode_id", -1)) == mode_id
                    and reverse_model.get(int(row["model_id"]), "")
                    not in {"led", "cfd"}
                }
            )
            if not model_ids:
                continue
            x_values = np.arange(len(windows), dtype=np.float64)
            width = 0.78 / max(1, len(model_ids))
            for model_position, model_id in enumerate(model_ids):
                model = reverse_model[model_id]
                values: list[float] = []
                excluded: list[bool] = []
                for window in windows:
                    window_index = window_id_to_index[str(window["id"])]
                    matches = [
                        row
                        for row in file_candidates
                        if int(row.get("mode_id", -1)) == mode_id
                        and int(row.get("model_id", -1)) == model_id
                        and int(row.get("window_id", -999)) == window_index
                        and np.isfinite(float(row.get("ctr_ps", np.nan)))
                    ]
                    if not matches:
                        values.append(float("nan"))
                        excluded.append(False)
                        continue
                    best = min(matches, key=lambda row: float(row["ctr_ps"]))
                    ctr = float(best["ctr_ps"])
                    bad = (
                        np.isfinite(led_ctr)
                        and led_ctr > 0.0
                        and ctr > float(ratio_limit) * led_ctr
                    )
                    values.append(float("nan") if bad else ctr)
                    excluded.append(bad)
                offsets = (
                    x_values - 0.39 + width / 2.0 + model_position * width
                    if len(model_ids) > 1
                    else x_values
                )
                bars = axis.bar(
                    offsets,
                    values,
                    width=(width if len(model_ids) > 1 else 0.62),
                    label=short_model_label(model),
                )
                for bar, ctr in zip(bars, values):
                    if np.isfinite(ctr):
                        axis.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            ctr,
                            f"{ctr:.0f}",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                        )
                for x_position, is_excluded in zip(offsets, excluded):
                    if is_excluded:
                        axis.text(
                            x_position,
                            0.02,
                            f">{float(ratio_limit):g}× LED",
                            transform=axis.get_xaxis_transform(),
                            ha="center",
                            va="bottom",
                            rotation=90,
                            fontsize=7,
                        )
            if np.isfinite(led_ctr):
                axis.axhline(
                    led_ctr,
                    linestyle="--",
                    linewidth=1.2,
                    label=f"LED {led_ctr:.0f} ps",
                )
            if np.isfinite(cfd_ctr):
                axis.axhline(
                    cfd_ctr,
                    linestyle=":",
                    linewidth=1.2,
                    label=f"CFD {cfd_ctr:.0f} ps",
                )
            axis.set_xticks(x_values, [window_label(window) for window in windows])
            axis.set_xlabel("Disjoint LED-relative window [ns]")
            axis.set_ylabel("Validation s-CTR [ps]")
            axis.set_title(short_mode_label(mode))
            axis.grid(axis="y", alpha=0.22)
            axis.legend(
                frameon=False,
                fontsize=8,
                ncol=min(4, len(model_ids) + 2),
            )
        fig.suptitle(
            f"{Path(file_name).stem} · disjoint-window validation scan",
            fontsize=13,
        )
        _save(fig, output_dir / f"{Path(file_name).stem}.png", dpi)


def plot_final_bars(path: Path, *, rows: list[dict[str, Any]], dpi: int) -> None:
    blind = [
        row
        for row in _series_rows(rows, "blind", True)
        if int(row.get("plot_included", 1)) == 1
    ]
    if not blind:
        return
    modes = sorted(set(row["mode"] for row in blind))
    fig, axes = plt.subplots(
        len(modes),
        1,
        figsize=(10.5, 4.5 * len(modes)),
        squeeze=False,
    )
    for axis, mode in zip(axes[:, 0], modes):
        subset = [row for row in blind if row["mode"] == mode]
        voltages = sorted(set(float(row["voltage_V"]) for row in subset))
        models = sorted(
            set(row["model"] for row in subset),
            key=lambda value: (value not in ("led", "cfd"), value),
        )
        width = 0.8 / max(1, len(models))
        center = np.arange(len(voltages))
        for index, model in enumerate(models):
            values: list[float] = []
            labels: list[tuple[int, str]] = []
            for voltage in voltages:
                match = next(
                    (
                        row
                        for row in subset
                        if float(row["voltage_V"]) == voltage
                        and row["model"] == model
                    ),
                    None,
                )
                values.append(float(match["ctr_ps"]) if match else np.nan)
                if (
                    match
                    and model not in ("led", "cfd")
                    and match.get("window_before_ns", "") != ""
                ):
                    labels.append(
                        (
                            len(values) - 1,
                            f"-{float(match['window_before_ns']):g}/"
                            f"+{float(match['window_after_ns']):g}",
                        )
                    )
            x_positions = center - 0.4 + width / 2 + index * width
            bars = axis.bar(
                x_positions,
                values,
                width=width,
                label=short_model_label(model),
            )
            for value_index, text in labels:
                value = values[value_index]
                if np.isfinite(value):
                    axis.text(
                        x_positions[value_index],
                        value,
                        text,
                        ha="center",
                        va="bottom",
                        rotation=90,
                        fontsize=6,
                    )
        axis.set_xticks(center, [f"{voltage:g}" for voltage in voltages])
        axis.set_xlabel("Bias voltage [V]")
        axis.set_ylabel("Blind CTR [ps]")
        axis.set_title(short_mode_label(mode))
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False, ncol=min(5, len(models)), fontsize=8)
    _save(fig, path, dpi)


def plot_selection_vs_blind(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    selection_stage: str,
    dpi: int,
) -> None:
    selected = [
        row
        for row in _series_rows(rows, selection_stage, True)
        if row["model"] not in ("led", "cfd")
        and int(row.get("plot_included", 1)) == 1
    ]
    blind = _series_rows(rows, "blind", True)
    pairs = []
    for row in selected:
        matching = next(
            (
                item
                for item in blind
                if item["file"] == row["file"]
                and item["mode"] == row["mode"]
                and item["model"] == row["model"]
                and int(item.get("plot_included", 1)) == 1
            ),
            None,
        )
        if matching:
            pairs.append((row, matching))
    if not pairs:
        return
    fig, axis = plt.subplots(figsize=(6.7, 6.0))
    for model in sorted(set(first["model"] for first, _second in pairs)):
        points = [(first, second) for first, second in pairs if first["model"] == model]
        axis.scatter(
            [float(first["ctr_ps"]) for first, _ in points],
            [float(second["ctr_ps"]) for _, second in points],
            label=short_model_label(model),
        )
    values = np.array([float(item["ctr_ps"]) for pair in pairs for item in pair])
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    padding = 0.05 * (high - low if high > low else 1.0)
    axis.plot(
        [low - padding, high + padding],
        [low - padding, high + padding],
        "--",
        linewidth=1,
    )
    x_values = np.asarray([float(first["ctr_ps"]) for first, _ in pairs])
    y_values = np.asarray([float(second["ctr_ps"]) for _, second in pairs])
    correlation = (
        float(np.corrcoef(x_values, y_values)[0, 1])
        if len(x_values) >= 3 and np.std(x_values) > 0 and np.std(y_values) > 0
        else np.nan
    )
    axis.set_xlabel(f"{selection_stage.replace('_', ' ').title()} CTR [ps]")
    axis.set_ylabel("Blind CTR [ps]")
    axis.set_title(
        "Selection vs blind"
        + (f" · r={correlation:.2f}" if np.isfinite(correlation) else "")
    )
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, fontsize=8)
    _save(fig, path, dpi)


def plot_correction_matrix(
    path: Path,
    *,
    corrections: dict[str, Any],
    dpi: int,
    title: str,
) -> None:
    normalized: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, value in corrections.items():
        if isinstance(value, tuple) and len(value) == 2:
            indices = np.asarray(value[0], dtype=np.int64)
            values = np.asarray(value[1], dtype=float)
            if len(indices) == len(values) and len(indices) >= 3:
                normalized[name] = indices, values
        else:
            values = np.asarray(value, dtype=float)
            if values.size >= 3:
                normalized[name] = np.arange(values.size, dtype=np.int64), values
    names = list(normalized)
    if len(names) < 2:
        return
    common = set(normalized[names[0]][0].tolist())
    for name in names[1:]:
        common.intersection_update(normalized[name][0].tolist())
    common_indices = np.asarray(sorted(common), dtype=np.int64)
    if common_indices.size < 3:
        return
    columns = []
    for name in names:
        indices, values = normalized[name]
        lookup = {int(index): float(value) for index, value in zip(indices, values)}
        columns.append(
            np.asarray([lookup[int(index)] for index in common_indices], dtype=float)
        )
    matrix = np.column_stack(columns)
    matrix = matrix[np.all(np.isfinite(matrix), axis=1)]
    if len(matrix) < 3:
        return
    correlation = np.corrcoef(matrix, rowvar=False)
    fig, axis = plt.subplots(
        figsize=(max(5, 1 + 0.75 * len(names)), max(4.5, 1 + 0.7 * len(names)))
    )
    image = axis.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
    axis.set_xticks(
        range(len(names)),
        [short_model_label(name) for name in names],
        rotation=35,
        ha="right",
    )
    axis.set_yticks(range(len(names)), [short_model_label(name) for name in names])
    axis.set_title(title + f" · n={len(matrix)}")
    for row in range(len(names)):
        for column in range(len(names)):
            axis.text(
                column,
                row,
                f"{correlation[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Pearson r")
    _save(fig, path, dpi)


def centered_correction_components(
    led_residual: np.ndarray,
    corrected_residual: np.ndarray,
    *,
    led_center_ps: float,
    correction_center_ps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return centered LED, centered correction, centered final and gain.

    Both centers must be learned on the development population and then frozen
    for blind evaluation.  This helper contains the full TOP/WORST ranking
    definition so plotting and any future CSV/audit output cannot diverge.
    """
    led = np.asarray(led_residual, dtype=np.float64).reshape(-1)
    final = np.asarray(corrected_residual, dtype=np.float64).reshape(-1)
    if led.size != final.size:
        raise ValueError("LED and corrected residual arrays must have equal length")
    raw_correction = final - led
    centered_led = led - float(led_center_ps)
    centered_correction = raw_correction - float(correction_center_ps)
    centered_final = centered_led + centered_correction
    gain = np.abs(centered_led) - np.abs(centered_final)
    return centered_led, centered_correction, centered_final, gain


def plot_correction_examples(
    path: Path,
    *,
    time_ps: np.ndarray,
    waveforms: np.ndarray,
    led_residual: np.ndarray,
    corrected_residual: np.ndarray,
    led_center_ps: float,
    correction_center_ps: float,
    model: str,
    mode: str,
    selection: str,
    k: int,
    dpi: int,
    window_before_ns: float | None = None,
    window_after_ns: float | None = None,
    event_ids: np.ndarray | None = None,
) -> None:
    """Render TOP/WORST events using development-centered linear correction.

    The caller must learn ``led_center_ps`` and ``correction_center_ps`` on the
    development population.  No blind statistic is used for centering.

    For each blind event:
        led_linear = led_residual - led_center_ps
        raw_correction = corrected_residual - led_residual
        correction_linear = raw_correction - correction_center_ps
        final_linear = led_linear + correction_linear
        gain = |led_linear| - |final_linear|

    TOP sorts by largest gain; WORST by smallest gain.
    """
    selection = str(selection).strip().lower()
    if selection not in {"top", "worst"}:
        raise ValueError(f"selection must be 'top' or 'worst', got {selection!r}")
    k = int(k)
    if k <= 0:
        return

    led = np.asarray(led_residual, dtype=np.float64).reshape(-1)
    final = np.asarray(corrected_residual, dtype=np.float64).reshape(-1)
    waves = np.asarray(waveforms)
    time_ns = np.asarray(time_ps, dtype=np.float64).reshape(-1) / 1000.0
    if led.size != final.size:
        raise ValueError("LED and corrected residual arrays must have equal length")
    if waves.ndim != 3 or waves.shape[0] != led.size or waves.shape[1] != 2:
        raise ValueError("waveforms must have shape (N, 2, samples)")
    if waves.shape[2] != time_ns.size:
        raise ValueError("time grid length must match waveform sample length")

    centered_led, centered_correction, centered_final, gain = (
        centered_correction_components(
            led,
            final,
            led_center_ps=led_center_ps,
            correction_center_ps=correction_center_ps,
        )
    )

    valid = np.flatnonzero(
        np.isfinite(gain)
        & np.isfinite(centered_led)
        & np.isfinite(centered_correction)
        & np.isfinite(centered_final)
    )
    if valid.size == 0:
        return
    ordered = valid[np.argsort(gain[valid])]
    if selection == "top":
        ordered = ordered[::-1]
    ordered = ordered[: min(k, ordered.size)]

    ids = None
    if event_ids is not None:
        ids = np.asarray(event_ids).reshape(-1)
        if ids.size != led.size:
            raise ValueError("event_ids length must match residual arrays")

    fig, axes = plt.subplots(
        ordered.size,
        1,
        figsize=(9.6, 3.0 * ordered.size),
        squeeze=False,
    )
    for rank, (axis, event_index) in enumerate(zip(axes[:, 0], ordered), start=1):
        if window_before_ns is not None and window_after_ns is not None:
            axis.axvspan(
                -float(window_before_ns),
                float(window_after_ns),
                alpha=0.08,
            )
        axis.plot(time_ns, waves[event_index, 0], linewidth=1.05, label="ch1")
        axis.plot(time_ns, waves[event_index, 1], linewidth=1.05, label="ch2")
        axis.axvline(0.0, linewidth=0.8, linestyle="--")
        axis.set_ylabel("mV")
        axis.grid(alpha=0.18)

        improvement = float(gain[event_index])
        if improvement >= 0.0:
            effect_text = f"{improvement:.0f} ps improvement"
        else:
            effect_text = f"{abs(improvement):.0f} ps worsening"
        event_text = f" · event {ids[event_index]}" if ids is not None else ""
        summary = (
            f"#{rank} LED {centered_led[event_index]:+.0f} ps "
            f"+ correction {centered_correction[event_index]:+.0f} ps "
            f"= {centered_final[event_index]:+.0f} ps · "
            f"|LED| {abs(centered_led[event_index]):.0f}→"
            f"{abs(centered_final[event_index]):.0f} ps · {effect_text}{event_text}"
        )
        axis.plot([], [], linestyle="none", marker="", label=summary)
        axis.legend(
            frameon=True,
            framealpha=0.86,
            fontsize=7.5,
            loc="upper right",
        )

    axes[-1, 0].set_xlabel("Time relative to LED-aligned native anchor [ns]")
    kind = "TOP" if selection == "top" else "WORST"
    fig.suptitle(
        f"{short_model_label(model)} · {short_mode_label(mode)} · {kind} corrections",
        fontsize=11,
    )
    _save(fig, path, dpi)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_summary_results(path: Path, rows: list[dict[str, Any]]) -> None:
    blind = [
        dict(row)
        for row in rows
        if row.get("stage_name") == "blind"
        and int(row.get("selected", 0)) == 1
    ]
    columns = [
        "file",
        "voltage_V",
        "mode",
        "model",
        "window_id",
        "window_before_ns",
        "window_after_ns",
        "subsampling",
        "hyperparameters_json",
        "validation_strategy",
        "validation_ctr_ps",
        "validation_ctr_uncertainty_ps",
        "n",
        "mean_ps",
        "std_ps",
        "ctr_ps",
        "ctr_uncertainty_ps",
        "rmse_ps",
        "led_ctr_ps",
        "ctr_over_led",
        "plot_included",
    ]
    output = [{key: row.get(key, "") for key in columns} for row in blind]
    write_csv(path, output)
