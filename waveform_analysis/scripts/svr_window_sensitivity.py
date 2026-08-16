from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from ml_pipeline.common import setup_logging
from ml_pipeline.prepared_data import input_channel_variant_dataset_view, prepare_file_dataset
from ml_pipeline.prediction import prediction_window_dataset_view
from ml_pipeline.study import (
    _aggregate_fold_stats,
    _candidate_training_config,
    _cleanup_training,
    _distribution_stats,
    _kfold,
    _normalization_for_fit_subset,
    _predict_indices,
    _random_dev_blind,
    _seed_for,
    _target_deltas,
    _train_in_memory,
    _voltage_from_name,
)
from ml_pipeline.study_config import CHANNEL_MODES, discover_root_files, load_study_config
from ml_pipeline.torch_data import Normalization


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} does not contain a JSON object")
    return value


def _resolve_from_project(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else PROJECT / path).resolve()


def _find_linear_svr_model_name(
    raw_config: dict[str, Any],
    requested_space_id: str | None,
) -> tuple[str, dict[str, Any]]:
    model_dir = _resolve_from_project(
        raw_config.get("model_spaces_dir", "config/model_spaces")
    )
    matches: list[tuple[str, dict[str, Any]]] = []
    for name in [str(v) for v in raw_config.get("models", [])]:
        space_path = model_dir / f"{name}.json"
        if not space_path.is_file():
            continue
        space = _read_json(space_path)
        if str(space.get("model_type")) != "linear_svr":
            continue
        if requested_space_id is not None and str(space.get("id")) != requested_space_id:
            continue
        matches.append((name, space))

    if not matches:
        suffix = f" with id {requested_space_id!r}" if requested_space_id is not None else ""
        raise RuntimeError(f"No linear_svr model space found in experiment{suffix}")
    if len(matches) > 1:
        ids = [str(space.get("id")) for _, space in matches]
        raise RuntimeError(
            "Experiment contains multiple linear_svr spaces. "
            f"Choose one with --svr-space-id. Available IDs: {ids}"
        )
    return matches[0]


def _load_svr_only_config(
    config_path: Path,
    requested_space_id: str | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate the experiment using only its Linear-SVR model space."""
    source = config_path.resolve()
    raw = _read_json(source)
    model_name, _raw_space = _find_linear_svr_model_name(raw, requested_space_id)

    pruned = json.loads(json.dumps(raw))
    pruned["models"] = [model_name]

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix=".svr_window_sensitivity_",
            dir=PROJECT,
            delete=False,
            encoding="utf-8",
        ) as stream:
            json.dump(pruned, stream, indent=2)
            temp_path = Path(stream.name)
        config = load_study_config(temp_path, PROJECT)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    if len(config["_model_spaces"]) != 1:
        raise RuntimeError("Expected exactly one validated Linear-SVR model space")
    space = config["_model_spaces"][0]
    if str(space["model_type"]) != "linear_svr":
        raise RuntimeError("Validated model space is not linear_svr")
    return config, space, raw


def _read_results(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Study results not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _selected_candidate(
    rows: list[dict[str, str]],
    manifest: dict[str, Any],
    *,
    file_id: int,
    mode: str,
    space_id: str,
) -> tuple[int, dict[str, Any], dict[str, str]]:
    codebooks = manifest["codebooks"]
    mode_id = int(codebooks["mode"][mode])
    model_id = int(codebooks["model"][space_id])
    selected = [
        row
        for row in rows
        if int(float(row["stage"])) == 0
        and int(float(row["file_id"])) == file_id
        and int(float(row["mode_id"])) == mode_id
        and int(float(row["model_id"])) == model_id
        and int(float(row.get("selected", "0"))) == 1
    ]
    if len(selected) != 1:
        raise RuntimeError(
            f"Expected exactly one selected OOF candidate for "
            f"file_id={file_id}, mode={mode}, model={space_id}; got {len(selected)}"
        )
    row = selected[0]
    candidate_id = int(float(row["candidate_id"]))
    params = manifest.get("candidate_parameters", {}).get(str(candidate_id))
    if not isinstance(params, dict):
        raise RuntimeError(f"candidate_parameters[{candidate_id}] missing from study manifest")
    if str(params.get("family")) != space_id or str(params.get("mode")) != mode:
        raise RuntimeError(f"Selected candidate descriptor is inconsistent: {params}")
    return candidate_id, params, row


def _window_label(start_ns: float, end_ns: float, index: int) -> str:
    def fmt(value: float) -> str:
        sign = "m" if value < 0 else "p"
        text = f"{abs(value):g}".replace(".", "p")
        return f"{sign}{text}"
    return f"w{index}_{fmt(start_ns)}_{fmt(end_ns)}"


def _scan_windows(
    config: dict[str, Any],
    *,
    start_ns: float | None,
    end_ns: list[float] | None,
) -> list[dict[str, Any]]:
    if end_ns is None:
        windows = [
            {
                "id": str(window["id"]),
                "before_ns": float(window["before_ns"]),
                "after_ns": float(window["after_ns"]),
            }
            for window in config["windows_ns"]
        ]
    else:
        if start_ns is None:
            raise ValueError("--start-ns is required when --end-ns is supplied")
        if start_ns >= 0:
            raise ValueError("--start-ns must be negative")
        windows = []
        for index, end in enumerate(end_ns):
            end = float(end)
            if end <= 0 or end <= start_ns:
                raise ValueError(f"Invalid scan window [{start_ns}, {end}] ns")
            windows.append(
                {
                    "id": _window_label(float(start_ns), end, index),
                    "before_ns": -float(start_ns),
                    "after_ns": end,
                }
            )

    seen: set[tuple[float, float]] = set()
    unique: list[dict[str, Any]] = []
    for window in windows:
        key = (float(window["before_ns"]), float(window["after_ns"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(window)
    if len(unique) < 2:
        raise RuntimeError(
            "Window sensitivity requires at least two distinct windows. "
            "Add more windows to the experiment config or use --start-ns/--end-ns."
        )
    return unique


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_correlation_csv(path: Path, labels: list[str], matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["window", *labels])
        for label, values in zip(labels, matrix):
            writer.writerow([label, *[f"{float(value):.12g}" for value in values]])


def _plot_correlation(
    path: Path,
    labels: list[str],
    matrix: np.ndarray,
    *,
    title: str,
    dpi: int,
) -> None:
    size = max(6.0, 0.8 * len(labels) + 3.0)
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(matrix, vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=8)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_ctr_by_file_mode(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    file_name: str,
    mode: str,
    dpi: int,
) -> None:
    subset = [row for row in rows if row["file"] == file_name and row["mode"] == mode]
    subset.sort(key=lambda row: float(row["window_width_ns"]))
    x = np.asarray([float(row["window_width_ns"]) for row in subset])
    y = np.asarray([float(row["ctr_ps"]) for row in subset])
    e = np.asarray([float(row["ctr_fold_std_ps"]) for row in subset])

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.errorbar(x, y, yerr=e, marker="o", capsize=3)
    ax.set_xlabel("Window width [ns]")
    ax.set_ylabel("Mean fold CTR FWHM [ps]")
    ax.set_title(f"{file_name} | {mode} | fixed selected Linear SVR")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_ctr_mode_summary(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    mode: str,
    dpi: int,
) -> None:
    subset = [row for row in rows if row["mode"] == mode]
    files = sorted(
        {row["file"] for row in subset},
        key=lambda name: min(float(r["voltage_V"]) for r in subset if r["file"] == name),
    )
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    for file_name in files:
        points = [row for row in subset if row["file"] == file_name]
        points.sort(key=lambda row: float(row["window_width_ns"]))
        voltage = float(points[0]["voltage_V"])
        ax.plot(
            [float(row["window_width_ns"]) for row in points],
            [float(row["ctr_ps"]) for row in points],
            marker="o",
            label=f"{file_name} ({voltage:g} V)",
        )
    ax.set_xlabel("Window width [ns]")
    ax.set_ylabel("Mean fold CTR FWHM [ps]")
    ax.set_title(f"{mode} | fixed selected Linear SVR")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _validate_fixed_svr_candidate(
    config: dict[str, Any],
    space: dict[str, Any],
    overrides: dict[str, Any],
    *,
    mode: str,
    subsampling: int,
    seed: int,
    work_dir: Path,
) -> None:
    cfg = _candidate_training_config(
        config,
        space,
        overrides,
        mode=mode,
        subsampling=subsampling,
        train_dir=work_dir,
        seed=seed,
        final=False,
    )
    if str(cfg["model"]["type"]) != "linear_svr":
        raise RuntimeError("Window sensitivity supports linear_svr only")
    eps = cfg["model"].get("epsilon_values")
    if isinstance(eps, list) and len(eps) != 1:
        raise RuntimeError(
            "The selected SVR configuration still contains multiple epsilon_values. "
            "This script requires a fully fixed model configuration so window is "
            "the only varied quantity."
        )


def _evaluate_one_window(
    *,
    config: dict[str, Any],
    space: dict[str, Any],
    dataset: Any,
    development: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    file_id: int,
    mode: str,
    window: dict[str, Any],
    variant: str,
    subsampling: int,
    overrides: dict[str, Any],
    selected_candidate_id: int,
    work_root: Path,
    logger: Any,
    normalization_cache: dict[str, Normalization],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    input_waveforms, target = CHANNEL_MODES[mode]
    source = input_channel_variant_dataset_view(dataset, input_waveforms, variant)
    view = prediction_window_dataset_view(
        source,
        input_waveforms=input_waveforms,
        target=target,
        before_ns=float(window["before_ns"]),
        after_ns=float(window["after_ns"]),
    )

    led_ps, _ = _target_deltas(
        dataset,
        mode,
        np.arange(dataset.event_id.size, dtype=np.int64),
    )
    led_residual = led_ps - float(dataset.true_tof_ps)
    oof_correction = np.full(dataset.event_id.size, np.nan, dtype=np.float64)
    oof_residual = np.full(dataset.event_id.size, np.nan, dtype=np.float64)
    fold_metrics: list[dict[str, Any]] = []
    base_seed = int(config["cross_validation"]["seed"])

    for fold_index, (train_pool, score_idx) in enumerate(folds):
        candidate_seed = _seed_for(
            base_seed,
            file_id,
            mode,
            window["id"],
            variant,
            subsampling,
            space["id"],
            selected_candidate_id,
            fold_index,
        )
        fold_dir = work_root / f"f{file_id}_{mode}_{window['id']}_fold{fold_index}"
        cfg = _candidate_training_config(
            config,
            space,
            overrides,
            mode=mode,
            subsampling=subsampling,
            train_dir=fold_dir,
            seed=candidate_seed,
            final=False,
        )
        fit_idx = np.asarray(train_pool, dtype=np.int64)
        fold_view = replace(
            view,
            train=fit_idx,
            validation=fit_idx,
            evaluation=np.asarray(score_idx, dtype=np.int64),
        )
        normalization = _normalization_for_fit_subset(
            fold_view,
            fit_idx,
            subsampling=subsampling,
            cache=normalization_cache,
        )
        model, normalization, _summary = _train_in_memory(
            cfg,
            fold_view,
            logger=logger,
            data_view={
                "stage": "svr_window_sensitivity",
                "fold": fold_index,
                "window": window["id"],
                "selected_candidate_id": selected_candidate_id,
            },
            normalization_override=normalization,
        )
        residual = _predict_indices(
            model,
            normalization,
            cfg,
            fold_view,
            np.asarray(score_idx, dtype=np.int64),
        )
        correction = led_residual[score_idx] - residual
        oof_residual[score_idx] = residual
        oof_correction[score_idx] = correction
        fold_metrics.append(
            _distribution_stats(
                residual,
                method=f"SVR window sensitivity {space['id']} {window['id']} fold {fold_index + 1}",
            )
        )
        _cleanup_training(model, fold_dir)

    if np.any(~np.isfinite(oof_correction[development])):
        raise RuntimeError(f"Missing OOF correction predictions for window {window['id']}")
    if np.any(~np.isfinite(oof_residual[development])):
        raise RuntimeError(f"Missing OOF residual predictions for window {window['id']}")

    metrics = _aggregate_fold_stats(
        fold_metrics,
        method=f"SVR window sensitivity {window['id']}",
    )
    return metrics, oof_correction, oof_residual


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the Linear-SVR configuration selected by an existing CTR study, "
            "vary only the waveform window, compare fold-wise CTR, and measure "
            "event-aligned OOF correction correlations between windows."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--study-dir", type=Path, default=None)
    parser.add_argument("--svr-space-id", type=str, default=None)
    parser.add_argument("--modes", nargs="+", default=None)
    parser.add_argument("--start-ns", type=float, default=None)
    parser.add_argument("--end-ns", nargs="+", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()

    config, space, raw_config = _load_svr_only_config(args.config, args.svr_space_id)
    space_id = str(space["id"])
    study_dir = (
        _resolve_from_project(raw_config["experiment"]["output_dir"])
        if args.study_dir is None
        else args.study_dir.resolve()
    )
    manifest = _read_json(study_dir / "manifest.json")
    study_rows = _read_results(study_dir / "results.csv")

    if manifest.get("status") != "complete":
        raise RuntimeError(
            f"Study is not complete: status={manifest.get('status')!r}. "
            "Use a completed study so the selected SVR configuration is frozen."
        )
    if space_id not in manifest.get("codebooks", {}).get("model", {}):
        raise RuntimeError(
            f"SVR space id {space_id!r} is not present in study codebooks. "
            "Make sure --config corresponds to this study."
        )

    output = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else study_dir / "svr_window_sensitivity"
    )
    if args.restart and output.exists():
        shutil.rmtree(output)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(
            f"Output directory already exists and is non-empty: {output}. "
            "Use --restart or choose --output-dir."
        )
    output.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(
        output / "window_sensitivity.log",
        config.get("logging", {}).get("level", "INFO"),
    )
    windows = _scan_windows(config, start_ns=args.start_ns, end_ns=args.end_ns)
    modes = [str(v) for v in args.modes] if args.modes is not None else [str(v) for v in config["channel_modes"]]
    unknown_modes = sorted(set(modes) - set(config["channel_modes"]))
    if unknown_modes:
        raise ValueError(f"Modes not present in experiment: {unknown_modes}")

    root_files = discover_root_files(config)
    if not root_files:
        raise FileNotFoundError("No ROOT input files discovered")

    codebooks = manifest["codebooks"]
    base_seed = int(config["cross_validation"]["seed"])
    n_splits = int(config["cross_validation"]["n_splits"])
    blind_fraction = float(config["cross_validation"]["blind_fraction"])
    dpi = int(config["reporting"]["dpi"])
    result_rows: list[dict[str, Any]] = []
    selected_configs: dict[str, Any] = {}
    work_root = output / ".work"
    work_root.mkdir(parents=True, exist_ok=True)

    logger.info(
        "SVR window sensitivity | study=%s | model=%s | windows=%d | modes=%s",
        study_dir, space_id, len(windows), modes,
    )
    logger.info("Protocol | development OOF only | same events/folds across windows | blind untouched")

    for root_file in root_files:
        if root_file.name not in codebooks["file"]:
            raise RuntimeError(f"Input file {root_file.name} not present in original study codebook")
        file_id = int(codebooks["file"][root_file.name])
        dataset = prepare_file_dataset(config, root_file, rebuild=False, logger=logger)
        development, _blind = _random_dev_blind(
            int(dataset.event_id.size),
            blind_fraction=blind_fraction,
            seed=_seed_for(base_seed, file_id, "devblind"),
        )
        folds = _kfold(
            development,
            n_splits=n_splits,
            seed=_seed_for(base_seed, file_id, "folds"),
        )
        voltage = _voltage_from_name(root_file.name, str(config["reporting"]["voltage_pattern"]))
        normalization_cache: dict[str, Normalization] = {}

        for mode in modes:
            selected_candidate_id, descriptor, selected_row = _selected_candidate(
                study_rows,
                manifest,
                file_id=file_id,
                mode=mode,
                space_id=space_id,
            )
            variant = str(descriptor["variant"])
            subsampling = int(descriptor["subsampling"])
            overrides = dict(descriptor.get("overrides", {}))
            original_window_id = str(descriptor["window"])
            original_ctr = float(selected_row["ctr_ps"])

            selected_configs[f"{root_file.name}:{mode}"] = {
                "file_id": file_id,
                "mode": mode,
                "space_id": space_id,
                "candidate_id": selected_candidate_id,
                "original_selected_window": original_window_id,
                "variant": variant,
                "subsampling": subsampling,
                "overrides": overrides,
                "selected_study_ctr_ps": original_ctr,
            }

            _validate_fixed_svr_candidate(
                config,
                space,
                overrides,
                mode=mode,
                subsampling=subsampling,
                seed=_seed_for(base_seed, file_id, mode, "fixed_config_validation"),
                work_dir=work_root / "_validate",
            )

            logger.info(
                "Selected fixed SVR | file=%s | mode=%s | candidate=%d | original_window=%s | "
                "variant=%s | subsampling=%d | overrides=%s",
                root_file.name, mode, selected_candidate_id, original_window_id,
                variant, subsampling, json.dumps(overrides, sort_keys=True),
            )

            corrections: list[np.ndarray] = []
            residuals: list[np.ndarray] = []
            labels: list[str] = []

            for window in windows:
                start_ns = -float(window["before_ns"])
                end_ns = float(window["after_ns"])
                width_ns = float(window["before_ns"]) + float(window["after_ns"])
                logger.info(
                    "Window | file=%s | mode=%s | %s [%.3f, %.3f] ns | width=%.3f ns",
                    root_file.name, mode, window["id"], start_ns, end_ns, width_ns,
                )

                metrics, oof_correction, oof_residual = _evaluate_one_window(
                    config=config,
                    space=space,
                    dataset=dataset,
                    development=development,
                    folds=folds,
                    file_id=file_id,
                    mode=mode,
                    window=window,
                    variant=variant,
                    subsampling=subsampling,
                    overrides=overrides,
                    selected_candidate_id=selected_candidate_id,
                    work_root=work_root,
                    logger=logger,
                    normalization_cache=normalization_cache,
                )

                labels.append(str(window["id"]))
                corrections.append(np.asarray(oof_correction[development], dtype=np.float64))
                residuals.append(np.asarray(oof_residual[development], dtype=np.float64))
                result_rows.append(
                    {
                        "file": root_file.name,
                        "file_id": file_id,
                        "voltage_V": voltage,
                        "mode": mode,
                        "model_space": space_id,
                        "selected_candidate_id": selected_candidate_id,
                        "window_id": str(window["id"]),
                        "window_start_ns": start_ns,
                        "window_end_ns": end_ns,
                        "window_width_ns": width_ns,
                        "is_original_selected_window": int(str(window["id"]) == original_window_id),
                        "variant": variant,
                        "subsampling": subsampling,
                        "n": int(metrics["n"]),
                        "mean_ps": float(metrics["mean_ps"]),
                        "std_ps": float(metrics["std_ps"]),
                        "ctr_ps": float(metrics["ctr_ps"]),
                        "ctr_fold_std_ps": float(metrics["ctr_fold_std_ps"]),
                        "rmse_ps": float(metrics["rmse_ps"]),
                        "rmse_fold_std_ps": float(metrics["rmse_fold_std_ps"]),
                        "selected_study_ctr_ps": original_ctr,
                        "delta_ctr_vs_selected_study_ps": float(metrics["ctr_ps"]) - original_ctr,
                        "overrides_json": json.dumps(overrides, sort_keys=True),
                    }
                )
                _write_csv(output / "results.csv", result_rows)
                logger.info(
                    "Result | file=%s | mode=%s | window=%s | CTR=%.3f +/- %.3f ps | delta_vs_selected=%+.3f ps",
                    root_file.name, mode, window["id"], float(metrics["ctr_ps"]),
                    float(metrics["ctr_fold_std_ps"]), float(metrics["ctr_ps"]) - original_ctr,
                )

            correction_matrix = np.stack(corrections, axis=0)
            residual_matrix = np.stack(residuals, axis=0)
            correlation = np.corrcoef(correction_matrix)
            safe_stem = root_file.stem.replace(" ", "_")
            safe_mode = mode.replace(" ", "_")
            pair_dir = output / "by_file_mode" / safe_stem / safe_mode
            pair_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                pair_dir / "oof_predictions.npz",
                window_labels=np.asarray(labels),
                development_indices=np.asarray(development, dtype=np.int64),
                event_id=np.asarray(dataset.event_id[development]),
                correction_ps=correction_matrix,
                residual_ps=residual_matrix,
            )
            _write_correlation_csv(pair_dir / "correction_correlation.csv", labels, correlation)
            _plot_correlation(
                pair_dir / "correction_correlation.png",
                labels,
                correlation,
                title=f"{root_file.name} | {mode}\nOOF predicted-correction correlation",
                dpi=dpi,
            )
            _plot_ctr_by_file_mode(
                pair_dir / "ctr_vs_window_width.png",
                result_rows,
                file_name=root_file.name,
                mode=mode,
                dpi=dpi,
            )

        normalization_cache.clear()

    _write_csv(output / "results.csv", result_rows)
    with (output / "selected_svr_configs.json").open("w", encoding="utf-8") as stream:
        json.dump(selected_configs, stream, indent=2, sort_keys=True)
    for mode in modes:
        _plot_ctr_mode_summary(
            output / "ctr_vs_window_width" / f"{mode}.png",
            result_rows,
            mode=mode,
            dpi=dpi,
        )

    shutil.rmtree(work_root, ignore_errors=True)
    logger.info("Complete | output=%s | rows=%d", output, len(result_rows))
    print(output)


if __name__ == "__main__":
    main()
