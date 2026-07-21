from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from .binary_io import (
    atomic_write_csv,
    canonical_run_id,
    discover_runs,
    file_signature,
    parse_run_info,
    read_csv,
)
from .cache import CACHE_SCHEMA_VERSION, load_state, mark_stage, save_state, signature, stage_migratable, stage_valid_any
from .config import stage_config
from .tabular import table_path
from .fitting import fit_timing, load_fit_csv, write_fit_csv
from .matching import (
    load_model,
    load_total_cache,
    count_model_corrected_alignments,
    load_training_csv,
    scan_matching_training,
    scan_streaming_matching_training,
    train_matching_models,
    write_model,
    write_model_metrics,
    write_total_cache,
    write_total_csv,
    write_training_csv,
)
from .models import EnergySelectionResult, PeakSelection, SelectionResult
from .plotting import (
    plot_matching_total,
    plot_matching_training,
    plot_peak_selection,
    plot_timing_fit,
)
from .preprocessing import (
    preprocess_binary,
    preprocess_candidates,
    preprocess_streaming_from_index,
)
from .streaming_cache import (
    build_streaming_candidate_index,
    candidate_index_outputs,
    collect_streaming_energy_measurements,
    decode_streaming_pulse_cache,
    pulse_cache_outputs,
)
from .selection import (
    collect_energy_measurements,
    collect_measurements,
    load_energy_selection_csv,
    load_selection_csv,
    select_energy_events,
    select_matched_events,
    write_energy_selection_csv,
    write_selection_csv,
)

SUMMARY_FIELDS = [
    "run_id",
    "Voltage",
    "AcquisitionMode",
    "E_th",
    "T_th",
    "gaussian_area_events",
    "gaussian_area_error_events",
    "gaussian_mean_ps",
    "gaussian_mean_error_ps",
    "gaussian_sigma_ps",
    "gaussian_sigma_error_ps",
    "CTR_ps",
    "CTR_error_ps",
    "average_delay_corrected_alignments"
]


def _normalise_run_ids(values: list[Any]) -> set[str]:
    output: set[str] = set()
    for value in values:
        output.add(canonical_run_id(str(value))[0])
    return output


def _log(run_id: str, stage: str, message: str) -> None:
    print(f"[{run_id}][{stage}] {message}", flush=True)


def _elapsed(start: float) -> str:
    return f"{time.perf_counter() - start:.2f} s"


def _selection_from_cache(measurements, duration_mask, alignment_mask, metadata: dict[str, Any]) -> SelectionResult:
    return SelectionResult(
        peak_a=PeakSelection(**metadata["peak_a"]),
        peak_b=PeakSelection(**metadata["peak_b"]),
        duration_mask=duration_mask,
        alignment_mask=alignment_mask,
        final_mask=duration_mask & alignment_mask,
        alignment_a_center_lsb=float(metadata["alignment_a_center_lsb"]),
        alignment_a_scale_lsb=float(metadata["alignment_a_scale_lsb"]),
        alignment_b_center_lsb=float(metadata["alignment_b_center_lsb"]),
        alignment_b_scale_lsb=float(metadata["alignment_b_scale_lsb"]),
    )


def _selection_metadata(selection: SelectionResult, toa_lsb_ps: float) -> dict[str, Any]:
    return {
        "toa_lsb_ps": toa_lsb_ps,
        "peak_a": {
            "low_lsb": selection.peak_a.low_lsb,
            "high_lsb": selection.peak_a.high_lsb,
            "peak_lsb": selection.peak_a.peak_lsb,
            "center_lsb": selection.peak_a.center_lsb,
            "scale_lsb": selection.peak_a.scale_lsb,
        },
        "peak_b": {
            "low_lsb": selection.peak_b.low_lsb,
            "high_lsb": selection.peak_b.high_lsb,
            "peak_lsb": selection.peak_b.peak_lsb,
            "center_lsb": selection.peak_b.center_lsb,
            "scale_lsb": selection.peak_b.scale_lsb,
        },
        "alignment_a_center_lsb": selection.alignment_a_center_lsb,
        "alignment_a_scale_lsb": selection.alignment_a_scale_lsb,
        "alignment_b_center_lsb": selection.alignment_b_center_lsb,
        "alignment_b_scale_lsb": selection.alignment_b_scale_lsb,
    }


def _number(value: Any) -> float | str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return number if math.isfinite(number) else ""


def _summary_row(run, run_info, fit, toa_lsb_ps: float| None, average_delay_corrected_alignments: int) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_id": run.run_id,
        "Voltage": run.voltage,
        "AcquisitionMode": run_info.acquisition_mode,
        "E_th": run_info.energy_threshold_mv,
        "T_th": run_info.timing_threshold_mv,
        "gaussian_area_events": _number(fit.area_events),
        "gaussian_area_error_events": _number(fit.area_error_events),
        "gaussian_mean_ps": _number(fit.mean_lsb * toa_lsb_ps),
        "gaussian_mean_error_ps": _number(fit.mean_error_lsb * toa_lsb_ps),
        "gaussian_sigma_ps": _number(fit.sigma_lsb * toa_lsb_ps),
        "gaussian_sigma_error_ps": _number(fit.sigma_error_lsb * toa_lsb_ps),
        "CTR_ps": _number(2.355 * fit.sigma_lsb * toa_lsb_ps),
        "CTR_error_ps": _number(2.355 * fit.sigma_error_lsb * toa_lsb_ps),
        "average_delay_corrected_alignments": int(average_delay_corrected_alignments),
    }
    return row


def _load_summary(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    return {row["run_id"]: row for row in read_csv(path)}


def _write_summary(path: Path, rows: dict[str, dict[str, Any]]) -> None:
    ordered = sorted(rows.values(), key=lambda row: int(canonical_run_id(str(row["run_id"]))[1]))
    atomic_write_csv(path, SUMMARY_FIELDS, ordered)


def _summary_rows_equal(left: dict[str, Any] | None, right: dict[str, Any]) -> bool:
    if left is None:
        return False
    return all(str(left.get(field, "")) == str(right.get(field, "")) for field in SUMMARY_FIELDS)


def _reuse_stage(
    state: dict[str, Any],
    stage: str,
    signatures: list[str],
    outputs: list[Path],
    overwrite: bool,
    migration_mode: bool,
) -> bool:
    if overwrite:
        return False
    return stage_valid_any(state, stage, signatures, outputs) or stage_migratable(
        state, stage, outputs, migration_mode
    )


def _refresh_stage_signature(
    state: dict[str, Any],
    stage: str,
    stage_signature: str,
    outputs: list[Path],
) -> bool:
    record = state["stages"][stage]
    if record.get("signature") == stage_signature:
        return False
    mark_stage(
        state,
        stage,
        stage_signature,
        outputs,
        dict(record.get("metadata", {})),
    )
    return True


def _energy_selection_metadata(
    selection: EnergySelectionResult,
    toa_lsb_ps: float,
) -> dict[str, Any]:
    return {
        "toa_lsb_ps": toa_lsb_ps,
        "peak_a": {
            "low_lsb": selection.peak_a.low_lsb,
            "high_lsb": selection.peak_a.high_lsb,
            "peak_lsb": selection.peak_a.peak_lsb,
            "center_lsb": selection.peak_a.center_lsb,
            "scale_lsb": selection.peak_a.scale_lsb,
        },
        "peak_b": {
            "low_lsb": selection.peak_b.low_lsb,
            "high_lsb": selection.peak_b.high_lsb,
            "peak_lsb": selection.peak_b.peak_lsb,
            "center_lsb": selection.peak_b.center_lsb,
            "scale_lsb": selection.peak_b.scale_lsb,
        },
    }


def _energy_selection_from_cache(
    duration_mask: np.ndarray,
    metadata: dict[str, Any],
) -> EnergySelectionResult:
    return EnergySelectionResult(
        peak_a=PeakSelection(**metadata["peak_a"]),
        peak_b=PeakSelection(**metadata["peak_b"]),
        duration_mask=duration_mask,
    )


def run_pipeline(cfg: dict) -> None:
    pipeline_start = time.perf_counter()
    input_dir = Path(cfg["paths"]["input_dir"])
    output_dir = Path(cfg["paths"]["output_dir"])
    summary_path = output_dir / "summary.csv"
    analysis_root = output_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(
        input_dir,
        cfg["files"]["data_pattern"],
        bool(cfg["files"]["recursive"]),
    )
    include = _normalise_run_ids(cfg["runs"]["include"])
    overwrite_runs = _normalise_run_ids(cfg["runs"]["overwrite"])
    if include:
        runs = [run for run in runs if run.run_id in include]
    if not runs:
        raise RuntimeError(f"No binary runs found in {input_dir}")

    summary_rows: dict[str, dict[str, Any]] = _load_summary(summary_path)
    completed = cached = failed = 0

    for run in runs:
        run_start = time.perf_counter()
        run_dir = analysis_root / run.run_id
        state_path = run_dir / "state.json"
        pulse_cache_dir = run_dir / "streaming_pulses"
        candidate_index_dir = run_dir / "streaming_candidates"
        candidate_path = (
            run_dir / "candidate_preprocessed" / f"{run.run_id}_candidates.dat"
        )
        preprocessed_path = run_dir / "preprocessed" / f"{run.run_id}_list.dat"
        csv_dir = run_dir / "csv"
        models_dir = run_dir / "models"
        plots_dir = run_dir / "plots"
        energy_selection_path = table_path(csv_dir, "energy_selection", cfg)
        training_path = table_path(csv_dir, "matching_training", cfg)
        model_metrics_path = csv_dir / "matching_model_metrics.csv"
        matching_total_path = table_path(csv_dir, "matching_total", cfg)
        matching_core_path = run_dir / "cache" / "matching_core.npz"
        model_a_path = models_dir / "ch1_ch3_average_delay.json"
        model_b_path = models_dir / "ch5_ch7_average_delay.json"
        selection_path = table_path(csv_dir, "selection", cfg)
        fit_path = csv_dir / "fit.csv"
        matching_train_plot_path = plots_dir / "matching_model_train.png"
        matching_total_plot_path = plots_dir / "matching_model_total.png"
        peak_plot_path = plots_dir / "peak_selection.png"
        timing_plot_path = plots_dir / "timing_fit.png"

        state = load_state(state_path)
        migration_mode = int(state.get("cache_schema_version", 1)) < CACHE_SCHEMA_VERSION
        force_run = run.run_id in overwrite_runs
        any_completed = False
        state_changed = False

        try:
            run_info = parse_run_info(
                run.info_path,
                cfg["thresholds"]["consistency"],
            )
            raw_signature = file_signature(run.data_path)

            # 1. Build the single compact STREAMING matched-event candidate cache.
            pulse_cache_signature = None
            pulse_outputs: list[Path] = []
            if run_info.acquisition_mode == "STREAMING":
                configured_channels = {
                    int(cfg["channels"][key])
                    for key in ("signal_a", "time_a", "signal_b", "time_b")
                }
                pulse_outputs = pulse_cache_outputs(
                    pulse_cache_dir,
                    configured_channels,
                )
                pulse_cache_signature = signature(
                    {
                        "data": raw_signature,
                        "acquisition_mode": run_info.acquisition_mode,
                    },
                    stage_config(cfg, "streaming_pulse_decode"),
                )
                pulse_overwrite = force_run or bool(
                    cfg["tasks"]["streaming_pulse_decode"]["overwrite"]
                )
                if _reuse_stage(
                    state,
                    "streaming_pulse_decode",
                    [pulse_cache_signature],
                    pulse_outputs,
                    pulse_overwrite,
                    False,
                ):
                    pulse_metadata = state["stages"]["streaming_pulse_decode"][
                        "metadata"
                    ]
                    _log(
                        run.run_id,
                        "streaming_pulse_decode",
                        "SKIPPED — overwrite=false and compact event cache is valid",
                    )
                elif not cfg["tasks"]["streaming_pulse_decode"]["enabled"]:
                    raise RuntimeError(
                        "STREAMING pulse decode is disabled and no valid cache exists"
                    )
                else:
                    start = time.perf_counter()
                    _log(
                        run.run_id,
                        "streaming_pulse_decode",
                        "STARTED — ordered energy-leading matching and timing candidate indexing",
                    )
                    pulse_metadata = decode_streaming_pulse_cache(
                        run.data_path,
                        pulse_cache_dir,
                        cfg,
                        run_info.acquisition_mode,
                    )
                    mark_stage(
                        state,
                        "streaming_pulse_decode",
                        pulse_cache_signature,
                        pulse_outputs,
                        pulse_metadata,
                    )
                    save_state(state_path, state)
                    any_completed = True
                    pulse_count = sum(
                        int(item["pulses"])
                        for item in pulse_metadata["channels"].values()
                    )
                    duplicate_count = sum(
                        int(item["duplicate_leading_edges_removed"])
                        + int(item["duplicate_trailing_edges_removed"])
                        for item in pulse_metadata["channels"].values()
                    )
                    invalid_tot_count = sum(
                        int(item["rejected_tot_too_short"])
                        + int(item["rejected_tot_too_long"])
                        for item in pulse_metadata["channels"].values()
                    )
                    recorded_tot_pairs = sum(
                        int(item.get("recorded_tot_pairs", 0))
                        for item in pulse_metadata["channels"].values()
                    )
                    fallback_pairs = sum(
                        int(item.get("fallback_pairs", 0))
                        for item in pulse_metadata["channels"].values()
                    )
                    inconsistent_recorded_tot = sum(
                        int(item.get("recorded_tot_no_matching_lead", 0))
                        for item in pulse_metadata["channels"].values()
                    )
                    _log(
                        run.run_id,
                        "streaming_pulse_decode",
                        f"COMPLETED in {_elapsed(start)} — "
                        f"{pulse_metadata['raw_records_read']} blocks, "
                        f"{pulse_count} in-memory reconstructed/leading records, "
                        f"{recorded_tot_pairs} energy pulses paired from recorded ToT, "
                        f"{fallback_pairs} paired by missing-ToT fallback, "
                        f"{inconsistent_recorded_tot} recorded-ToT records without a compatible lead, "
                        f"{duplicate_count} duplicate edges removed, "
                        f"{invalid_tot_count} invalid-ToT pulses rejected",
                    )

            # 2. Candidate-preserving preprocessing: no timing choice is made here.
            candidate_stage_cfg = stage_config(cfg, "candidate_preprocessing")
            if run_info.acquisition_mode == "STREAMING":
                candidate_inputs = {
                    "pulse_cache": [file_signature(path) for path in pulse_outputs],
                    "acquisition_mode": run_info.acquisition_mode,
                }
                candidate_outputs = pulse_outputs
            else:
                candidate_stage_cfg = dict(candidate_stage_cfg)
                candidate_stage_cfg.pop("streaming_physical_time", None)
                candidate_inputs = {
                    "data": raw_signature,
                    "acquisition_mode": run_info.acquisition_mode,
                }
                candidate_outputs = [candidate_path]
            candidate_signature = signature(candidate_inputs, candidate_stage_cfg)
            candidate_overwrite = force_run or bool(
                cfg["tasks"]["candidate_preprocessing"]["overwrite"]
            )
            if _reuse_stage(
                state,
                "candidate_preprocessing",
                [candidate_signature],
                candidate_outputs,
                candidate_overwrite,
                False,
            ):
                candidate_metadata = state["stages"]["candidate_preprocessing"][
                    "metadata"
                ]
                toa_lsb_ps = float(candidate_metadata["toa_lsb_ps"])
                _log(
                    run.run_id,
                    "candidate_preprocessing",
                    "SKIPPED — overwrite=false and cached result is valid",
                )
            elif not cfg["tasks"]["candidate_preprocessing"]["enabled"]:
                raise RuntimeError(
                    "Candidate preprocessing is disabled and no valid cached result exists"
                )
            else:
                start = time.perf_counter()
                if run_info.acquisition_mode == "STREAMING":
                    _log(
                        run.run_id,
                        "candidate_preprocessing",
                        "STARTED — validating integrated compact candidate cache",
                    )
                    candidate_metadata = build_streaming_candidate_index(
                        pulse_cache_dir, candidate_index_dir, cfg
                    )
                else:
                    candidate_metadata = preprocess_candidates(
                        run.data_path,
                        candidate_path,
                        cfg,
                        run_info.acquisition_mode,
                    )
                mark_stage(
                    state,
                    "candidate_preprocessing",
                    candidate_signature,
                    candidate_outputs,
                    candidate_metadata,
                )
                save_state(state_path, state)
                toa_lsb_ps = float(candidate_metadata["toa_lsb_ps"])
                any_completed = True
                if run_info.acquisition_mode == "STREAMING":
                    pairing_note = (
                        f" — {candidate_metadata['energy_leading_pairs_total']} "
                        "ordered ch1-ch5 leading-edge pairs, "
                        f"{candidate_metadata['candidate_events']} events with "
                        "timing candidates on both sides, "
                        f"{candidate_metadata['candidate_references_a']} ch3 and "
                        f"{candidate_metadata['candidate_references_b']} ch7 references"
                    )
                else:
                    pairing_note = (
                        f" — {candidate_metadata['raw_records_read']} triggers, "
                        f"{candidate_metadata['candidate_events_written']} "
                        "candidate events"
                    )
                _log(
                    run.run_id,
                    "candidate_preprocessing",
                    f"COMPLETED in {_elapsed(start)}{pairing_note}",
                )

            # 3. Energy peak selection before timing matching.
            if run_info.acquisition_mode == "STREAMING":
                candidate_source_signature = [
                    file_signature(path) for path in candidate_outputs
                ]
            else:
                candidate_source_signature = file_signature(candidate_path)
            energy_selection_signature = signature(
                {"candidate_preprocessed": candidate_source_signature},
                stage_config(cfg, "energy_selection"),
            )
            energy_selection_outputs = [energy_selection_path]
            energy_selection_overwrite = force_run or bool(
                cfg["tasks"]["energy_selection"]["overwrite"]
            )
            if _reuse_stage(
                state,
                "energy_selection",
                [energy_selection_signature],
                energy_selection_outputs,
                energy_selection_overwrite,
                False,
            ):
                energy_measurements, duration_mask = load_energy_selection_csv(
                    energy_selection_path
                )
                energy_metadata = state["stages"]["energy_selection"]["metadata"]
                energy_selection = _energy_selection_from_cache(
                    duration_mask,
                    energy_metadata,
                )
                toa_lsb_ps = float(energy_metadata["toa_lsb_ps"])
                _log(
                    run.run_id,
                    "energy_selection",
                    "SKIPPED — overwrite=false and cached result is valid",
                )
            elif not cfg["tasks"]["energy_selection"]["enabled"]:
                raise RuntimeError(
                    "Energy selection is disabled and no valid cached result exists"
                )
            else:
                start = time.perf_counter()
                if run_info.acquisition_mode == "STREAMING":
                    energy_measurements, toa_lsb_ps = (
                        collect_streaming_energy_measurements(
                            pulse_cache_dir, candidate_index_dir, cfg
                        )
                    )
                else:
                    energy_measurements, toa_lsb_ps = collect_energy_measurements(
                        candidate_path, cfg
                    )
                energy_selection = select_energy_events(energy_measurements, cfg)
                write_energy_selection_csv(
                    energy_selection_path,
                    energy_measurements,
                    energy_selection,
                    cfg["analysis_output"]["diagnostic_mode"],
                )
                energy_metadata = _energy_selection_metadata(
                    energy_selection,
                    toa_lsb_ps,
                )
                mark_stage(
                    state,
                    "energy_selection",
                    energy_selection_signature,
                    energy_selection_outputs,
                    energy_metadata,
                )
                save_state(state_path, state)
                any_completed = True
                _log(
                    run.run_id,
                    "energy_selection",
                    f"COMPLETED in {_elapsed(start)}",
                )

            selected_event_indices = set(
                int(value)
                for value in energy_measurements.event_index[
                    energy_selection.duration_mask
                ]
            )
            if not selected_event_indices:
                raise RuntimeError("Energy peak selection retained no events")

            # Peak plot belongs to the pre-matching energy-selection stage.
            if cfg["plots"]["peak_selection"]["enabled"]:
                peak_plot_signature = signature(
                    {"energy_selection": file_signature(energy_selection_path)},
                    stage_config(cfg, "plot_peak_selection"),
                )
                peak_outputs = [peak_plot_path]
                peak_overwrite = force_run or bool(
                    cfg["plots"]["peak_selection"]["overwrite"]
                )
                if _reuse_stage(
                    state,
                    "plot_peak_selection",
                    [peak_plot_signature],
                    peak_outputs,
                    peak_overwrite,
                    False,
                ):
                    _log(
                        run.run_id,
                        "plot_peak_selection",
                        "SKIPPED — overwrite=false and cached result is valid",
                    )
                else:
                    start = time.perf_counter()
                    plot_peak_selection(
                        peak_plot_path,
                        run.run_id,
                        energy_measurements,
                        energy_selection,
                        toa_lsb_ps,
                        cfg,
                    )
                    mark_stage(
                        state,
                        "plot_peak_selection",
                        peak_plot_signature,
                        peak_outputs,
                    )
                    save_state(state_path, state)
                    any_completed = True
                    _log(
                        run.run_id,
                        "plot_peak_selection",
                        f"COMPLETED in {_elapsed(start)}",
                    )
            else:
                _log(run.run_id, "plot_peak_selection", "SKIPPED — disabled")

            # 3. Training labels are collected only from energy-selected events.
            training_signature = signature(
                {
                    "candidate_preprocessed": candidate_source_signature,
                    "energy_selection": file_signature(energy_selection_path),
                    "acquisition_mode": run_info.acquisition_mode,
                },
                stage_config(cfg, "matching_training"),
            )
            training_outputs = [training_path]
            training_overwrite = force_run or bool(
                cfg["tasks"]["matching_training"]["overwrite"]
            )
            if _reuse_stage(
                state,
                "matching_training",
                [training_signature],
                training_outputs,
                training_overwrite,
                False,
            ):
                matching_samples = load_training_csv(training_path)
                training_metadata = state["stages"]["matching_training"]["metadata"]
                toa_lsb_ps = float(training_metadata["toa_lsb_ps"])
                _log(
                    run.run_id,
                    "matching_training",
                    "SKIPPED — overwrite=false and cached result is valid",
                )
            elif not cfg["tasks"]["matching_training"]["enabled"]:
                raise RuntimeError(
                    "Matching training is disabled and no valid cached result exists"
                )
            else:
                start = time.perf_counter()
                if run_info.acquisition_mode == "STREAMING":
                    matching_samples, training_metadata = (
                        scan_streaming_matching_training(
                            pulse_cache_dir,
                            candidate_index_dir,
                            cfg,
                            selected_event_indices,
                        )
                    )
                else:
                    matching_samples, training_metadata = scan_matching_training(
                        candidate_path,
                        run_info.acquisition_mode,
                        cfg,
                        selected_event_indices,
                    )
                training_rows = training_metadata.pop("rows")
                write_training_csv(
                    training_path,
                    training_rows,
                    cfg["analysis_output"]["diagnostic_mode"],
                )
                mark_stage(
                    state,
                    "matching_training",
                    training_signature,
                    training_outputs,
                    training_metadata,
                )
                save_state(state_path, state)
                toa_lsb_ps = float(training_metadata["toa_lsb_ps"])
                any_completed = True
                _log(
                    run.run_id,
                    "matching_training",
                    f"COMPLETED in {_elapsed(start)}",
                )

            # 4. Per-run average-delay calibration.
            model_signature = signature(
                {"training": file_signature(training_path)},
                stage_config(cfg, "matching_model"),
            )
            model_outputs = [
                model_a_path,
                model_b_path,
                model_metrics_path,
            ]
            model_overwrite = force_run or bool(
                cfg["tasks"]["matching_model"]["overwrite"]
            )
            if _reuse_stage(
                state,
                "matching_model",
                [model_signature],
                model_outputs,
                model_overwrite,
                False,
            ):
                matching_models = {
                    "a": load_model(model_a_path),
                    "b": load_model(model_b_path),
                }
                filtered_matching_samples = matching_samples
                _log(
                    run.run_id,
                    "matching_model",
                    "SKIPPED — overwrite=false and cached result is valid",
                )
            elif not cfg["tasks"]["matching_model"]["enabled"]:
                raise RuntimeError(
                    "Matching model is disabled and no valid cached result exists"
                )
            else:
                start = time.perf_counter()
                matching_models, filtered_matching_samples = train_matching_models(
                    matching_samples,
                    cfg,
                )
                write_model(model_a_path, matching_models["a"])
                write_model(model_b_path, matching_models["b"])
                write_model_metrics(model_metrics_path, matching_models)
                mark_stage(
                    state,
                    "matching_model",
                    model_signature,
                    model_outputs,
                )
                save_state(state_path, state)
                any_completed = True
                _log(
                    run.run_id,
                    "matching_model",
                    f"COMPLETED in {_elapsed(start)}",
                )

            if cfg["plots"]["matching_train"]["enabled"]:
                train_plot_signature = signature(
                    {
                        "training": file_signature(training_path),
                        "model_a": file_signature(model_a_path),
                        "model_b": file_signature(model_b_path),
                        "energy_selection": file_signature(energy_selection_path),
                    },
                    stage_config(cfg, "plot_matching_train"),
                )
                train_plot_outputs = [matching_train_plot_path]
                train_plot_overwrite = force_run or bool(
                    cfg["plots"]["matching_train"]["overwrite"]
                )
                if _reuse_stage(
                    state,
                    "plot_matching_train",
                    [train_plot_signature],
                    train_plot_outputs,
                    train_plot_overwrite,
                    False,
                ):
                    _log(
                        run.run_id,
                        "plot_matching_train",
                        "SKIPPED — overwrite=false and cached result is valid",
                    )
                else:
                    start = time.perf_counter()
                    plot_matching_training(
                        matching_train_plot_path,
                        run.run_id,
                        filtered_matching_samples,
                        matching_models,
                        toa_lsb_ps,
                        cfg,
                        energy_selection,
                    )
                    mark_stage(
                        state,
                        "plot_matching_train",
                        train_plot_signature,
                        train_plot_outputs,
                    )
                    save_state(state_path, state)
                    any_completed = True
                    _log(
                        run.run_id,
                        "plot_matching_train",
                        f"COMPLETED in {_elapsed(start)}",
                    )
            else:
                _log(run.run_id, "plot_matching_train", "SKIPPED — disabled")

            # 5. Matching is finally applied only to energy-selected events.
            preprocessing_signature = signature(
                {
                    "candidate_preprocessed": candidate_source_signature,
                    "energy_selection": file_signature(energy_selection_path),
                    "model_a": file_signature(model_a_path),
                    "model_b": file_signature(model_b_path),
                    "acquisition_mode": run_info.acquisition_mode,
                },
                stage_config(cfg, "preprocessing"),
            )
            preprocessing_outputs = [preprocessed_path, matching_core_path, matching_total_path]
            preprocessing_overwrite = force_run or bool(
                cfg["tasks"]["preprocessing"]["overwrite"]
            )
            if _reuse_stage(
                state,
                "preprocessing",
                [preprocessing_signature],
                preprocessing_outputs,
                preprocessing_overwrite,
                False,
            ):
                _log(
                    run.run_id,
                    "preprocessing",
                    "SKIPPED — overwrite=false and cached result is valid",
                )
            elif not cfg["tasks"]["preprocessing"]["enabled"]:
                raise RuntimeError(
                    "Final matching preprocessing is disabled and no valid cached result exists"
                )
            else:
                start = time.perf_counter()
                if run_info.acquisition_mode == "STREAMING":
                    preprocessing_metadata, matching_total_rows = (
                        preprocess_streaming_from_index(
                            run.data_path,
                            pulse_cache_dir,
                            candidate_index_dir,
                            preprocessed_path,
                            cfg,
                            matching_models,
                            selected_event_indices,
                        )
                    )
                else:
                    preprocessing_metadata, matching_total_rows = preprocess_binary(
                        candidate_path,
                        preprocessed_path,
                        cfg,
                        run_info.acquisition_mode,
                        matching_models,
                        selected_event_indices,
                    )
                write_total_cache(matching_core_path, matching_total_rows)
                write_total_csv(
                    matching_total_path,
                    matching_total_rows,
                    cfg["analysis_output"]["diagnostic_mode"],
                )
                mark_stage(
                    state,
                    "preprocessing",
                    preprocessing_signature,
                    preprocessing_outputs,
                    preprocessing_metadata,
                )
                save_state(state_path, state)
                any_completed = True
                _log(
                    run.run_id,
                    "preprocessing",
                    f"COMPLETED in {_elapsed(start)}",
                )

            if cfg["plots"]["matching_total"]["enabled"]:
                total_plot_signature = signature(
                    {
                        "total_cache": file_signature(matching_core_path),
                        "total_diagnostic": file_signature(matching_total_path),
                        "model_a": file_signature(model_a_path),
                        "model_b": file_signature(model_b_path),
                        "energy_selection": file_signature(energy_selection_path),
                    },
                    stage_config(cfg, "plot_matching_total"),
                )
                total_plot_outputs = [matching_total_plot_path]
                total_plot_overwrite = force_run or bool(
                    cfg["plots"]["matching_total"]["overwrite"]
                )
                if _reuse_stage(
                    state,
                    "plot_matching_total",
                    [total_plot_signature],
                    total_plot_outputs,
                    total_plot_overwrite,
                    False,
                ):
                    _log(
                        run.run_id,
                        "plot_matching_total",
                        "SKIPPED — overwrite=false and cached result is valid",
                    )
                else:
                    start = time.perf_counter()
                    plot_matching_total(
                        matching_total_plot_path,
                        run.run_id,
                        load_total_cache(matching_core_path),
                        matching_models,
                        toa_lsb_ps,
                        cfg,
                        energy_selection,
                    )
                    mark_stage(
                        state,
                        "plot_matching_total",
                        total_plot_signature,
                        total_plot_outputs,
                    )
                    save_state(state_path, state)
                    any_completed = True
                    _log(
                        run.run_id,
                        "plot_matching_total",
                        f"COMPLETED in {_elapsed(start)}",
                    )
            else:
                _log(run.run_id, "plot_matching_total", "SKIPPED — disabled")

            # 6. Alignment selection and all timing analysis use the matched binary.
            selection_signature = signature(
                {
                    "preprocessed": file_signature(preprocessed_path),
                    "energy_selection": file_signature(energy_selection_path),
                },
                stage_config(cfg, "selection"),
            )
            selection_outputs = [selection_path]
            selection_overwrite = force_run or bool(
                cfg["tasks"]["selection"]["overwrite"]
            )
            if _reuse_stage(
                state,
                "selection",
                [selection_signature],
                selection_outputs,
                selection_overwrite,
                migration_mode,
            ):
                measurements, duration_mask, alignment_mask = load_selection_csv(
                    selection_path
                )
                metadata = state["stages"]["selection"]["metadata"]
                selection = _selection_from_cache(
                    measurements,
                    duration_mask,
                    alignment_mask,
                    metadata,
                )
                toa_lsb_ps = float(metadata["toa_lsb_ps"])
                state_changed |= _refresh_stage_signature(
                    state,
                    "selection",
                    selection_signature,
                    selection_outputs,
                )
                _log(
                    run.run_id,
                    "selection",
                    "SKIPPED — overwrite=false and cached result is valid",
                )
            elif not cfg["tasks"]["selection"]["enabled"]:
                raise RuntimeError(
                    "Post-matching selection is disabled and no valid cached result exists"
                )
            else:
                start = time.perf_counter()
                measurements, toa_lsb_ps = collect_measurements(
                    preprocessed_path,
                    cfg,
                )
                selection = select_matched_events(
                    measurements,
                    energy_selection.peak_a,
                    energy_selection.peak_b,
                    cfg,
                )
                write_selection_csv(
                    selection_path,
                    measurements,
                    selection,
                    cfg["analysis_output"]["diagnostic_mode"],
                )
                metadata = _selection_metadata(selection, toa_lsb_ps)
                mark_stage(
                    state,
                    "selection",
                    selection_signature,
                    selection_outputs,
                    metadata,
                )
                save_state(state_path, state)
                any_completed = True
                _log(run.run_id, "selection", f"COMPLETED in {_elapsed(start)}")

            selection_file_signature = file_signature(selection_path)
            fit_signature = signature(
                {"selection": selection_file_signature},
                stage_config(cfg, "fit"),
            )
            fit_outputs = [fit_path]
            fit_overwrite = force_run or bool(cfg["tasks"]["fit"]["overwrite"])
            if _reuse_stage(
                state,
                "fit",
                [fit_signature],
                fit_outputs,
                fit_overwrite,
                migration_mode,
            ):
                fit = load_fit_csv(fit_path)
                state_changed |= _refresh_stage_signature(
                    state,
                    "fit",
                    fit_signature,
                    fit_outputs,
                )
                _log(
                    run.run_id,
                    "fit",
                    "SKIPPED — overwrite=false and cached result is valid",
                )
            elif not cfg["tasks"]["fit"]["enabled"]:
                raise RuntimeError("Fit is disabled and no valid cached result exists")
            else:
                start = time.perf_counter()
                fit = fit_timing(
                    measurements.timing_lsb[selection.final_mask],
                    cfg["fit"],
                )
                write_fit_csv(
                    fit_path, fit, cfg["analysis_output"]["diagnostic_mode"]
                )
                mark_stage(state, "fit", fit_signature, fit_outputs)
                save_state(state_path, state)
                any_completed = True
                _log(run.run_id, "fit", f"COMPLETED in {_elapsed(start)}")

            if cfg["plots"]["timing_fit"]["enabled"]:
                timing_plot_signature = signature(
                    file_signature(fit_path),
                    stage_config(cfg, "plot_timing_fit"),
                )
                timing_outputs = [timing_plot_path]
                timing_overwrite = force_run or bool(
                    cfg["plots"]["timing_fit"]["overwrite"]
                )
                if _reuse_stage(
                    state,
                    "plot_timing_fit",
                    [timing_plot_signature],
                    timing_outputs,
                    timing_overwrite,
                    migration_mode,
                ):
                    state_changed |= _refresh_stage_signature(
                        state,
                        "plot_timing_fit",
                        timing_plot_signature,
                        timing_outputs,
                    )
                    _log(
                        run.run_id,
                        "plot_timing_fit",
                        "SKIPPED — overwrite=false and cached result is valid",
                    )
                else:
                    start = time.perf_counter()
                    plot_timing_fit(
                        timing_plot_path,
                        run.run_id,
                        fit,
                        toa_lsb_ps,
                        cfg,
                    )
                    mark_stage(
                        state,
                        "plot_timing_fit",
                        timing_plot_signature,
                        timing_outputs,
                    )
                    save_state(state_path, state)
                    any_completed = True
                    _log(
                        run.run_id,
                        "plot_timing_fit",
                        f"COMPLETED in {_elapsed(start)}",
                    )
            else:
                _log(run.run_id, "plot_timing_fit", "SKIPPED — disabled")

        
            average_delay_corrected_alignments = count_model_corrected_alignments(
                load_total_cache(matching_core_path),
                center_a_lsb=selection.alignment_a_center_lsb,
                scale_a_lsb=selection.alignment_a_scale_lsb,
                center_b_lsb=selection.alignment_b_center_lsb,
                scale_b_lsb=selection.alignment_b_scale_lsb,
                z_threshold=float(cfg["alignment_filter"]["z_threshold"]),
            )
            new_summary_row = _summary_row(
                run,
                run_info,
                fit,
                toa_lsb_ps,
                average_delay_corrected_alignments,
            )
            if _summary_rows_equal(summary_rows.get(run.run_id), new_summary_row):
                _log(run.run_id, "summary", "SKIPPED — unchanged")
            else:
                summary_rows[run.run_id] = new_summary_row
                _write_summary(summary_path, summary_rows)
                any_completed = True
                _log(run.run_id, "summary", "UPDATED")

            if migration_mode or state_changed:
                state["cache_schema_version"] = CACHE_SCHEMA_VERSION
                save_state(state_path, state)

            if any_completed:
                completed += 1
            else:
                cached += 1
            _log(run.run_id, "run", f"COMPLETED in {_elapsed(run_start)}")
        except Exception as exc:
            failed += 1
            _log(run.run_id, "run", f"FAILED — {exc}")

    print(
        "Pipeline completed\n"
        f"Runs discovered: {len(runs)}\n"
        f"Runs with completed work: {completed}\n"
        f"Runs fully cached: {cached}\n"
        f"Runs failed: {failed}\n"
        f"Total time: {_elapsed(pipeline_start)}",
        flush=True,
    )
    if failed:
        raise RuntimeError(f"{failed} run(s) failed")
