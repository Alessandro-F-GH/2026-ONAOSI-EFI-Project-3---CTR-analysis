from __future__ import annotations

import sys

import hashlib
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils_fit import fit_delta_times_ps
from utils_fit.io import load_fit_csv, write_fit_csv
from utils_fit.plotting import plot_gaussian_fit

from .binary_io import (
    HEADER_SIZE,
    DataError,
    atomic_write_csv,
    canonical_run_id,
    discover_runs,
    file_signature,
    parse_run_info,
    read_csv,
    read_meta,
)
from .cache import load_state, mark_stage, save_state, signature, stage_valid_any
from .config import stage_config
from .tabular import table_path
from .matching import (
    MatchingSamples,
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
from .models import (
    EnergyMeasurements,
    EnergySelectionResult,
    PeakSelection,
    SelectionResult,
)
from .plotting import (
    plot_matching_total,
    plot_matching_training,
    plot_peak_selection,
)
from .preprocessing import preprocess_binary, preprocess_streaming_from_index
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
from .streaming_cache import (
    candidate_index_outputs,
    collect_streaming_energy_measurements,
    pulse_cache_outputs,
)

_RUN_NAME_RE = re.compile(r"(?i)^Run[_-]?(\d+)$")


@dataclass(slots=True)
class CompatibleRun:
    run_id: str
    run_number: str
    voltage: int
    acquisition_mode: str
    energy_threshold_mv: float
    timing_threshold_mv: float
    data_path: Path
    info_path: Path
    output_dir: Path
    toa_lsb_ps: float
    tot_lsb_ps: float
    measurement_mode: int
    candidate_kind: str
    candidate_path: Path | None
    pulse_cache_dir: Path | None
    candidate_index_dir: Path | None
    energy_measurements: EnergyMeasurements | None = None
    global_offset: int = 0
    global_stop: int = 0


MANIFEST_FIELDS = [
    "run_id",
    "run_number",
    "Voltage",
    "AcquisitionMode",
    "E_th",
    "T_th",
    "toa_lsb_ps",
    "tot_lsb_ps",
    "measurement_mode",
    "candidate_kind",
    "candidate_source",
    "candidate_events",
    "global_event_offset",
    "global_event_stop",
]

GROUP_SUMMARY_FILENAME = "grouped_results_summary.csv"
GROUP_SUMMARY_FIELDS = [
    "group_id",
    "group",
    "group_status_code",
    "group_status",
    "error",
    "run_count",
    "run_ids",
    "Voltage",
    "AcquisitionMode",
    "E_th",
    "T_th",
    "fit_metric",
    "measurement_mode",
    "toa_lsb_ps",
    "tot_lsb_ps",
    "candidate_events",
    "energy_selected_events",
    "energy_selection_fraction",
    "matched_events",
    "matching_efficiency",
    "fit_selected_events",
    "fit_selection_fraction",
    "overall_selection_fraction",
    "peak_a_low_ps",
    "peak_a_high_ps",
    "peak_b_low_ps",
    "peak_b_high_ps",
    "alignment_a_center_ps",
    "alignment_a_scale_ps",
    "alignment_b_center_ps",
    "alignment_b_scale_ps",
    "average_delay_a_ps",
    "average_delay_a_std_ps",
    "average_delay_a_training_events",
    "average_delay_b_ps",
    "average_delay_b_std_ps",
    "average_delay_b_training_events",
    "fit_success",
    "fit_status",
    "gaussian_area_events",
    "gaussian_area_error_events",
    "gaussian_mean_ps",
    "gaussian_mean_error_ps",
    "gaussian_sigma_ps",
    "gaussian_sigma_error_ps",
    "CTR_ps",
    "CTR_error_ps",
    "chi_square",
    "ndof",
    "reduced_chi_square",
    "average_delay_corrected_alignments",
    "result_dir",
]

GROUP_STATUS_COMPLETE = 0
GROUP_STATUS_FAILED = 1



def _log(group_name: str, stage: str, message: str) -> None:
    print(f"[{group_name}][{stage}] {message}", flush=True)


def _elapsed(start: float) -> str:
    return f"{time.perf_counter() - start:.2f} s"


def _format_token(value: float | int) -> str:
    number = float(value)
    if math.isclose(number, round(number), rel_tol=0.0, abs_tol=1e-9):
        return str(int(round(number)))
    return f"{number:.6g}".replace("-", "m").replace(".", "p")


def group_name(
    voltage: int,
    timing_threshold_mv: float,
    energy_threshold_mv: float,
    acquisition_mode: str | None = None,
    measurement_mode: int | None = None,
    toa_lsb_ps: float | None = None,
    tot_lsb_ps: float | None = None,
) -> str:
    name = (
        f"{int(voltage)}V_"
        f"Ti{_format_token(timing_threshold_mv)}_"
        f"E{_format_token(energy_threshold_mv)}"
    )
    if acquisition_mode is None:
        return name
    return (
        f"{name}_{str(acquisition_mode).upper()}_"
        f"M{int(measurement_mode) if measurement_mode is not None else 'unknown'}_"
        f"ToA{_format_token(toa_lsb_ps if toa_lsb_ps is not None else 0)}ps_"
        f"ToT{_format_token(tot_lsb_ps if tot_lsb_ps is not None else 0)}ps"
    )


def _find_run_dir(path: str | Path) -> Path:
    current = Path(path).expanduser().resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if _RUN_NAME_RE.fullmatch(candidate.name):
            return candidate
    raise DataError(
        f"Cannot find a RunXXXX output directory in {Path(path)!s}"
    )


def _float_equal(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-9)


def _array_digest(values: np.ndarray | list[int] | set[int]) -> str:
    array = np.asarray(sorted(values) if isinstance(values, set) else values, dtype=np.int64)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _files_signature(paths: list[Path]) -> list[dict[str, Any]]:
    return [file_signature(path) for path in paths]


def _filter_configured_runs(runs: list[Any], cfg: dict) -> list[Any]:
    include = {
        canonical_run_id(str(value))[0]
        for value in cfg.get("runs", {}).get("include", [])
    }
    if not include:
        return runs
    return [run for run in runs if run.run_id in include]


def _energy_metadata(selection: EnergySelectionResult, toa_lsb_ps: float) -> dict[str, Any]:
    return {
        "toa_lsb_ps": float(toa_lsb_ps),
        "peak_a": {
            "low_lsb": int(selection.peak_a.low_lsb),
            "high_lsb": int(selection.peak_a.high_lsb),
            "peak_lsb": float(selection.peak_a.peak_lsb),
            "center_lsb": float(selection.peak_a.center_lsb),
            "scale_lsb": float(selection.peak_a.scale_lsb),
        },
        "peak_b": {
            "low_lsb": int(selection.peak_b.low_lsb),
            "high_lsb": int(selection.peak_b.high_lsb),
            "peak_lsb": float(selection.peak_b.peak_lsb),
            "center_lsb": float(selection.peak_b.center_lsb),
            "scale_lsb": float(selection.peak_b.scale_lsb),
        },
    }


def _energy_from_cache(mask: np.ndarray, metadata: dict[str, Any]) -> EnergySelectionResult:
    return EnergySelectionResult(
        peak_a=PeakSelection(**metadata["peak_a"]),
        peak_b=PeakSelection(**metadata["peak_b"]),
        duration_mask=np.asarray(mask, dtype=bool),
    )


def _selection_metadata(selection: SelectionResult, toa_lsb_ps: float) -> dict[str, Any]:
    metadata = _energy_metadata(
        EnergySelectionResult(selection.peak_a, selection.peak_b, selection.duration_mask),
        toa_lsb_ps,
    )
    metadata.update(
        {
            "alignment_a_center_lsb": float(selection.alignment_a_center_lsb),
            "alignment_a_scale_lsb": float(selection.alignment_a_scale_lsb),
            "alignment_b_center_lsb": float(selection.alignment_b_center_lsb),
            "alignment_b_scale_lsb": float(selection.alignment_b_scale_lsb),
        }
    )
    return metadata


def _selection_from_cache(
    duration_mask: np.ndarray,
    alignment_mask: np.ndarray,
    metadata: dict[str, Any],
) -> SelectionResult:
    duration_mask = np.asarray(duration_mask, dtype=bool)
    alignment_mask = np.asarray(alignment_mask, dtype=bool)
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


def _required_candidate_outputs(run: CompatibleRun, cfg: dict) -> list[Path]:
    if run.acquisition_mode == "STREAMING":
        channels = {
            int(cfg["channels"][key])
            for key in ("signal_a", "time_a", "signal_b", "time_b")
        }
        assert run.pulse_cache_dir is not None
        assert run.candidate_index_dir is not None
        return [
            *pulse_cache_outputs(run.pulse_cache_dir, channels),
            *candidate_index_outputs(run.candidate_index_dir),
        ]
    assert run.candidate_path is not None
    return [run.candidate_path]


def _candidate_signature(run: CompatibleRun, cfg: dict) -> list[dict[str, Any]]:
    return _files_signature(_required_candidate_outputs(run, cfg))


def _discover_compatible_runs(
    main_run_dir: Path,
    cfg: dict,
    skip_missing: bool,
) -> list[CompatibleRun]:
    main_run_id, _ = canonical_run_id(main_run_dir.name)
    analysis_root = main_run_dir.parent
    discovered = _filter_configured_runs(
        discover_runs(
            cfg["paths"]["input_dir"],
            cfg["files"]["data_pattern"],
            bool(cfg["files"]["recursive"]),
        ),
        cfg,
    )
    run_map = {run.run_id: run for run in discovered}
    if main_run_id not in run_map:
        raise DataError(
            f"{main_run_id} is not present in configured input directory "
            f"{cfg['paths']['input_dir']}"
        )

    main_input = run_map[main_run_id]
    main_info = parse_run_info(main_input.info_path, cfg["thresholds"]["consistency"])
    main_meta = read_meta(main_input.data_path, main_info.acquisition_mode)

    compatible: list[CompatibleRun] = []
    missing: list[str] = []
    for run_input in discovered:
        run_info = parse_run_info(
            run_input.info_path, cfg["thresholds"]["consistency"]
        )
        if run_input.voltage != main_input.voltage:
            continue
        if run_info.acquisition_mode != main_info.acquisition_mode:
            continue
        if not _float_equal(
            run_info.energy_threshold_mv, main_info.energy_threshold_mv
        ):
            continue
        if not _float_equal(
            run_info.timing_threshold_mv, main_info.timing_threshold_mv
        ):
            continue
        meta = read_meta(run_input.data_path, run_info.acquisition_mode)
        if meta.measurement_mode != main_meta.measurement_mode:
            continue
        if not _float_equal(meta.toa_lsb_ps, main_meta.toa_lsb_ps):
            continue
        if not _float_equal(meta.tot_lsb_ps, main_meta.tot_lsb_ps):
            continue

        output_dir = analysis_root / run_input.run_id
        if run_info.acquisition_mode == "STREAMING":
            candidate_kind = "streaming_event_cache"
            candidate_path = None
            pulse_cache_dir = output_dir / "streaming_pulses"
            candidate_index_dir = output_dir / "streaming_candidates"
        else:
            candidate_kind = "candidate_binary"
            candidate_path = (
                output_dir
                / "candidate_preprocessed"
                / f"{run_input.run_id}_candidates.dat"
            )
            pulse_cache_dir = None
            candidate_index_dir = None

        candidate = CompatibleRun(
            run_id=run_input.run_id,
            run_number=run_input.run_number,
            voltage=run_input.voltage,
            acquisition_mode=run_info.acquisition_mode,
            energy_threshold_mv=run_info.energy_threshold_mv,
            timing_threshold_mv=run_info.timing_threshold_mv,
            data_path=run_input.data_path,
            info_path=run_input.info_path,
            output_dir=output_dir,
            toa_lsb_ps=meta.toa_lsb_ps,
            tot_lsb_ps=meta.tot_lsb_ps,
            measurement_mode=meta.measurement_mode,
            candidate_kind=candidate_kind,
            candidate_path=candidate_path,
            pulse_cache_dir=pulse_cache_dir,
            candidate_index_dir=candidate_index_dir,
        )
        required = _required_candidate_outputs(candidate, cfg)
        absent = [path for path in required if not path.exists()]
        if absent:
            missing.append(
                f"{run_input.run_id}: " + ", ".join(str(path) for path in absent)
            )
            continue
        compatible.append(candidate)

    if missing and not skip_missing:
        raise RuntimeError(
            "Compatible runs are missing candidate-preprocessing outputs. "
            "Run main.py for those runs first, or use --skip-missing:\n  "
            + "\n  ".join(missing)
        )
    if not compatible:
        raise RuntimeError("No compatible runs with valid preprocessed candidates found")
    if main_run_id not in {run.run_id for run in compatible}:
        raise RuntimeError(f"Main run {main_run_id} has no valid candidate outputs")
    return sorted(compatible, key=lambda item: int(item.run_number))


def _collect_group_energy(
    runs: list[CompatibleRun], cfg: dict
) -> EnergyMeasurements:
    event_parts: list[np.ndarray] = []
    duration_a_parts: list[np.ndarray] = []
    duration_b_parts: list[np.ndarray] = []
    energy_a_parts: list[np.ndarray] = []
    energy_b_parts: list[np.ndarray] = []
    offset = 0
    for run in runs:
        if run.acquisition_mode == "STREAMING":
            assert run.pulse_cache_dir is not None
            assert run.candidate_index_dir is not None
            measurements, toa_lsb_ps = collect_streaming_energy_measurements(
                run.pulse_cache_dir, run.candidate_index_dir, cfg
            )
        else:
            assert run.candidate_path is not None
            measurements, toa_lsb_ps = collect_energy_measurements(
                run.candidate_path, cfg
            )
        if not _float_equal(toa_lsb_ps, run.toa_lsb_ps):
            raise DataError(f"Unexpected ToA LSB for {run.run_id}")
        run.energy_measurements = measurements
        run.global_offset = offset
        run.global_stop = offset + measurements.size
        event_parts.append(np.arange(offset, run.global_stop, dtype=np.int64))
        duration_a_parts.append(measurements.duration_a_lsb)
        duration_b_parts.append(measurements.duration_b_lsb)
        energy_a_parts.append(measurements.energy_a_lsb)
        energy_b_parts.append(measurements.energy_b_lsb)
        offset = run.global_stop

    return EnergyMeasurements(
        event_index=np.concatenate(event_parts) if event_parts else np.empty(0, dtype=np.int64),
        duration_a_lsb=np.concatenate(duration_a_parts) if duration_a_parts else np.empty(0, dtype=np.int64),
        duration_b_lsb=np.concatenate(duration_b_parts) if duration_b_parts else np.empty(0, dtype=np.int64),
        energy_a_lsb=np.concatenate(energy_a_parts) if energy_a_parts else np.empty(0, dtype=np.int64),
        energy_b_lsb=np.concatenate(energy_b_parts) if energy_b_parts else np.empty(0, dtype=np.int64),
    )


def _restore_run_offsets_from_manifest(
    runs: list[CompatibleRun], manifest_path: Path
) -> None:
    rows = {row["run_id"]: row for row in read_csv(manifest_path)}
    for run in runs:
        row = rows.get(run.run_id)
        if row is None:
            raise DataError(f"Run {run.run_id} is missing from cached group manifest")
        run.global_offset = int(row["global_event_offset"])
        run.global_stop = int(row["global_event_stop"])


def _write_manifest(path: Path, runs: list[CompatibleRun]) -> None:
    rows: list[dict[str, Any]] = []
    for run in runs:
        source = (
            run.pulse_cache_dir
            if run.acquisition_mode == "STREAMING"
            else run.candidate_path
        )
        rows.append(
            {
                "run_id": run.run_id,
                "run_number": run.run_number,
                "Voltage": run.voltage,
                "AcquisitionMode": run.acquisition_mode,
                "E_th": run.energy_threshold_mv,
                "T_th": run.timing_threshold_mv,
                "toa_lsb_ps": run.toa_lsb_ps,
                "tot_lsb_ps": run.tot_lsb_ps,
                "measurement_mode": run.measurement_mode,
                "candidate_kind": run.candidate_kind,
                "candidate_source": str(source),
                "candidate_events": run.global_stop - run.global_offset,
                "global_event_offset": run.global_offset,
                "global_event_stop": run.global_stop,
            }
        )
    atomic_write_csv(path, MANIFEST_FIELDS, rows)


def _selected_local_indices(
    run: CompatibleRun,
    measurements: EnergyMeasurements,
    mask: np.ndarray,
    cfg: dict,
) -> set[int]:
    local_mask = np.asarray(mask[run.global_offset : run.global_stop], dtype=bool)
    if run.energy_measurements is None:
        # Re-read only the compact candidate representation to recover exact local
        # event indices. This matters when a candidate binary contains a skipped or
        # malformed event and keeps cached group selection scientifically correct.
        if run.acquisition_mode == "STREAMING":
            assert run.pulse_cache_dir is not None
            assert run.candidate_index_dir is not None
            local_measurements, _ = collect_streaming_energy_measurements(
                run.pulse_cache_dir, run.candidate_index_dir, cfg
            )
        else:
            assert run.candidate_path is not None
            local_measurements, _ = collect_energy_measurements(run.candidate_path, cfg)
        run.energy_measurements = local_measurements
    local_event_index = run.energy_measurements.event_index
    if local_event_index.size != local_mask.size:
        raise DataError(
            f"Cached group manifest is inconsistent with {run.run_id}: "
            f"{local_mask.size} pooled rows versus {local_event_index.size} local rows"
        )
    return set(int(value) for value in local_event_index[local_mask])


def _collect_group_training(
    runs: list[CompatibleRun],
    cfg: dict,
    selected_by_run: dict[str, set[int]],
) -> dict[str, MatchingSamples]:
    parts: dict[str, dict[str, list[np.ndarray]]] = {
        pair: {"event": [], "duration": [], "delay": []} for pair in ("a", "b")
    }
    for run in runs:
        selected = selected_by_run[run.run_id]
        if run.acquisition_mode == "STREAMING":
            assert run.pulse_cache_dir is not None
            assert run.candidate_index_dir is not None
            samples, _ = scan_streaming_matching_training(
                run.pulse_cache_dir,
                run.candidate_index_dir,
                cfg,
                selected,
            )
        else:
            assert run.candidate_path is not None
            samples, _ = scan_matching_training(
                run.candidate_path,
                run.acquisition_mode,
                cfg,
                selected,
            )
        for pair in ("a", "b"):
            local = samples[pair]
            parts[pair]["event"].append(
                local.event_index.astype(np.int64, copy=False) + run.global_offset
            )
            parts[pair]["duration"].append(local.energy_duration_lsb)
            parts[pair]["delay"].append(local.delay_lsb)

    result: dict[str, MatchingSamples] = {}
    for pair in ("a", "b"):
        result[pair] = MatchingSamples(
            event_index=np.concatenate(parts[pair]["event"])
            if parts[pair]["event"]
            else np.empty(0, dtype=np.int64),
            energy_duration_lsb=np.concatenate(parts[pair]["duration"])
            if parts[pair]["duration"]
            else np.empty(0, dtype=np.int64),
            delay_lsb=np.concatenate(parts[pair]["delay"])
            if parts[pair]["delay"]
            else np.empty(0, dtype=np.int64),
        )
    return result


def _training_rows(samples: dict[str, MatchingSamples], cfg: dict) -> list[dict[str, Any]]:
    channel_pairs = {
        "a": (int(cfg["channels"]["signal_a"]), int(cfg["channels"]["time_a"])),
        "b": (int(cfg["channels"]["signal_b"]), int(cfg["channels"]["time_b"])),
    }
    rows: list[dict[str, Any]] = []
    for pair in ("a", "b"):
        energy_channel, timing_channel = channel_pairs[pair]
        pair_samples = samples[pair]
        rows.extend(
            {
                "event_index": int(event_index),
                "pair": pair,
                "energy_channel": energy_channel,
                "timing_channel": timing_channel,
                "energy_duration_lsb": int(duration),
                "delay_lsb": int(delay),
                "energy_leading_lsb": "",
                "timing_leading_lsb": "",
            }
            for event_index, duration, delay in zip(
                pair_samples.event_index,
                pair_samples.energy_duration_lsb,
                pair_samples.delay_lsb,
            )
        )
    return rows


def _concatenate_binary_files(
    sources: list[Path], destination: Path, expected_mode: str
) -> dict[str, Any]:
    if not sources:
        raise RuntimeError("No matched per-run binaries are available to concatenate")
    metas = [read_meta(path, expected_mode) for path in sources]
    reference = metas[0]
    for path, meta in zip(sources[1:], metas[1:]):
        if (
            meta.acquisition_mode != reference.acquisition_mode
            or meta.measurement_mode != reference.measurement_mode
            or meta.time_unit != reference.time_unit
            or not _float_equal(meta.toa_lsb_ps, reference.toa_lsb_ps)
            or not _float_equal(meta.tot_lsb_ps, reference.tot_lsb_ps)
        ):
            raise DataError(f"Incompatible matched binary header: {path}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(".group_list.tmp")
    total_payload_bytes = 0
    try:
        with temporary.open("wb") as target:
            target.write(reference.raw_header)
            for source_path in sources:
                with source_path.open("rb") as source:
                    source.seek(HEADER_SIZE)
                    while True:
                        block = source.read(1024 * 1024)
                        if not block:
                            break
                        target.write(block)
                        total_payload_bytes += len(block)
            target.flush()
            os.fsync(target.fileno())
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "source_files": len(sources),
        "payload_bytes": total_payload_bytes,
        "toa_lsb_ps": reference.toa_lsb_ps,
        "acquisition_mode": expected_mode,
    }


def _write_group_total(
    path: Path,
    rows: list[dict[str, Any]],
    cfg: dict,
) -> None:
    write_total_csv(
        path, rows, cfg["analysis_output"]["diagnostic_mode"]
    )


def _finite_number(value: Any) -> float | str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return number if math.isfinite(number) else ""


def _ratio(numerator: int | float, denominator: int | float) -> float | str:
    denominator = float(denominator)
    if denominator <= 0:
        return ""
    return float(numerator) / denominator


def _group_id(runs: list[CompatibleRun]) -> str:
    first = runs[0]
    return (
        f"V{first.voltage}|mode={first.acquisition_mode}|"
        f"E={first.energy_threshold_mv:.12g}|T={first.timing_threshold_mv:.12g}|"
        f"measurement={first.measurement_mode}|"
        f"toa={first.toa_lsb_ps:.12g}|tot={first.tot_lsb_ps:.12g}"
    )


def _summary_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    def number(name: str) -> float:
        try:
            return float(row.get(name, math.inf))
        except (TypeError, ValueError):
            return math.inf

    return (
        number("Voltage"),
        number("E_th"),
        number("T_th"),
        str(row.get("AcquisitionMode", "")),
        number("measurement_mode"),
        number("toa_lsb_ps"),
        str(row.get("group_id", "")),
    )


def _update_group_summary(path: Path, row: dict[str, Any]) -> None:
    existing: dict[str, dict[str, Any]] = {}
    if path.exists():
        for previous in read_csv(path):
            key = str(previous.get("group_id", previous.get("group", "")))
            if key:
                existing[key] = previous
    existing[str(row["group_id"])] = row
    ordered = sorted(existing.values(), key=_summary_sort_key)
    atomic_write_csv(path, GROUP_SUMMARY_FIELDS, ordered)


def _build_summary_row(
    group_dir: Path,
    group: str,
    runs: list[CompatibleRun],
    energy_measurements: EnergyMeasurements,
    energy_selection: EnergySelectionResult,
    measurements,
    selection: SelectionResult,
    fit,
    models: dict[str, Any],
    toa_lsb_ps: float,
    matching_total_rows: list[dict[str, Any]],
    cfg: dict,
) -> dict[str, Any]:
    candidate_events = int(energy_measurements.size)
    energy_selected_events = int(np.count_nonzero(energy_selection.duration_mask))
    matched_events = int(measurements.size)
    fit_selected_events = int(np.count_nonzero(selection.final_mask))
    model_a = models["a"]
    model_b = models["b"]
    first = runs[0]
    return {
        "group_id": _group_id(runs),
        "group": group,
        "group_status_code": GROUP_STATUS_COMPLETE,
        "group_status": "complete",
        "error": "",
        "run_count": len(runs),
        "run_ids": ";".join(run.run_id for run in runs),
        "Voltage": first.voltage,
        "AcquisitionMode": first.acquisition_mode,
        "E_th": first.energy_threshold_mv,
        "T_th": first.timing_threshold_mv,
        "fit_metric": "common_bin_integrated_gaussian_all_events",
        "measurement_mode": first.measurement_mode,
        "toa_lsb_ps": first.toa_lsb_ps,
        "tot_lsb_ps": first.tot_lsb_ps,
        "candidate_events": candidate_events,
        "energy_selected_events": energy_selected_events,
        "energy_selection_fraction": _ratio(energy_selected_events, candidate_events),
        "matched_events": matched_events,
        "matching_efficiency": _ratio(matched_events, energy_selected_events),
        "fit_selected_events": fit_selected_events,
        "fit_selection_fraction": _ratio(fit_selected_events, matched_events),
        "overall_selection_fraction": _ratio(fit_selected_events, candidate_events),
        "peak_a_low_ps": energy_selection.peak_a.low_lsb * toa_lsb_ps,
        "peak_a_high_ps": energy_selection.peak_a.high_lsb * toa_lsb_ps,
        "peak_b_low_ps": energy_selection.peak_b.low_lsb * toa_lsb_ps,
        "peak_b_high_ps": energy_selection.peak_b.high_lsb * toa_lsb_ps,
        "alignment_a_center_ps": selection.alignment_a_center_lsb * toa_lsb_ps,
        "alignment_a_scale_ps": selection.alignment_a_scale_lsb * toa_lsb_ps,
        "alignment_b_center_ps": selection.alignment_b_center_lsb * toa_lsb_ps,
        "alignment_b_scale_ps": selection.alignment_b_scale_lsb * toa_lsb_ps,
        "average_delay_a_ps": model_a.average_delay_lsb * toa_lsb_ps,
        "average_delay_a_std_ps": model_a.delay_std_lsb * toa_lsb_ps,
        "average_delay_a_training_events": model_a.training_samples,
        "average_delay_b_ps": model_b.average_delay_lsb * toa_lsb_ps,
        "average_delay_b_std_ps": model_b.delay_std_lsb * toa_lsb_ps,
        "average_delay_b_training_events": model_b.training_samples,
        "fit_success": int(bool(fit.success)),
        "fit_status": "success" if fit.success else str(fit.message),
        "gaussian_area_events": _finite_number(fit.n_fit),
        "gaussian_area_error_events": "",
        "gaussian_mean_ps": _finite_number(fit.mean_ps),
        "gaussian_mean_error_ps": _finite_number(fit.mean_error_ps),
        "gaussian_sigma_ps": _finite_number(fit.sigma_ps),
        "gaussian_sigma_error_ps": _finite_number(fit.sigma_error_ps),
        "CTR_ps": _finite_number(fit.ctr_ps),
        "CTR_error_ps": _finite_number(fit.ctr_error_ps),
        "chi_square": _finite_number(fit.chi2),
        "ndof": fit.ndof,
        "reduced_chi_square": _finite_number(fit.chi2_ndof),
        "average_delay_corrected_alignments": count_model_corrected_alignments(
            matching_total_rows,
            center_a_lsb=selection.alignment_a_center_lsb,
            scale_a_lsb=selection.alignment_a_scale_lsb,
            center_b_lsb=selection.alignment_b_center_lsb,
            scale_b_lsb=selection.alignment_b_scale_lsb,
            z_threshold=float(cfg["alignment_filter"]["z_threshold"]),
        ),
        "result_dir": str(group_dir),
    }


def _failed_summary_row(
    group_dir: Path,
    group: str,
    runs: list[CompatibleRun],
    error: Exception,
) -> dict[str, Any]:
    first = runs[0]
    row = {field: "" for field in GROUP_SUMMARY_FIELDS}
    row.update(
        {
            "group_id": _group_id(runs),
            "group": group,
            "group_status_code": GROUP_STATUS_FAILED,
            "group_status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "run_count": len(runs),
            "run_ids": ";".join(run.run_id for run in runs),
            "Voltage": first.voltage,
            "AcquisitionMode": first.acquisition_mode,
            "E_th": first.energy_threshold_mv,
            "T_th": first.timing_threshold_mv,
            "measurement_mode": first.measurement_mode,
            "toa_lsb_ps": first.toa_lsb_ps,
            "tot_lsb_ps": first.tot_lsb_ps,
            "result_dir": str(group_dir),
        }
    )
    return row


def run_compatible_group_analysis(
    main_run_output: str | Path,
    cfg: dict,
    *,
    overwrite: bool = False,
    skip_missing: bool = False,
    output_root: str | Path | None = None,
) -> Path:
    """Pool compatible candidate-preprocessed runs and run one global analysis.

    Compatibility requires equal voltage, acquisition mode, energy/timing
    thresholds, measurement mode and ToA/ToT LSB. Energy peak selection and the
    matching model are estimated from the pooled candidates. The global model is
    then applied separately to each run, the matched binaries are concatenated,
    and post-matching selection plus timing fit are performed on the concatenated
    dataset.
    """
    main_run_dir = _find_run_dir(main_run_output)
    runs = _discover_compatible_runs(main_run_dir, cfg, skip_missing)
    first = runs[0]
    name = group_name(
        first.voltage,
        first.timing_threshold_mv,
        first.energy_threshold_mv,
        first.acquisition_mode,
        first.measurement_mode,
        first.toa_lsb_ps,
        first.tot_lsb_ps,
    )
    analysis_root = main_run_dir.parent
    group_dir = (
        Path(output_root).expanduser().resolve() / name
        if output_root is not None
        else analysis_root.parent / "grouped_analysis" / name
    )
    csv_dir = group_dir / "csv"
    models_dir = group_dir / "models"
    plots_dir = group_dir / "plots"
    matched_runs_dir = group_dir / "matched_runs"
    preprocessed_dir = group_dir / "preprocessed"
    state_path = group_dir / "state.json"
    manifest_path = group_dir / "manifest.csv"
    energy_selection_path = table_path(csv_dir, "energy_selection", cfg)
    training_path = table_path(csv_dir, "matching_training", cfg)
    model_metrics_path = csv_dir / "matching_model_metrics.csv"
    model_a_path = models_dir / "ch1_ch3_average_delay.json"
    model_b_path = models_dir / "ch5_ch7_average_delay.json"
    matching_total_path = table_path(csv_dir, "matching_total", cfg)
    concatenated_path = preprocessed_dir / "group_list.dat"
    selection_path = table_path(csv_dir, "selection", cfg)
    fit_path = csv_dir / "fit.csv"
    summary_path = group_dir.parent / GROUP_SUMMARY_FILENAME
    legacy_summary_path = group_dir / "summary.csv"
    peak_plot_path = plots_dir / "peak_selection.png"
    matching_train_plot_path = plots_dir / "matching_model_train.png"
    matching_total_plot_path = plots_dir / "matching_model_total.png"
    timing_plot_path = plots_dir / "timing_fit.png"

    group_dir.mkdir(parents=True, exist_ok=True)
    state = load_state(state_path)
    previous_mode = state.get("metadata", {}).get("AcquisitionMode")
    if previous_mode not in (None, first.acquisition_mode):
        raise RuntimeError(
            f"Group folder {group_dir} already belongs to acquisition mode "
            f"{previous_mode}, while the main run uses {first.acquisition_mode}. "
            "Choose a different --output-root to avoid mixing binary formats."
        )
    state["metadata"] = {
        "group_name": name,
        "main_run": canonical_run_id(main_run_dir.name)[0],
        "run_ids": [run.run_id for run in runs],
        "Voltage": first.voltage,
        "AcquisitionMode": first.acquisition_mode,
        "E_th": first.energy_threshold_mv,
        "T_th": first.timing_threshold_mv,
    }

    candidate_signatures = {
        run.run_id: _candidate_signature(run, cfg) for run in runs
    }
    manifest_signature = signature(
        {
            "runs": [
                {
                    "run_id": run.run_id,
                    "data": file_signature(run.data_path),
                    "info": file_signature(run.info_path),
                    "candidate": candidate_signatures[run.run_id],
                }
                for run in runs
            ]
        },
        {"group_analysis_version": 1},
    )

    # 1. Virtual concatenation of candidate-preprocessed data and global energy selection.
    energy_signature = signature(
        {"manifest": manifest_signature}, stage_config(cfg, "energy_selection")
    )
    if not overwrite and stage_valid_any(
        state, "energy_selection", [energy_signature], [manifest_path, energy_selection_path]
    ):
        energy_measurements, duration_mask = load_energy_selection_csv(
            energy_selection_path
        )
        _restore_run_offsets_from_manifest(runs, manifest_path)
        energy_selection = _energy_from_cache(
            duration_mask, state["stages"]["energy_selection"]["metadata"]
        )
        _log(name, "energy_selection", "SKIPPED — cached pooled selection is valid")
    else:
        start = time.perf_counter()
        energy_measurements = _collect_group_energy(runs, cfg)
        if energy_measurements.size == 0:
            raise RuntimeError("Compatible candidate data contain no energy events")
        energy_selection = select_energy_events(energy_measurements, cfg)
        _write_manifest(manifest_path, runs)
        write_energy_selection_csv(
            energy_selection_path,
            energy_measurements,
            energy_selection,
            cfg["analysis_output"]["diagnostic_mode"],
        )
        mark_stage(
            state,
            "energy_selection",
            energy_signature,
            [manifest_path, energy_selection_path],
            _energy_metadata(energy_selection, first.toa_lsb_ps),
        )
        save_state(state_path, state)
        _log(
            name,
            "energy_selection",
            f"COMPLETED in {_elapsed(start)} — {energy_measurements.size} pooled candidates, "
            f"{int(np.count_nonzero(energy_selection.duration_mask))} selected",
        )

    selected_by_run = {
        run.run_id: _selected_local_indices(
            run, energy_measurements, energy_selection.duration_mask, cfg
        )
        for run in runs
    }

    # Peak selection plot.
    if cfg["plots"]["peak_selection"]["enabled"]:
        plot_signature = signature(
            file_signature(energy_selection_path),
            stage_config(cfg, "plot_peak_selection"),
        )
        if overwrite or not stage_valid_any(
            state, "plot_peak_selection", [plot_signature], [peak_plot_path]
        ):
            plot_peak_selection(
                peak_plot_path,
                name,
                energy_measurements,
                energy_selection,
                first.toa_lsb_ps,
                cfg,
            )
            mark_stage(
                state,
                "plot_peak_selection",
                plot_signature,
                [peak_plot_path],
            )
            save_state(state_path, state)

    # 2. Global matching training labels.
    training_signature = signature(
        {
            "energy_selection": file_signature(energy_selection_path),
            "candidates": candidate_signatures,
        },
        stage_config(cfg, "matching_training"),
    )
    if not overwrite and stage_valid_any(
        state, "matching_training", [training_signature], [training_path]
    ):
        matching_samples = load_training_csv(training_path)
        _log(name, "matching_training", "SKIPPED — cached pooled labels are valid")
    else:
        start = time.perf_counter()
        matching_samples = _collect_group_training(runs, cfg, selected_by_run)
        write_training_csv(
            training_path,
            _training_rows(matching_samples, cfg),
            cfg["analysis_output"]["diagnostic_mode"],
        )
        mark_stage(
            state,
            "matching_training",
            training_signature,
            [training_path],
            {
                "pair_a_samples": matching_samples["a"].size,
                "pair_b_samples": matching_samples["b"].size,
                "toa_lsb_ps": first.toa_lsb_ps,
            },
        )
        save_state(state_path, state)
        _log(
            name,
            "matching_training",
            f"COMPLETED in {_elapsed(start)} — a={matching_samples['a'].size}, "
            f"b={matching_samples['b'].size}",
        )

    # 3. One global matching model per channel pair.
    model_signature = signature(
        file_signature(training_path), stage_config(cfg, "matching_model")
    )
    model_outputs = [
        model_a_path,
        model_b_path,
        model_metrics_path,
    ]
    if not overwrite and stage_valid_any(
        state, "matching_model", [model_signature], model_outputs
    ):
        models = {"a": load_model(model_a_path), "b": load_model(model_b_path)}
        filtered_samples = matching_samples
        _log(name, "matching_model", "SKIPPED — cached global models are valid")
    else:
        start = time.perf_counter()
        models, filtered_samples = train_matching_models(matching_samples, cfg)
        write_model(model_a_path, models["a"])
        write_model(model_b_path, models["b"])
        write_model_metrics(model_metrics_path, models)
        mark_stage(state, "matching_model", model_signature, model_outputs)
        save_state(state_path, state)
        _log(name, "matching_model", f"COMPLETED in {_elapsed(start)}")

    if cfg["plots"]["matching_train"]["enabled"]:
        plot_signature = signature(
            {
                "training": file_signature(training_path),
                "model_a": file_signature(model_a_path),
                "model_b": file_signature(model_b_path),
                "selection": file_signature(energy_selection_path),
            },
            stage_config(cfg, "plot_matching_train"),
        )
        if overwrite or not stage_valid_any(
            state, "plot_matching_train", [plot_signature], [matching_train_plot_path]
        ):
            plot_matching_training(
                matching_train_plot_path,
                name,
                filtered_samples,
                models,
                first.toa_lsb_ps,
                cfg,
                energy_selection,
            )
            mark_stage(
                state,
                "plot_matching_train",
                plot_signature,
                [matching_train_plot_path],
            )
            save_state(state_path, state)

    # 4. Apply the global model run-by-run. Each run has an independent cache.
    all_total_rows: list[dict[str, Any]] = []
    matched_files: list[Path] = []
    for run in runs:
        matched_path = matched_runs_dir / f"{run.run_id}_list.dat"
        diagnostics_path = table_path(
            matched_runs_dir, f"{run.run_id}_matching_total", cfg
        )
        core_path = matched_runs_dir / f"{run.run_id}_matching_core.npz"
        selected_local = selected_by_run[run.run_id]
        run_stage = f"matching_{run.run_id}"
        run_signature = signature(
            {
                "candidate": candidate_signatures[run.run_id],
                "selected_indices": _array_digest(selected_local),
                "model_a": file_signature(model_a_path),
                "model_b": file_signature(model_b_path),
                "raw_header": file_signature(run.data_path),
            },
            stage_config(cfg, "preprocessing"),
        )
        if not overwrite and stage_valid_any(
            state, run_stage, [run_signature], [matched_path, core_path, diagnostics_path]
        ):
            run_rows = load_total_cache(core_path)
            _log(name, run_stage, "SKIPPED — cached matched run is valid")
        else:
            start = time.perf_counter()
            if run.acquisition_mode == "STREAMING":
                assert run.pulse_cache_dir is not None
                assert run.candidate_index_dir is not None
                metadata, run_rows = preprocess_streaming_from_index(
                    run.data_path,
                    run.pulse_cache_dir,
                    run.candidate_index_dir,
                    matched_path,
                    cfg,
                    models,
                    selected_local,
                )
            else:
                assert run.candidate_path is not None
                metadata, run_rows = preprocess_binary(
                    run.candidate_path,
                    matched_path,
                    cfg,
                    run.acquisition_mode,
                    models,
                    selected_local,
                )
            write_total_cache(core_path, run_rows)
            write_total_csv(
                diagnostics_path,
                run_rows,
                cfg["analysis_output"]["diagnostic_mode"],
            )
            mark_stage(
                state,
                run_stage,
                run_signature,
                [matched_path, core_path, diagnostics_path],
                metadata,
            )
            save_state(state_path, state)
            _log(
                name,
                run_stage,
                f"COMPLETED in {_elapsed(start)} — {metadata['events_written']} events",
            )
        matched_files.append(matched_path)
        for row in run_rows:
            local_event_index = int(row["event_index"])
            enriched = dict(row)
            enriched["run_id"] = run.run_id
            enriched["local_event_index"] = local_event_index
            enriched["global_event_index"] = run.global_offset + local_event_index
            enriched["event_index"] = run.global_offset + local_event_index
            all_total_rows.append(enriched)

    total_signature = signature(
        {
            "matched_runs": [file_signature(path) for path in matched_files],
            "matching_core": [
                file_signature(matched_runs_dir / f"{run.run_id}_matching_core.npz")
                for run in runs
            ],
            "diagnostics": [
                file_signature(
                    table_path(matched_runs_dir, f"{run.run_id}_matching_total", cfg)
                )
                for run in runs
            ],
        },
        {"group_total_version": 2},
    )
    if overwrite or not stage_valid_any(
        state, "matching_total", [total_signature], [matching_total_path]
    ):
        _write_group_total(matching_total_path, all_total_rows, cfg)
        mark_stage(state, "matching_total", total_signature, [matching_total_path])
        save_state(state_path, state)

    if cfg["plots"]["matching_total"]["enabled"]:
        plot_signature = signature(
            {
                "total": file_signature(matching_total_path),
                "model_a": file_signature(model_a_path),
                "model_b": file_signature(model_b_path),
                "selection": file_signature(energy_selection_path),
            },
            stage_config(cfg, "plot_matching_total"),
        )
        if overwrite or not stage_valid_any(
            state, "plot_matching_total", [plot_signature], [matching_total_plot_path]
        ):
            plot_matching_total(
                matching_total_plot_path,
                name,
                all_total_rows,
                models,
                first.toa_lsb_ps,
                cfg,
                energy_selection,
            )
            mark_stage(
                state,
                "plot_matching_total",
                plot_signature,
                [matching_total_plot_path],
            )
            save_state(state_path, state)

    # 5. Concatenate the matched preprocessed binaries.
    concatenation_signature = signature(
        [file_signature(path) for path in matched_files],
        {"binary_concatenation_version": 1},
    )
    if not overwrite and stage_valid_any(
        state,
        "concatenate_preprocessed",
        [concatenation_signature],
        [concatenated_path],
    ):
        _log(name, "concatenate_preprocessed", "SKIPPED — cached binary is valid")
    else:
        start = time.perf_counter()
        concat_metadata = _concatenate_binary_files(
            matched_files, concatenated_path, first.acquisition_mode
        )
        mark_stage(
            state,
            "concatenate_preprocessed",
            concatenation_signature,
            [concatenated_path],
            concat_metadata,
        )
        save_state(state_path, state)
        _log(name, "concatenate_preprocessed", f"COMPLETED in {_elapsed(start)}")

    # 6. Post-matching selection and fit on the concatenated binary.
    selection_signature = signature(
        {
            "preprocessed": file_signature(concatenated_path),
            "energy_selection": file_signature(energy_selection_path),
        },
        stage_config(cfg, "selection"),
    )
    if not overwrite and stage_valid_any(
        state, "selection", [selection_signature], [selection_path]
    ):
        measurements, duration_mask, alignment_mask = load_selection_csv(selection_path)
        selection = _selection_from_cache(
            duration_mask,
            alignment_mask,
            state["stages"]["selection"]["metadata"],
        )
        _log(name, "selection", "SKIPPED — cached post-matching selection is valid")
    else:
        start = time.perf_counter()
        measurements, toa_lsb_ps = collect_measurements(concatenated_path, cfg)
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
        mark_stage(
            state,
            "selection",
            selection_signature,
            [selection_path],
            _selection_metadata(selection, toa_lsb_ps),
        )
        save_state(state_path, state)
        _log(
            name,
            "selection",
            f"COMPLETED in {_elapsed(start)} — {measurements.size} matched, "
            f"{int(np.count_nonzero(selection.final_mask))} retained",
        )

    fit_signature = signature(
        file_signature(selection_path), stage_config(cfg, "fit")
    )
    if not overwrite and stage_valid_any(state, "fit", [fit_signature], [fit_path]):
        fit = load_fit_csv(fit_path)
        _log(name, "fit", "SKIPPED — cached fit is valid")
    else:
        start = time.perf_counter()
        timing_ps = (
            measurements.timing_lsb[selection.final_mask].astype(np.float64)
            * float(first.toa_lsb_ps)
        )
        fit = fit_delta_times_ps(
            timing_ps,
            method="Pico-TDC LED grouped",
            parameter=float(first.timing_threshold_mv),
            n_total=int(measurements.size),
            n_selected=int(np.count_nonzero(selection.final_mask)),
            config=cfg["fit"],
        )
        if not fit.success:
            raise RuntimeError(f"Common Gaussian fit failed: {fit.message}")
        write_fit_csv(
            fit_path, fit, cfg["analysis_output"]["diagnostic_mode"]
        )
        mark_stage(state, "fit", fit_signature, [fit_path])
        save_state(state_path, state)
        _log(name, "fit", f"COMPLETED in {_elapsed(start)}")

    if cfg["plots"]["timing_fit"]["enabled"]:
        plot_signature = signature(
            file_signature(fit_path), stage_config(cfg, "plot_timing_fit")
        )
        if overwrite or not stage_valid_any(
            state, "plot_timing_fit", [plot_signature], [timing_plot_path]
        ):
            plot_gaussian_fit(
                fit,
                timing_plot_path,
                dpi=int(cfg["plots"]["dpi"]),
                title=f"{name} — Pico-TDC grouped timing fit",
                xlabel="ch7 − ch3 [ps]",
            )
            mark_stage(
                state,
                "plot_timing_fit",
                plot_signature,
                [timing_plot_path],
            )
            save_state(state_path, state)

    summary_row = _build_summary_row(
        group_dir,
        name,
        runs,
        energy_measurements,
        energy_selection,
        measurements,
        selection,
        fit,
        models,
        first.toa_lsb_ps,
        all_total_rows,
        cfg,
    )
    _update_group_summary(summary_path, summary_row)
    legacy_summary_path.unlink(missing_ok=True)
    state["metadata"]["summary"] = str(summary_path)
    save_state(state_path, state)
    _log(
        name,
        "group",
        f"COMPLETED — {len(runs)} compatible runs; results in {group_dir}",
    )
    return group_dir



def _compatibility_key(run: CompatibleRun) -> tuple[Any, ...]:
    return (
        run.voltage,
        run.acquisition_mode,
        run.energy_threshold_mv,
        run.timing_threshold_mv,
        run.measurement_mode,
        run.toa_lsb_ps,
        run.tot_lsb_ps,
    )


def _discover_all_group_members(cfg: dict) -> list[list[CompatibleRun]]:
    analysis_root = Path(cfg["paths"]["output_dir"]) / "analysis"
    discovered = _filter_configured_runs(
        discover_runs(
            cfg["paths"]["input_dir"],
            cfg["files"]["data_pattern"],
            bool(cfg["files"]["recursive"]),
        ),
        cfg,
    )
    groups: dict[tuple[Any, ...], list[CompatibleRun]] = {}
    for run_input in discovered:
        run_info = parse_run_info(
            run_input.info_path, cfg["thresholds"]["consistency"]
        )
        meta = read_meta(run_input.data_path, run_info.acquisition_mode)
        output_dir = analysis_root / run_input.run_id
        if run_info.acquisition_mode == "STREAMING":
            candidate = CompatibleRun(
                run_id=run_input.run_id,
                run_number=run_input.run_number,
                voltage=run_input.voltage,
                acquisition_mode=run_info.acquisition_mode,
                energy_threshold_mv=run_info.energy_threshold_mv,
                timing_threshold_mv=run_info.timing_threshold_mv,
                data_path=run_input.data_path,
                info_path=run_input.info_path,
                output_dir=output_dir,
                toa_lsb_ps=meta.toa_lsb_ps,
                tot_lsb_ps=meta.tot_lsb_ps,
                measurement_mode=meta.measurement_mode,
                candidate_kind="streaming_compact_index",
                candidate_path=None,
                pulse_cache_dir=output_dir / "streaming_pulses",
                candidate_index_dir=output_dir / "streaming_candidates",
            )
        else:
            candidate = CompatibleRun(
                run_id=run_input.run_id,
                run_number=run_input.run_number,
                voltage=run_input.voltage,
                acquisition_mode=run_info.acquisition_mode,
                energy_threshold_mv=run_info.energy_threshold_mv,
                timing_threshold_mv=run_info.timing_threshold_mv,
                data_path=run_input.data_path,
                info_path=run_input.info_path,
                output_dir=output_dir,
                toa_lsb_ps=meta.toa_lsb_ps,
                tot_lsb_ps=meta.tot_lsb_ps,
                measurement_mode=meta.measurement_mode,
                candidate_kind="candidate_binary",
                candidate_path=(
                    output_dir
                    / "candidate_preprocessed"
                    / f"{run_input.run_id}_candidates.dat"
                ),
                pulse_cache_dir=None,
                candidate_index_dir=None,
            )
        groups.setdefault(_compatibility_key(candidate), []).append(candidate)
    return [
        sorted(members, key=lambda item: int(item.run_number))
        for _, members in sorted(groups.items(), key=lambda item: item[0])
    ]


def run_all_compatible_group_analyses(
    cfg: dict,
    *,
    overwrite: bool = False,
    skip_missing: bool = False,
    output_root: str | Path | None = None,
) -> Path:
    """Analyze every compatibility group and write one consolidated summary."""
    groups = _discover_all_group_members(cfg)
    if not groups:
        raise RuntimeError("No runs were found for grouped analysis")

    summary_root = (
        Path(output_root).expanduser().resolve()
        if output_root is not None
        else Path(cfg["paths"]["output_dir"]) / "grouped_analysis"
    )
    summary_root.mkdir(parents=True, exist_ok=True)
    summary_path = summary_root / GROUP_SUMMARY_FILENAME
    # Dataset-wide execution is authoritative: remove rows for groups that are no
    # longer present in the configured input/include set.
    summary_path.unlink(missing_ok=True)
    failures: list[str] = []

    for members in groups:
        first = members[0]
        name = group_name(
            first.voltage,
            first.timing_threshold_mv,
            first.energy_threshold_mv,
            first.acquisition_mode,
            first.measurement_mode,
            first.toa_lsb_ps,
            first.tot_lsb_ps,
        )
        group_dir = summary_root / name
        available = [
            run
            for run in members
            if all(path.exists() for path in _required_candidate_outputs(run, cfg))
        ]
        representative = available[0] if available else first
        try:
            run_compatible_group_analysis(
                representative.output_dir,
                cfg,
                overwrite=overwrite,
                skip_missing=skip_missing,
                output_root=summary_root,
            )
        except Exception as exc:
            _update_group_summary(
                summary_path,
                _failed_summary_row(group_dir, name, members, exc),
            )
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            _log(name, "group", f"FAILED — {type(exc).__name__}: {exc}")

    if failures:
        raise RuntimeError(
            f"{len(failures)} grouped analyses failed; partial results and failure "
            f"rows were written to {summary_path}:\n  " + "\n  ".join(failures)
        )
    return summary_path
