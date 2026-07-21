from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .binary_io import (
    DataError,
    LEAD_TRAIL,
    STREAMING,
    acquisition_mode_name,
    iter_events,
    read_header,
    write_event,
    write_header,
)
from .matching import AverageDelayModel, window_ns_to_lsb
from .models import BinaryMeta, Event, Hit
from .pulses import (
    Pulse,
    earliest_energy_pair,
    leading_hits_before,
)


def _channel_pairs(cfg: dict) -> dict[str, tuple[int, int]]:
    channels = cfg["channels"]
    return {
        "a": (int(channels["signal_a"]), int(channels["time_a"])),
        "b": (int(channels["signal_b"]), int(channels["time_b"])),
    }


def _trigger_candidate_event(
    event: Event,
    cfg: dict,
    timing_window_lsb: int,
) -> tuple[Event | None, dict[str, Any]]:
    """Historical Trigger-Matching behaviour: one candidate per trigger."""
    pairs = _channel_pairs(cfg)
    energy_pair_a = earliest_energy_pair(event.hits, pairs["a"][0])
    energy_pair_b = earliest_energy_pair(event.hits, pairs["b"][0])
    if energy_pair_a is None:
        return None, {"status": "missing_energy_pulse_a"}
    if energy_pair_b is None:
        return None, {"status": "missing_energy_pulse_b"}

    timing_candidates: dict[str, list[Hit]] = {}
    for pair_key, energy_pair in (("a", energy_pair_a), ("b", energy_pair_b)):
        timing_channel = pairs[pair_key][1]
        candidates = leading_hits_before(
            event.hits,
            timing_channel,
            energy_pair[0].toa_lsb,
            timing_window_lsb,
        )
        if not candidates:
            return None, {"status": f"missing_timing_candidate_{pair_key}"}
        timing_candidates[pair_key] = candidates

    output = [
        *energy_pair_a,
        *energy_pair_b,
        *timing_candidates["a"],
        *timing_candidates["b"],
    ]
    output.sort(key=lambda hit: (hit.toa_lsb, hit.channel, -hit.edge))
    return (
        Event(event.event_index, event.timestamp_lsb, event.trigger_id, output),
        {
            "status": "accepted",
            "candidate_count_a": len(timing_candidates["a"]),
            "candidate_count_b": len(timing_candidates["b"]),
        },
    )


def preprocess_candidates(
    input_path: str | Path,
    output_path: str | Path,
    cfg: dict,
    acquisition_mode: str,
    streaming_pulse_cache_dir: str | Path | None = None,
) -> dict[str, int | float | str | bool]:
    """Build the candidate-preserving binary for TRG_MATCHING runs.

    STREAMING uses the integrated ``streaming_events.npz`` path and deliberately
    no longer writes a candidate binary.
    """
    del streaming_pulse_cache_dir
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")

    try:
        with input_path.open("rb") as source, temporary.open("wb") as destination:
            meta = read_header(source, acquisition_mode)
            if meta.measurement_mode != LEAD_TRAIL:
                raise DataError("Timing analysis requires LEAD_TRAIL measurement mode")
            write_header(destination, meta)

            if meta.acquisition_mode == STREAMING:
                raise DataError(
                    "STREAMING candidate binaries were removed. Use the integrated "
                    "streaming_events.npz cache built by decode_streaming_pulse_cache()."
                )
            else:
                records_read = 0
                events_written = 0
                candidate_a_total = 0
                candidate_b_total = 0
                detail = {
                    "physical_time_source": "trigger_relative_toa",
                    "event_timestamp_used_for_time": False,
                }
                timing_window_lsb = window_ns_to_lsb(
                    float(cfg["matching_model"]["inference"]["candidate_window_ns"]),
                    meta.toa_lsb_ps,
                )
                for event in iter_events(source, meta):
                    records_read += 1
                    candidate_event, event_meta = _trigger_candidate_event(
                        event, cfg, timing_window_lsb
                    )
                    if candidate_event is None:
                        continue
                    if write_event(destination, candidate_event, meta):
                        events_written += 1
                        candidate_a_total += int(event_meta["candidate_count_a"])
                        candidate_b_total += int(event_meta["candidate_count_b"])
        temporary.replace(output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    result: dict[str, int | float | str | bool] = {
        "events_read": records_read,
        "raw_records_read": records_read,
        "events_written": events_written,
        "candidate_events_written": events_written,
        "timing_candidates_a": candidate_a_total,
        "timing_candidates_b": candidate_b_total,
        "candidate_window_ns": float(
            cfg["matching_model"]["inference"]["candidate_window_ns"]
        ),
        "toa_lsb_ps": meta.toa_lsb_ps,
        "timestamp_lsb_ps": meta.timestamp_lsb_ps,
        "acquisition_mode": acquisition_mode_name(meta.acquisition_mode),
        **detail,
    }
    return result



def _base_diagnostic(
    event_index: int,
    model: AverageDelayModel,
    energy_duration_lsb: int,
    candidate_count: int,
    maximum_deviation_lsb: float | None,
) -> dict[str, Any]:
    return {
        "event_index": int(event_index),
        "pair": model.pair,
        "energy_channel": model.energy_channel,
        "timing_channel": model.timing_channel,
        "energy_duration_lsb": int(energy_duration_lsb),
        "average_delay_lsb": float(model.average_delay_lsb),
        "selected_delay_lsb": "",
        "baseline_delay_lsb": "",
        "candidate_changed": 0,
        "deviation_lsb": "",
        "maximum_deviation_lsb": (
            "" if maximum_deviation_lsb is None else float(maximum_deviation_lsb)
        ),
        "candidate_count": int(candidate_count),
        "accepted": 0,
        "event_accepted": 0,
        "status": "unknown",
    }


def _maximum_deviation_lsb(cfg: dict, toa_lsb_ps: float) -> float | None:
    value_ns = cfg["matching_model"]["inference"].get("maximum_deviation_ns")
    if value_ns is None:
        return None
    return float(value_ns) * 1000.0 / float(toa_lsb_ps)


def _match_trigger_timing_lead(
    event: Event,
    energy_pair: Pulse,
    timing_channel: int,
    model: AverageDelayModel,
    window_lsb: int,
    maximum_deviation_lsb: float | None,
) -> tuple[Hit | None, dict[str, Any]]:
    energy_lead_lsb = energy_pair[0].toa_lsb
    energy_duration_lsb = energy_pair[1].toa_lsb - energy_lead_lsb
    candidates = leading_hits_before(
        event.hits,
        timing_channel,
        energy_lead_lsb,
        window_lsb,
    )
    row = _base_diagnostic(
        event.event_index,
        model,
        energy_duration_lsb,
        len(candidates),
        maximum_deviation_lsb,
    )
    if not candidates:
        row["status"] = "no_timing_candidate"
        return None, row

    selected = min(
        candidates,
        key=lambda hit: (
            abs((energy_lead_lsb - hit.toa_lsb) - model.average_delay_lsb),
            -hit.toa_lsb,
        ),
    )
    # Simple reference choice retained for impact studies: latest preceding hit.
    baseline = max(candidates, key=lambda hit: (hit.toa_lsb, -hit.channel))
    selected_delay_lsb = energy_lead_lsb - selected.toa_lsb
    baseline_delay_lsb = energy_lead_lsb - baseline.toa_lsb
    deviation_lsb = abs(selected_delay_lsb - model.average_delay_lsb)
    row["selected_delay_lsb"] = selected_delay_lsb
    row["baseline_delay_lsb"] = baseline_delay_lsb
    row["candidate_changed"] = int(selected.toa_lsb != baseline.toa_lsb)
    row["deviation_lsb"] = deviation_lsb
    if maximum_deviation_lsb is not None and deviation_lsb > maximum_deviation_lsb:
        row["status"] = "deviation_too_large"
        return None, row
    row["accepted"] = 1
    row["status"] = "accepted"
    return selected, row


def match_selected_event(
    event: Event,
    cfg: dict,
    models: dict[str, AverageDelayModel],
    toa_lsb_ps: float,
    streaming_physical_mode: bool,
) -> tuple[Event | None, list[dict[str, Any]]]:
    inference_cfg = cfg["matching_model"]["inference"]
    window_lsb = window_ns_to_lsb(
        float(inference_cfg["candidate_window_ns"]), toa_lsb_ps
    )
    maximum_deviation_lsb = _maximum_deviation_lsb(cfg, toa_lsb_ps)
    channel_pairs = _channel_pairs(cfg)
    energy_pair_a = earliest_energy_pair(event.hits, channel_pairs["a"][0])
    energy_pair_b = earliest_energy_pair(event.hits, channel_pairs["b"][0])
    if energy_pair_a is None or energy_pair_b is None:
        return None, []

    output: list[Hit] = []
    diagnostics: list[dict[str, Any]] = []
    matched: dict[str, Hit] = {}
    for pair, energy_pair in (("a", energy_pair_a), ("b", energy_pair_b)):
        timing_lead, row = _match_trigger_timing_lead(
            event,
            energy_pair,
            channel_pairs[pair][1],
            models[pair],
            window_lsb,
            maximum_deviation_lsb,
        )
        diagnostics.append(row)
        if timing_lead is None:
            for previous in diagnostics:
                previous["event_accepted"] = 0
            return None, diagnostics
        matched[pair] = timing_lead
    output.extend(energy_pair_a)
    output.extend(energy_pair_b)
    output.extend((matched["a"], matched["b"]))
    output.sort(key=lambda hit: (hit.toa_lsb, hit.channel, -hit.edge))
    return Event(
        event.event_index,
        0 if streaming_physical_mode else event.timestamp_lsb,
        0 if streaming_physical_mode else event.trigger_id,
        output,
    ), diagnostics


def preprocess_binary(
    input_path: str | Path,
    output_path: str | Path,
    cfg: dict,
    acquisition_mode: str,
    models: dict[str, AverageDelayModel],
    selected_event_indices: set[int],
) -> tuple[dict[str, int | float | str | bool], list[dict[str, Any]]]:
    """Match selected events using nearest calibrated average delay."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    read_count = 0
    selected_count = 0
    written_count = 0
    diagnostics: list[dict[str, Any]] = []
    try:
        with input_path.open("rb") as source, temporary.open("wb") as destination:
            meta = read_header(source, acquisition_mode)
            if meta.measurement_mode != LEAD_TRAIL:
                raise DataError("Timing analysis requires LEAD_TRAIL measurement mode")
            write_header(destination, meta)
            streaming_physical_mode = meta.acquisition_mode == STREAMING
            for event in iter_events(source, meta):
                read_count += 1
                if event.event_index not in selected_event_indices:
                    continue
                selected_count += 1
                cleaned, event_rows = match_selected_event(
                    event,
                    cfg,
                    models,
                    meta.toa_lsb_ps,
                    streaming_physical_mode,
                )
                event_written = bool(
                    cleaned is not None and write_event(destination, cleaned, meta)
                )
                if event_written:
                    written_count += 1
                for row in event_rows:
                    row["event_accepted"] = int(event_written)
                    if cleaned is not None and not event_written:
                        row["status"] = "write_failed"
                diagnostics.extend(event_rows)
        temporary.replace(output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    return (
        {
            "events_read": read_count,
            "energy_selected_events": selected_count,
            "events_written": written_count,
            "events_discarded_matching": selected_count - written_count,
            "toa_lsb_ps": meta.toa_lsb_ps,
            "candidate_window_ns": float(
                cfg["matching_model"]["inference"]["candidate_window_ns"]
            ),
            "matching_method": "average_delay_nearest_candidate",
            "acquisition_mode": acquisition_mode_name(meta.acquisition_mode),
            "physical_time_source": (
                "streaming_hit_toa_uint64"
                if meta.acquisition_mode == STREAMING
                else "trigger_relative_toa"
            ),
            "event_timestamp_used_for_time": False,
        },
        diagnostics,
    )


def preprocess_streaming_from_index(
    raw_input_path: str | Path,
    pulse_cache_dir: str | Path,
    candidate_index_dir: str | Path,
    output_path: str | Path,
    cfg: dict,
    models: dict[str, AverageDelayModel],
    selected_event_indices: set[int],
) -> tuple[dict[str, int | float | str | bool], list[dict[str, Any]]]:
    """Apply independent nearest-average-delay matching to STREAMING events.

    The only persistent raw-preprocessing cache contains matched ch1/ch5 energy
    pulses and CSR timing-leading candidates for each side.  ch3 and ch7 are
    never pre-paired with one another.
    """
    del candidate_index_dir
    from .streaming_cache import (
        load_streaming_event_cache,
        streaming_energy_duration_lsb,
    )

    raw_input_path = Path(raw_input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    arrays, cache_metadata = load_streaming_event_cache(pulse_cache_dir)
    channel_pairs = _channel_pairs(cfg)
    toa_lsb_ps = float(cache_metadata["toa_lsb_ps"])
    inference_cfg = cfg["matching_model"]["inference"]
    inference_window_lsb = window_ns_to_lsb(
        float(inference_cfg["candidate_window_ns"]), toa_lsb_ps
    )
    maximum_deviation_lsb = _maximum_deviation_lsb(cfg, toa_lsb_ps)
    event_count = int(arrays["energy_lead_a"].size)
    duration_by_side = {
        side: streaming_energy_duration_lsb(arrays, cache_metadata, side)
        for side in ("a", "b")
    }
    selected_sorted = sorted(
        index for index in selected_event_indices if 0 <= index < event_count
    )

    diagnostics: list[dict[str, Any]] = []
    written_count = 0
    missing_candidate_events = 0
    deviation_rejected_events = 0
    try:
        with raw_input_path.open("rb") as source, temporary.open("wb") as destination:
            meta = read_header(source, "STREAMING")
            if meta.measurement_mode != LEAD_TRAIL:
                raise DataError("Timing analysis requires LEAD_TRAIL measurement mode")
            write_header(destination, meta)

            for event_index in selected_sorted:
                event_rows: list[dict[str, Any]] = []
                selected_timing: dict[str, int] = {}
                event_valid = True
                event_missing = False
                event_deviation = False

                for side in ("a", "b"):
                    energy_lead = int(arrays[f"energy_lead_{side}"][event_index])
                    energy_trail = int(arrays[f"energy_trail_{side}"][event_index])
                    offsets = arrays[f"timing_offsets_{side}"]
                    candidates_all = arrays[f"timing_leads_{side}"][
                        int(offsets[event_index]) : int(offsets[event_index + 1])
                    ].astype(np.int64, copy=False)
                    delays_all = energy_lead - candidates_all
                    valid_mask = (delays_all > 0) & (delays_all < inference_window_lsb)
                    candidates = candidates_all[valid_mask]
                    delays = delays_all[valid_mask]
                    row = _base_diagnostic(
                        event_index,
                        models[side],
                        int(duration_by_side[side][event_index]),
                        int(candidates.size),
                        maximum_deviation_lsb,
                    )
                    if candidates.size == 0:
                        row["status"] = "no_timing_candidate"
                        event_valid = False
                        event_missing = True
                        event_rows.append(row)
                        continue

                    deviations = np.abs(
                        delays.astype(float) - float(models[side].average_delay_lsb)
                    )
                    best_index = int(
                        min(
                            range(candidates.size),
                            key=lambda index: (
                                float(deviations[index]),
                                -int(candidates[index]),
                            ),
                        )
                    )
                    baseline_index = int(np.argmax(candidates))
                    selected_lead = int(candidates[best_index])
                    selected_delay = int(delays[best_index])
                    baseline_delay = int(delays[baseline_index])
                    deviation = float(deviations[best_index])
                    row["selected_delay_lsb"] = selected_delay
                    row["baseline_delay_lsb"] = baseline_delay
                    row["candidate_changed"] = int(best_index != baseline_index)
                    row["deviation_lsb"] = deviation
                    if (
                        maximum_deviation_lsb is not None
                        and deviation > maximum_deviation_lsb
                    ):
                        row["status"] = "deviation_too_large"
                        event_valid = False
                        event_deviation = True
                    else:
                        row["accepted"] = 1
                        row["status"] = "accepted"
                        selected_timing[side] = selected_lead
                    event_rows.append(row)

                if not event_valid or len(selected_timing) != 2:
                    if event_missing:
                        missing_candidate_events += 1
                    elif event_deviation:
                        deviation_rejected_events += 1
                    for row in event_rows:
                        row["accepted"] = 0
                        row["event_accepted"] = 0
                        if row["status"] == "accepted":
                            row["status"] = (
                                "other_side_missing_candidate"
                                if event_missing
                                else "other_side_deviation_too_large"
                            )
                    diagnostics.extend(event_rows)
                    continue

                output_hits = [
                    Hit(
                        int(arrays["energy_board_a"][event_index]),
                        channel_pairs["a"][0],
                        1,
                        int(arrays["energy_lead_a"][event_index]),
                        0,
                    ),
                    Hit(
                        int(arrays["energy_board_a"][event_index]),
                        channel_pairs["a"][0],
                        0,
                        int(arrays["energy_trail_a"][event_index]),
                        int(arrays["energy_tot_a"][event_index]),
                    ),
                    Hit(
                        int(arrays["energy_board_b"][event_index]),
                        channel_pairs["b"][0],
                        1,
                        int(arrays["energy_lead_b"][event_index]),
                        0,
                    ),
                    Hit(
                        int(arrays["energy_board_b"][event_index]),
                        channel_pairs["b"][0],
                        0,
                        int(arrays["energy_trail_b"][event_index]),
                        int(arrays["energy_tot_b"][event_index]),
                    ),
                    Hit(
                        int(arrays["energy_board_a"][event_index]),
                        channel_pairs["a"][1],
                        1,
                        selected_timing["a"],
                        0,
                    ),
                    Hit(
                        int(arrays["energy_board_b"][event_index]),
                        channel_pairs["b"][1],
                        1,
                        selected_timing["b"],
                        0,
                    ),
                ]
                output_hits.sort(key=lambda hit: (hit.toa_lsb, hit.channel, -hit.edge))
                event_written = write_event(
                    destination,
                    Event(written_count, 0, 0, output_hits),
                    meta,
                )
                if event_written:
                    written_count += 1
                for row in event_rows:
                    row["accepted"] = int(event_written)
                    row["event_accepted"] = int(event_written)
                    row["status"] = "accepted" if event_written else "write_failed"
                diagnostics.extend(event_rows)
        temporary.replace(output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    return (
        {
            "events_read": event_count,
            "energy_selected_events": len(selected_sorted),
            "events_written": written_count,
            "events_discarded_matching": len(selected_sorted) - written_count,
            "events_missing_timing_candidate": missing_candidate_events,
            "events_rejected_maximum_deviation": deviation_rejected_events,
            "toa_lsb_ps": toa_lsb_ps,
            "energy_pairing_rule": cache_metadata.get("energy_pairing_rule", ""),
            "energy_pulse_reconstruction_rule": cache_metadata.get(
                "energy_pulse_reconstruction_rule", ""
            ),
            "energy_leading_pair_window_ns": cache_metadata.get(
                "energy_leading_pair_window_ns", ""
            ),
            "candidate_window_ns": float(inference_cfg["candidate_window_ns"]),
            "matching_method": "average_delay_nearest_independent_candidate",
            "acquisition_mode": "STREAMING",
            "physical_time_source": "streaming_hit_toa_uint64",
            "event_timestamp_used_for_time": False,
            "candidate_storage": "single_streaming_events_npz",
            "timing_sides_prepaired": False,
        },
        diagnostics,
    )

