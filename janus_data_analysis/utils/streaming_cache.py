from __future__ import annotations

from array import array
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from .binary_io import DataError, LEAD_TRAIL, STREAMING, iter_events, read_header
from .models import EnergyMeasurements

# Pulse columns used internally while constructing the cache.
BOARD_COL = 0
LEAD_COL = 1
TRAIL_COL = 2
TOT_COL = 3
PULSE_COLUMNS = 4

# STREAMING now stores one compact final candidate cache, rather than one pulse
# array per channel plus a second CSR index cache.
STREAMING_EVENT_CACHE_FILE = "streaming_events.npz"
STREAMING_EVENT_CACHE_VERSION = 2


def pulse_cache_paths(
    cache_dir: str | Path,
    channels: list[int] | tuple[int, ...] | set[int],
) -> dict[int, Path]:
    """Compatibility helper.

    Per-channel pulse files are intentionally no longer produced.  The same
    compact cache path is returned for every requested channel so old callers
    that only use this function to build signatures continue to work.
    """
    path = Path(cache_dir) / STREAMING_EVENT_CACHE_FILE
    return {int(channel): path for channel in sorted(channels)}


def pulse_cache_metadata_path(cache_dir: str | Path) -> Path:
    """Metadata is embedded in the single NPZ cache."""
    return Path(cache_dir) / STREAMING_EVENT_CACHE_FILE


def pulse_cache_outputs(
    cache_dir: str | Path,
    channels: list[int] | tuple[int, ...] | set[int],
) -> list[Path]:
    del channels
    return [Path(cache_dir) / STREAMING_EVENT_CACHE_FILE]


def candidate_index_paths(index_dir: str | Path) -> dict[str, Path]:
    """Legacy path names retained for import compatibility only."""
    root = Path(index_dir)
    return {"metadata": root / "deprecated_streaming_candidate_index.json"}


def candidate_index_outputs(index_dir: str | Path) -> list[Path]:
    del index_dir
    return []


def _atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _deduplicate_sorted_edges(
    boards: np.ndarray,
    times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    if times.size == 0:
        return boards, times, 0
    keep = np.ones(times.size, dtype=bool)
    if times.size > 1:
        keep[1:] = (boards[1:] != boards[:-1]) | (times[1:] != times[:-1])
    duplicates = int(times.size - np.count_nonzero(keep))
    return boards[keep], times[keep], duplicates


def _sorted_unique_edges(
    boards_buffer: array,
    times_buffer: array,
    deduplicate: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    boards = np.frombuffer(boards_buffer, dtype=np.uint8)
    times = np.frombuffer(times_buffer, dtype=np.uint64)
    if times.size != boards.size:
        raise DataError("Streaming edge buffers have inconsistent sizes")
    if times.size == 0:
        return boards.copy(), times.copy(), 0
    order = np.lexsort((times, boards))
    boards_sorted = np.asarray(boards[order], dtype=np.uint8)
    times_sorted = np.asarray(times[order], dtype=np.uint64)
    if not deduplicate:
        return boards_sorted, times_sorted, 0
    return _deduplicate_sorted_edges(boards_sorted, times_sorted)


def _sorted_unique_trailing_records(
    boards_buffer: array,
    times_buffer: array,
    tots_buffer: array,
    deduplicate: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Sort STREAMING trailing records while preserving their recorded ToT.

    A trailing record is considered an exact duplicate only when board, ToA,
    and ToT are all equal.  Treating records with the same trailing timestamp
    but a different ToT as duplicates would discard information needed to
    resolve missing-edge ambiguities.
    """
    boards = np.frombuffer(boards_buffer, dtype=np.uint8)
    times = np.frombuffer(times_buffer, dtype=np.uint64)
    tots = np.frombuffer(tots_buffer, dtype=np.uint16)
    if not (boards.size == times.size == tots.size):
        raise DataError("Streaming trailing-edge buffers have inconsistent sizes")
    if times.size == 0:
        return boards.copy(), times.copy(), tots.copy(), 0

    order = np.lexsort((tots, times, boards))
    boards_sorted = np.asarray(boards[order], dtype=np.uint8)
    times_sorted = np.asarray(times[order], dtype=np.uint64)
    tots_sorted = np.asarray(tots[order], dtype=np.uint16)
    if not deduplicate:
        return boards_sorted, times_sorted, tots_sorted, 0

    keep = np.ones(times_sorted.size, dtype=bool)
    if times_sorted.size > 1:
        keep[1:] = (
            (boards_sorted[1:] != boards_sorted[:-1])
            | (times_sorted[1:] != times_sorted[:-1])
            | (tots_sorted[1:] != tots_sorted[:-1])
        )
    duplicates = int(times_sorted.size - np.count_nonzero(keep))
    return (
        boards_sorted[keep],
        times_sorted[keep],
        tots_sorted[keep],
        duplicates,
    )


def _pair_one_board_longest_valid_indices(
    lead_times: np.ndarray,
    trail_times: np.ndarray,
    minimum_duration_lsb: int,
    maximum_duration_lsb: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Return local lead/trail index pairs for the legacy fallback rule.

    Keeping indices instead of reconstructing them from timestamps avoids
    expensive Python dictionaries and remains correct when duplicate times are
    present.  The matching semantics are unchanged: every trailing edge takes
    the earliest still-unused leading edge that gives a valid duration.
    """
    lead_times = np.asarray(lead_times, dtype=np.uint64)
    trail_times = np.asarray(trail_times, dtype=np.uint64)
    n_leads = int(lead_times.size)
    n_trails = int(trail_times.size)
    if n_leads == 0 or n_trails == 0:
        return np.empty((0, 2), dtype=np.int64), {
            "paired": 0,
            "unmatched_leading_edges": n_leads,
            "unmatched_trailing_edges": n_trails,
            "rejected_too_short": 0,
            "rejected_too_long": 0,
            "nested_leading_edges_skipped": 0,
        }

    used = np.zeros(n_leads, dtype=bool)
    pairs: list[tuple[int, int]] = []
    rejected_too_short = 0
    rejected_too_long = 0
    nested_skipped = 0
    first_live = 0
    preceding_end = 0

    for trail_index, trail_value in enumerate(trail_times):
        trail = int(trail_value)
        minimum_lead = trail - int(maximum_duration_lsb)
        maximum_lead = trail - int(minimum_duration_lsb)

        while first_live < n_leads and int(lead_times[first_live]) < minimum_lead:
            if not used[first_live]:
                rejected_too_long += 1
            first_live += 1

        chosen = -1
        index = first_live
        while index < n_leads and int(lead_times[index]) <= maximum_lead:
            if not used[index]:
                chosen = index
                break
            index += 1

        # Both arrays are sorted, so maintain the insertion point in linear
        # time rather than calling searchsorted for every trailing record.
        while preceding_end < n_leads and int(lead_times[preceding_end]) < trail:
            preceding_end += 1

        if chosen < 0:
            if preceding_end > 0:
                rejected_too_short += 1
            continue

        if preceding_end > chosen + 1:
            nested_skipped += int(np.count_nonzero(~used[chosen + 1 : preceding_end]))

        used[chosen] = True
        pairs.append((chosen, trail_index))

    paired_indices = (
        np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        if pairs
        else np.empty((0, 2), dtype=np.int64)
    )
    return paired_indices, {
        "paired": int(paired_indices.shape[0]),
        "unmatched_leading_edges": int(n_leads - np.count_nonzero(used)),
        "unmatched_trailing_edges": int(n_trails - paired_indices.shape[0]),
        "rejected_too_short": int(rejected_too_short),
        "rejected_too_long": int(rejected_too_long),
        "nested_leading_edges_skipped": int(nested_skipped),
    }


def _pair_one_board_longest_valid(
    lead_times: np.ndarray,
    trail_times: np.ndarray,
    minimum_duration_lsb: int,
    maximum_duration_lsb: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Associate each trail with the earliest unmatched physically valid lead.

    This preserves a long real signal in the common pattern
    ``L_signal, L_noise, T_signal`` instead of replacing it with the nested
    short noise crossing.
    """
    lead_times = np.asarray(lead_times, dtype=np.uint64)
    trail_times = np.asarray(trail_times, dtype=np.uint64)
    paired_indices, statistics = _pair_one_board_longest_valid_indices(
        lead_times,
        trail_times,
        minimum_duration_lsb,
        maximum_duration_lsb,
    )
    if not paired_indices.size:
        return np.empty((0, 2), dtype=np.uint64), statistics
    return (
        np.column_stack(
            (
                lead_times[paired_indices[:, 0]],
                trail_times[paired_indices[:, 1]],
            )
        ).astype(np.uint64, copy=False),
        statistics,
    )


# Backward-compatible private alias used by earlier tests.
_pair_one_board_latest_lead = _pair_one_board_longest_valid


def _pair_one_board_recorded_tot(
    lead_times: np.ndarray,
    trail_times: np.ndarray,
    trail_tots: np.ndarray,
    minimum_duration_lsb: int,
    maximum_duration_lsb: int,
    toa_lsb_ps: float,
    tot_lsb_ps: float,
    tot_match_tolerance_lsb: int,
    fallback_when_tot_missing: bool,
    fallback_when_tot_inconsistent: bool,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Reconstruct pulses using the ToT stored on STREAMING trails.

    Janus writes a 16-bit ToT on the trailing record in STREAMING +
    LEAD_TRAIL mode.  Its expected leading timestamp is therefore

    ``trail_toa - tot * tot_lsb_ps / toa_lsb_ps``.

    Expected leading timestamps are matched monotonically to the recorded
    leading array.  The implementation is linear after sorting and avoids one
    ``searchsorted`` call per hit, which is essential for multi-million-hit
    streaming files.
    """
    lead_times = np.asarray(lead_times, dtype=np.uint64)
    trail_times = np.asarray(trail_times, dtype=np.uint64)
    trail_tots = np.asarray(trail_tots, dtype=np.uint16)
    if trail_times.size != trail_tots.size:
        raise DataError("Trailing ToA and ToT arrays have inconsistent sizes")

    n_leads = int(lead_times.size)
    n_trails = int(trail_times.size)
    used_leads = np.zeros(n_leads, dtype=bool)
    used_trails = np.zeros(n_trails, dtype=bool)
    pair_parts: list[np.ndarray] = []

    nonzero_indices = np.flatnonzero(trail_tots > 0)
    recorded_tot_records = int(nonzero_indices.size)
    missing_tot_records = int(n_trails - recorded_tot_records)
    recorded_tot_too_short = 0
    recorded_tot_too_long = 0
    recorded_tot_no_matching_lead = 0
    residual_sum_lsb = 0.0
    residual_max_lsb = 0.0

    direct_pairs = np.empty((0, 3), dtype=np.uint64)
    if n_leads and nonzero_indices.size:
        nominal_duration = (
            trail_tots[nonzero_indices].astype(np.float64)
            * float(tot_lsb_ps)
            / float(toa_lsb_ps)
        )
        too_short = nominal_duration < float(minimum_duration_lsb)
        too_long = nominal_duration > float(maximum_duration_lsb)
        recorded_tot_too_short = int(np.count_nonzero(too_short))
        recorded_tot_too_long = int(np.count_nonzero(too_long))

        nominal_valid_mask = ~(too_short | too_long)
        nominal_valid_indices = nonzero_indices[nominal_valid_mask]
        nominal_valid_duration = nominal_duration[nominal_valid_mask]
        nominal_valid_count = int(nominal_valid_indices.size)

        if nominal_valid_count:
            # Only the short duration is represented as float.  Absolute ToA
            # timestamps remain uint64, so runs beyond 2**53 retain exact LSBs.
            rounded_duration = np.rint(nominal_valid_duration).astype(np.uint64)
            corresponding_trails = trail_times[nominal_valid_indices]
            no_underflow = rounded_duration <= corresponding_trails
            matchable_indices = nominal_valid_indices[no_underflow]
            matchable_duration = rounded_duration[no_underflow]

            if matchable_indices.size:
                expected_leads = trail_times[matchable_indices] - matchable_duration
                expected_order = np.argsort(expected_leads, kind="stable")
                expected_sorted = expected_leads[expected_order]
                trail_indices_sorted = matchable_indices[expected_order]

                local_pairs, _, _ = _pair_one_board_ordered_nearest(
                    lead_times,
                    expected_sorted,
                    int(tot_match_tolerance_lsb),
                    prefer_later_first_on_tie=True,
                )
                if local_pairs.size:
                    lead_indices = local_pairs[:, 0]
                    expected_indices = local_pairs[:, 1]
                    trail_indices = trail_indices_sorted[expected_indices]
                    selected_leads = lead_times[lead_indices]
                    selected_trails = trail_times[trail_indices]
                    selected_expected = expected_sorted[expected_indices]

                    before_trail = selected_leads < selected_trails
                    actual_duration = np.zeros(selected_leads.size, dtype=np.uint64)
                    actual_duration[before_trail] = (
                        selected_trails[before_trail] - selected_leads[before_trail]
                    )
                    physically_valid = (
                        before_trail
                        & (actual_duration >= np.uint64(minimum_duration_lsb))
                        & (actual_duration <= np.uint64(maximum_duration_lsb))
                    )

                    lead_indices = lead_indices[physically_valid]
                    trail_indices = trail_indices[physically_valid]
                    selected_leads = selected_leads[physically_valid]
                    selected_trails = selected_trails[physically_valid]
                    selected_expected = selected_expected[physically_valid]

                    if selected_leads.size:
                        residual = np.where(
                            selected_leads >= selected_expected,
                            selected_leads - selected_expected,
                            selected_expected - selected_leads,
                        )
                        used_leads[lead_indices] = True
                        used_trails[trail_indices] = True
                        direct_pairs = np.column_stack(
                            (
                                selected_leads,
                                selected_trails,
                                trail_tots[trail_indices].astype(np.uint64),
                            )
                        ).astype(np.uint64, copy=False)
                        residual_sum_lsb = float(
                            np.sum(residual, dtype=np.float64)
                        )
                        residual_max_lsb = float(np.max(residual))
                        pair_parts.append(direct_pairs)

            recorded_tot_no_matching_lead = int(
                nominal_valid_count - direct_pairs.shape[0]
            )

    fallback_mask = np.zeros(n_trails, dtype=bool)
    if fallback_when_tot_missing:
        fallback_mask |= trail_tots == 0
    if fallback_when_tot_inconsistent:
        fallback_mask |= (~used_trails) & (trail_tots > 0)

    fallback_indices = np.flatnonzero(fallback_mask & ~used_trails)
    remaining_lead_indices = np.flatnonzero(~used_leads)
    fallback_stats = {
        "paired": 0,
        "unmatched_leading_edges": int(remaining_lead_indices.size),
        "unmatched_trailing_edges": int(fallback_indices.size),
        "rejected_too_short": 0,
        "rejected_too_long": 0,
        "nested_leading_edges_skipped": 0,
    }
    fallback_pairs = np.empty((0, 3), dtype=np.uint64)
    if fallback_indices.size and remaining_lead_indices.size:
        fallback_local, fallback_stats = _pair_one_board_longest_valid_indices(
            lead_times[remaining_lead_indices],
            trail_times[fallback_indices],
            minimum_duration_lsb,
            maximum_duration_lsb,
        )
        if fallback_local.size:
            lead_indices = remaining_lead_indices[fallback_local[:, 0]]
            trail_indices = fallback_indices[fallback_local[:, 1]]
            used_leads[lead_indices] = True
            used_trails[trail_indices] = True
            fallback_pairs = np.column_stack(
                (
                    lead_times[lead_indices],
                    trail_times[trail_indices],
                    trail_tots[trail_indices].astype(np.uint64),
                )
            ).astype(np.uint64, copy=False)
            pair_parts.append(fallback_pairs)

    if pair_parts:
        paired = np.concatenate(pair_parts, axis=0)
        paired = paired[np.argsort(paired[:, 0], kind="stable")]
    else:
        paired = np.empty((0, 3), dtype=np.uint64)

    recorded_tot_pairs = int(direct_pairs.shape[0])
    fallback_pair_count = int(fallback_pairs.shape[0])
    return paired, {
        "paired": int(paired.shape[0]),
        "unmatched_leading_edges": int(n_leads - np.count_nonzero(used_leads)),
        "unmatched_trailing_edges": int(n_trails - np.count_nonzero(used_trails)),
        "rejected_too_short": int(
            recorded_tot_too_short + int(fallback_stats["rejected_too_short"])
        ),
        "rejected_too_long": int(
            recorded_tot_too_long + int(fallback_stats["rejected_too_long"])
        ),
        "nested_leading_edges_skipped": int(
            fallback_stats["nested_leading_edges_skipped"]
        ),
        "recorded_tot_records": recorded_tot_records,
        "missing_recorded_tot_records": missing_tot_records,
        "recorded_tot_pairs": recorded_tot_pairs,
        "fallback_pairs": fallback_pair_count,
        "recorded_tot_no_matching_lead": int(recorded_tot_no_matching_lead),
        "recorded_tot_too_short": int(recorded_tot_too_short),
        "recorded_tot_too_long": int(recorded_tot_too_long),
        "recorded_tot_residual_mean_lsb": (
            float(residual_sum_lsb / recorded_tot_pairs)
            if recorded_tot_pairs
            else 0.0
        ),
        "recorded_tot_residual_max_lsb": float(residual_max_lsb),
    }


def _pair_edges_same_board(
    lead_boards: np.ndarray,
    lead_times: np.ndarray,
    trail_boards: np.ndarray,
    trail_times: np.ndarray,
    trail_tots: np.ndarray,
    minimum_duration_lsb: int,
    maximum_duration_lsb: int,
    toa_lsb_ps: float,
    tot_lsb_ps: float,
    tot_match_tolerance_lsb: int,
    fallback_when_tot_missing: bool,
    fallback_when_tot_inconsistent: bool,
) -> tuple[np.ndarray, dict[str, int | float]]:
    boards = np.union1d(lead_boards, trail_boards)
    pulse_parts: list[np.ndarray] = []
    statistics = {
        "paired": 0,
        "unmatched_leading_edges": 0,
        "unmatched_trailing_edges": 0,
        "rejected_too_short": 0,
        "rejected_too_long": 0,
        "nested_leading_edges_skipped": 0,
        "recorded_tot_records": 0,
        "missing_recorded_tot_records": 0,
        "recorded_tot_pairs": 0,
        "fallback_pairs": 0,
        "recorded_tot_no_matching_lead": 0,
        "recorded_tot_too_short": 0,
        "recorded_tot_too_long": 0,
        "recorded_tot_residual_mean_lsb": 0.0,
        "recorded_tot_residual_max_lsb": 0.0,
    }
    residual_weighted_sum = 0.0
    for board_value in boards:
        board = int(board_value)
        board_mask = trail_boards == board_value
        paired, board_stats = _pair_one_board_recorded_tot(
            lead_times[lead_boards == board_value],
            trail_times[board_mask],
            trail_tots[board_mask],
            minimum_duration_lsb,
            maximum_duration_lsb,
            toa_lsb_ps,
            tot_lsb_ps,
            tot_match_tolerance_lsb,
            fallback_when_tot_missing,
            fallback_when_tot_inconsistent,
        )
        residual_weighted_sum += float(board_stats["recorded_tot_residual_mean_lsb"]) * int(
            board_stats["recorded_tot_pairs"]
        )
        for key in statistics:
            if key == "recorded_tot_residual_mean_lsb":
                continue
            if key == "recorded_tot_residual_max_lsb":
                statistics[key] = max(float(statistics[key]), float(board_stats[key]))
            else:
                statistics[key] += int(board_stats[key])
        if paired.size:
            board_column = np.full((paired.shape[0], 1), board, dtype=np.uint64)
            pulse_parts.append(np.hstack((board_column, paired)))

    if int(statistics["recorded_tot_pairs"]) > 0:
        statistics["recorded_tot_residual_mean_lsb"] = (
            residual_weighted_sum / int(statistics["recorded_tot_pairs"])
        )

    if not pulse_parts:
        return np.empty((0, PULSE_COLUMNS), dtype=np.uint64), statistics
    pulses = np.concatenate(pulse_parts, axis=0)
    order = np.lexsort((pulses[:, LEAD_COL], pulses[:, BOARD_COL]))
    return pulses[order], statistics


def _duration_limit_lsb(cfg: dict, channel: int, toa_lsb_ps: float) -> tuple[int, int]:
    channels = cfg["channels"]
    reconstruction = cfg["preprocessing"]["pulse_reconstruction"]
    is_energy = channel in {
        int(channels["signal_a"]),
        int(channels["signal_b"]),
    }
    maximum_ns = float(
        reconstruction["maximum_energy_tot_ns"]
        if is_energy
        else reconstruction.get("maximum_timing_tot_ns", reconstruction["maximum_energy_tot_ns"])
    )
    minimum_ns = float(reconstruction["minimum_tot_ns"])
    minimum_lsb = max(1, int(math.ceil(minimum_ns * 1000.0 / toa_lsb_ps)))
    maximum_lsb = max(
        minimum_lsb,
        int(math.floor(maximum_ns * 1000.0 / toa_lsb_ps)),
    )
    return minimum_lsb, maximum_lsb


def _pair_one_board_ordered_nearest(
    first_leads: np.ndarray,
    second_leads: np.ndarray,
    maximum_delta_lsb: int,
    prefer_later_first_on_tie: bool = False,
) -> tuple[np.ndarray, int, int]:
    """Monotonic one-to-one nearest matching for two ordered leading arrays."""
    first_leads = np.asarray(first_leads, dtype=np.uint64)
    second_leads = np.asarray(second_leads, dtype=np.uint64)
    pairs: list[tuple[int, int]] = []
    i = 0
    j = 0
    skipped_first = 0
    skipped_second = 0
    n_first = int(first_leads.size)
    n_second = int(second_leads.size)

    while i < n_first and j < n_second:
        first = int(first_leads[i])
        second = int(second_leads[j])
        delta = first - second
        if delta < -maximum_delta_lsb:
            skipped_first += 1
            i += 1
            continue
        if delta > maximum_delta_lsb:
            skipped_second += 1
            j += 1
            continue

        current_distance = abs(delta)
        if i + 1 < n_first:
            next_first_distance = abs(int(first_leads[i + 1]) - second)
            if (
                next_first_distance <= maximum_delta_lsb
                and (
                    next_first_distance < current_distance
                    or (
                        prefer_later_first_on_tie
                        and next_first_distance == current_distance
                    )
                )
            ):
                skipped_first += 1
                i += 1
                continue
        if j + 1 < n_second:
            next_second_distance = abs(first - int(second_leads[j + 1]))
            if (
                next_second_distance <= maximum_delta_lsb
                and next_second_distance < current_distance
            ):
                skipped_second += 1
                j += 1
                continue

        pairs.append((i, j))
        i += 1
        j += 1

    skipped_first += n_first - i
    skipped_second += n_second - j
    output = (
        np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        if pairs
        else np.empty((0, 2), dtype=np.int64)
    )
    return output, skipped_first, skipped_second


def pair_ordered_energy_leads(
    first: np.ndarray,
    second: np.ndarray,
    maximum_delta_lsb: int,
    require_same_board: bool = True,
) -> tuple[np.ndarray, dict[str, int]]:
    """Pair ch1/ch5 pulses using only ordered leading-edge timestamps."""
    first = np.asarray(first)
    second = np.asarray(second)
    parts: list[np.ndarray] = []
    skipped_first = 0
    skipped_second = 0

    if require_same_board:
        boards = np.intersect1d(first[:, BOARD_COL], second[:, BOARD_COL])
        first_board_values = set(int(value) for value in np.unique(first[:, BOARD_COL]))
        second_board_values = set(int(value) for value in np.unique(second[:, BOARD_COL]))
        for board in boards:
            first_indices = np.flatnonzero(first[:, BOARD_COL] == board)
            second_indices = np.flatnonzero(second[:, BOARD_COL] == board)
            local, local_skip_first, local_skip_second = _pair_one_board_ordered_nearest(
                first[first_indices, LEAD_COL],
                second[second_indices, LEAD_COL],
                maximum_delta_lsb,
            )
            skipped_first += local_skip_first
            skipped_second += local_skip_second
            if local.size:
                parts.append(
                    np.column_stack(
                        (first_indices[local[:, 0]], second_indices[local[:, 1]])
                    ).astype(np.int64, copy=False)
                )
        for board in first_board_values - second_board_values:
            skipped_first += int(np.count_nonzero(first[:, BOARD_COL] == board))
        for board in second_board_values - first_board_values:
            skipped_second += int(np.count_nonzero(second[:, BOARD_COL] == board))
    else:
        local, skipped_first, skipped_second = _pair_one_board_ordered_nearest(
            first[:, LEAD_COL], second[:, LEAD_COL], maximum_delta_lsb
        )
        parts.append(local)

    if parts:
        paired = np.concatenate(parts, axis=0)
        order = np.argsort(first[paired[:, 0], LEAD_COL], kind="stable")
        paired = paired[order]
    else:
        paired = np.empty((0, 2), dtype=np.int64)
    return paired, {
        "pairs": int(paired.shape[0]),
        "unpaired_first": int(skipped_first),
        "unpaired_second": int(skipped_second),
    }


def pair_overlapping_pulse_arrays(
    first: np.ndarray,
    second: np.ndarray,
    require_same_board: bool = True,
) -> tuple[np.ndarray, dict[str, int]]:
    """Legacy overlap matcher retained for external imports and comparisons."""
    first = np.asarray(first)
    second = np.asarray(second)
    parts: list[np.ndarray] = []
    if require_same_board:
        boards = np.intersect1d(first[:, BOARD_COL], second[:, BOARD_COL])
    else:
        boards = np.asarray([0], dtype=np.int64)
    for board in boards:
        if require_same_board:
            first_indices = np.flatnonzero(first[:, BOARD_COL] == board)
            second_indices = np.flatnonzero(second[:, BOARD_COL] == board)
        else:
            first_indices = np.arange(first.shape[0])
            second_indices = np.arange(second.shape[0])
        i = j = 0
        local: list[tuple[int, int]] = []
        while i < first_indices.size and j < second_indices.size:
            fi = int(first_indices[i])
            sj = int(second_indices[j])
            if int(first[fi, LEAD_COL]) <= int(second[sj, TRAIL_COL]) and int(
                second[sj, LEAD_COL]
            ) <= int(first[fi, TRAIL_COL]):
                local.append((fi, sj))
                i += 1
                j += 1
            elif int(first[fi, TRAIL_COL]) < int(second[sj, LEAD_COL]):
                i += 1
            else:
                j += 1
        if local:
            parts.append(np.asarray(local, dtype=np.int64))
    paired = np.concatenate(parts, axis=0) if parts else np.empty((0, 2), dtype=np.int64)
    return paired, {
        "pairs": int(paired.shape[0]),
        "unpaired_first": int(first.shape[0] - paired.shape[0]),
        "unpaired_second": int(second.shape[0] - paired.shape[0]),
    }


def _board_times(boards: np.ndarray, times: np.ndarray) -> dict[int, np.ndarray]:
    return {
        int(board): np.asarray(times[boards == board], dtype=np.uint64)
        for board in np.unique(boards)
    }


def _preceding_candidates(
    by_board: dict[int, np.ndarray],
    board: int,
    energy_lead: int,
    window_lsb: int,
) -> np.ndarray:
    values = by_board.get(int(board))
    if values is None or values.size == 0:
        return np.empty(0, dtype=np.uint64)
    lower = max(0, energy_lead - window_lsb)
    start = int(np.searchsorted(values, lower, side="right"))
    stop = int(np.searchsorted(values, energy_lead, side="left"))
    return np.asarray(values[start:stop], dtype=np.uint64)


def decode_streaming_pulse_cache(
    input_path: str | Path,
    cache_dir: str | Path,
    cfg: dict,
    acquisition_mode: str,
) -> dict[str, Any]:
    """Create the only persistent STREAMING preprocessing cache.

    Processing is performed in memory:
    1. collect ordered leading edges for ch1/ch5 and timing channels;
    2. reconstruct only the two energy pulses, because their ToT is required;
    3. pair energy pulses by ordered leading times within a configurable window;
    4. build independent timing-candidate lists relative to each energy side;
    5. store only matched energy events and their candidate leading timestamps.

    Per-channel pulse arrays and ch3-ch7 overlap-pair caches are not written.
    """
    input_path = Path(input_path)
    cache_dir = Path(cache_dir)
    channels_cfg = cfg["channels"]
    signal_a = int(channels_cfg["signal_a"])
    time_a = int(channels_cfg["time_a"])
    signal_b = int(channels_cfg["signal_b"])
    time_b = int(channels_cfg["time_b"])
    channels = (signal_a, time_a, signal_b, time_b)
    energy_channels = {signal_a, signal_b}
    reconstruction = cfg["preprocessing"]["pulse_reconstruction"]
    deduplicate = bool(reconstruction["deduplicate_exact_edges"])

    lead_times: dict[int, array] = {channel: array("Q") for channel in channels}
    lead_boards: dict[int, array] = {channel: array("B") for channel in channels}
    trail_times: dict[int, array] = {channel: array("Q") for channel in energy_channels}
    trail_boards: dict[int, array] = {channel: array("B") for channel in energy_channels}
    trail_tots: dict[int, array] = {channel: array("H") for channel in energy_channels}
    raw_leading = {channel: 0 for channel in channels}
    raw_trailing = {channel: 0 for channel in channels}
    nonzero_tot_on_leading = {channel: 0 for channel in channels}
    records_read = 0
    relevant_hits = 0

    with input_path.open("rb") as source:
        meta = read_header(source, acquisition_mode)
        if meta.acquisition_mode != STREAMING:
            raise DataError("STREAMING cache requested for a non-STREAMING run")
        if meta.measurement_mode != LEAD_TRAIL:
            raise DataError("Timing analysis requires LEAD_TRAIL measurement mode")
        for record in iter_events(source, meta):
            records_read += 1
            for hit in record.hits:
                channel = int(hit.channel)
                if channel not in lead_times:
                    continue
                relevant_hits += 1
                if int(hit.edge) == 1:
                    raw_leading[channel] += 1
                    if hit.tot_lsb not in (None, 0):
                        nonzero_tot_on_leading[channel] += 1
                    lead_times[channel].append(int(hit.toa_lsb))
                    lead_boards[channel].append(int(hit.board))
                elif int(hit.edge) == 0:
                    raw_trailing[channel] += 1
                    if channel in energy_channels:
                        if hit.tot_lsb is None:
                            raise DataError(
                                "STREAMING LEAD_TRAIL trailing record is missing its ToT field"
                            )
                        trail_times[channel].append(int(hit.toa_lsb))
                        trail_boards[channel].append(int(hit.board))
                        trail_tots[channel].append(int(hit.tot_lsb))

    sorted_lead_boards: dict[int, np.ndarray] = {}
    sorted_leads: dict[int, np.ndarray] = {}
    duplicate_leads: dict[int, int] = {}
    for channel in channels:
        boards, values, duplicates = _sorted_unique_edges(
            lead_boards[channel], lead_times[channel], deduplicate
        )
        sorted_lead_boards[channel] = boards
        sorted_leads[channel] = values
        duplicate_leads[channel] = duplicates

    energy_pulses: dict[int, np.ndarray] = {}
    energy_pair_stats: dict[int, dict[str, int | float]] = {}
    duplicate_trails: dict[int, int] = {}
    reconstruction_cfg = cfg["preprocessing"]["pulse_reconstruction"]
    tot_match_tolerance_ns = float(
        reconstruction_cfg["tot_lead_match_tolerance_ns"]
    )
    configured_tot_match_tolerance_lsb = max(
        0,
        int(math.ceil(tot_match_tolerance_ns * 1000.0 / float(meta.toa_lsb_ps))),
    )
    # The recorded ToT may be quantized more coarsely than the ToA.  Never use
    # a matching tolerance smaller than one ToT output bin expressed on the
    # ToA axis, otherwise a perfectly valid edge pair can be rejected solely
    # because of output quantization.
    tot_quantization_tolerance_lsb = max(
        1,
        int(math.ceil(float(meta.tot_lsb_ps) / float(meta.toa_lsb_ps))),
    )
    tot_match_tolerance_lsb = max(
        configured_tot_match_tolerance_lsb,
        tot_quantization_tolerance_lsb,
    )
    fallback_when_tot_missing = bool(
        reconstruction_cfg["fallback_when_tot_missing"]
    )
    fallback_when_tot_inconsistent = bool(
        reconstruction_cfg["fallback_when_tot_inconsistent"]
    )
    for channel in energy_channels:
        trail_board, trails, tots, duplicates = _sorted_unique_trailing_records(
            trail_boards[channel],
            trail_times[channel],
            trail_tots[channel],
            deduplicate,
        )
        duplicate_trails[channel] = duplicates
        minimum_lsb, maximum_lsb = _duration_limit_lsb(cfg, channel, meta.toa_lsb_ps)
        pulses, statistics = _pair_edges_same_board(
            sorted_lead_boards[channel],
            sorted_leads[channel],
            trail_board,
            trails,
            tots,
            minimum_lsb,
            maximum_lsb,
            float(meta.toa_lsb_ps),
            float(meta.tot_lsb_ps),
            tot_match_tolerance_lsb,
            fallback_when_tot_missing,
            fallback_when_tot_inconsistent,
        )
        energy_pulses[channel] = pulses
        energy_pair_stats[channel] = statistics

    streaming_cfg = cfg["preprocessing"]["streaming_physical_time"]
    energy_window_ns = float(streaming_cfg["energy_leading_pair_window_ns"])
    energy_window_lsb = max(
        1, int(math.ceil(energy_window_ns * 1000.0 / float(meta.toa_lsb_ps)))
    )
    require_same_board = bool(streaming_cfg["require_same_board"])
    energy_pairs, energy_stats = pair_ordered_energy_leads(
        energy_pulses[signal_a],
        energy_pulses[signal_b],
        energy_window_lsb,
        require_same_board=require_same_board,
    )

    candidate_window_ns = max(
        float(cfg["matching_model"]["training"]["window_ns"]),
        float(cfg["matching_model"]["inference"]["candidate_window_ns"]),
    )
    candidate_window_lsb = max(
        1, int(math.ceil(candidate_window_ns * 1000.0 / float(meta.toa_lsb_ps)))
    )
    timing_by_board_a = _board_times(sorted_lead_boards[time_a], sorted_leads[time_a])
    timing_by_board_b = _board_times(sorted_lead_boards[time_b], sorted_leads[time_b])

    event_board_a: list[int] = []
    event_energy_lead_a: list[int] = []
    event_energy_trail_a: list[int] = []
    event_energy_tot_a: list[int] = []
    event_board_b: list[int] = []
    event_energy_lead_b: list[int] = []
    event_energy_trail_b: list[int] = []
    event_energy_tot_b: list[int] = []
    timing_candidates_a = array("Q")
    timing_candidates_b = array("Q")
    offsets_a = array("q", [0])
    offsets_b = array("q", [0])
    missing_a = 0
    missing_b = 0
    missing_either = 0

    for pair in energy_pairs:
        pulse_a = energy_pulses[signal_a][int(pair[0])]
        pulse_b = energy_pulses[signal_b][int(pair[1])]
        board_a = int(pulse_a[BOARD_COL])
        board_b = int(pulse_b[BOARD_COL])
        lead_a = int(pulse_a[LEAD_COL])
        lead_b = int(pulse_b[LEAD_COL])
        candidates_a = _preceding_candidates(
            timing_by_board_a, board_a, lead_a, candidate_window_lsb
        )
        candidates_b = _preceding_candidates(
            timing_by_board_b, board_b, lead_b, candidate_window_lsb
        )
        if candidates_a.size == 0:
            missing_a += 1
        if candidates_b.size == 0:
            missing_b += 1
        if candidates_a.size == 0 or candidates_b.size == 0:
            missing_either += 1
            continue

        event_board_a.append(board_a)
        event_energy_lead_a.append(lead_a)
        event_energy_trail_a.append(int(pulse_a[TRAIL_COL]))
        event_energy_tot_a.append(int(pulse_a[TOT_COL]))
        event_board_b.append(board_b)
        event_energy_lead_b.append(lead_b)
        event_energy_trail_b.append(int(pulse_b[TRAIL_COL]))
        event_energy_tot_b.append(int(pulse_b[TOT_COL]))
        timing_candidates_a.extend(int(value) for value in candidates_a)
        timing_candidates_b.extend(int(value) for value in candidates_b)
        offsets_a.append(len(timing_candidates_a))
        offsets_b.append(len(timing_candidates_b))

    channel_metadata: dict[str, Any] = {}
    for channel in channels:
        if channel in energy_channels:
            pair_stats = energy_pair_stats[channel]
            pulse_count = int(energy_pulses[channel].shape[0])
            trail_duplicates = int(duplicate_trails[channel])
        else:
            pair_stats = {
                "unmatched_leading_edges": 0,
                "unmatched_trailing_edges": 0,
                "rejected_too_short": 0,
                "rejected_too_long": 0,
                "nested_leading_edges_skipped": 0,
                "recorded_tot_records": 0,
                "missing_recorded_tot_records": 0,
                "recorded_tot_pairs": 0,
                "fallback_pairs": 0,
                "recorded_tot_no_matching_lead": 0,
                "recorded_tot_too_short": 0,
                "recorded_tot_too_long": 0,
                "recorded_tot_residual_mean_lsb": 0.0,
                "recorded_tot_residual_max_lsb": 0.0,
            }
            pulse_count = int(sorted_leads[channel].size)
            trail_duplicates = 0
        channel_metadata[str(channel)] = {
            "leading_edges_raw": int(raw_leading[channel]),
            "trailing_edges_raw": int(raw_trailing[channel]),
            "duplicate_leading_edges_removed": int(duplicate_leads[channel]),
            "duplicate_trailing_edges_removed": trail_duplicates,
            "leading_edges_unique": int(sorted_leads[channel].size),
            "pulses": pulse_count,
            "unmatched_leading_edges": int(pair_stats["unmatched_leading_edges"]),
            "unmatched_trailing_edges": int(pair_stats["unmatched_trailing_edges"]),
            "rejected_tot_too_short": int(pair_stats["rejected_too_short"]),
            "rejected_tot_too_long": int(pair_stats["rejected_too_long"]),
            "nested_leading_edges_skipped": int(
                pair_stats["nested_leading_edges_skipped"]
            ),
            "nonzero_tot_on_leading_records": int(nonzero_tot_on_leading[channel]),
            "recorded_tot_records": int(pair_stats["recorded_tot_records"]),
            "missing_recorded_tot_records": int(
                pair_stats["missing_recorded_tot_records"]
            ),
            "recorded_tot_pairs": int(pair_stats["recorded_tot_pairs"]),
            "fallback_pairs": int(pair_stats["fallback_pairs"]),
            "recorded_tot_no_matching_lead": int(
                pair_stats["recorded_tot_no_matching_lead"]
            ),
            "recorded_tot_too_short": int(pair_stats["recorded_tot_too_short"]),
            "recorded_tot_too_long": int(pair_stats["recorded_tot_too_long"]),
            "recorded_tot_residual_mean_lsb": float(
                pair_stats["recorded_tot_residual_mean_lsb"]
            ),
            "recorded_tot_residual_max_lsb": float(
                pair_stats["recorded_tot_residual_max_lsb"]
            ),
            "stored_energy_pulses": bool(channel in energy_channels),
        }

    metadata: dict[str, Any] = {
        "raw_records_read": int(records_read),
        "relevant_hits_read": int(relevant_hits),
        "toa_lsb_ps": float(meta.toa_lsb_ps),
        "tot_lsb_ps": float(meta.tot_lsb_ps),
        "timestamp_lsb_ps": float(meta.timestamp_lsb_ps),
        "channels": channel_metadata,
        "physical_time_source": "streaming_hit_toa_uint64",
        "event_timestamp_used_for_time": False,
        "energy_pulse_reconstruction_rule": (
            "recorded_trailing_tot_nearest_expected_lead_with_configured_fallback"
        ),
        "tot_lead_match_tolerance_ns": tot_match_tolerance_ns,
        "configured_tot_lead_match_tolerance_lsb": (
            configured_tot_match_tolerance_lsb
        ),
        "tot_quantization_tolerance_lsb": tot_quantization_tolerance_lsb,
        "tot_lead_match_tolerance_lsb": tot_match_tolerance_lsb,
        "fallback_when_tot_missing": fallback_when_tot_missing,
        "fallback_when_tot_inconsistent": fallback_when_tot_inconsistent,
        "energy_pairing_rule": "ordered_leading_edges_nearest_within_window",
        "energy_leading_pair_window_ns": energy_window_ns,
        "energy_leading_pair_window_lsb": energy_window_lsb,
        "energy_leading_pairs_total": int(energy_pairs.shape[0]),
        "energy_unpaired_a": int(energy_stats["unpaired_first"]),
        "energy_unpaired_b": int(energy_stats["unpaired_second"]),
        "timing_candidate_rule": "independent_preceding_leads_per_energy_side",
        "candidate_window_ns": candidate_window_ns,
        "candidate_window_lsb": candidate_window_lsb,
        "energy_pairs_without_timing_candidate_a": int(missing_a),
        "energy_pairs_without_timing_candidate_b": int(missing_b),
        "energy_pairs_without_complete_timing_candidates": int(missing_either),
        "candidate_events": int(len(event_energy_lead_a)),
        "candidate_references_a": int(len(timing_candidates_a)),
        "candidate_references_b": int(len(timing_candidates_b)),
        "maximum_candidates_a": int(max(np.diff(np.asarray(offsets_a, dtype=np.int64)), default=0)),
        "maximum_candidates_b": int(max(np.diff(np.asarray(offsets_b, dtype=np.int64)), default=0)),
        "require_same_board": require_same_board,
        "cache_format": "streaming_matched_energy_recorded_tot_independent_timing_csr_v2",
        "cached_intermediate_pulse_arrays": False,
        "cached_timing_pair_index": False,
        "version": STREAMING_EVENT_CACHE_VERSION,
    }

    arrays = {
        "energy_board_a": np.asarray(event_board_a, dtype=np.uint8),
        "energy_lead_a": np.asarray(event_energy_lead_a, dtype=np.uint64),
        "energy_trail_a": np.asarray(event_energy_trail_a, dtype=np.uint64),
        "energy_tot_a": np.asarray(event_energy_tot_a, dtype=np.uint16),
        "energy_board_b": np.asarray(event_board_b, dtype=np.uint8),
        "energy_lead_b": np.asarray(event_energy_lead_b, dtype=np.uint64),
        "energy_trail_b": np.asarray(event_energy_trail_b, dtype=np.uint64),
        "energy_tot_b": np.asarray(event_energy_tot_b, dtype=np.uint16),
        "timing_offsets_a": np.asarray(offsets_a, dtype=np.int64),
        "timing_leads_a": np.frombuffer(timing_candidates_a, dtype=np.uint64).copy(),
        "timing_offsets_b": np.asarray(offsets_b, dtype=np.int64),
        "timing_leads_b": np.frombuffer(timing_candidates_b, dtype=np.uint64).copy(),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    cache_path = cache_dir / STREAMING_EVENT_CACHE_FILE
    _atomic_save_npz(cache_path, arrays)
    # Remove legacy per-channel caches after the new cache is safely committed.
    for legacy in cache_dir.glob("ch*_pulses.npy"):
        legacy.unlink(missing_ok=True)
    (cache_dir / "metadata.json").unlink(missing_ok=True)
    return metadata


def load_streaming_event_cache(
    cache_dir: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    path = Path(cache_dir) / STREAMING_EVENT_CACHE_FILE
    if not path.exists():
        raise DataError(f"Missing STREAMING event cache: {path}")
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "energy_board_a",
            "energy_lead_a",
            "energy_trail_a",
            "energy_tot_a",
            "energy_board_b",
            "energy_lead_b",
            "energy_trail_b",
            "energy_tot_b",
            "timing_offsets_a",
            "timing_leads_a",
            "timing_offsets_b",
            "timing_leads_b",
            "metadata_json",
        }
        missing = required - set(archive.files)
        if missing:
            raise DataError(
                "Invalid STREAMING event cache; missing: " + ", ".join(sorted(missing))
            )
        arrays = {
            key: np.asarray(archive[key])
            for key in required
            if key != "metadata_json"
        }
        metadata = json.loads(str(archive["metadata_json"].item()))

    if int(metadata.get("version", -1)) != STREAMING_EVENT_CACHE_VERSION:
        raise DataError(
            "Obsolete STREAMING event cache version; rerun streaming_pulse_decode "
            "to rebuild it with recorded-ToT pulse reconstruction"
        )

    event_count = int(arrays["energy_lead_a"].size)
    for key in (
        "energy_board_a",
        "energy_trail_a",
        "energy_tot_a",
        "energy_board_b",
        "energy_lead_b",
        "energy_trail_b",
        "energy_tot_b",
    ):
        if arrays[key].size != event_count:
            raise DataError(f"STREAMING event cache has inconsistent {key} length")
    for side in ("a", "b"):
        offsets = arrays[f"timing_offsets_{side}"]
        leads = arrays[f"timing_leads_{side}"]
        if offsets.size != event_count + 1 or int(offsets[0]) != 0:
            raise DataError(f"Invalid STREAMING timing offsets for side {side}")
        if int(offsets[-1]) != leads.size or np.any(np.diff(offsets) < 0):
            raise DataError(f"Invalid STREAMING timing CSR data for side {side}")
    return arrays, metadata


def streaming_energy_duration_lsb(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    side: str,
) -> np.ndarray:
    """Return matched-edge durations in ToA-LSB units for one cached side.

    The recorded ToT is used to identify the correct leading edge.  Once the
    pair is known, ``trail - lead`` is retained as the operational duration so
    downstream selection uses the full ToA resolution and remains exactly
    consistent with the matched binary written by preprocessing.
    """
    if side not in {"a", "b"}:
        raise DataError(f"Invalid STREAMING side {side!r}")
    lead = arrays[f"energy_lead_{side}"].astype(np.int64, copy=False)
    trail = arrays[f"energy_trail_{side}"].astype(np.int64, copy=False)
    del metadata
    return (trail - lead).astype(np.int64, copy=False)


def load_streaming_pulse_cache(
    cache_dir: str | Path,
    channels: list[int] | tuple[int, ...] | set[int],
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Compatibility view for energy channels only.

    New code should use :func:`load_streaming_event_cache`.  Timing pulse arrays
    are deliberately unavailable because only candidate leading timestamps are
    persisted.
    """
    arrays, metadata = load_streaming_event_cache(cache_dir)
    requested = set(int(value) for value in channels)
    channel_meta = metadata.get("channels", {})
    known = sorted(int(value) for value in channel_meta)
    if len(known) != 4:
        raise DataError("STREAMING cache channel metadata is incomplete")
    # Infer energy channels as the two channels with reconstructed trailing data.
    energy_channels = [
        channel
        for channel in known
        if bool(channel_meta[str(channel)].get("stored_energy_pulses", False))
    ]
    if len(energy_channels) != 2 or not requested.issubset(set(energy_channels)):
        raise DataError(
            "The compact STREAMING cache no longer stores per-channel timing pulses; "
            "use load_streaming_event_cache()"
        )
    output: dict[int, np.ndarray] = {}
    for channel, side in zip(sorted(energy_channels), ("a", "b")):
        if channel not in requested:
            continue
        output[channel] = np.column_stack(
            (
                arrays[f"energy_board_{side}"],
                arrays[f"energy_lead_{side}"],
                arrays[f"energy_trail_{side}"],
            )
        ).astype(np.uint64, copy=False)
    return output, metadata


def build_streaming_candidate_index(
    pulse_cache_dir: str | Path,
    index_dir: str | Path,
    cfg: dict,
) -> dict[str, Any]:
    """Compatibility stage: candidate indexing is already inside the one cache."""
    del cfg
    _, metadata = load_streaming_event_cache(pulse_cache_dir)
    # The former directory held four large NPY index arrays.  It is no longer
    # part of the cache and is removed when an old analysis tree is upgraded.
    legacy_root = Path(index_dir)
    if legacy_root.exists():
        for path in legacy_root.iterdir():
            if path.is_file():
                path.unlink(missing_ok=True)
        try:
            legacy_root.rmdir()
        except OSError:
            pass
    return metadata


def load_streaming_candidate_index(
    index_dir: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    del index_dir
    raise DataError(
        "Separate STREAMING candidate-index caches were removed; "
        "use load_streaming_event_cache()"
    )


def collect_streaming_energy_measurements(
    pulse_cache_dir: str | Path,
    index_dir: str | Path,
    cfg: dict,
) -> tuple[EnergyMeasurements, float]:
    del index_dir, cfg
    arrays, metadata = load_streaming_event_cache(pulse_cache_dir)
    event_count = int(arrays["energy_lead_a"].size)
    measurements = EnergyMeasurements(
        event_index=np.arange(event_count, dtype=np.int64),
        duration_a_lsb=streaming_energy_duration_lsb(arrays, metadata, "a"),
        duration_b_lsb=streaming_energy_duration_lsb(arrays, metadata, "b"),
        energy_a_lsb=arrays["energy_lead_a"].astype(np.int64),
        energy_b_lsb=arrays["energy_lead_b"].astype(np.int64),
    )
    return measurements, float(metadata["toa_lsb_ps"])
