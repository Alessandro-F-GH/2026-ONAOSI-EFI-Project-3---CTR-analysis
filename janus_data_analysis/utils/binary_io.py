from __future__ import annotations

import csv
import hashlib
import math
import os
import re
import struct
from collections.abc import Iterator
from pathlib import Path
from typing import Any, BinaryIO

from .models import BinaryMeta, Event, Hit, RunInfo, RunInput

HEADER_SIZE = 33
TRIGGER_EVENT_HEADER_STRUCT = struct.Struct("<HQQH")
TRIGGER_EVENT_HEADER_SIZE = TRIGGER_EVENT_HEADER_STRUCT.size
STREAMING_EVENT_HEADER_STRUCT = struct.Struct("<HQH")
STREAMING_EVENT_HEADER_SIZE = STREAMING_EVENT_HEADER_STRUCT.size
TRIGGER_MATCHING = 0x32
STREAMING = 0x22
TIME_UNIT_LSB = 0x00
LEAD_ONLY = 0x01
LEAD_TRAIL = 0x03
STREAMING_HIT_LSB_LEAD_ONLY_STRUCT = struct.Struct("<BBBQ")
STREAMING_HIT_LSB_LEAD_TRAIL_STRUCT = struct.Struct("<BBBQH")
TRIGGER_HIT_LSB_LEAD_ONLY_STRUCT = struct.Struct("<BBBI")
TRIGGER_HIT_LSB_WITH_TOT_STRUCT = struct.Struct("<BBBIH")
SUPPORTED_ACQUISITION_MODES = {TRIGGER_MATCHING, STREAMING}
SUPPORTED_MEASUREMENT_MODES = {0x01, 0x03, 0x05, 0x09}
ACQUISITION_MODE_NAMES = {
    TRIGGER_MATCHING: "TRG_MATCHING",
    STREAMING: "STREAMING",
}
ACQUISITION_MODE_CODES = {
    "TRG_MATCHING": TRIGGER_MATCHING,
    "TRIGGER_MATCHING": TRIGGER_MATCHING,
    "STREAMING": STREAMING,
}
RUN_RE = re.compile(r"(?i)Run[_-]?(\d+)")
DEFAULT_THRESHOLD_RE = re.compile(r"^\s*DiscrThreshold\s+([-+]?\d+(?:\.\d+)?)\b", re.I)
CHANNEL_THRESHOLD_RE = re.compile(
    r"^\s*DiscrThreshold\s*\[\s*(\d+)\s*\]\s*\[\s*(\d+)\s*\]\s+([-+]?\d+(?:\.\d+)?)\b",
    re.I,
)
ACQUISITION_MODE_RE = re.compile(r"^\s*AcquisitionMode\s+([A-Za-z0-9_]+)\b", re.I)


class DataError(RuntimeError):
    pass


def canonical_run_id(text: str) -> tuple[str, str]:
    stripped = str(text).strip()
    if stripped.isdigit():
        return f"Run{stripped}", stripped
    match = RUN_RE.search(stripped)
    if not match:
        raise DataError(f"Cannot extract run ID from {text!r}")
    number = match.group(1)
    return f"Run{number}", number


def voltage_from_run_number(number: str) -> int:
    if len(number) < 2:
        raise DataError(f"Run number {number!r} has fewer than two digits")
    return int(number[:2])


def discover_runs(input_dir: str | Path, pattern: str, recursive: bool) -> list[RunInput]:
    input_dir = Path(input_dir)
    data_files = list(input_dir.rglob(pattern) if recursive else input_dir.glob(pattern))
    runs: list[RunInput] = []
    seen: set[str] = set()
    for data_path in sorted(data_files, key=lambda path: path.name.lower()):
        run_id, number = canonical_run_id(data_path.name)
        if run_id in seen:
            raise DataError(f"Duplicate data files found for {run_id}")
        candidates = [
            path
            for path in data_path.parent.iterdir()
            if path.is_file() and path.name.lower() == f"{run_id}_info.txt".lower()
        ]
        if len(candidates) != 1:
            raise DataError(f"Expected one {run_id}_Info.txt beside {data_path.name}, found {len(candidates)}")
        seen.add(run_id)
        runs.append(
            RunInput(
                run_id=run_id,
                run_number=number,
                voltage=voltage_from_run_number(number),
                data_path=data_path.resolve(),
                info_path=candidates[0].resolve(),
            )
        )
    return runs


def parse_run_info(info_path: str | Path, consistency: str) -> RunInfo:
    info_path = Path(info_path)
    default_threshold: float | None = None
    overrides: dict[int, float] = {}
    acquisition_mode: str | None = None
    with info_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            channel_match = CHANNEL_THRESHOLD_RE.match(line)
            if channel_match:
                if int(channel_match.group(1)) == 0:
                    overrides[int(channel_match.group(2))] = float(channel_match.group(3))
                continue
            default_match = DEFAULT_THRESHOLD_RE.match(line)
            if default_match:
                default_threshold = float(default_match.group(1))
                continue
            mode_match = ACQUISITION_MODE_RE.match(line)
            if mode_match:
                raw_mode = mode_match.group(1).upper()
                if raw_mode not in ACQUISITION_MODE_CODES:
                    raise DataError(f"Unsupported AcquisitionMode {raw_mode!r} in {info_path.name}")
                acquisition_mode = ACQUISITION_MODE_NAMES[ACQUISITION_MODE_CODES[raw_mode]]
    if default_threshold is None:
        raise DataError(f"No default DiscrThreshold found in {info_path}")
    if acquisition_mode is None:
        raise DataError(f"No AcquisitionMode found in {info_path}")
    selected = {
        channel: overrides.get(channel, default_threshold)
        for channel in (1, 3, 5, 7)
    }
    mismatches: list[str] = []
    if not math.isclose(selected[1], selected[5], rel_tol=0.0, abs_tol=1e-12):
        mismatches.append(f"ch1={selected[1]:g}, ch5={selected[5]:g}")
    if not math.isclose(selected[3], selected[7], rel_tol=0.0, abs_tol=1e-12):
        mismatches.append(f"ch3={selected[3]:g}, ch7={selected[7]:g}")
    if mismatches:
        message = f"Threshold inconsistency in {info_path.name}: " + "; ".join(mismatches)
        if consistency.lower() == "error":
            raise DataError(message)
        print(f"[thresholds] WARNING: {message}", flush=True)
    return RunInfo(acquisition_mode, selected[1], selected[3])


def acquisition_mode_code(name: str) -> int:
    normalised = str(name).strip().upper()
    try:
        return ACQUISITION_MODE_CODES[normalised]
    except KeyError as exc:
        raise DataError(f"Unsupported acquisition mode {name!r}") from exc


def acquisition_mode_name(code: int) -> str:
    try:
        return ACQUISITION_MODE_NAMES[int(code)]
    except KeyError as exc:
        raise DataError(f"Unsupported acquisition mode 0x{int(code):04x}") from exc


def file_signature(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    stat = path.stat()
    chunk_size = 65536
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(chunk_size))
        if stat.st_size > chunk_size:
            handle.seek(max(0, stat.st_size - chunk_size))
            digest.update(handle.read(chunk_size))
    return {
        "size": stat.st_size,
        "edge_sha256": digest.hexdigest(),
    }


def legacy_file_signature(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _read_exact(handle: BinaryIO, size: int, label: str) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise DataError(f"Truncated {label}: expected {size} bytes, found {len(data)}")
    return data


def read_header(handle: BinaryIO, expected_acquisition_mode: str | None = None) -> BinaryMeta:
    raw = _read_exact(handle, HEADER_SIZE, "binary header")
    data_format_major = int(raw[0])
    data_format_minor = int(raw[1])
    if data_format_major != 3 or data_format_minor < 2:
        raise DataError(
            "Unsupported Janus data-format version "
            f"{data_format_major}.{data_format_minor}; this decoder requires >= 3.2"
        )
    (
        _fers_version,
        run_number,
        acquisition_mode,
        measurement_mode,
        time_unit,
        toa_lsb_ps,
        tot_lsb_ps,
        timestamp_lsb_ps,
        _start_run_ms,
    ) = struct.unpack_from("<HHHBBfffQ", raw, 5)
    if acquisition_mode not in SUPPORTED_ACQUISITION_MODES:
        raise DataError(f"Unsupported acquisition mode 0x{acquisition_mode:04x}")
    if expected_acquisition_mode is not None:
        expected_code = acquisition_mode_code(expected_acquisition_mode)
        if acquisition_mode != expected_code:
            raise DataError(
                f"Acquisition mode mismatch: Info file says {acquisition_mode_name(expected_code)}, "
                f"binary header says {acquisition_mode_name(acquisition_mode)}"
            )
    if measurement_mode not in SUPPORTED_MEASUREMENT_MODES:
        raise DataError(f"Unsupported measurement mode 0x{measurement_mode:02x}")
    if time_unit != TIME_UNIT_LSB:
        raise DataError("Binary input must store ToA in integer LSB units")
    if not math.isfinite(toa_lsb_ps) or toa_lsb_ps <= 0:
        raise DataError(f"Invalid ToA LSB: {toa_lsb_ps}")
    if measurement_mode == LEAD_TRAIL and (
        not math.isfinite(tot_lsb_ps) or tot_lsb_ps <= 0
    ):
        raise DataError(f"Invalid ToT LSB for LEAD_TRAIL data: {tot_lsb_ps}")
    if not math.isfinite(timestamp_lsb_ps) or timestamp_lsb_ps <= 0:
        raise DataError(f"Invalid timestamp LSB: {timestamp_lsb_ps}")
    return BinaryMeta(
        raw_header=raw,
        run_number=int(run_number),
        acquisition_mode=int(acquisition_mode),
        measurement_mode=int(measurement_mode),
        time_unit=int(time_unit),
        toa_lsb_ps=float(toa_lsb_ps),
        tot_lsb_ps=float(tot_lsb_ps),
        timestamp_lsb_ps=float(timestamp_lsb_ps),
    )


def _has_tot(meta: BinaryMeta) -> bool:
    if meta.acquisition_mode == STREAMING:
        return meta.measurement_mode == LEAD_TRAIL
    return meta.measurement_mode != LEAD_ONLY


def _event_header_size(meta: BinaryMeta) -> int:
    return STREAMING_EVENT_HEADER_SIZE if meta.acquisition_mode == STREAMING else TRIGGER_EVENT_HEADER_SIZE


def hit_size(meta: BinaryMeta) -> int:
    if meta.acquisition_mode == STREAMING:
        return (
            STREAMING_HIT_LSB_LEAD_TRAIL_STRUCT.size
            if _has_tot(meta)
            else STREAMING_HIT_LSB_LEAD_ONLY_STRUCT.size
        )
    return (
        TRIGGER_HIT_LSB_WITH_TOT_STRUCT.size
        if _has_tot(meta)
        else TRIGGER_HIT_LSB_LEAD_ONLY_STRUCT.size
    )


def _read_event_header(handle: BinaryIO, meta: BinaryMeta, event_size: int, index: int) -> tuple[int, int, int]:
    if meta.acquisition_mode == STREAMING:
        remainder = _read_exact(handle, STREAMING_EVENT_HEADER_SIZE - 2, f"event {index} header")
        timestamp_lsb, number_of_hits = struct.unpack("<QH", remainder)
        return int(timestamp_lsb), 0, int(number_of_hits)
    remainder = _read_exact(handle, TRIGGER_EVENT_HEADER_SIZE - 2, f"event {index} header")
    timestamp_lsb, trigger_id, number_of_hits = struct.unpack("<QQH", remainder)
    return int(timestamp_lsb), int(trigger_id), int(number_of_hits)


def _read_hit(raw: bytes, meta: BinaryMeta, event_index: int) -> Hit:
    board, channel, edge = struct.unpack_from("<BBB", raw, 0)
    if edge not in (0, 1):
        raise DataError(f"Invalid edge code {edge} in event {event_index}")
    if meta.acquisition_mode == STREAMING:
        toa_lsb = struct.unpack_from("<Q", raw, 3)[0]
        tot_offset = 11
    else:
        toa_lsb = struct.unpack_from("<I", raw, 3)[0]
        tot_offset = 7
    tot_lsb = struct.unpack_from("<H", raw, tot_offset)[0] if _has_tot(meta) else None
    return Hit(int(board), int(channel), int(edge), int(toa_lsb), None if tot_lsb is None else int(tot_lsb))


def iter_events(handle: BinaryIO, meta: BinaryMeta) -> Iterator[Event]:
    one_hit_size = hit_size(meta)
    header_size = _event_header_size(meta)
    index = 0
    while True:
        size_raw = handle.read(2)
        if not size_raw:
            return
        if len(size_raw) != 2:
            raise DataError(f"Truncated event size after event {index}")
        event_size = struct.unpack("<H", size_raw)[0]
        if event_size < header_size:
            raise DataError(f"Event {index} size {event_size} is smaller than its header {header_size}")
        timestamp_lsb, trigger_id, number_of_hits = _read_event_header(handle, meta, event_size, index)
        expected = header_size + number_of_hits * one_hit_size
        if event_size != expected:
            mode = acquisition_mode_name(meta.acquisition_mode)
            raise DataError(
                f"Event {index} has inconsistent size for {mode}: header reports "
                f"{event_size} bytes, but {number_of_hits} hits require exactly "
                f"{expected} bytes ({header_size}-byte event header + "
                f"{number_of_hits} x {one_hit_size}-byte hits). This usually means "
                "the acquisition/measurement/time-unit format was decoded incorrectly "
                "or the file is truncated/corrupted."
            )
        hits = [
            _read_hit(_read_exact(handle, one_hit_size, f"event {index} hit {hit_index}"), meta, index)
            for hit_index in range(number_of_hits)
        ]
        yield Event(index, timestamp_lsb, trigger_id, hits)
        index += 1


def write_header(handle: BinaryIO, meta: BinaryMeta) -> None:
    handle.write(meta.raw_header)


def _write_hit(handle: BinaryIO, hit: Hit, meta: BinaryMeta) -> None:
    if meta.acquisition_mode == STREAMING:
        handle.write(struct.pack("<BBBQ", hit.board, hit.channel, hit.edge, hit.toa_lsb))
    else:
        if hit.toa_lsb > 0xFFFFFFFF:
            raise DataError(f"ToA {hit.toa_lsb} does not fit Trigger Matching uint32 format")
        handle.write(struct.pack("<BBBI", hit.board, hit.channel, hit.edge, hit.toa_lsb))
    if _has_tot(meta):
        tot_lsb = 0 if hit.tot_lsb is None else int(hit.tot_lsb)
        if not 0 <= tot_lsb <= 0xFFFF:
            raise DataError(f"ToT {tot_lsb} does not fit uint16 format")
        handle.write(struct.pack("<H", tot_lsb))


def write_event(handle: BinaryIO, event: Event, meta: BinaryMeta) -> bool:
    if not event.hits:
        return False
    event_size = _event_header_size(meta) + len(event.hits) * hit_size(meta)
    if event_size > 0xFFFF:
        raise DataError(f"Event {event.event_index} is too large")
    if meta.acquisition_mode == STREAMING:
        handle.write(STREAMING_EVENT_HEADER_STRUCT.pack(event_size, event.timestamp_lsb, len(event.hits)))
    else:
        handle.write(TRIGGER_EVENT_HEADER_STRUCT.pack(event_size, event.timestamp_lsb, event.trigger_id, len(event.hits)))
    for hit in event.hits:
        _write_hit(handle, hit, meta)
    return True


def read_meta(path: str | Path, expected_acquisition_mode: str | None = None) -> BinaryMeta:
    with Path(path).open("rb") as handle:
        return read_header(handle, expected_acquisition_mode)


def atomic_write_csv(path: str | Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))
