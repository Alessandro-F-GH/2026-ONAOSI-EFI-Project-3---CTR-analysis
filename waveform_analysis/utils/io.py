from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import awkward as ak
import numpy as np
import uproot

FORMAT_VERSION = "trc-singlefile-v1"


@dataclass(frozen=True)
class EventChunk:
    event_index: np.ndarray
    event_id: np.ndarray
    source_file_id: np.ndarray
    sample_count: np.ndarray
    vertical_gain_v_per_count: np.ndarray
    vertical_offset_v: np.ndarray
    horizontal_interval_s: np.ndarray
    horizontal_offset_s: np.ndarray
    samples: tuple[ak.Array, ak.Array, ak.Array, ak.Array]


def _to_python(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, ak.Array):
        return ak.to_list(value)
    if hasattr(value, "to_list"):
        return value.to_list()
    return value


def read_metadata(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with uproot.open(source) as root_file:
        if "metadata" not in root_file:
            raise KeyError("ROOT file does not contain the 'metadata' TTree")
        arrays = root_file["metadata"].arrays(library="ak")
        metadata = {field: _to_python(arrays[field][0]) for field in arrays.fields}
    version = metadata.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(f"Unsupported format_version {version!r}; expected {FORMAT_VERSION!r}")
    return metadata


def event_count(path: str | Path) -> int:
    with uproot.open(path) as root_file:
        return int(root_file["events"].num_entries)


def iterate_chunks(
    path: str | Path,
    *,
    step_size: int | str = "128 MB",
    entry_start: int | None = None,
    entry_stop: int | None = None,
) -> Iterator[EventChunk]:
    branches = [
        "event_index",
        "event_id",
        "source_file_id",
        "sample_count",
        "vertical_gain_v_per_count",
        "vertical_offset_v",
        "horizontal_interval_s",
        "horizontal_offset_s",
        "samples_ch1",
        "samples_ch2",
        "samples_ch3",
        "samples_ch4",
    ]
    source = Path(path)
    with uproot.open(source) as root_file:
        if "events" not in root_file:
            raise KeyError("ROOT file does not contain the 'events' TTree")
        tree = root_file["events"]
        for arrays in tree.iterate(
            filter_name=branches,
            step_size=step_size,
            entry_start=entry_start,
            entry_stop=entry_stop,
            library="ak",
        ):
            yield EventChunk(
                event_index=np.asarray(ak.to_numpy(arrays["event_index"]), dtype=np.int64),
                event_id=np.asarray(ak.to_numpy(arrays["event_id"]), dtype=np.int64),
                source_file_id=np.asarray(ak.to_numpy(arrays["source_file_id"]), dtype=np.int64),
                sample_count=np.asarray(ak.to_numpy(arrays["sample_count"]), dtype=np.int32),
                vertical_gain_v_per_count=np.asarray(
                    ak.to_numpy(arrays["vertical_gain_v_per_count"]), dtype=np.float64
                ),
                vertical_offset_v=np.asarray(
                    ak.to_numpy(arrays["vertical_offset_v"]), dtype=np.float64
                ),
                horizontal_interval_s=np.asarray(
                    ak.to_numpy(arrays["horizontal_interval_s"]), dtype=np.float64
                ),
                horizontal_offset_s=np.asarray(
                    ak.to_numpy(arrays["horizontal_offset_s"]), dtype=np.float64
                ),
                samples=(
                    arrays["samples_ch1"],
                    arrays["samples_ch2"],
                    arrays["samples_ch3"],
                    arrays["samples_ch4"],
                ),
            )


def decode_voltage_mV(
    raw_samples: np.ndarray,
    vertical_gain_v_per_count: float,
    vertical_offset_v: float,
) -> np.ndarray:
    """Convert integer ADC samples to physical voltage in millivolts.

    Timing is intentionally not decoded here. The CTR pipeline keeps crossing
    timestamps as int64 femtoseconds and converts to picoseconds only when the
    final fit result is reported.
    """
    raw = np.asarray(raw_samples, dtype=np.float64)
    return (raw * vertical_gain_v_per_count - vertical_offset_v) * 1000.0


def decode_waveform(
    raw_samples: np.ndarray,
    vertical_gain_v_per_count: float,
    vertical_offset_v: float,
    horizontal_interval_s: float,
    horizontal_offset_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode a waveform for plotting/inspection.

    The returned time axis is in nanoseconds for user-facing plots. The CTR
    extraction path uses :func:`decode_voltage_mV` plus integer-femtosecond
    timestamps instead.
    """
    raw = np.asarray(raw_samples, dtype=np.float64)
    voltage_mV = decode_voltage_mV(
        raw,
        vertical_gain_v_per_count,
        vertical_offset_v,
    )
    time_ns = (
        np.arange(raw.size, dtype=np.float64) * horizontal_interval_s
        + horizontal_offset_s
    ) * 1.0e9
    return time_ns, voltage_mV


def get_event(path: str | Path, entry: int) -> dict[str, Any]:
    if entry < 0:
        raise ValueError("entry must be non-negative")
    chunks = list(iterate_chunks(path, step_size=1, entry_start=entry, entry_stop=entry + 1))
    if not chunks or chunks[0].event_id.size == 0:
        raise IndexError(f"event entry {entry} does not exist")
    chunk = chunks[0]
    waveforms: list[tuple[np.ndarray, np.ndarray]] = []
    for channel in range(4):
        raw = np.asarray(ak.to_numpy(chunk.samples[channel][0]), dtype=np.int16)
        waveforms.append(
            decode_waveform(
                raw,
                chunk.vertical_gain_v_per_count[0, channel],
                chunk.vertical_offset_v[0, channel],
                chunk.horizontal_interval_s[0, channel],
                chunk.horizontal_offset_s[0, channel],
            )
        )
    return {
        "event_index": int(chunk.event_index[0]),
        "event_id": int(chunk.event_id[0]),
        "source_file_id": chunk.source_file_id[0].copy(),
        "waveforms": waveforms,
    }
