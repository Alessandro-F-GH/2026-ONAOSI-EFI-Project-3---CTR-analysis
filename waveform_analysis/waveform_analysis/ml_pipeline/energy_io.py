from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import awkward as ak
import numpy as np
import uproot


@dataclass(frozen=True)
class EnergyChunk:
    event_index: np.ndarray
    event_id: np.ndarray
    source_file_id: np.ndarray
    vertical_gain_v_per_count: np.ndarray
    vertical_offset_v: np.ndarray
    horizontal_interval_s: np.ndarray
    horizontal_offset_s: np.ndarray
    samples: tuple[ak.Array, ak.Array]


def energy_event_count(path: Path) -> int:
    with uproot.open(path) as root_file:
        if "events" not in root_file:
            raise KeyError("ROOT file does not contain the 'events' TTree")
        return int(root_file["events"].num_entries)


def iterate_energy_chunks(
    path: Path,
    *,
    energy_channels_one_based: tuple[int, int],
    step_size: int | str,
    entry_stop: int | None,
) -> Iterator[EnergyChunk]:
    channel_indices = np.asarray(energy_channels_one_based, dtype=np.int64) - 1
    sample_names = [f"samples_ch{int(channel)}" for channel in energy_channels_one_based]
    branches = [
        "event_index",
        "event_id",
        "source_file_id",
        "vertical_gain_v_per_count",
        "vertical_offset_v",
        "horizontal_interval_s",
        "horizontal_offset_s",
        *sample_names,
    ]
    with uproot.open(path) as root_file:
        if "events" not in root_file:
            raise KeyError("ROOT file does not contain the 'events' TTree")
        tree = root_file["events"]
        for arrays in tree.iterate(
            filter_name=branches,
            step_size=step_size,
            entry_stop=entry_stop,
            library="ak",
        ):
            source_full = np.asarray(ak.to_numpy(arrays["source_file_id"]), dtype=np.int64)
            gain_full = np.asarray(
                ak.to_numpy(arrays["vertical_gain_v_per_count"]), dtype=np.float64
            )
            offset_full = np.asarray(
                ak.to_numpy(arrays["vertical_offset_v"]), dtype=np.float64
            )
            interval_full = np.asarray(
                ak.to_numpy(arrays["horizontal_interval_s"]), dtype=np.float64
            )
            horizontal_offset_full = np.asarray(
                ak.to_numpy(arrays["horizontal_offset_s"]), dtype=np.float64
            )
            yield EnergyChunk(
                event_index=np.asarray(ak.to_numpy(arrays["event_index"]), dtype=np.int64),
                event_id=np.asarray(ak.to_numpy(arrays["event_id"]), dtype=np.int64),
                source_file_id=source_full[:, channel_indices],
                vertical_gain_v_per_count=gain_full[:, channel_indices],
                vertical_offset_v=offset_full[:, channel_indices],
                horizontal_interval_s=interval_full[:, channel_indices],
                horizontal_offset_s=horizontal_offset_full[:, channel_indices],
                samples=(arrays[sample_names[0]], arrays[sample_names[1]]),
            )
