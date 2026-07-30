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
    source_run_index: np.ndarray
    bias_voltage_V: np.ndarray
    vertical_gain_v_per_count: np.ndarray
    vertical_offset_v: np.ndarray
    horizontal_interval_s: np.ndarray
    horizontal_offset_s: np.ndarray
    samples: tuple[ak.Array, ak.Array]
    timing_vertical_gain_v_per_count: np.ndarray | None
    timing_vertical_offset_v: np.ndarray | None
    timing_horizontal_interval_s: np.ndarray | None
    timing_horizontal_offset_s: np.ndarray | None
    timing_samples: tuple[ak.Array, ak.Array] | None


def energy_event_count(path: Path) -> int:
    with uproot.open(path) as root_file:
        if "events" not in root_file:
            raise KeyError("ROOT file does not contain the 'events' TTree")
        return int(root_file["events"].num_entries)


def iterate_energy_chunks(
    path: Path,
    *,
    energy_channels_one_based: tuple[int, int],
    timing_channels_one_based: tuple[int, int] | None = None,
    step_size: int | str,
    entry_stop: int | None,
) -> Iterator[EnergyChunk]:
    energy_indices = np.asarray(energy_channels_one_based, dtype=np.int64) - 1
    timing_indices = (
        None
        if timing_channels_one_based is None
        else np.asarray(timing_channels_one_based, dtype=np.int64) - 1
    )
    energy_sample_names = [
        f"samples_ch{int(channel)}" for channel in energy_channels_one_based
    ]
    timing_sample_names = (
        []
        if timing_channels_one_based is None
        else [f"samples_ch{int(channel)}" for channel in timing_channels_one_based]
    )
    required_branches = [
        "event_index",
        "event_id",
        "source_file_id",
        "vertical_gain_v_per_count",
        "vertical_offset_v",
        "horizontal_interval_s",
        "horizontal_offset_s",
        *energy_sample_names,
        *timing_sample_names,
    ]
    # Preserve order while avoiding duplicate branch requests if configurations overlap.
    required_branches = list(dict.fromkeys(required_branches))
    with uproot.open(path) as root_file:
        if "events" not in root_file:
            raise KeyError("ROOT file does not contain the 'events' TTree")
        tree = root_file["events"]
        available = set(tree.keys())
        missing = [name for name in required_branches if name not in available]
        if missing:
            raise KeyError("ROOT events tree is missing branches: " + ", ".join(missing))
        optional_branches = [
            name for name in ("source_run_index", "bias_voltage_V") if name in available
        ]
        branches = [*required_branches, *optional_branches]
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
            event_id = np.asarray(ak.to_numpy(arrays["event_id"]), dtype=np.int64)
            n_events = int(event_id.size)
            if "source_run_index" in arrays.fields:
                source_run_index = np.asarray(
                    ak.to_numpy(arrays["source_run_index"]), dtype=np.int32
                )
            else:
                source_run_index = np.zeros(n_events, dtype=np.int32)
            if "bias_voltage_V" in arrays.fields:
                bias_voltage = np.asarray(
                    ak.to_numpy(arrays["bias_voltage_V"]), dtype=np.float64
                )
            else:
                bias_voltage = np.full(n_events, np.nan, dtype=np.float64)

            if timing_indices is None:
                timing_gain = None
                timing_offset = None
                timing_interval = None
                timing_horizontal_offset = None
                timing_samples = None
            else:
                timing_gain = gain_full[:, timing_indices]
                timing_offset = offset_full[:, timing_indices]
                timing_interval = interval_full[:, timing_indices]
                timing_horizontal_offset = horizontal_offset_full[:, timing_indices]
                timing_samples = (
                    arrays[timing_sample_names[0]],
                    arrays[timing_sample_names[1]],
                )

            yield EnergyChunk(
                event_index=np.asarray(ak.to_numpy(arrays["event_index"]), dtype=np.int64),
                event_id=event_id,
                source_file_id=source_full[:, energy_indices],
                source_run_index=source_run_index,
                bias_voltage_V=bias_voltage,
                vertical_gain_v_per_count=gain_full[:, energy_indices],
                vertical_offset_v=offset_full[:, energy_indices],
                horizontal_interval_s=interval_full[:, energy_indices],
                horizontal_offset_s=horizontal_offset_full[:, energy_indices],
                samples=(arrays[energy_sample_names[0]], arrays[energy_sample_names[1]]),
                timing_vertical_gain_v_per_count=timing_gain,
                timing_vertical_offset_v=timing_offset,
                timing_horizontal_interval_s=timing_interval,
                timing_horizontal_offset_s=timing_horizontal_offset,
                timing_samples=timing_samples,
            )
