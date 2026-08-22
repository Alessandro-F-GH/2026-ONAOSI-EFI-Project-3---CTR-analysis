from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class Hit:
    board: int
    channel: int
    edge: int
    toa_lsb: int
    tot_lsb: int | None = None


@dataclass(slots=True)
class Event:
    event_index: int
    timestamp_lsb: int
    trigger_id: int
    hits: list[Hit] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class BinaryMeta:
    raw_header: bytes
    run_number: int
    acquisition_mode: int
    measurement_mode: int
    time_unit: int
    toa_lsb_ps: float
    tot_lsb_ps: float
    timestamp_lsb_ps: float


@dataclass(slots=True, frozen=True)
class RunInput:
    run_id: str
    run_number: str
    voltage: int
    data_path: Path
    info_path: Path


@dataclass(slots=True, frozen=True)
class RunInfo:
    acquisition_mode: str
    energy_threshold_mv: float
    timing_threshold_mv: float




@dataclass(slots=True)
class EnergyMeasurements:
    event_index: np.ndarray
    duration_a_lsb: np.ndarray
    duration_b_lsb: np.ndarray
    energy_a_lsb: np.ndarray
    energy_b_lsb: np.ndarray

    @property
    def size(self) -> int:
        return int(self.event_index.size)


@dataclass(slots=True)
class Measurements:
    event_index: np.ndarray
    duration_a_lsb: np.ndarray
    duration_b_lsb: np.ndarray
    energy_a_lsb: np.ndarray
    time_a_lsb: np.ndarray
    energy_b_lsb: np.ndarray
    time_b_lsb: np.ndarray

    @property
    def size(self) -> int:
        return int(self.event_index.size)

    @property
    def alignment_a_lsb(self) -> np.ndarray:
        return self.time_a_lsb - self.energy_a_lsb

    @property
    def alignment_b_lsb(self) -> np.ndarray:
        return self.time_b_lsb - self.energy_b_lsb

    @property
    def timing_lsb(self) -> np.ndarray:
        return self.time_b_lsb - self.time_a_lsb

    def take(self, indices: np.ndarray) -> Measurements:
        return Measurements(
            event_index=self.event_index[indices],
            duration_a_lsb=self.duration_a_lsb[indices],
            duration_b_lsb=self.duration_b_lsb[indices],
            energy_a_lsb=self.energy_a_lsb[indices],
            time_a_lsb=self.time_a_lsb[indices],
            energy_b_lsb=self.energy_b_lsb[indices],
            time_b_lsb=self.time_b_lsb[indices],
        )


@dataclass(slots=True, frozen=True)
class PeakSelection:
    low_lsb: int
    high_lsb: int
    peak_lsb: float
    center_lsb: float
    scale_lsb: float


@dataclass(slots=True, frozen=True)
class EnergySelectionResult:
    peak_a: PeakSelection
    peak_b: PeakSelection
    duration_mask: np.ndarray


@dataclass(slots=True, frozen=True)
class SelectionResult:
    peak_a: PeakSelection
    peak_b: PeakSelection
    duration_mask: np.ndarray
    alignment_mask: np.ndarray
    final_mask: np.ndarray
    alignment_a_center_lsb: float
    alignment_a_scale_lsb: float
    alignment_b_center_lsb: float
    alignment_b_scale_lsb: float
