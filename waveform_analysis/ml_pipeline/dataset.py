from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .common import read_json
DATASET_FORMAT_VERSION = 7
_SUPPORTED_DATASET_FORMAT_VERSIONS = {1, 2, 3, 4, 5, 6, 7}
_ARRAY_NAMES = (
    "event_id", "event_index", "source_file_id", "source_run_index",
    "bias_voltage_V", "amplitude_mV", "noise_rms_mV", "trigger_index",
    "windows_mV",
)
_OPTIONAL_ARRAY_NAMES = (
    "energy_led_time_fs",
    "timing_led_time_fs",
    "energy_cfd_time_fs",
    "timing_cfd_time_fs",
    "energy_window_anchor_time_fs",
    "timing_aligned_energy_window_anchor_time_fs",
    "timing_window_anchor_time_fs",
    "timing_aligned_energy_windows_mV",
    "timing_windows_mV",
    "denoised_windows_mV",
    "denoised_timing_aligned_energy_windows_mV",
    "denoised_timing_windows_mV",
    "raw_energy_led_time_fs",
    "raw_timing_led_time_fs",
    "raw_energy_cfd_time_fs",
    "raw_timing_cfd_time_fs",
    "raw_energy_window_anchor_time_fs",
    "raw_timing_aligned_energy_window_anchor_time_fs",
    "raw_timing_window_anchor_time_fs",
    "raw_energy_windows_mV",
    "raw_timing_aligned_energy_windows_mV",
    "raw_timing_windows_mV",
)


@dataclass(frozen=True)
class PreparedDataset:
    directory: Path
    manifest: dict[str, Any]
    event_id: np.ndarray
    event_index: np.ndarray
    source_file_id: np.ndarray
    source_run_index: np.ndarray
    bias_voltage_V: np.ndarray
    amplitude_mV: np.ndarray
    noise_rms_mV: np.ndarray
    trigger_index: np.ndarray
    led_time_fs: np.ndarray
    cfd_time_fs: np.ndarray
    windows_mV: np.ndarray
    relative_time_ps: np.ndarray
    energy_led_time_fs: np.ndarray | None = None
    timing_led_time_fs: np.ndarray | None = None
    energy_cfd_time_fs: np.ndarray | None = None
    timing_cfd_time_fs: np.ndarray | None = None
    energy_window_anchor_time_fs: np.ndarray | None = None
    timing_aligned_energy_window_anchor_time_fs: np.ndarray | None = None
    timing_window_anchor_time_fs: np.ndarray | None = None
    window_anchor_time_fs: np.ndarray | None = None
    timing_aligned_energy_windows_mV: np.ndarray | None = None
    timing_windows_mV: np.ndarray | None = None
    denoised_windows_mV: np.ndarray | None = None
    denoised_timing_aligned_energy_windows_mV: np.ndarray | None = None
    denoised_timing_windows_mV: np.ndarray | None = None
    raw_energy_led_time_fs: np.ndarray | None = None
    raw_timing_led_time_fs: np.ndarray | None = None
    raw_energy_cfd_time_fs: np.ndarray | None = None
    raw_timing_cfd_time_fs: np.ndarray | None = None
    raw_energy_window_anchor_time_fs: np.ndarray | None = None
    raw_timing_aligned_energy_window_anchor_time_fs: np.ndarray | None = None
    raw_timing_window_anchor_time_fs: np.ndarray | None = None
    raw_energy_windows_mV: np.ndarray | None = None
    raw_timing_aligned_energy_windows_mV: np.ndarray | None = None
    raw_timing_windows_mV: np.ndarray | None = None
    timing_relative_time_ps: np.ndarray | None = None
    train: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    validation: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    test: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    evaluation: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))

    @property
    def input_length(self) -> int:
        return int(self.windows_mV.shape[2])

    @property
    def true_tof_ps(self) -> float:
        return float(self.manifest["true_tof_ps"])


def _load_array(directory: Path, name: str) -> np.ndarray:
    path = directory / f"{name}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Prepared dataset array not found: {path}")
    return np.load(path, mmap_mode="r")


def _load_optional_array(directory: Path, name: str) -> np.ndarray | None:
    path = directory / f"{name}.npy"
    return np.load(path, mmap_mode="r") if path.is_file() else None


def load_prepared_dataset(directory: str | Path) -> PreparedDataset:
    directory = Path(directory).resolve()
    manifest_path = directory / "manifest.json"
    split_path = directory / "splits.npz"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Not a prepared ML dataset: {directory}")
    manifest = read_json(manifest_path)
    if int(manifest.get("format_version", -1)) not in _SUPPORTED_DATASET_FORMAT_VERSIONS:
        raise ValueError(f"Unsupported prepared dataset version in {directory}")
    event_id = _load_array(directory, "event_id")
    if split_path.is_file():
        with np.load(split_path, allow_pickle=False) as splits:
            split_values = {
                name: splits[name].astype(np.int64)
                for name in ("train", "validation", "test", "evaluation")
            }
    else:
        # Format v5 permanent datasets intentionally contain no ML split. The
        # experiment runner creates all random partitions in memory.
        all_indices = np.arange(event_id.size, dtype=np.int64)
        split_values = {
            "train": all_indices,
            "validation": np.empty(0, dtype=np.int64),
            "test": np.empty(0, dtype=np.int64),
            "evaluation": all_indices,
        }
    return PreparedDataset(
        directory=directory,
        manifest=manifest,
        event_id=event_id,
        event_index=_load_array(directory, "event_index"),
        source_file_id=_load_array(directory, "source_file_id"),
        source_run_index=_load_array(directory, "source_run_index"),
        bias_voltage_V=_load_array(directory, "bias_voltage_V"),
        amplitude_mV=_load_array(directory, "amplitude_mV"),
        noise_rms_mV=_load_array(directory, "noise_rms_mV"),
        trigger_index=_load_array(directory, "trigger_index"),
        # Format v6 stores energy LED/CFD only once.  The generic timing fields
        # remain an in-memory compatibility view used by target-specific dataset
        # views; old formats that physically stored them still load unchanged.
        led_time_fs=(
            _load_optional_array(directory, "led_time_fs")
            if (directory / "led_time_fs.npy").is_file()
            else _load_array(directory, "energy_led_time_fs")
        ),
        cfd_time_fs=(
            _load_optional_array(directory, "cfd_time_fs")
            if (directory / "cfd_time_fs.npy").is_file()
            else _load_array(directory, "energy_cfd_time_fs")
        ),
        windows_mV=_load_array(directory, "windows_mV"),
        relative_time_ps=_load_array(directory, "relative_time_ps"),
        energy_led_time_fs=_load_optional_array(directory, "energy_led_time_fs"),
        timing_led_time_fs=_load_optional_array(directory, "timing_led_time_fs"),
        energy_cfd_time_fs=_load_optional_array(directory, "energy_cfd_time_fs"),
        timing_cfd_time_fs=_load_optional_array(directory, "timing_cfd_time_fs"),
        energy_window_anchor_time_fs=_load_optional_array(
            directory, "energy_window_anchor_time_fs"
        ),
        timing_aligned_energy_window_anchor_time_fs=_load_optional_array(
            directory, "timing_aligned_energy_window_anchor_time_fs"
        ),
        timing_window_anchor_time_fs=_load_optional_array(
            directory, "timing_window_anchor_time_fs"
        ),
        window_anchor_time_fs=_load_optional_array(
            directory, "energy_window_anchor_time_fs"
        ),
        timing_aligned_energy_windows_mV=_load_optional_array(
            directory, "timing_aligned_energy_windows_mV"
        ),
        timing_windows_mV=_load_optional_array(directory, "timing_windows_mV"),
        denoised_windows_mV=_load_optional_array(directory, "denoised_windows_mV"),
        denoised_timing_aligned_energy_windows_mV=_load_optional_array(directory, "denoised_timing_aligned_energy_windows_mV"),
        denoised_timing_windows_mV=_load_optional_array(directory, "denoised_timing_windows_mV"),
        raw_energy_led_time_fs=_load_optional_array(directory, "raw_energy_led_time_fs"),
        raw_timing_led_time_fs=_load_optional_array(directory, "raw_timing_led_time_fs"),
        raw_energy_cfd_time_fs=_load_optional_array(directory, "raw_energy_cfd_time_fs"),
        raw_timing_cfd_time_fs=_load_optional_array(directory, "raw_timing_cfd_time_fs"),
        raw_energy_window_anchor_time_fs=_load_optional_array(directory, "raw_energy_window_anchor_time_fs"),
        raw_timing_aligned_energy_window_anchor_time_fs=_load_optional_array(
            directory, "raw_timing_aligned_energy_window_anchor_time_fs"
        ),
        raw_timing_window_anchor_time_fs=_load_optional_array(directory, "raw_timing_window_anchor_time_fs"),
        raw_energy_windows_mV=_load_optional_array(directory, "raw_energy_windows_mV"),
        raw_timing_aligned_energy_windows_mV=_load_optional_array(
            directory, "raw_timing_aligned_energy_windows_mV"
        ),
        raw_timing_windows_mV=_load_optional_array(directory, "raw_timing_windows_mV"),
        timing_relative_time_ps=_load_optional_array(directory, "timing_relative_time_ps"),
        train=split_values["train"],
        validation=split_values["validation"],
        test=split_values["test"],
        evaluation=split_values["evaluation"],
    )


def load_prepared_dataset_spec(spec: str | Path | dict[str, Any]) -> PreparedDataset:
    """Load a canonical prepared dataset from a path or dataset object."""

    if isinstance(spec, (str, Path)):
        return load_prepared_dataset(spec)
    if not isinstance(spec, dict) or not str(spec.get("dataset", "")).strip():
        raise ValueError(
            "Dataset specification must be a path or an object containing 'dataset'"
        )
    return load_prepared_dataset(spec["dataset"])
