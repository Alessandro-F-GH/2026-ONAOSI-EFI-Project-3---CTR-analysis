from __future__ import annotations

import numpy as np

from ..dataset import PreparedDataset


def select_precomputed_cfd_times(
    energy_cfd_time_fs: np.ndarray,
    timing_cfd_time_fs: np.ndarray | None,
) -> np.ndarray:
    """Select the CFD timestamps materialized by preprocessing.

    The prepared standard CFD must follow the same waveform family selected for
    the prepared LED.  When timing-channel preprocessing is enabled,
    ``timing_cfd_time_fs`` is passed and is selected. Otherwise the energy CFD
    timestamps remain the standard-method artifact.

    This function does not inspect manifest metadata and never recomputes a
    crossing from saved waveform windows.
    """

    energy = np.asarray(energy_cfd_time_fs, dtype=np.int64)
    if energy.ndim != 1:
        raise ValueError(
            f"Expected one energy CFD timestamp per detector, got shape {energy.shape}"
        )
    if timing_cfd_time_fs is None:
        return energy

    timing = np.asarray(timing_cfd_time_fs, dtype=np.int64)
    if timing.shape != energy.shape:
        raise ValueError(
            "Timing and energy CFD timestamp shapes differ: "
            f"{timing.shape} vs {energy.shape}"
        )
    return timing


def cfd_delta_ps(dataset: PreparedDataset, indices: np.ndarray) -> np.ndarray:
    """Return the detector-pair CFD difference from precomputed timestamps."""

    idx = np.asarray(indices, dtype=np.int64)
    return (
        np.asarray(dataset.cfd_time_fs[idx, 0], dtype=np.float64)
        - np.asarray(dataset.cfd_time_fs[idx, 1], dtype=np.float64)
    ) / 1000.0
