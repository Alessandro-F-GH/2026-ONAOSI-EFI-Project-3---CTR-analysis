from __future__ import annotations

import numpy as np

from utils.cnn_preprocessing import (
    direct_pair_crop,
    first_rising_crossing_ns,
    invariant_pair_crop,
    relative_grid_ns,
)


def _pulse(time_ns: np.ndarray, arrival_ns: float, amplitude: float = 100.0) -> np.ndarray:
    rise = 1.0 / (1.0 + np.exp(-(time_ns - arrival_ns) / 0.08))
    decay = np.exp(-np.maximum(time_ns - arrival_ns, 0.0) / 4.0)
    return amplitude * rise * decay


def test_direct_crop_encodes_single_detector_shift() -> None:
    time = np.arange(-5.0, 5.0, 0.01)
    signal3 = _pulse(time, 0.10)
    signal4 = _pulse(time, -0.05)
    threshold = 10.0
    t3 = first_rising_crossing_ns(time, signal3, threshold)
    t4 = first_rising_crossing_ns(time, signal4, threshold)
    grid = relative_grid_ns(3.5, 20.0)
    crop = direct_pair_crop(
        time,
        signal3,
        time,
        signal4,
        t3_cross_ns=t3,
        t4_cross_ns=t4,
        shift_ps=80,
        shifted_timing_channel=3,
        relative_grid=grid,
        interpolation="cubic",
    )
    assert crop is not None
    crop_t3 = first_rising_crossing_ns(grid, crop[0], threshold)
    crop_t4 = first_rising_crossing_ns(grid, crop[1], threshold)
    assert np.isclose((crop_t3 - crop_t4) * 1000.0, (t3 - t4) * 1000.0 + 80.0, atol=1.0)


def test_invariant_crop_is_unchanged_by_translation() -> None:
    time = np.arange(-5.0, 5.0, 0.01)
    signal3 = _pulse(time, 0.10)
    signal4 = _pulse(time, -0.05)
    threshold = 10.0
    t3 = first_rising_crossing_ns(time, signal3, threshold)
    t4 = first_rising_crossing_ns(time, signal4, threshold)
    grid = relative_grid_ns(3.5, 20.0)
    original = invariant_pair_crop(
        time,
        signal3,
        time,
        signal4,
        t3_cross_ns=t3,
        t4_cross_ns=t4,
        relative_grid=grid,
        interpolation="cubic",
    )
    shift_ns = 0.08
    shifted_signal3 = np.interp(time - shift_ns, time, signal3)
    shifted = invariant_pair_crop(
        time,
        shifted_signal3,
        time,
        signal4,
        t3_cross_ns=t3 + shift_ns,
        t4_cross_ns=t4,
        relative_grid=grid,
        interpolation="cubic",
    )
    assert original is not None and shifted is not None
    assert np.allclose(original[0], shifted[0], atol=0.25)
    assert np.allclose(original[1], shifted[1], atol=0.01)
