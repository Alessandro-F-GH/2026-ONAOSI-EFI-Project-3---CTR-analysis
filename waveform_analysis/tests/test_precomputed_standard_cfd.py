from __future__ import annotations

import numpy as np
import pytest

from ml_pipeline.standard_methods.cfd import select_precomputed_cfd_times


def test_precomputed_standard_cfd_prefers_timing_timestamps() -> None:
    energy = np.asarray([11_000, 21_000], dtype=np.int64)
    timing = np.asarray([31_000, 41_000], dtype=np.int64)
    selected = select_precomputed_cfd_times(energy, timing)
    np.testing.assert_array_equal(selected, timing)


def test_precomputed_standard_cfd_uses_energy_without_timing_channels() -> None:
    energy = np.asarray([11_000, 21_000], dtype=np.int64)
    selected = select_precomputed_cfd_times(energy, None)
    np.testing.assert_array_equal(selected, energy)


def test_precomputed_standard_cfd_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shapes differ"):
        select_precomputed_cfd_times(
            np.asarray([11_000, 21_000], dtype=np.int64),
            np.asarray([31_000], dtype=np.int64),
        )
