from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml_pipeline.common import atomic_json
from ml_pipeline.splitting import contiguous_development_blind_split
from ml_pipeline.study import _assert_resume_data_compatibility


def test_contiguous_cv_split_uses_only_one_guard_gap() -> None:
    development, blind = contiguous_development_blind_split(
        n_events=100,
        blind_fraction=0.20,
        guard_gap_events=10,
    )

    np.testing.assert_array_equal(development, np.arange(72, dtype=np.int64))
    np.testing.assert_array_equal(blind, np.arange(82, 100, dtype=np.int64))
    assert development.size + blind.size == 90
    assert blind[0] - development[-1] - 1 == 10


def _study_config(*, legacy_initial_validation: bool) -> dict:
    split = {
        "strategy": "contiguous_blocks",
        "seed": 7,
        "blind_fraction": 0.15,
        "guard_gap_events": 1000,
    }
    if legacy_initial_validation:
        split["initial_validation_fraction"] = 0.15
    return {
        "data": {
            "root_folder": "/data",
            "root_glob": "*.root",
            "channels": {"energy": [1, 2], "timing": [3, 4]},
            "true_tof_ps": 0.0,
        },
        "preprocessing": {"common": {}, "energy": {}, "timing": {}},
        "split": split,
        "windows_ns": [
            {
                "id": "w1",
                "start_ns": -6.0,
                "end_ns": 32.0,
                "before_ns": 6.0,
                "after_ns": 32.0,
            }
        ],
    }


def test_resume_rejects_legacy_initial_validation_protocol(tmp_path: Path) -> None:
    atomic_json(
        tmp_path / "resolved_study_config.json",
        _study_config(legacy_initial_validation=True),
    )

    with pytest.raises(RuntimeError, match="--restart"):
        _assert_resume_data_compatibility(
            _study_config(legacy_initial_validation=False),
            tmp_path,
        )


def test_resume_accepts_same_development_blind_protocol(tmp_path: Path) -> None:
    config = _study_config(legacy_initial_validation=False)
    atomic_json(tmp_path / "resolved_study_config.json", config)
    _assert_resume_data_compatibility(config, tmp_path)
