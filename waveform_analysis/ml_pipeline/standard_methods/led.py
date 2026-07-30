from __future__ import annotations
import numpy as np
from ..dataset import PreparedDataset

def led_delta_ps(dataset: PreparedDataset, indices: np.ndarray) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64)
    return (
        np.asarray(dataset.led_time_fs[idx, 0], dtype=np.float64)
        - np.asarray(dataset.led_time_fs[idx, 1], dtype=np.float64)
    ) / 1000.0
