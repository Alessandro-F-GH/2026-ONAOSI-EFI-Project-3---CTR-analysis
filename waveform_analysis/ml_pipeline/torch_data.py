from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import PreparedDataset
from .input_transform import apply_input_transform, normalize_input_transform


@dataclass(frozen=True)
class Normalization:
    mean_mV: float
    std_mV: float

    def as_dict(self) -> dict[str, float | str]:
        return {
            "mode": "global affine transform shared by both channels",
            "mean_mV": float(self.mean_mV),
            "std_mV": float(self.std_mV),
        }


def compute_normalization(
    datasets_and_indices: Iterable[tuple[PreparedDataset, np.ndarray]],
    *,
    chunk_size: int = 2048,
) -> Normalization:
    count = 0
    total = 0.0
    total_square = 0.0
    for dataset, indices in datasets_and_indices:
        values_indices = np.asarray(indices, dtype=np.int64)
        for start in range(0, values_indices.size, chunk_size):
            chunk_indices = values_indices[start : start + chunk_size]
            values = np.asarray(dataset.windows_mV[chunk_indices], dtype=np.float64)
            count += values.size
            total += float(np.sum(values))
            total_square += float(np.sum(values * values))
    if count == 0:
        raise ValueError("Cannot normalize an empty training dataset")
    mean = total / count
    variance = max(total_square / count - mean * mean, 1e-12)
    return Normalization(float(mean), float(np.sqrt(variance)))


class CorrectionDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    """View of a post-selection prepared dataset.

    The target is the complete LED timing error relative to the known TOF:
    target = (t1_LED - t2_LED) - true_TOF.
    """

    def __init__(
        self,
        dataset: PreparedDataset,
        indices: np.ndarray,
        normalization: Normalization,
        input_transform: str = "none",
    ) -> None:
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.normalization = normalization
        self.input_transform = normalize_input_transform(input_transform)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, position: int):
        index = int(self.indices[position])
        pair = np.asarray(self.dataset.windows_mV[index], dtype=np.float32)
        pair = np.asarray(
            apply_input_transform(pair, self.input_transform), dtype=np.float32
        ).copy()
        pair = (pair - np.float32(self.normalization.mean_mV)) / np.float32(self.normalization.std_mV)
        led_delta_ps = (
            int(self.dataset.led_time_fs[index, 0]) - int(self.dataset.led_time_fs[index, 1])
        ) / 1000.0
        cfd_delta_ps = (
            int(self.dataset.cfd_time_fs[index, 0]) - int(self.dataset.cfd_time_fs[index, 1])
        ) / 1000.0
        true_tof_ps = self.dataset.true_tof_ps
        target_ps = led_delta_ps - true_tof_ps
        return (
            torch.from_numpy(pair),
            torch.tensor(target_ps, dtype=torch.float32),
            torch.tensor(led_delta_ps, dtype=torch.float32),
            torch.tensor(cfd_delta_ps, dtype=torch.float32),
            torch.tensor(true_tof_ps, dtype=torch.float32),
        )

