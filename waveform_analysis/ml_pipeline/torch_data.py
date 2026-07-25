from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

if TYPE_CHECKING:
    from .data import EnergyCache


@dataclass(frozen=True)
class Normalization:
    mean_mV: float
    std_mV: float

    def as_dict(self) -> dict[str, float | str]:
        return {
            "mode": "one_global_affine_transform_shared_by_both_channels",
            "mean_mV": self.mean_mV,
            "std_mV": self.std_mV,
        }


def compute_normalization(
    windows: np.ndarray, indices: np.ndarray, *, chunk_size: int = 2048
) -> Normalization:
    total = 0
    total_sum = 0.0
    total_sq = 0.0
    indices = np.asarray(indices, dtype=np.int64)
    for start in range(0, indices.size, chunk_size):
        chunk_indices = indices[start : start + chunk_size]
        values = np.asarray(windows[chunk_indices], dtype=np.float64)
        total += values.size
        total_sum += float(np.sum(values))
        total_sq += float(np.sum(values * values))
    if total == 0:
        raise ValueError("Cannot normalize an empty training dataset")
    mean = total_sum / total
    variance = max(total_sq / total - mean * mean, 1e-12)
    return Normalization(mean_mV=float(mean), std_mV=float(np.sqrt(variance)))


class CorrectionDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        cache: EnergyCache,
        indices: np.ndarray,
        normalization: Normalization,
        led_center_ps: float,
        *,
        duplicate_swapped_channels: bool = False,
    ) -> None:
        # Store paths instead of memory-map objects. DataLoader workers on Windows
        # use spawn; reopening the maps lazily avoids pickling the full arrays.
        self.cache_directory = Path(cache.directory)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.normalization = normalization
        self.led_center_ps = float(led_center_ps)
        self.duplicate_swapped_channels = bool(duplicate_swapped_channels)
        self.base_length = int(self.indices.size)
        self._windows: np.ndarray | None = None
        self._led_times: np.ndarray | None = None

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_windows"] = None
        state["_led_times"] = None
        return state

    def _ensure_open(self) -> None:
        if self._windows is None:
            self._windows = np.load(
                self.cache_directory / "windows_mV.npy", mmap_mode="r"
            )
        if self._led_times is None:
            self._led_times = np.load(
                self.cache_directory / "led_time_fs.npy", mmap_mode="r"
            )

    def __len__(self) -> int:
        multiplier = 2 if self.duplicate_swapped_channels else 1
        return multiplier * self.base_length

    def __getitem__(
        self, position: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._ensure_open()
        assert self._windows is not None
        assert self._led_times is not None
        if position < 0 or position >= len(self):
            raise IndexError(position)
        swapped = self.duplicate_swapped_channels and position >= self.base_length
        base_position = position % self.base_length
        index = int(self.indices[base_position])
        pair = np.asarray(self._windows[index], dtype=np.float32).copy()
        led_delta_ps = (
            int(self._led_times[index, 0]) - int(self._led_times[index, 1])
        ) / 1000.0
        target_ps = led_delta_ps - self.led_center_ps
        if swapped:
            pair = pair[[1, 0], :]
            # Swapping the ordered detector pair reverses every signed timing
            # quantity.  Negating the already centered target is equivalent to
            # using the swapped LED center -led_center_ps.
            led_delta_ps = -led_delta_ps
            target_ps = -target_ps
        pair = (pair - np.float32(self.normalization.mean_mV)) / np.float32(
            self.normalization.std_mV
        )
        return (
            torch.from_numpy(pair),
            torch.tensor(target_ps, dtype=torch.float32),
            torch.tensor(led_delta_ps, dtype=torch.float32),
        )


class EpochRandomSampler(Sampler[int]):
    """Deterministic per-epoch shuffle, enabling reproducible batch resume."""

    def __init__(self, dataset_length: int, seed: int) -> None:
        self.dataset_length = int(dataset_length)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.dataset_length, generator=generator).tolist()
        return iter(order)

    def __len__(self) -> int:
        return self.dataset_length


class EpochSymmetricBatchSampler(Sampler[list[int]]):
    """Yield batches containing each canonical event and its swapped copy.

    The associated ``CorrectionDataset`` must have
    ``duplicate_swapped_channels=True`` and therefore length ``2 * base_length``.
    Every optimization batch is exactly symmetric: if position ``i`` appears,
    position ``i + base_length`` appears in the same batch.  This is especially
    useful for a mini-batch standard-deviation objective because the signed LED
    targets and antisymmetric model outputs then have exactly zero batch mean.
    """

    def __init__(self, base_length: int, batch_size: int, seed: int) -> None:
        self.base_length = int(base_length)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        if self.base_length <= 0:
            raise ValueError("base_length must be positive")
        if self.batch_size < 2 or self.batch_size % 2 != 0:
            raise ValueError(
                "Symmetric channel-swap batches require an even batch_size >= 2"
            )
        self.canonical_per_batch = self.batch_size // 2

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.base_length, generator=generator).tolist()
        for start in range(0, self.base_length, self.canonical_per_batch):
            canonical = order[start : start + self.canonical_per_batch]
            batch: list[int] = []
            for position in canonical:
                batch.extend((position, position + self.base_length))
            yield batch

    def __len__(self) -> int:
        return (self.base_length + self.canonical_per_batch - 1) // self.canonical_per_batch
