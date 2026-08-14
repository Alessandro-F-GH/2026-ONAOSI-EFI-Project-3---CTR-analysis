from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import PreparedDataset
from .input_transform import (
    apply_component_subsampling,
    apply_input_transform,
    normalize_input_transform,
    normalize_subsampling_factor,
    transformed_component_lengths,
)


ArrayLikeStat = float | np.ndarray


def _stat_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def _json_stat(value: ArrayLikeStat) -> float | list[float]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        return float(array)
    return array.tolist()


@dataclass(frozen=True)
class Normalization:
    """Waveform normalization learned from training events only.

    ``global`` stores one scalar mean/std shared by every sample. ``feature``
    stores one mean/std per time position and broadcasts it over the two detector
    channels, preserving detector-swap symmetry.
    """

    mean_mV: ArrayLikeStat
    std_mV: ArrayLikeStat
    mode: str = "global"

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower()
        if mode not in {"global", "feature"}:
            raise ValueError("Normalization mode must be 'global' or 'feature'")
        mean = _stat_array(self.mean_mV)
        std = _stat_array(self.std_mV)
        if mean.shape != std.shape:
            raise ValueError("Normalization mean and std must have matching shapes")
        if mean.ndim > 1:
            raise ValueError("Normalization statistics must be scalar or one-dimensional")
        if mode == "global" and mean.ndim != 0:
            raise ValueError("Global normalization requires scalar statistics")
        if mode == "feature" and mean.ndim != 1:
            raise ValueError("Feature normalization requires one-dimensional statistics")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
            raise ValueError("Normalization statistics must be finite")
        if np.any(std <= 0.0):
            raise ValueError("Normalization standard deviation must be positive")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "mean_mV", float(mean) if mean.ndim == 0 else mean)
        object.__setattr__(self, "std_mV", float(std) if std.ndim == 0 else std)

    def as_dict(self) -> dict[str, Any]:
        description = (
            "per-time-position z-score shared by both detector channels"
            if self.mode == "feature"
            else "global affine transform shared by both channels"
        )
        return {
            # Keep the historical human-readable ``mode`` field for external
            # metadata consumers and add an explicit machine-readable strategy.
            "mode": description,
            "strategy": self.mode,
            "mean_mV": _json_stat(self.mean_mV),
            "std_mV": _json_stat(self.std_mV),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Normalization":
        mean = value["mean_mV"]
        std = value["std_mV"]
        strategy = str(value.get("strategy", "")).strip().lower()
        if strategy not in {"global", "feature"}:
            raw_mode = str(value.get("mode", "global")).strip().lower()
            strategy = "feature" if raw_mode == "feature" else "global"
        if isinstance(mean, list) or isinstance(std, list):
            strategy = "feature"
        return cls(mean_mV=mean, std_mV=std, mode=strategy)


def compute_normalization(
    datasets_and_indices: Iterable[tuple[PreparedDataset, np.ndarray]],
    *,
    chunk_size: int = 2048,
    featurewise: bool = False,
    subsampling_factor: int = 1,
) -> Normalization:
    entries = list(datasets_and_indices)
    if not entries:
        raise ValueError("Cannot normalize an empty training dataset")
    factor = normalize_subsampling_factor(subsampling_factor)

    if featurewise:
        lengths = {
            int(
                sum(
                    (int(value) + factor - 1) // factor
                    for value in (
                        dataset.manifest.get("input_component_lengths")
                        if isinstance(dataset.manifest.get("input_component_lengths"), list)
                        else [int(dataset.windows_mV.shape[-1])]
                    )
                )
            )
            for dataset, _ in entries
        }
        if len(lengths) != 1:
            raise ValueError(
                "Feature normalization requires equal waveform lengths, got "
                f"{sorted(lengths)}"
            )
        length = lengths.pop()
        count = 0
        total = np.zeros(length, dtype=np.float64)
        total_square = np.zeros(length, dtype=np.float64)
        for dataset, indices in entries:
            values_indices = np.asarray(indices, dtype=np.int64)
            for start in range(0, values_indices.size, chunk_size):
                chunk_indices = values_indices[start : start + chunk_size]
                values = np.asarray(dataset.windows_mV[chunk_indices], dtype=np.float64)
                values = apply_component_subsampling(
                    values,
                    factor,
                    dataset.manifest.get("input_component_lengths"),
                )
                # Pool events and detector channels, but keep every time position
                # as an independent feature.
                count += int(values.shape[0] * values.shape[1])
                total += np.sum(values, axis=(0, 1))
                total_square += np.sum(values * values, axis=(0, 1))
        if count == 0:
            raise ValueError("Cannot normalize an empty training dataset")
        mean = total / count
        variance = np.maximum(total_square / count - mean * mean, 0.0)
        # Constant features carry no information. A scale of one maps them to
        # exactly zero without amplifying floating-point noise.
        std = np.sqrt(variance)
        std = np.where(std > 1e-6, std, 1.0)
        return Normalization(mean.astype(np.float32), std.astype(np.float32), mode="feature")

    count = 0
    total = 0.0
    total_square = 0.0
    for dataset, indices in entries:
        values_indices = np.asarray(indices, dtype=np.int64)
        for start in range(0, values_indices.size, chunk_size):
            chunk_indices = values_indices[start : start + chunk_size]
            values = np.asarray(dataset.windows_mV[chunk_indices], dtype=np.float64)
            values = apply_component_subsampling(
                values,
                factor,
                dataset.manifest.get("input_component_lengths"),
            )
            count += values.size
            total += float(np.sum(values))
            total_square += float(np.sum(values * values))
    if count == 0:
        raise ValueError("Cannot normalize an empty training dataset")
    mean = total / count
    variance = max(total_square / count - mean * mean, 1e-12)
    return Normalization(float(mean), float(np.sqrt(variance)), mode="global")




def window_anchor_shift_pair_ps(
    dataset: PreparedDataset, indices: np.ndarray | int
) -> np.ndarray:
    """Return the exact interpolated-LED minus native-anchor pair shift.

    For detector ``j``, ``delta_j = t_LED,j - t_anchor,j``.  The returned
    ordered-pair quantity is ``delta_1 - delta_2`` in ps.  Legacy/synthetic
    datasets without anchor timestamps receive zero shift for compatibility.
    """

    selected = np.asarray(indices, dtype=np.int64)
    if dataset.window_anchor_time_fs is None:
        return np.zeros(selected.shape, dtype=np.float64)
    led = np.asarray(dataset.led_time_fs[selected], dtype=np.float64)
    anchor = np.asarray(dataset.window_anchor_time_fs[selected], dtype=np.float64)
    per_detector_shift_fs = led - anchor
    return (per_detector_shift_fs[..., 0] - per_detector_shift_fs[..., 1]) / 1000.0


def window_anchor_delta_pair_ps(
    dataset: PreparedDataset, indices: np.ndarray | int
) -> np.ndarray:
    """Return the native-window anchor pair difference in ps.

    The ML waveform for detector ``j`` is translated by an integer number of
    acquisition samples so that its native anchor sample becomes the local
    origin.  Consequently the global pair offset

        Delta t_anchor = t_anchor,1 - t_anchor,2

    is no longer present in the waveform values and must be handled
    analytically.  Legacy/synthetic datasets without anchor timestamps receive
    zero so their previous behaviour is preserved.
    """

    selected = np.asarray(indices, dtype=np.int64)
    if dataset.window_anchor_time_fs is None:
        return np.zeros(selected.shape, dtype=np.float64)
    anchor = np.asarray(dataset.window_anchor_time_fs[selected], dtype=np.float64)
    return (anchor[..., 0] - anchor[..., 1]) / 1000.0


def factored_correction_target_ps(
    dataset: PreparedDataset, indices: np.ndarray
) -> np.ndarray:
    """Relative LED target with native-grid alignment error removed.

    For detector ``j`` the continuous shift needed to place the interpolated
    LED at the local origin is ``t_LED,j`` while the native-grid window can only
    apply the discrete shift ``t_anchor,j``. Their residual mismatch is

        delta_j = t_LED,j - t_anchor,j.

    The ordered-pair mismatch still encoded as sub-sample phase in the shifted
    waveforms is therefore

        Delta delta = delta_1 - delta_2.

    Remove that directly LED-derived alignment residual from the supervision
    target:

        g(s1) - g(s2) = Delta t_LED - Delta delta - TOF_true.

    No anchor or alignment term is added analytically at inference. The model
    output itself is the complete learned correction applied to Delta t_LED.
    """

    selected = np.asarray(indices, dtype=np.int64)
    led_delta_ps = (
        np.asarray(dataset.led_time_fs[selected, 0], dtype=np.float64)
        - np.asarray(dataset.led_time_fs[selected, 1], dtype=np.float64)
    ) / 1000.0
    alignment_residual_ps = window_anchor_shift_pair_ps(dataset, selected)
    return led_delta_ps - alignment_residual_ps - float(dataset.true_tof_ps)


class CorrectionDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    """View of a post-selection prepared dataset.

    The target removes only the pairwise residual between the continuous LED
    alignment that would be required and the discrete native-sample shift that
    was actually applied. No LED-derived shift is added back at inference.
    """

    def __init__(
        self,
        dataset: PreparedDataset,
        indices: np.ndarray,
        normalization: Normalization,
        input_transform: str = "none",
        subsampling_factor: int = 1,
    ) -> None:
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.normalization = normalization
        self.input_transform = normalize_input_transform(input_transform)
        self.subsampling_factor = normalize_subsampling_factor(subsampling_factor)
        raw_lengths = dataset.manifest.get("input_component_lengths")
        source_lengths = (
            [int(value) for value in raw_lengths]
            if isinstance(raw_lengths, list)
            else [int(dataset.input_length)]
        )
        self.transformed_component_lengths = transformed_component_lengths(
            source_lengths, self.input_transform
        )

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, position: int):
        index = int(self.indices[position])
        pair = np.asarray(self.dataset.windows_mV[index], dtype=np.float32)
        pair = np.asarray(
            apply_input_transform(pair, self.input_transform), dtype=np.float32
        )
        pair = np.asarray(
            apply_component_subsampling(
                pair,
                self.subsampling_factor,
                self.transformed_component_lengths,
            ),
            dtype=np.float32,
        ).copy()
        mean = _stat_array(self.normalization.mean_mV)
        std = _stat_array(self.normalization.std_mV)
        if mean.ndim == 1 and int(mean.size) != int(pair.shape[-1]):
            raise ValueError(
                "Feature normalization length does not match waveform input: "
                f"{mean.size} != {pair.shape[-1]}"
            )
        pair = (pair - mean) / std
        led_delta_ps = (
            int(self.dataset.led_time_fs[index, 0]) - int(self.dataset.led_time_fs[index, 1])
        ) / 1000.0
        cfd_delta_ps = (
            int(self.dataset.cfd_time_fs[index, 0]) - int(self.dataset.cfd_time_fs[index, 1])
        ) / 1000.0
        true_tof_ps = self.dataset.true_tof_ps
        alignment_residual_ps = float(window_anchor_shift_pair_ps(self.dataset, index))
        target_ps = led_delta_ps - alignment_residual_ps - true_tof_ps
        return (
            torch.from_numpy(pair),
            torch.tensor(target_ps, dtype=torch.float32),
            torch.tensor(led_delta_ps, dtype=torch.float32),
            torch.tensor(cfd_delta_ps, dtype=torch.float32),
            torch.tensor(true_tof_ps, dtype=torch.float32),
            torch.tensor(alignment_residual_ps, dtype=torch.float32),
        )
