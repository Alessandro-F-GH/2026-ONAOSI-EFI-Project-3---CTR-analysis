from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml_pipeline.dataset import PreparedDataset
from ml_pipeline.prediction import prediction_dataset_view
from ml_pipeline.torch_data import (
    CorrectionDataset,
    Normalization,
    factored_correction_target_ps,
    window_anchor_shift_pair_ps,
)
from ml_pipeline.training_utils import predict_loader


def _dataset() -> PreparedDataset:
    n = 1
    zeros_i64 = np.zeros(n, dtype=np.int64)
    zeros_pair_i64 = np.zeros((n, 2), dtype=np.int64)
    zeros_pair_f32 = np.zeros((n, 2), dtype=np.float32)
    windows = np.zeros((n, 2, 4), dtype=np.float32)
    energy_led = np.asarray([[10_400_000, 6_100_000]], dtype=np.int64)
    timing_led = np.asarray([[20_250_000, 16_050_000]], dtype=np.int64)
    energy_anchor = np.asarray([[10_000_000, 6_000_000]], dtype=np.int64)
    timing_energy_anchor = np.asarray([[20_000_000, 16_000_000]], dtype=np.int64)
    timing_anchor = np.asarray([[20_000_000, 16_000_000]], dtype=np.int64)
    return PreparedDataset(
        directory=Path("synthetic"),
        manifest={
            "fingerprint": "synthetic",
            "true_tof_ps": 4000.0,
            "led_timestamp_source": "energy_channels",
            "timing_channel_waveforms_saved": True,
            "waveform_grid": "native_acquisition_samples",
        },
        event_id=zeros_i64,
        event_index=zeros_i64,
        source_file_id=zeros_pair_i64,
        source_run_index=zeros_i64,
        bias_voltage_V=np.zeros(n, dtype=np.float64),
        amplitude_mV=zeros_pair_f32,
        noise_rms_mV=zeros_pair_f32,
        trigger_index=np.zeros((n, 2), dtype=np.int32),
        led_time_fs=energy_led,
        cfd_time_fs=energy_led,
        windows_mV=windows,
        relative_time_ps=np.asarray([-1000.0, 0.0, 1000.0, 2000.0]),
        energy_led_time_fs=energy_led,
        timing_led_time_fs=timing_led,
        energy_cfd_time_fs=energy_led,
        timing_cfd_time_fs=timing_led,
        energy_window_anchor_time_fs=energy_anchor,
        timing_aligned_energy_window_anchor_time_fs=timing_energy_anchor,
        timing_window_anchor_time_fs=timing_anchor,
        window_anchor_time_fs=energy_anchor,
        timing_aligned_energy_windows_mV=windows.copy(),
        timing_windows_mV=windows.copy(),
        timing_relative_time_ps=np.asarray([-1000.0, 0.0, 1000.0, 2000.0]),
        train=np.asarray([0], dtype=np.int64),
        validation=np.asarray([0], dtype=np.int64),
        evaluation=np.asarray([0], dtype=np.int64),
    )


class _ZeroLearnedCorrection(torch.nn.Module):
    def forward(self, waveform_pair: torch.Tensor) -> torch.Tensor:
        return torch.zeros(waveform_pair.shape[0], dtype=waveform_pair.dtype)


class _RawLedBaseline(_ZeroLearnedCorrection):
    apply_window_anchor_shift = False


def test_anchor_shift_is_removed_from_target_and_added_back_to_led_correction() -> None:
    dataset = _dataset()
    indices = np.asarray([0], dtype=np.int64)

    # LED pair = 4300 ps, anchor pair = 4000 ps, so the known shift is 300 ps.
    np.testing.assert_allclose(window_anchor_shift_pair_ps(dataset, indices), [300.0])
    np.testing.assert_allclose(factored_correction_target_ps(dataset, indices), [0.0])

    correction_dataset = CorrectionDataset(
        dataset,
        indices,
        Normalization(mean_mV=0.0, std_mV=1.0),
    )
    batch = next(iter(DataLoader(correction_dataset, batch_size=1)))
    assert float(batch[1][0]) == 0.0
    assert float(batch[5][0]) == 300.0

    result = predict_loader(
        _ZeroLearnedCorrection(),
        DataLoader(correction_dataset, batch_size=1),
        torch.device("cpu"),
    )
    np.testing.assert_allclose(result["prediction_ps"], [0.0])
    np.testing.assert_allclose(result["total_led_correction_ps"], [300.0])
    np.testing.assert_allclose(result["corrected_ps"], [4000.0])

    baseline = predict_loader(
        _RawLedBaseline(),
        DataLoader(correction_dataset, batch_size=1),
        torch.device("cpu"),
    )
    np.testing.assert_allclose(baseline["corrected_ps"], [4300.0])


def test_prediction_view_uses_anchor_matching_the_selected_waveform_alignment() -> None:
    dataset = _dataset()

    energy_view = prediction_dataset_view(
        dataset, input_waveforms="energy", target="energy_led"
    )
    np.testing.assert_array_equal(
        energy_view.window_anchor_time_fs,
        dataset.energy_window_anchor_time_fs,
    )

    energy_to_timing_view = prediction_dataset_view(
        dataset, input_waveforms="energy", target="timing_led"
    )
    np.testing.assert_array_equal(
        energy_to_timing_view.window_anchor_time_fs,
        dataset.timing_aligned_energy_window_anchor_time_fs,
    )

    timing_view = prediction_dataset_view(
        dataset, input_waveforms="timing", target="timing_led"
    )
    np.testing.assert_array_equal(
        timing_view.window_anchor_time_fs,
        dataset.timing_window_anchor_time_fs,
    )
