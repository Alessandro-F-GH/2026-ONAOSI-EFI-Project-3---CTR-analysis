from __future__ import annotations

import numpy as np

from ml_pipeline.signal import _native_window, _timing_from_basic, relative_window_grid_ps
from utils.signal import BasicFeatures, prepare_timing_features


def _basic(signal: np.ndarray) -> BasicFeatures:
    return BasicFeatures(
        baseline_mV=0.0,
        noise_rms_mV=0.0,
        amplitude_mV=float(np.max(signal)),
        peak_index=int(np.argmax(signal)),
        trigger_index=1,
        trigger_time_fs=np.int64(1_000_000),
        corrected_signal_mV=np.asarray(signal, dtype=np.float64),
    )


def test_led_and_cfd_use_only_local_linear_crossing_interpolation() -> None:
    # Samples are at 0, 1, 2, 3 ns. LED=7 mV crosses halfway between
    # (1 ns, 4 mV) and (2 ns, 10 mV). CFD=0.5*10=5 mV crosses 1/6 of that interval.
    signal = np.asarray([0.0, 4.0, 10.0, 8.0])
    timing = _timing_from_basic(
        _basic(signal),
        horizontal_interval_s=1.0e-9,
        horizontal_offset_s=0.0,
        extraction_config={
            "analysis_crop_ns": {"before": 2.0, "after": 3.0},
            "led_threshold_mV": 7.0,
            "cfd_fraction": 0.5,
        },
    )
    assert timing.valid
    assert int(timing.led_time_fs) == 1_500_000
    assert int(timing.cfd_time_fs) == 1_166_667


def test_prepared_window_stays_on_native_sample_grid() -> None:
    config = {"ml_window_ns": {"before": 2.0, "after": 1.0}}
    grid = relative_window_grid_ps(config, native_interval_s=1.0e-9)
    np.testing.assert_allclose(grid, [-2000.0, -1000.0, 0.0, 1000.0], atol=1e-9)

    signal = np.asarray([0.0, 4.0, 10.0, 8.0, 3.0])
    window = _native_window(
        signal,
        horizontal_interval_s=1.0e-9,
        horizontal_offset_s=0.0,
        alignment_ns=2.4,
        relative_grid_ps=grid,
    )
    # 2.4 ns is aligned to native sample index 2; values are copied, not interpolated.
    np.testing.assert_array_equal(window, signal[:4].astype(np.float32))


def test_legacy_upsample_argument_is_ignored_for_timing() -> None:
    kwargs = {
        "trigger_index": 1,
        "horizontal_interval_s": 1.0e-9,
        "horizontal_offset_s": 0.0,
        "crop_before_ns": 2.0,
        "crop_after_ns": 3.0,
        "led_thresholds_mV": np.asarray([7.0]),
        "cfd_fractions": np.asarray([0.5]),
    }
    signal = np.asarray([0.0, 4.0, 10.0, 8.0])
    fine = prepare_timing_features(signal, upsample_step_ps=2.5, **kwargs)
    coarse = prepare_timing_features(signal, upsample_step_ps=500.0, **kwargs)
    np.testing.assert_array_equal(fine.led_times_fs, coarse.led_times_fs)
    np.testing.assert_array_equal(fine.cfd_times_fs, coarse.cfd_times_fs)


def test_cfd_uses_last_rising_crossing_before_main_peak() -> None:
    # The 5 mV CFD threshold is crossed once by an early pre-pulse and again by
    # the main pulse. CFD must use the crossing attached to the peak that defines
    # the amplitude, not the first crossing in the analysis crop.
    signal = np.asarray([0.0, 6.0, 2.0, 4.0, 10.0, 8.0])
    timing = _timing_from_basic(
        _basic(signal),
        horizontal_interval_s=1.0e-9,
        horizontal_offset_s=0.0,
        extraction_config={
            "analysis_crop_ns": {"before": 2.0, "after": 6.0},
            "led_threshold_mV": 7.0,
            "cfd_fraction": 0.5,
        },
    )
    assert timing.valid
    # Main rising edge: 4 mV at 3 ns to 10 mV at 4 ns; 5 mV is at 3 + 1/6 ns.
    assert int(timing.cfd_time_fs) == 3_166_667


def test_prepare_timing_features_cfd_ignores_early_low_threshold_crossing() -> None:
    signal = np.asarray([0.0, 6.0, 2.0, 4.0, 10.0, 8.0])
    timing = prepare_timing_features(
        signal,
        trigger_index=1,
        horizontal_interval_s=1.0e-9,
        horizontal_offset_s=0.0,
        crop_before_ns=2.0,
        crop_after_ns=6.0,
        led_thresholds_mV=np.asarray([7.0]),
        cfd_fractions=np.asarray([0.5]),
    )
    assert int(timing.cfd_times_fs[0]) == 3_166_667


def test_timing_channel_materializes_cfd_for_standard_method() -> None:
    from ml_pipeline.signal import extract_timing_channel

    waveform_config = {
        "baseline_samples": 2,
        "search_trigger_threshold_mV": 3.0,
        "analysis_crop_ns": {"before": 2.0, "after": 3.0},
        "led_threshold_mV": 7.0,
        "cfd_fraction": 0.5,
        "ml_window_ns": {"before": 2.0, "after": 1.0},
        "denoising": {"enabled": False},
        "timing_channel_led": {"enabled": True},
    }
    # Gain 1 mV/count. Baseline is zero; the main edge is 4 -> 10 mV.
    raw = np.asarray([0, 0, 0, 4, 10, 8, 3], dtype=np.int16)
    relative_grid_ps = np.asarray([-2000.0, -1000.0, 0.0, 1000.0])
    extracted = extract_timing_channel(
        raw,
        vertical_gain_v_per_count=1.0e-3,
        vertical_offset_v=0.0,
        horizontal_interval_s=1.0e-9,
        horizontal_offset_s=0.0,
        polarity=1,
        waveform_config=waveform_config,
        relative_grid_ps=relative_grid_ps,
    )
    assert extracted.valid
    assert int(extracted.led_time_fs) == 3_500_000
    assert int(extracted.cfd_time_fs) == 3_166_667
