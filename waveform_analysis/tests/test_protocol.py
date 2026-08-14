from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ml_pipeline.models.cnn_regressor import AntisymmetricCNNRegressor
from ml_pipeline.models.constructive_mlp_encoder import AntisymmetricConstructiveMLPEncoder
from ml_pipeline.models.linear_svr import LinearPairSVR
from ml_pipeline.dataset import PreparedDataset, load_prepared_dataset
from ml_pipeline.event_selection import apply_energy_preselection
from ml_pipeline.prepared_data import (
    _raw_preprocess_config,
    input_channel_variant_dataset_view,
    materialize_selected_dataset,
    raw_dataset_view,
)
from ml_pipeline.study import (
    _fit_early_split,
    _kfold,
    _aggregate_fold_stats,
    _distribution_stats,
    _ml_input_variant_for_mode,
    _random_dev_blind,
    _voltage_from_name,
    run_study,
)
from ml_pipeline.torch_data import (
    Normalization,
    CorrectionDataset,
    factored_correction_target_ps,
    window_anchor_delta_pair_ps,
)
from utils.fit import fit_delta_times_integer_fs


class SplitProtocolTests(unittest.TestCase):
    def test_cv_score_never_enters_fit_or_early_stop(self) -> None:
        development, blind = _random_dev_blind(200, blind_fraction=0.2, seed=7)
        self.assertEqual(len(np.intersect1d(development, blind)), 0)
        self.assertEqual(len(development) + len(blind), 200)

        folds = _kfold(development, n_splits=5, seed=11)
        seen_score: list[np.ndarray] = []
        for fold_index, (train_pool, score) in enumerate(folds):
            fit, early = _fit_early_split(train_pool, fraction=0.15, seed=100 + fold_index)
            self.assertEqual(len(np.intersect1d(fit, early)), 0)
            self.assertEqual(len(np.intersect1d(fit, score)), 0)
            self.assertEqual(len(np.intersect1d(early, score)), 0)
            np.testing.assert_array_equal(np.sort(np.concatenate([fit, early])), np.sort(train_pool))
            seen_score.append(score)
        np.testing.assert_array_equal(
            np.sort(np.concatenate(seen_score)), np.sort(development)
        )

    def test_filename_voltage(self) -> None:
        pattern = r"(?P<voltage>\d+(?:\.\d+)?)V"
        self.assertEqual(_voltage_from_name("45V-400mV.root", pattern), 45.0)
        self.assertEqual(_voltage_from_name("47.5V-470mV.root", pattern), 47.5)




class ChannelVariantPolicyTests(unittest.TestCase):
    def test_mode_resolves_single_channel_variant(self) -> None:
        study = {
            "preprocessing": {
                "input_variant_by_channel": {
                    "energy": "denoised",
                    "timing": "raw",
                }
            }
        }
        self.assertEqual(_ml_input_variant_for_mode(study, "energy_to_energy"), "denoised")
        self.assertEqual(_ml_input_variant_for_mode(study, "energy_to_timing"), "denoised")
        self.assertEqual(_ml_input_variant_for_mode(study, "timing_to_timing"), "raw")


    def test_raw_multithreshold_view_cannot_switch_variant(self) -> None:
        n, length = 2, 4
        raw = np.zeros((n, 2, length), dtype=np.float32)
        dataset = PreparedDataset(
            directory=Path("."), manifest={"true_tof_ps": 0.0},
            event_id=np.arange(n), event_index=np.arange(n),
            source_file_id=np.zeros((n, 2), dtype=np.int64),
            source_run_index=np.zeros(n, dtype=np.int64),
            bias_voltage_V=np.zeros(n), amplitude_mV=np.zeros((n, 2)),
            noise_rms_mV=np.zeros((n, 2)), trigger_index=np.zeros((n, 2), dtype=np.int64),
            led_time_fs=np.zeros((n, 2), dtype=np.int64), cfd_time_fs=np.zeros((n, 2), dtype=np.int64),
            windows_mV=raw, relative_time_ps=np.arange(length, dtype=np.float64),
            denoised_windows_mV=np.ones_like(raw),
        )
        view = raw_dataset_view(dataset)
        self.assertIs(view.windows_mV, raw)
        self.assertEqual(view.manifest["ml_input_variant"], "raw")

    def test_raw_multithreshold_view_uses_dedicated_raw_energy_when_ml_is_denoised(self) -> None:
        n, length = 2, 4
        ml_energy = np.ones((n, 2, length), dtype=np.float32)
        raw_energy = np.zeros((n, 2, length), dtype=np.float32)
        ml_led = np.full((n, 2), 100, dtype=np.int64)
        raw_led = np.full((n, 2), 200, dtype=np.int64)
        dataset = PreparedDataset(
            directory=Path("."),
            manifest={
                "true_tof_ps": 0.0,
                "input_variant_by_channel": {"energy": "denoised", "timing": "raw"},
            },
            event_id=np.arange(n), event_index=np.arange(n),
            source_file_id=np.zeros((n, 2), dtype=np.int64),
            source_run_index=np.zeros(n, dtype=np.int64),
            bias_voltage_V=np.zeros(n), amplitude_mV=np.zeros((n, 2)),
            noise_rms_mV=np.zeros((n, 2)), trigger_index=np.zeros((n, 2), dtype=np.int64),
            led_time_fs=ml_led, cfd_time_fs=ml_led.copy(),
            windows_mV=ml_energy, relative_time_ps=np.arange(length, dtype=np.float64),
            energy_led_time_fs=ml_led, energy_cfd_time_fs=ml_led.copy(),
            raw_energy_led_time_fs=raw_led,
            raw_energy_cfd_time_fs=raw_led.copy(),
            raw_energy_windows_mV=raw_energy,
        )
        view = raw_dataset_view(dataset)
        self.assertIs(view.windows_mV, raw_energy)
        self.assertIs(view.energy_led_time_fs, raw_led)
        self.assertEqual(view.manifest["energy_led_signal_variant"], "raw")

    def test_channel_specific_view_does_not_require_other_family_denoising(self) -> None:
        n, length = 3, 8
        raw_energy = np.zeros((n, 2, length), dtype=np.float32)
        denoised_energy = np.ones((n, 2, length), dtype=np.float32)
        raw_timing = np.full((n, 2, length), 2.0, dtype=np.float32)
        denoised_timing = np.full((n, 2, length), 3.0, dtype=np.float32)
        basic = dict(
            directory=Path("."),
            manifest={"true_tof_ps": 0.0},
            event_id=np.arange(n),
            event_index=np.arange(n),
            source_file_id=np.zeros((n, 2), dtype=np.int64),
            source_run_index=np.zeros(n, dtype=np.int64),
            bias_voltage_V=np.zeros(n),
            amplitude_mV=np.zeros((n, 2)),
            noise_rms_mV=np.zeros((n, 2)),
            trigger_index=np.zeros((n, 2), dtype=np.int64),
            led_time_fs=np.zeros((n, 2), dtype=np.int64),
            cfd_time_fs=np.zeros((n, 2), dtype=np.int64),
            windows_mV=raw_energy,
            relative_time_ps=np.arange(length, dtype=np.float64),
            timing_windows_mV=raw_timing,
            timing_relative_time_ps=np.arange(length, dtype=np.float64),
        )
        energy_only = PreparedDataset(**basic, denoised_windows_mV=denoised_energy)
        energy_view = input_channel_variant_dataset_view(energy_only, "energy", "denoised")
        self.assertIs(energy_view.windows_mV, denoised_energy)
        self.assertIs(energy_view.timing_windows_mV, raw_timing)

        timing_only = PreparedDataset(**basic, denoised_timing_windows_mV=denoised_timing)
        timing_view = input_channel_variant_dataset_view(timing_only, "timing", "denoised")
        self.assertIs(timing_view.windows_mV, raw_energy)
        self.assertIs(timing_view.timing_windows_mV, denoised_timing)




class ClassicalCTRTests(unittest.TestCase):
    def test_ctr_uses_classical_sample_standard_deviation(self) -> None:
        values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
        metrics = _distribution_stats(values, method="synthetic")
        expected_std = float(np.std(values, ddof=1))
        expected_ctr = 2.0 * np.sqrt(2.0 * np.log(2.0)) * expected_std
        self.assertAlmostEqual(metrics["mean_ps"], 0.0)
        self.assertAlmostEqual(metrics["std_ps"], expected_std)
        self.assertAlmostEqual(metrics["ctr_ps"], expected_ctr)
        self.assertNotIn("bias_ps", metrics)

    def test_cv_summary_averages_fold_metrics_without_pooling_outputs(self) -> None:
        # Deliberately give the two model folds very different centers. Pooling
        # their raw outputs would inflate CTR; averaging fold CTRs must not.
        fold_a = _distribution_stats(np.array([-1.0, 0.0, 1.0]), method="a")
        fold_b = _distribution_stats(np.array([99.0, 100.0, 101.0]), method="b")
        summary = _aggregate_fold_stats([fold_a, fold_b], method="cv")
        self.assertAlmostEqual(summary["ctr_ps"], fold_a["ctr_ps"])
        self.assertAlmostEqual(summary["ctr_fold_std_ps"], 0.0)
        self.assertAlmostEqual(summary["mean_ps"], 50.0)

class SharedFitTests(unittest.TestCase):
    def test_fit_uses_every_event_including_tails(self) -> None:
        rng = np.random.default_rng(123)
        values_ps = np.concatenate([rng.normal(0.0, 30.0, 1000), np.array([-400.0, 450.0])])
        values_fs = np.rint(values_ps * 1000.0).astype(np.int64)
        fit = fit_delta_times_integer_fs(
            values_fs,
            method="synthetic",
            parameter=0.0,
            n_total=len(values_fs),
            n_selected=len(values_fs),
            config={
                "min_events": 10,
                "histogram_bin_ps": 10.0,
                "bin_phase_count": 10,
            },
        )
        self.assertTrue(fit.success)
        self.assertEqual(fit.n_fit, len(values_fs))
        self.assertEqual(int(np.sum(fit.counts)), len(values_fs))
        self.assertGreater(fit.ctr_ps, 0.0)
        self.assertAlmostEqual(fit.bin_width_ps, 10.0)
        self.assertGreaterEqual(fit.bin_phase_ps, 0.0)
        self.assertLess(fit.bin_phase_ps, 10.0)

    def test_broad_non_gaussian_candidate_does_not_crash_fitter(self) -> None:
        rng = np.random.default_rng(321)
        values_ps = np.concatenate([
            rng.normal(-80.0, 700.0, 1500),
            rng.normal(2500.0, 300.0, 40),
        ])
        values_fs = np.rint(values_ps * 1000.0).astype(np.int64)
        fit = fit_delta_times_integer_fs(
            values_fs, method="poor_candidate", parameter=0.0,
            n_total=len(values_fs), n_selected=len(values_fs),
            config={
                "min_events": 100,
                "histogram_bin_ps": 10.0,
                "bin_phase_count": 10,
            },
        )
        self.assertTrue(fit.success)
        self.assertEqual(fit.n_fit, len(values_fs))
        self.assertTrue(np.isfinite(fit.ctr_ps))


class PreparedDatasetStorageTests(unittest.TestCase):
    def test_v6_does_not_require_duplicate_generic_led_cfd_files(self) -> None:
        import json

        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            n, length = 5, 8
            manifest = {
                "format_version": 6,
                "true_tof_ps": 0.0,
                "event_count": n,
            }
            (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            required = {
                "event_id": np.arange(n, dtype=np.int64),
                "event_index": np.arange(n, dtype=np.int64),
                "source_file_id": np.zeros((n, 2), dtype=np.int64),
                "source_run_index": np.zeros(n, dtype=np.int32),
                "bias_voltage_V": np.full(n, 45.0),
                "amplitude_mV": np.ones((n, 2), dtype=np.float32),
                "noise_rms_mV": np.ones((n, 2), dtype=np.float32),
                "trigger_index": np.zeros((n, 2), dtype=np.int32),
                "windows_mV": np.zeros((n, 2, length), dtype=np.float32),
                "relative_time_ps": np.arange(length, dtype=np.float64),
                "energy_led_time_fs": np.arange(n * 2, dtype=np.int64).reshape(n, 2),
                "energy_cfd_time_fs": (100 + np.arange(n * 2, dtype=np.int64)).reshape(n, 2),
            }
            for name, values in required.items():
                np.save(directory / f"{name}.npy", values)
            dataset = load_prepared_dataset(directory)
            np.testing.assert_array_equal(dataset.led_time_fs, required["energy_led_time_fs"])
            np.testing.assert_array_equal(dataset.cfd_time_fs, required["energy_cfd_time_fs"])
            self.assertFalse((directory / "led_time_fs.npy").exists())
            self.assertFalse((directory / "cfd_time_fs.npy").exists())





class ChannelSelectiveMaterializationTests(unittest.TestCase):
    def test_denoising_is_not_reapplied_after_canonical_preprocessing(self) -> None:
        n, length = 6, 32
        rng = np.random.default_rng(2)
        raw_energy = rng.normal(size=(n, 2, length)).astype(np.float32)
        raw_timing = rng.normal(size=(n, 2, length)).astype(np.float32)
        cache = SimpleNamespace(
            valid=np.ones(n, dtype=bool),
            event_id=np.arange(n, dtype=np.int64),
            event_index=np.arange(n, dtype=np.int64),
            source_file_id=np.zeros((n, 2), dtype=np.int64),
            source_run_index=np.zeros(n, dtype=np.int32),
            bias_voltage_V=np.full(n, 45.0),
            amplitude_mV=np.full((n, 2), 100.0, dtype=np.float32),
            noise_rms_mV=np.ones((n, 2), dtype=np.float32),
            trigger_index=np.full((n, 2), 10, dtype=np.int32),
            windows_mV=raw_energy,
            timing_aligned_energy_windows_mV=raw_energy.copy(),
            timing_windows_mV=raw_timing,
            relative_time_ps=np.arange(length, dtype=np.float64) * 10.0,
            timing_relative_time_ps=np.arange(length, dtype=np.float64) * 10.0,
            energy_led_time_fs=np.tile(np.array([[1000, 900]], dtype=np.int64), (n, 1)),
            timing_led_time_fs=np.tile(np.array([[1100, 1000]], dtype=np.int64), (n, 1)),
            energy_cfd_time_fs=np.tile(np.array([[1000, 900]], dtype=np.int64), (n, 1)),
            timing_cfd_time_fs=np.tile(np.array([[1100, 1000]], dtype=np.int64), (n, 1)),
            energy_window_anchor_time_fs=np.zeros((n, 2), dtype=np.int64),
            timing_aligned_energy_window_anchor_time_fs=np.zeros((n, 2), dtype=np.int64),
            timing_window_anchor_time_fs=np.zeros((n, 2), dtype=np.int64),
            manifest={
                "fingerprint": "fake",
                "energy_channels_one_based": [1, 2],
                "waveform_grid": "native_samples",
                "native_sample_interval_ps": 10.0,
                "timing_native_sample_interval_ps": 10.0,
            },
        )
        config = {
            "source_root": "fake.root",
            "true_tof_ps": 0.0,
            "selection": {"minimum_events": 1},
            "photopeak": {"enabled": False},
            "input_variant_by_channel": {"energy": "denoised", "timing": "raw"},
            "denoising": {
                "enabled": True, "method": "butterworth_lowpass",
                "cutoff_GHz": 1.0, "order": 2,
            },
            "materialization_chunk_size": 3,
        }
        with tempfile.TemporaryDirectory() as directory_name:
            out = Path(directory_name) / "prepared"
            materialize_selected_dataset(
                cache, output=out, config=config, rebuild=True,
                logger=SimpleNamespace(info=lambda *args, **kwargs: None),
            )
            self.assertFalse((out / "denoised_windows_mV.npy").exists())
            self.assertFalse((out / "denoised_timing_aligned_energy_windows_mV.npy").exists())
            self.assertFalse((out / "denoised_timing_windows_mV.npy").exists())
            manifest = __import__("json").loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["denoising_stage"], "before_led_cfd_and_window_extraction")
            self.assertEqual(manifest["energy_led_signal_variant"], "denoised")
            self.assertEqual(manifest["timing_led_signal_variant"], "raw")

    def test_root_preprocessing_applies_channel_variant_before_led_extraction(self) -> None:
        study = {
            "data": {
                "channels": {"energy": [1, 2], "polarities": [1, 1], "timing": [3, 4], "timing_polarities": [1, 1]},
                "true_tof_ps": 0.0,
            },
            "windows_ns": [{"before_ns": 4.0, "after_ns": 20.0}],
            "preprocessing": {
                "common": {
                    "baseline_samples": 10,
                    "search_trigger_threshold_mV": 5.0,
                    "analysis_crop_ns": {"before": 2.0, "after": 10.0},
                    "led_threshold_mV": 7.0,
                    "cfd_fraction": 0.1,
                },
                "energy": {},
                "timing": {},
                "input_variant_by_channel": {"energy": "denoised", "timing": "raw"},
                "denoising": {"method": "butterworth_lowpass", "cutoff_GHz": 1.0, "order": 4},
                "io": {},
                "parallelization": {},
            },
        }
        cfg = _raw_preprocess_config(study, Path("fake.root"), Path("cache"))
        self.assertTrue(cfg["waveform"]["denoising"]["enabled"])
        self.assertFalse(cfg["waveform"]["timing_channel_led"]["denoising"]["enabled"])
        self.assertIn("selection", cfg)
        self.assertIn("photopeak", cfg)


class PhotopeakFirstPassTests(unittest.TestCase):
    def test_photopeak_preselection_uses_raw_energy_features_before_timing(self) -> None:
        rng = np.random.default_rng(91)
        # Dominant photopeak around 100 mV plus a low-amplitude population that
        # should be rejected before timing preprocessing.
        main = rng.normal(100.0, 3.0, size=(180, 2))
        low = rng.normal(35.0, 2.0, size=(30, 2))
        amplitudes = np.vstack([main, low]).astype(np.float32)
        noise = np.full_like(amplitudes, 1.0)
        trigger = np.full(amplitudes.shape, 50, dtype=np.int32)
        mask, summary = apply_energy_preselection(
            amplitudes,
            noise,
            trigger,
            energy_channels=(1, 2),
            selection={"minimum_events": 20},
            photopeak={
                "enabled": True,
                "histogram_bin_mV": 2.0,
                "search_quantile_min": 0.5,
                "smoothing_sigma_bins": 1.0,
                "initial_half_width_mV": 12.0,
                "max_iterations": 8,
                "iteration_sigma": 2.5,
                "convergence_tolerance_mV": 0.05,
                "selection_sigma_low": -3.0,
                "selection_sigma_high": 3.0,
            },
            logger=SimpleNamespace(info=lambda *args, **kwargs: None),
        )
        self.assertGreater(np.count_nonzero(mask[:180]), 160)
        self.assertEqual(np.count_nonzero(mask[180:]), 0)
        self.assertEqual(summary["stage"], "raw_energy_first_pass_before_timing_preprocessing")
        self.assertEqual(summary["source_signal_variant"], "raw_energy")
        self.assertEqual(len(summary["photopeak"]), 2)

    def test_materialization_does_not_refit_photopeak_after_expensive_preprocessing(self) -> None:
        n, length = 6, 8
        led = np.tile(np.array([[1000, 900]], dtype=np.int64), (n, 1))
        cache = SimpleNamespace(
            valid=np.ones(n, dtype=bool),
            event_id=np.arange(n, dtype=np.int64),
            event_index=np.arange(n, dtype=np.int64),
            source_file_id=np.zeros((n, 2), dtype=np.int64),
            source_run_index=np.zeros(n, dtype=np.int32),
            bias_voltage_V=np.full(n, 45.0),
            # Deliberately incompatible with the original photopeak. If the
            # second stage tried to fit/select again this test would fail.
            amplitude_mV=np.full((n, 2), -999.0, dtype=np.float32),
            noise_rms_mV=np.ones((n, 2), dtype=np.float32),
            trigger_index=np.full((n, 2), 4, dtype=np.int32),
            windows_mV=np.zeros((n, 2, length), dtype=np.float32),
            timing_aligned_energy_windows_mV=None,
            timing_windows_mV=None,
            relative_time_ps=np.arange(length, dtype=np.float64) * 10.0,
            timing_relative_time_ps=None,
            energy_led_time_fs=led,
            timing_led_time_fs=None,
            energy_cfd_time_fs=led.copy(),
            timing_cfd_time_fs=None,
            energy_window_anchor_time_fs=np.zeros((n, 2), dtype=np.int64),
            timing_aligned_energy_window_anchor_time_fs=None,
            timing_window_anchor_time_fs=None,
            manifest={
                "fingerprint": "photopeak-first",
                "event_count": n,
                "energy_channels_one_based": [1, 2],
                "waveform_grid": "native_samples",
                "native_sample_interval_ps": 10.0,
                "photopeak_preselection": {
                    "stage": "raw_energy_first_pass_before_timing_preprocessing",
                    "selected_events": n,
                    "photopeak": [{"channel": 1}, {"channel": 2}],
                },
            },
        )
        config = {
            "source_root": "fake.root",
            "true_tof_ps": 0.0,
            "selection": {"minimum_events": 1},
            # Intentionally incomplete config: it must not be consulted here.
            "photopeak": {"enabled": True},
            "input_variant_by_channel": {"energy": "raw", "timing": "raw"},
            "denoising": {"enabled": False},
            "materialization_chunk_size": 4,
        }
        with tempfile.TemporaryDirectory() as directory_name:
            dataset = materialize_selected_dataset(
                cache,
                output=Path(directory_name) / "prepared",
                config=config,
                rebuild=True,
                logger=SimpleNamespace(info=lambda *args, **kwargs: None),
            )
            self.assertEqual(dataset.event_id.size, n)
            self.assertEqual(
                dataset.manifest["selection"]["photopeak_preselection"]["selected_events"], n
            )


class DatasetLevelLedOutlierTests(unittest.TestCase):
    def test_robust_zscore_removes_gross_led_mismatch_before_materialization(self) -> None:
        n, length = 21, 16
        rng = np.random.default_rng(42)
        # Typical pair differences are ~100 ps with a few-ps spread; one event
        # is an obvious acquisition mismatch at +5 ns.
        delta_ps = np.concatenate([rng.normal(100.0, 5.0, n - 1), [5000.0]])
        ch1_fs = np.rint(delta_ps * 1000.0).astype(np.int64)
        led = np.column_stack([ch1_fs, np.zeros(n, dtype=np.int64)])
        cache = SimpleNamespace(
            valid=np.ones(n, dtype=bool),
            event_id=np.arange(n, dtype=np.int64),
            event_index=np.arange(n, dtype=np.int64),
            source_file_id=np.zeros((n, 2), dtype=np.int64),
            source_run_index=np.zeros(n, dtype=np.int32),
            bias_voltage_V=np.full(n, 45.0),
            amplitude_mV=np.full((n, 2), 100.0, dtype=np.float32),
            noise_rms_mV=np.ones((n, 2), dtype=np.float32),
            trigger_index=np.full((n, 2), 8, dtype=np.int32),
            windows_mV=np.zeros((n, 2, length), dtype=np.float32),
            timing_aligned_energy_windows_mV=None,
            timing_windows_mV=None,
            relative_time_ps=np.arange(length, dtype=np.float64) * 10.0,
            timing_relative_time_ps=None,
            energy_led_time_fs=led,
            timing_led_time_fs=None,
            energy_cfd_time_fs=led.copy(),
            timing_cfd_time_fs=None,
            energy_window_anchor_time_fs=np.zeros((n, 2), dtype=np.int64),
            timing_aligned_energy_window_anchor_time_fs=None,
            timing_window_anchor_time_fs=None,
            manifest={
                "fingerprint": "fake-zscore",
                "energy_channels_one_based": [1, 2],
                "waveform_grid": "native_samples",
                "native_sample_interval_ps": 10.0,
            },
        )
        config = {
            "source_root": "fake.root",
            "true_tof_ps": 0.0,
            "selection": {
                "minimum_events": 3,
                "led_outlier_rejection": {"enabled": True, "zscore_limit": 6.0},
            },
            "photopeak": {"enabled": False},
            "input_variant_by_channel": {"energy": "raw", "timing": "raw"},
            "denoising": {"enabled": False},
            "materialization_chunk_size": 8,
        }
        with tempfile.TemporaryDirectory() as directory_name:
            out = Path(directory_name) / "prepared"
            dataset = materialize_selected_dataset(
                cache, output=out, config=config, rebuild=True,
                logger=SimpleNamespace(info=lambda *args, **kwargs: None),
            )
            self.assertEqual(dataset.event_id.size, n - 1)
            self.assertNotIn(n - 1, set(np.asarray(dataset.event_id).tolist()))
            summary = dataset.manifest["selection"]["led_outlier_rejection"]
            self.assertTrue(summary["enabled"])
            self.assertEqual(summary["families"][0]["rejected"], 1)
            self.assertEqual(summary["families"][0]["zscore_limit"], 6.0)


class ProgressiveResumeTests(unittest.TestCase):
    @staticmethod
    def _dataset() -> PreparedDataset:
        n, length = 20, 8
        led = np.column_stack([np.arange(n) * 1000 + 20000, np.arange(n) * 1000])
        return PreparedDataset(
            directory=Path("prepared"),
            manifest={"true_tof_ps": 0.0, "request_fingerprint": "synthetic-fp"},
            event_id=np.arange(n), event_index=np.arange(n),
            source_file_id=np.zeros((n, 2), dtype=np.int64),
            source_run_index=np.zeros(n, dtype=np.int64),
            bias_voltage_V=np.full(n, 45.0), amplitude_mV=np.ones((n, 2)),
            noise_rms_mV=np.ones((n, 2)), trigger_index=np.zeros((n, 2), dtype=np.int64),
            led_time_fs=led, cfd_time_fs=led.copy(),
            windows_mV=np.zeros((n, 2, length), dtype=np.float32),
            relative_time_ps=np.arange(length, dtype=np.float64),
            energy_led_time_fs=led, energy_cfd_time_fs=led.copy(),
            energy_window_anchor_time_fs=np.zeros((n, 2), dtype=np.int64),
        )

    def test_completed_candidate_is_persisted_and_skipped_on_resume(self) -> None:
        logger = SimpleNamespace(
            info=lambda *args, **kwargs: None,
            warning=lambda *args, **kwargs: None,
            debug=lambda *args, **kwargs: None,
        )
        with tempfile.TemporaryDirectory() as directory_name:
            output = Path(directory_name) / "study"
            config = {
                "experiment": {"name": "resume_test", "output_dir": str(output)},
                "data": {"root_folder": ".", "root_glob": "*.root", "true_tof_ps": 0.0},
                "_model_spaces": [{
                    "id": "fake_model",
                    "search": {"method": "grid", "parameters": {}},
                    "base_train_config": {},
                }],
                "channel_modes": ["energy_to_energy"],
                "windows_ns": [
                    {"id": "w1", "before_ns": 1.0, "after_ns": 1.0},
                    {"id": "w2", "before_ns": 2.0, "after_ns": 2.0},
                ],
                "preprocessing": {
                    "prepared_dir": str(Path(directory_name) / "prepared"),
                    "input_variant_by_channel": {"energy": "raw", "timing": "raw"},
                    "subsampling_factors": [1],
                },
                "cross_validation": {
                    "blind_fraction": 0.2, "n_splits": 2,
                    "early_stop_fraction": 0.15, "seed": 11,
                },
                "multithreshold": {"enabled": False},
                "reporting": {
                    "dpi": 80, "voltage_pattern": r"(?P<voltage>\d+)V",
                    "save_final_fit_plots": False, "xai": {"enabled": False},
                },
                "_config_hash": "resume-test-hash",
                "_config_path": "synthetic.json",
            }
            dataset = self._dataset()
            calls: list[str] = []

            def interrupted_candidate(*args, **kwargs):
                window_id = kwargs["window"]["id"]
                calls.append(window_id)
                if window_id == "w2":
                    raise RuntimeError("synthetic interruption")
                return {
                    "n": 8, "mean_ps": 0.0, "std_ps": 10.0, "ctr_ps": 23.5482,
                    "ctr_fold_std_ps": 0.5, "rmse_ps": 10.0, "rmse_fold_std_ps": 0.5,
                }

            common_patches = (
                patch("ml_pipeline.study.discover_root_files", return_value=[Path("45V-test.root")]),
                patch("ml_pipeline.study.prepare_file_dataset", return_value=dataset),
                patch("ml_pipeline.study.plot_prepared_signal_examples"),
                patch("ml_pipeline.study._plot_ctr_vs_voltage"),
            )
            with common_patches[0], common_patches[1], common_patches[2], common_patches[3], \
                 patch("ml_pipeline.study._waveform_oof_candidate", side_effect=interrupted_candidate):
                with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                    run_study(
                        config, dry_run=False, resume=False, restart=False,
                        rebuild_preprocessing=False, logger=logger,
                    )

            self.assertEqual(calls, ["w1", "w2"])
            results_text = (output / "results.csv").read_text(encoding="utf-8")
            self.assertIn("23.5482", results_text)
            manifest = __import__("json").loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "in_progress")
            self.assertEqual(manifest["row_count"], 1)

            resumed_calls: list[str] = []
            def resumed_candidate(*args, **kwargs):
                window_id = kwargs["window"]["id"]
                resumed_calls.append(window_id)
                return {
                    "n": 8, "mean_ps": 0.0, "std_ps": 11.0, "ctr_ps": 25.90302,
                    "ctr_fold_std_ps": 0.6, "rmse_ps": 11.0, "rmse_fold_std_ps": 0.6,
                }

            def fake_final(*args, **kwargs):
                blind = np.asarray(args[3])
                residual = np.zeros(blind.size, dtype=np.float64)
                return residual, {
                    "n": int(blind.size), "mean_ps": 0.0, "std_ps": 0.0,
                    "ctr_ps": 0.0, "rmse_ps": 0.0,
                }, {}, None

            with patch("ml_pipeline.study.discover_root_files", return_value=[Path("45V-test.root")]), \
                 patch("ml_pipeline.study.prepare_file_dataset", return_value=dataset), \
                 patch("ml_pipeline.study.plot_prepared_signal_examples"), \
                 patch("ml_pipeline.study._plot_ctr_vs_voltage"), \
                 patch("ml_pipeline.study._waveform_oof_candidate", side_effect=resumed_candidate), \
                 patch("ml_pipeline.study._waveform_final", side_effect=fake_final):
                result = run_study(
                    config, dry_run=False, resume=True, restart=False,
                    rebuild_preprocessing=False, logger=logger,
                )

            self.assertEqual(resumed_calls, ["w2"])
            self.assertGreaterEqual(result["row_count"], 4)
            manifest = __import__("json").loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["completed_files"], [0])


class NativeAnchorFactoredTargetTests(unittest.TestCase):
    def test_target_factors_native_anchor_pair_and_reconstructs_full_correction(self) -> None:
        # LED pair = 34 ps. Per-channel continuous-minus-discrete alignment
        # residuals are +12 ps and -2 ps, hence Delta residual = 14 ps.
        # Relative target = 34 - 14 - 5 = 15 ps.
        dataset = PreparedDataset(
            directory=Path("."),
            manifest={"true_tof_ps": 5.0},
            event_id=np.array([0]), event_index=np.array([0]),
            source_file_id=np.zeros((1, 2), dtype=np.int64),
            source_run_index=np.zeros(1, dtype=np.int64),
            bias_voltage_V=np.array([45.0]),
            amplitude_mV=np.ones((1, 2), dtype=np.float32),
            noise_rms_mV=np.ones((1, 2), dtype=np.float32),
            trigger_index=np.zeros((1, 2), dtype=np.int64),
            led_time_fs=np.array([[112000, 78000]], dtype=np.int64),
            cfd_time_fs=np.array([[112000, 78000]], dtype=np.int64),
            windows_mV=np.zeros((1, 2, 4), dtype=np.float32),
            relative_time_ps=np.arange(4, dtype=np.float64),
            window_anchor_time_fs=np.array([[100000, 80000]], dtype=np.int64),
        )
        indices = np.array([0], dtype=np.int64)
        anchor = window_anchor_delta_pair_ps(dataset, indices)
        target = factored_correction_target_ps(dataset, indices)
        self.assertAlmostEqual(float(anchor[0]), 20.0)
        self.assertAlmostEqual(float(target[0]), 15.0)

        item = CorrectionDataset(
            dataset, indices, Normalization(0.0, 1.0),
        )[0]
        self.assertAlmostEqual(float(item[1]), 15.0)
        self.assertAlmostEqual(float(item[5]), 14.0)

    def test_missing_anchor_preserves_legacy_full_target(self) -> None:
        dataset = PreparedDataset(
            directory=Path("."), manifest={"true_tof_ps": 0.0},
            event_id=np.array([0]), event_index=np.array([0]),
            source_file_id=np.zeros((1, 2), dtype=np.int64),
            source_run_index=np.zeros(1, dtype=np.int64),
            bias_voltage_V=np.array([45.0]), amplitude_mV=np.ones((1, 2), dtype=np.float32),
            noise_rms_mV=np.ones((1, 2), dtype=np.float32),
            trigger_index=np.zeros((1, 2), dtype=np.int64),
            led_time_fs=np.array([[30000, 10000]], dtype=np.int64),
            cfd_time_fs=np.array([[30000, 10000]], dtype=np.int64),
            windows_mV=np.zeros((1, 2, 4), dtype=np.float32),
            relative_time_ps=np.arange(4, dtype=np.float64),
            window_anchor_time_fs=None,
        )
        target = factored_correction_target_ps(dataset, np.array([0], dtype=np.int64))
        self.assertAlmostEqual(float(target[0]), 20.0)


class AntisymmetryTests(unittest.TestCase):
    @staticmethod
    def _assert_antisymmetric(model: torch.nn.Module, length: int) -> None:
        torch.manual_seed(5)
        pair = torch.randn(8, 2, length)
        with torch.no_grad():
            forward = model(pair)
            reverse = model(pair[:, [1, 0], :])
        torch.testing.assert_close(reverse, -forward, rtol=1e-5, atol=1e-5)

    def test_linear_svr_exact_antisymmetry(self) -> None:
        model = LinearPairSVR(32)
        with torch.no_grad():
            model.weight.copy_(torch.randn(32))
        self._assert_antisymmetric(model, 32)
        self.assertFalse(any("bias" in key.lower() for key in model.state_dict()))

    def test_constructive_exact_antisymmetry(self) -> None:
        model = AntisymmetricConstructiveMLPEncoder(
            {"activation": "silu", "unit_bias": True, "_trained_units": 3}, 32
        )
        self._assert_antisymmetric(model, 32)

    def test_cnn_exact_antisymmetry(self) -> None:
        model = AntisymmetricCNNRegressor(
            {
                "channels": [4, 8],
                "kernel_sizes": [5, 3],
                "strides": [2, 2],
                "dilations": [1, 1],
                "activation": "silu",
                "normalization": "none",
                "conv_dropout": 0.0,
                "adaptive_pool_length": 4,
                "dense_units": [8],
                "dense_dropout": 0.0,
                "max_abs_single_channel_output_ps": None,
            },
            32,
        )
        model.eval()
        self._assert_antisymmetric(model, 32)
        self.assertFalse(any("pair_output_bias" in key for key in model.state_dict()))


if __name__ == "__main__":
    unittest.main()
