from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ml_pipeline.models.cnn_regressor import AntisymmetricCNNRegressor
from ml_pipeline.models.constructive_mlp_encoder import AntisymmetricConstructiveMLPEncoder
from ml_pipeline.models.linear_svr import LinearPairSVR
from ml_pipeline.dataset import PreparedDataset, load_prepared_dataset
from ml_pipeline.prepared_data import (
    input_channel_variant_dataset_view,
    materialize_selected_dataset,
    raw_dataset_view,
)
from ml_pipeline.study import (
    _fit_early_split,
    _kfold,
    _ml_input_variant_for_mode,
    _random_dev_blind,
    _voltage_from_name,
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
                "adaptive_binning": {
                    "enabled": True,
                    "bins_per_fwhm": 10.0,
                    "min_bin_ps": 1.0,
                    "max_bin_ps": 25.0,
                    "phase_count": 8,
                },
            },
        )
        self.assertTrue(fit.success)
        self.assertEqual(fit.n_fit, len(values_fs))
        self.assertEqual(int(np.sum(fit.counts)), len(values_fs))
        self.assertGreater(fit.ctr_ps, 0.0)


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
    def test_only_requested_channel_family_is_denoised(self) -> None:
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
            self.assertTrue((out / "denoised_windows_mV.npy").is_file())
            self.assertTrue((out / "denoised_timing_aligned_energy_windows_mV.npy").is_file())
            self.assertFalse((out / "denoised_timing_windows_mV.npy").exists())


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
