from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ml_pipeline.config import validate_model_config
from ml_pipeline.model import (
    AntisymmetricCorrectionCNN,
    AntisymmetricCorrectionTimeSeriesMLP,
    build_correction_model,
)


class _AntisymmetryMixin:
    def assert_antisymmetric(self, model: torch.nn.Module, length: int) -> None:
        model.eval()
        pair = torch.randn(7, 2, length)
        with torch.no_grad():
            canonical = model(pair)
            swapped = model(pair[:, [1, 0], :])
        self.assertTrue(torch.allclose(canonical, -swapped, atol=1e-7, rtol=1e-7))

    def assert_zero_for_identical(self, model: torch.nn.Module, length: int) -> None:
        model.eval()
        waveform = torch.randn(5, length)
        pair = torch.stack([waveform, waveform], dim=1)
        with torch.no_grad():
            correction = model(pair)
        self.assertTrue(torch.equal(correction, torch.zeros_like(correction)))


class AntisymmetricCorrectionCNNTest(_AntisymmetryMixin, unittest.TestCase):
    def config(self) -> dict:
        return {
            "model_type": "cnn",
            "architecture": {
                "conv_channels": [4, 8],
                "kernel_sizes": [5, 3],
                "pool_sizes": [2, 2],
                "batch_norm": False,
                "activation": "relu",
                "dropout": 0.0,
                "dense_units": [4],
                "max_abs_single_channel_output_ps": None,
            },
        }

    def test_swap_reverses_sign(self) -> None:
        self.assert_antisymmetric(AntisymmetricCorrectionCNN(self.config()), 64)

    def test_identical_channels_have_zero_correction(self) -> None:
        self.assert_zero_for_identical(AntisymmetricCorrectionCNN(self.config()), 64)


class AntisymmetricTimeSeriesMLPTest(_AntisymmetryMixin, unittest.TestCase):
    def config(self) -> dict:
        return {
            "model_type": "time_series_mlp",
            "architecture": {
                "hidden_units": [16, 8],
                "batch_norm": False,
                "activation": "relu",
                "dropout": 0.0,
                "max_abs_single_channel_output_ps": None,
            },
        }

    def test_swap_reverses_sign(self) -> None:
        self.assert_antisymmetric(
            AntisymmetricCorrectionTimeSeriesMLP(self.config(), input_length=40), 40
        )

    def test_identical_channels_have_zero_correction(self) -> None:
        self.assert_zero_for_identical(
            AntisymmetricCorrectionTimeSeriesMLP(self.config(), input_length=40), 40
        )

    def test_factory_builds_time_series_model(self) -> None:
        model = build_correction_model(self.config(), input_length=40)
        pair = torch.randn(3, 2, 40)
        output = model(pair)
        self.assertEqual(tuple(output.shape), (3,))


class ModelConfigValidationTest(unittest.TestCase):
    def test_time_series_config_is_valid(self) -> None:
        config = {
            "model_type": "time_series_mlp",
            "architecture": {
                "hidden_units": [32, 16],
                "batch_norm": True,
                "activation": "relu",
                "dropout": 0.01,
                "max_abs_single_channel_output_ps": 500.0,
            },
            "optimizer": {"name": "adamw", "learning_rate": 3e-4},
            "scheduler": {"name": "reduce_on_plateau"},
            "training": {"epochs": 5, "batch_size": 32},
            "checkpointing": {"every_batches": 0},
        }
        validate_model_config(config)


class StandardDeviationLossTest(unittest.TestCase):
    def test_loss_is_in_ps_and_invariant_to_constant_bias(self) -> None:
        from ml_pipeline.losses import residual_std_loss_ps

        target = torch.tensor([-2.0, -1.0, 1.0, 2.0])
        prediction = torch.tensor([-1.0, -2.0, 2.0, 1.0])
        base = residual_std_loss_ps(prediction, target)
        shifted = residual_std_loss_ps(prediction + 123.0, target)
        self.assertTrue(torch.allclose(base, shifted, atol=1e-6, rtol=0.0))

        residual = (prediction - target).numpy()
        expected = float(residual.std(ddof=0))
        self.assertAlmostEqual(float(base), expected, places=6)


class SymmetricStandardDeviationObjectiveTest(unittest.TestCase):
    def test_paired_std_equals_canonical_rmse(self) -> None:
        from ml_pipeline.losses import residual_std_loss_ps

        residual = torch.tensor([3.0, -1.0, 2.0, 5.0])
        target = torch.zeros_like(residual)
        paired_prediction = torch.cat([residual, -residual])
        paired_target = torch.cat([target, -target])
        loss = residual_std_loss_ps(paired_prediction, paired_target)
        expected = torch.sqrt(torch.mean(residual * residual))
        self.assertTrue(torch.allclose(loss, expected, atol=1e-6, rtol=0.0))


class Catch22RandomForestTest(unittest.TestCase):
    def test_shared_forest_is_exactly_antisymmetric(self) -> None:
        import numpy as np
        from sklearn.ensemble import RandomForestRegressor

        from ml_pipeline.catch22_random_forest import SharedCatch22RandomForest

        rng = np.random.default_rng(123)
        pair = rng.normal(size=(80, 2, 6))
        target = pair[:, 0, 0] - pair[:, 1, 0]
        x = np.concatenate([pair[:, 0, :], pair[:, 1, :]], axis=0)
        y = np.concatenate([0.5 * target, -0.5 * target], axis=0)
        forest = RandomForestRegressor(n_estimators=20, random_state=123, n_jobs=1)
        forest.fit(x, y)
        model = SharedCatch22RandomForest(
            forests=[forest], stage_weights=[1.0]
        )
        canonical = model.predict_pair(pair)
        swapped = model.predict_pair(pair[:, [1, 0], :])
        self.assertTrue(np.array_equal(canonical, -swapped))
        identical = np.stack([pair[:, 0, :], pair[:, 0, :]], axis=1)
        self.assertTrue(np.array_equal(model.predict_pair(identical), np.zeros(80)))

    def test_random_forest_config_is_valid(self) -> None:
        config = {
            "model_type": "catch22_random_forest",
            "features": {
                "implementation": "aeon",
                "catch24": False,
                "chunk_events": 64,
                "n_jobs": 1,
                "parallel_backend": "threading",
            },
            "random_forest": {
                "n_estimators": 10,
                "criterion": "squared_error",
                "bootstrap": True,
                "n_jobs": 1,
            },
            "training": {
                "stages": 2,
                "stage_learning_rate": 0.5,
                "monitor": "validation_loss",
            },
            "checkpointing": {"every_trees": 5},
        }
        validate_model_config(config)


class ChannelSwapAugmentationTest(unittest.TestCase):
    def test_dataset_swaps_pair_and_negates_signed_values(self) -> None:
        from ml_pipeline.torch_data import CorrectionDataset, Normalization

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            windows = np.array(
                [
                    [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]],
                    [[4.0, 5.0, 6.0], [40.0, 50.0, 60.0]],
                ],
                dtype=np.float32,
            )
            led = np.array([[12000, 2000], [3000, 8000]], dtype=np.int64)
            np.save(directory / "windows_mV.npy", windows)
            np.save(directory / "led_time_fs.npy", led)
            cache = SimpleNamespace(directory=directory)
            dataset = CorrectionDataset(
                cache,
                np.array([0, 1], dtype=np.int64),
                Normalization(mean_mV=0.0, std_mV=1.0),
                led_center_ps=2.0,
                duplicate_swapped_channels=True,
            )
            self.assertEqual(len(dataset), 4)
            pair, target, led_delta = dataset[0]
            swapped_pair, swapped_target, swapped_led = dataset[2]
            self.assertTrue(torch.equal(swapped_pair, pair[[1, 0], :]))
            self.assertEqual(float(swapped_target), -float(target))
            self.assertEqual(float(swapped_led), -float(led_delta))

    def test_symmetric_batch_sampler_keeps_pairs_together(self) -> None:
        from ml_pipeline.torch_data import EpochSymmetricBatchSampler

        sampler = EpochSymmetricBatchSampler(base_length=5, batch_size=4, seed=123)
        batches = list(iter(sampler))
        flattened = [value for batch in batches for value in batch]
        self.assertEqual(sorted(flattened), list(range(10)))
        for batch in batches:
            for position in batch:
                if position < 5:
                    self.assertIn(position + 5, batch)

    def test_random_forest_pair_duplication_reverses_signs(self) -> None:
        from ml_pipeline.catch22_random_forest import _duplicate_swapped_pairs

        pair = np.arange(24, dtype=np.float64).reshape(2, 2, 6)
        led = np.array([10.0, -3.0])
        target = np.array([7.0, -6.0])
        augmented_pair, augmented_led, augmented_target = _duplicate_swapped_pairs(
            pair, led, target
        )
        self.assertTrue(np.array_equal(augmented_pair[2:], pair[:, [1, 0], :]))
        self.assertTrue(np.array_equal(augmented_led[2:], -led))
        self.assertTrue(np.array_equal(augmented_target[2:], -target))


if __name__ == "__main__":
    unittest.main()
