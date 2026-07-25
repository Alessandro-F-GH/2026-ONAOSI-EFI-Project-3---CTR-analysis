from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from utils.ml_dataset import finalize_and_write_dataset, regularized_polynomial_fit


class PolynomialFitTests(unittest.TestCase):
    def test_recovers_known_polynomial_without_regularization(self) -> None:
        relative_ns = np.linspace(-2.5, 2.5, 501)
        expected = np.asarray([7.0, 10.0, -2.0, 1.0, 0.5])
        normalized = relative_ns / 2.5
        signal = sum(expected[index] * normalized**index for index in range(expected.size))

        coefficients, rmse, r2 = regularized_polynomial_fit(
            relative_ns,
            signal,
            half_width_ns=2.5,
            degree=4,
            l2_regularization=0.0,
            penalize_intercept=False,
        )

        np.testing.assert_allclose(coefficients, expected, rtol=0.0, atol=1.0e-10)
        self.assertLess(rmse, 1.0e-10)
        self.assertAlmostEqual(r2, 1.0, places=12)

    def test_regularization_reduces_nonconstant_coefficient_norm(self) -> None:
        rng = np.random.default_rng(1234)
        relative_ns = np.linspace(-2.5, 2.5, 101)
        normalized = relative_ns / 2.5
        signal = 7.0 + 15.0 * normalized - 5.0 * normalized**2
        signal += rng.normal(0.0, 0.5, size=signal.size)

        unregularized, _, _ = regularized_polynomial_fit(
            relative_ns,
            signal,
            half_width_ns=2.5,
            degree=8,
            l2_regularization=0.0,
            penalize_intercept=False,
        )
        regularized, _, _ = regularized_polynomial_fit(
            relative_ns,
            signal,
            half_width_ns=2.5,
            degree=8,
            l2_regularization=1.0e-2,
            penalize_intercept=False,
        )

        self.assertLess(
            float(np.linalg.norm(regularized[1:])),
            float(np.linalg.norm(unregularized[1:])),
        )


class DatasetWritingTests(unittest.TestCase):
    def test_target_is_centered_and_internal_tof_is_not_written(self) -> None:
        project = Path(__file__).resolve().parents[1]
        config = json.loads((project / "config" / "analysis.json").read_text())
        rows = [
            {
                "meta_event_index": 0,
                "meta_event_id": 10,
                "meta_source_file_id": "1",
                "timing_ch3_poly_c0": 7.0,
                "_led_tof_ps": 10.0,
            },
            {
                "meta_event_index": 1,
                "meta_event_id": 11,
                "meta_source_file_id": "1",
                "timing_ch3_poly_c0": 7.1,
                "_led_tof_ps": 14.0,
            },
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "dataset.csv"
            summary = finalize_and_write_dataset(rows, output, config)
            with output.open(newline="", encoding="utf-8") as stream:
                written = list(csv.DictReader(stream))

        self.assertEqual(summary["target_center_led_tof_ps"], 12.0)
        self.assertNotIn("_led_tof_ps", written[0])
        targets = [float(item["target_led_residual_ps"]) for item in written]
        self.assertEqual(targets, [-2.0, 2.0])


if __name__ == "__main__":
    unittest.main()


def test_filter_rows_by_led_mad_rejects_largest_outlier():
    from utils.ml_dataset import filter_rows_by_led_mad

    values = [0.0, 1.0, -1.0, 0.5, -0.5, 100.0]
    rows = [
        {
            "meta_event_index": index,
            "meta_event_id": 1000 + index,
            "_led_tof_ps": value,
        }
        for index, value in enumerate(values)
    ]
    filtered, summary, worst = filter_rows_by_led_mad(rows, threshold=5.0)
    assert worst == 5
    assert summary["events_rejected"] == 1
    assert [row["meta_event_index"] for row in filtered] == [0, 1, 2, 3, 4]
