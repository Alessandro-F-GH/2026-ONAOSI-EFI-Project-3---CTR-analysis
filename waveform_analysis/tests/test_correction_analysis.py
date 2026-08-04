from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from ml_pipeline.correction_analysis import (
    analyze_right_corrections,
    save_correction_analysis,
)


def test_right_corrections_are_ranked_by_error_reduction() -> None:
    dataset = SimpleNamespace(
        event_id=np.asarray([10, 11, 12, 13]),
        event_index=np.asarray([100, 101, 102, 103]),
        source_file_id=np.asarray([0, 0, 1, 1]),
        source_run_index=np.asarray([0, 1, 0, 1]),
        amplitude_mV=np.asarray(
            [[100.0, 90.0], [100.0, 90.0], [100.0, 90.0], [100.0, 90.0]]
        ),
        noise_rms_mV=np.ones((4, 2), dtype=np.float64),
    )
    raw = np.asarray([100.0, 100.0, 100.0, -100.0])
    predicted = np.asarray([50.0, 2.0, -20.0, -60.0])
    corrected = raw - predicted

    result = analyze_right_corrections(
        dataset,
        np.arange(4, dtype=np.int64),
        raw_ps=raw,
        corrected_ps=corrected,
        predicted_correction_ps=predicted,
        true_tof_ps=0.0,
        top_n=3,
    )

    assert [row["event_id"] for row in result.top_events] == [13, 10, 11]
    assert [row["improvement_ps"] for row in result.top_events] == [60.0, 50.0, 2.0]
    assert result.summary["top_right_correction_ps"] == 60.0
    assert result.summary["right_correction_count"] == 3
    assert result.summary["wrong_correction_count"] == 1


def test_correction_analysis_writes_waveform_artifacts(tmp_path) -> None:
    time = np.linspace(-100.0, 200.0, 7)
    dataset = SimpleNamespace(
        event_id=np.asarray([42]),
        event_index=np.asarray([7]),
        source_file_id=np.asarray([2]),
        source_run_index=np.asarray([3]),
        amplitude_mV=np.asarray([[120.0, 110.0]]),
        noise_rms_mV=np.asarray([[1.0, 1.2]]),
        windows_mV=np.asarray(
            [[[0.0, 1.0, 4.0, 8.0, 5.0, 2.0, 0.0],
              [0.0, 0.5, 3.0, 7.0, 6.0, 2.5, 0.0]]]
        ),
        relative_time_ps=time,
    )
    result = analyze_right_corrections(
        dataset,
        np.asarray([0], dtype=np.int64),
        raw_ps=np.asarray([80.0]),
        corrected_ps=np.asarray([20.0]),
        predicted_correction_ps=np.asarray([60.0]),
        true_tof_ps=0.0,
        top_n=1,
    )
    payload = save_correction_analysis(
        result,
        dataset,
        output_dir=tmp_path / "analysis",
        input_transform="none",
        input_waveform_source="energy",
        prediction_target="timing_led",
        model_name="test_model",
        dpi=72,
    )

    assert (tmp_path / "analysis" / "top_right_corrections.csv").is_file()
    assert (tmp_path / "analysis" / "correction_analysis.json").is_file()
    assert len(payload["waveform_plots"]) == 1
    plots = list((tmp_path / "analysis" / "waveforms").glob("*.png"))
    assert len(plots) == 1
    assert plots[0].stat().st_size > 0
