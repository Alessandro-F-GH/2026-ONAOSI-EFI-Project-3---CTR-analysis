from __future__ import annotations

import logging
from pathlib import Path

from ml_pipeline.study import _plot_results


def _summary_row(split: str, statistic: str, ctr_ps: float, improvement: float) -> dict[str, object]:
    return {
        "record_type": "summary",
        "root_id": "run_a",
        "channel_mode": "energy_to_timing",
        "model_id": "linear_svr",
        "loss_id": "mse",
        "input_transform": "none",
        "window_start_ns": -4.0,
        "window_end_ns": 12.0,
        "split": split,
        "statistic": statistic,
        "is_selected_hyperparameters": 1,
        "is_selected_window": 1,
        "ctr_ps": ctr_ps,
        "relative_improvement_pct": improvement,
    }


def test_plot_results_accepts_internal_rows_after_compact_csv_refactor(tmp_path: Path) -> None:
    rows = [
        _summary_row("validation", "mean", 130.0, 4.0),
        _summary_row("validation", "sem", 1.5, 0.3),
        _summary_row("blind", "mean", 132.0, 3.0),
        _summary_row("blind", "sem", 1.8, 0.4),
    ]

    _plot_results(rows, tmp_path, logging.getLogger("test-study-plot"))

    mode_dir = tmp_path / "plots" / "energy_to_timing"
    assert (mode_dir / "run_a_linear_svr_mse_none_windows.png").is_file()
    assert (mode_dir / "run_a_linear_svr_mse_none_cv_vs_blind.png").is_file()
    assert (mode_dir / "run_a_mse_none_best_models.png").is_file()
