from __future__ import annotations

import logging
from pathlib import Path

from ml_pipeline.study import (
    _keep_study_checkpoints,
    _prune_configuration_windows,
    _prune_file_runs_to_summary_winners,
    _prune_window_trials,
    _trial_run_directory,
)


def _touch_checkpoint(trial_dir: Path) -> None:
    checkpoint = trial_dir / "fold_0" / "checkpoints" / "best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"checkpoint")


def test_results_only_is_default_and_can_be_overridden() -> None:
    assert _keep_study_checkpoints({}) is False
    assert _keep_study_checkpoints({"storage": {"keep_checkpoints": False}}) is False
    assert _keep_study_checkpoints({"storage": {"keep_checkpoints": True}}) is True


def test_pruning_keeps_only_selected_trial_and_window(tmp_path: Path) -> None:
    logger = logging.getLogger("checkpoint-pruning-test")
    run_root = tmp_path / "runs"
    common = dict(
        run_root=run_root,
        root_id="root",
        mode_id="timing_to_timing",
        model_id="linear_svr",
        loss_id="mse",
        transform="none",
    )
    for window_id in ("w1", "w2"):
        for trial_id in ("t1", "t2"):
            _touch_checkpoint(
                _trial_run_directory(
                    **common, window_id=window_id, trial_id=trial_id
                )
            )

    _prune_window_trials(
        **common,
        window_id="w1",
        keep_trial_id="t2",
        logger=logger,
    )
    assert not _trial_run_directory(**common, window_id="w1", trial_id="t1").exists()
    assert _trial_run_directory(**common, window_id="w1", trial_id="t2").exists()

    _prune_configuration_windows(
        **common,
        keep_window_id="w1",
        logger=logger,
    )
    assert _trial_run_directory(**common, window_id="w1", trial_id="t2").exists()
    assert not (run_root / "root" / "timing_to_timing" / "linear_svr" / "mse" / "none" / "w2").exists()


def test_file_pruning_keeps_one_overall_winner_per_mode(tmp_path: Path) -> None:
    logger = logging.getLogger("file-winner-pruning-test")
    run_root = tmp_path / "runs"
    root_id = "root"

    winner = _trial_run_directory(
        run_root,
        root_id,
        "timing_to_timing",
        "linear_svr",
        "mse",
        "none",
        "w1",
        "t1",
    )
    loser = _trial_run_directory(
        run_root,
        root_id,
        "timing_to_timing",
        "mlp",
        "mse",
        "normalize",
        "w2",
        "t2",
    )
    _touch_checkpoint(winner)
    _touch_checkpoint(loser)

    rows = [
        {
            "record_type": "summary",
            "split": "validation",
            "statistic": "mean",
            "status": "completed",
            "root_id": root_id,
            "channel_mode": "timing_to_timing",
            "model_id": "linear_svr",
            "loss_id": "mse",
            "input_transform": "none",
            "window_id": "w1",
            "trial_id": "t1",
            "is_selected_hyperparameters": 1,
            "is_selected_window": 1,
            "ctr_ps": 55.0,
        },
        {
            "record_type": "summary",
            "split": "validation",
            "statistic": "mean",
            "status": "completed",
            "root_id": root_id,
            "channel_mode": "timing_to_timing",
            "model_id": "mlp",
            "loss_id": "mse",
            "input_transform": "normalize",
            "window_id": "w2",
            "trial_id": "t2",
            "is_selected_hyperparameters": 1,
            "is_selected_window": 1,
            "ctr_ps": 60.0,
        },
    ]
    config = {
        "channel_modes": ["timing_to_timing"],
        "selection": {"window_metric": "ctr_ps"},
    }

    _prune_file_runs_to_summary_winners(
        config=config,
        rows=rows,
        root_id=root_id,
        run_root=run_root,
        logger=logger,
    )

    assert winner.exists()
    assert not loser.exists()
