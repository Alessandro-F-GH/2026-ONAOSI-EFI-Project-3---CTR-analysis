from pathlib import Path

from ml_pipeline.study import (
    SUMMARY_RESULT_FIELDS,
    _best_configuration_row,
    _extract_voltage,
    _read_summary_results,
    _update_summary_results,
)


def test_voltage_extraction_from_optional_filename_convention() -> None:
    reporting = {
        "voltage_from_filename": {
            "enabled": True,
            "pattern": r"^(?P<voltage>\d+(?:\.\d+)?)V",
            "group": "voltage",
        }
    }
    assert _extract_voltage(Path("45V_400mV.root"), reporting) == 45.0
    assert _extract_voltage(Path("52.5V_run.root"), reporting) == 52.5
    assert _extract_voltage(Path("run_45V.root"), reporting) != _extract_voltage(
        Path("run_45V.root"), reporting
    )


def test_best_configuration_is_selected_only_from_cv_summary() -> None:
    rows = [
        {
            "record_type": "summary",
            "split": "validation",
            "statistic": "mean",
            "status": "completed",
            "root_id": "r1",
            "channel_mode": "energy_to_timing",
            "is_selected_hyperparameters": 1,
            "is_selected_window": 1,
            "ctr_ps": 120.0,
            "model_id": "mlp",
        },
        {
            "record_type": "summary",
            "split": "validation",
            "statistic": "mean",
            "status": "completed",
            "root_id": "r1",
            "channel_mode": "energy_to_timing",
            "is_selected_hyperparameters": 1,
            "is_selected_window": 1,
            "ctr_ps": 110.0,
            "model_id": "linear_svr",
        },
        {
            "record_type": "summary",
            "split": "blind",
            "statistic": "mean",
            "status": "completed",
            "root_id": "r1",
            "channel_mode": "energy_to_timing",
            "is_selected_hyperparameters": 1,
            "is_selected_window": 1,
            "ctr_ps": 90.0,
            "model_id": "blind_only_winner",
        },
    ]
    selected = _best_configuration_row(rows, "r1", "energy_to_timing", "ctr_ps")
    assert selected is not None
    assert selected["model_id"] == "linear_svr"


def test_summary_csv_contains_no_fold_level_columns_and_updates_per_file(tmp_path: Path) -> None:
    assert "fold_id" not in SUMMARY_RESULT_FIELDS
    path = tmp_path / "summary_results.csv"
    base = {field: "" for field in SUMMARY_RESULT_FIELDS}
    first = {**base, "file_name": "45V_a.root", "channel_mode": "energy_to_energy", "model_id": "mlp"}
    replacement = {**base, "file_name": "45V_a.root", "channel_mode": "energy_to_energy", "model_id": "linear_svr"}
    second_file = {**base, "file_name": "50V_b.root", "channel_mode": "energy_to_energy", "model_id": "mlp"}

    roots = [str(tmp_path / "45V_a.root"), str(tmp_path / "50V_b.root")]
    _update_summary_results(path, Path("45V_a.root"), [first], roots)
    _update_summary_results(path, Path("50V_b.root"), [second_file], roots)
    _update_summary_results(path, Path("45V_a.root"), [replacement], roots)

    rows = _read_summary_results(path)
    assert len(rows) == 2
    assert rows[0]["file_name"] == "45V_a.root"
    assert rows[0]["model_id"] == "linear_svr"
    assert rows[1]["file_name"] == "50V_b.root"


def _summary_row(
    *,
    root_id: str = "r1",
    mode: str = "energy_to_timing",
    model: str = "linear_svr",
    loss: str = "mse",
    transform: str = "none",
    window: str = "w10",
    start: float = -2.0,
    end: float = 8.0,
    split: str = "validation",
    statistic: str = "mean",
    ctr: float = 120.0,
) -> dict[str, object]:
    return {
        "record_type": "summary",
        "split": split,
        "statistic": statistic,
        "status": "completed",
        "root_id": root_id,
        "channel_mode": mode,
        "model_id": model,
        "model_type": model,
        "loss_id": loss,
        "loss_type": loss,
        "input_transform": transform,
        "window_id": window,
        "window_start_ns": start,
        "window_end_ns": end,
        "trial_id": "trial_0001",
        "is_selected_hyperparameters": 1,
        "is_selected_window": 1,
        "n_events": 100,
        "loss": 1.0,
        "bias_ps": 0.5,
        "ctr_ps": ctr,
        "baseline_ctr_ps": 150.0,
        "relative_improvement_pct": 20.0,
    }


def test_model_loss_compact_selection_uses_cv_and_picks_best_transform() -> None:
    from ml_pipeline.study import _best_model_loss_configuration_row

    rows = [
        _summary_row(transform="none", ctr=125.0),
        _summary_row(transform="normalize", ctr=110.0),
        _summary_row(transform="differentiate", split="blind", ctr=80.0),
    ]
    selected = _best_model_loss_configuration_row(
        rows,
        "r1",
        "energy_to_timing",
        "linear_svr",
        "mse",
        "ctr_ps",
    )
    assert selected is not None
    assert selected["input_transform"] == "normalize"
    assert selected["ctr_ps"] == 110.0


def test_model_loss_table_and_best_ctr_window_plot(tmp_path: Path) -> None:
    from ml_pipeline.study import (
        _generate_model_loss_records,
        _plot_file_best_ctr_vs_window,
        _update_model_loss_results,
    )

    root_file = Path("47V-run.root")
    rows: list[dict[str, object]] = []
    for model, loss, transform, window, start, end, validation_ctr, blind_ctr in (
        ("linear_svr", "mse", "none", "w10", -2.0, 8.0, 120.0, 122.0),
        ("linear_svr", "mse", "normalize", "w10", -2.0, 8.0, 112.0, 114.0),
        ("linear_svr", "mse", "none", "w16", -4.0, 12.0, 108.0, 110.0),
        ("mlp", "mse", "none", "w10", -2.0, 8.0, 118.0, 121.0),
        ("mlp", "mse", "none", "w16", -4.0, 12.0, 111.0, 115.0),
    ):
        validation = _summary_row(
            model=model,
            loss=loss,
            transform=transform,
            window=window,
            start=start,
            end=end,
            ctr=validation_ctr,
        )
        blind = {
            **validation,
            "split": "blind",
            "ctr_ps": blind_ctr,
        }
        sem_validation = {**validation, "statistic": "sem", "ctr_ps": 1.5}
        sem_blind = {**blind, "statistic": "sem", "ctr_ps": 2.0}
        rows.extend((validation, blind, sem_validation, sem_blind))

    config = {
        "selection": {"window_metric": "ctr_ps"},
        "channel_modes": ["energy_to_timing"],
        "models": ["linear_svr", "mlp"],
        "losses": [{"id": "mse", "type": "mse"}],
        "windows_ns": [
            {"id": "w10", "start_ns": -2.0, "end_ns": 8.0},
            {"id": "w16", "start_ns": -4.0, "end_ns": 12.0},
        ],
        "root_files": [str(root_file)],
        "reporting": {
            "dpi": 80,
            "plot_best_ctr_vs_window": True,
            "voltage_from_filename": {
                "enabled": True,
                "pattern": r"^(?P<voltage>\d+(?:\.\d+)?)V",
                "group": "voltage",
            },
        },
    }
    records = _generate_model_loss_records(
        config=config,
        rows=rows,
        root_file=root_file,
        root_id="r1",
    )
    assert len(records) == 2
    svr = next(record for record in records if record["model_id"] == "linear_svr")
    assert svr["window_id"] == "w16"
    assert svr["validation_ctr_mean_ps"] == 108.0
    assert svr["window_size_ns"] == 16.0

    table_path = tmp_path / "model_loss_results.csv"
    _update_model_loss_results(table_path, root_file, records, config["root_files"])
    table_rows = _read_summary_results(table_path)
    assert len(table_rows) == 2
    assert {row["model_id"] for row in table_rows} == {"linear_svr", "mlp"}

    class _Logger:
        def warning(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

    _plot_file_best_ctr_vs_window(
        config=config,
        rows=rows,
        root_file=root_file,
        root_id="r1",
        output=tmp_path,
        logger=_Logger(),
    )
    assert (tmp_path / "summary_plots" / "47v_run_best_ctr_vs_window.png").is_file()


def test_completed_file_marker_requires_every_current_grid_block() -> None:
    from ml_pipeline.study import _root_has_complete_requested_blocks

    config = {
        "channel_modes": ["energy_to_timing"],
        "models": ["linear_svr"],
        "model_spaces": {
            "linear_svr": {"supported_losses": ["mse"]},
        },
        "losses": [{"id": "mse", "type": "mse"}],
        "input_transforms": ["none", "normalize"],
        "windows_ns": [{"id": "w10", "start_ns": -2.0, "end_ns": 8.0}],
    }
    validation_none = _summary_row(transform="none", window="w10", ctr=120.0)
    blind_none = {**validation_none, "split": "blind", "ctr_ps": 122.0}
    rows = [validation_none, blind_none]
    assert not _root_has_complete_requested_blocks(config, rows, "r1")

    validation_normalize = _summary_row(
        transform="normalize", window="w10", ctr=110.0
    )
    blind_normalize = {
        **validation_normalize,
        "split": "blind",
        "ctr_ps": 112.0,
    }
    rows.extend((validation_normalize, blind_normalize))
    assert _root_has_complete_requested_blocks(config, rows, "r1")
