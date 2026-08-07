from __future__ import annotations

import json
from pathlib import Path

from ml_pipeline.study import _effective_train_config
from ml_pipeline.study_config import load_model_space


def test_linear_svr_model_space_maps_common_losses(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[1]
    space = load_model_space(project / "config" / "model_spaces" / "linear_svr.json")

    mse = _effective_train_config(
        space, {"id": "mse", "type": "mse"}, "none",
        {"input_waveforms": "energy", "target": "energy_led"},
        "svr_mse", tmp_path / "mse", 7,
    )
    assert mse["model"]["loss"]["type"] == "rmse"

    var_bias = _effective_train_config(
        space,
        {
            "id": "vb",
            "type": "var_bias",
            "bias_weight": 0.01,
            "bias_normalization": "target_std",
            "minimum_scale": 1e-8,
        },
        "differentiate",
        {"input_waveforms": "timing", "target": "timing_led"},
        "svr_vb", tmp_path / "vb", 8,
    )
    assert var_bias["model"]["loss"] == {
        "type": "variance_bias",
        "bias_weight": 0.01,
        "bias_normalization": "target_std",
        "minimum_scale": 1e-8,
    }


def test_linear_svr_model_space_json_is_valid() -> None:
    project = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (project / "config" / "model_spaces" / "linear_svr.json").read_text()
    )
    assert payload["id"] == "linear_svr"
    assert payload["model_type"] == "linear_svr"
    assert payload["search"]["parameters"]["model.C"]["type"] == "loguniform"
