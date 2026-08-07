from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from .common import canonical_hash, read_json
from .input_transform import normalize_input_transform, normalize_subsampling_factor


CHANNEL_MODES: dict[str, dict[str, str]] = {
    "energy_to_energy": {
        "input_waveforms": "energy",
        "target": "energy_led",
        "baseline": "energy_led",
    },
    "energy_to_timing": {
        "input_waveforms": "energy",
        "target": "timing_led",
        "baseline": "timing_led",
    },
    "timing_to_timing": {
        "input_waveforms": "timing",
        "target": "timing_led",
        "baseline": "timing_led",
    },
    "energy_timing_to_timing": {
        "input_waveforms": "energy_timing",
        "target": "timing_led",
        "baseline": "timing_led",
    },
}


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def discover_root_files(config: dict[str, Any]) -> list[Path]:
    data = config["data"]
    folder = Path(data["root_folder"])
    pattern = str(data.get("root_glob", "*.root"))
    recursive = bool(data.get("recursive", False))
    files = sorted(folder.rglob(pattern) if recursive else folder.glob(pattern))
    files = [path.resolve() for path in files if path.is_file()]
    if not files:
        raise FileNotFoundError(
            f"No ROOT files matched {pattern!r} under {folder} (recursive={recursive})"
        )
    return files


def load_model_space(path: Path) -> dict[str, Any]:
    space = read_json(path)
    for key in ("id", "model_type", "base_train_config", "search"):
        if key not in space:
            raise ValueError(f"Model space {path} is missing {key!r}")
    if not isinstance(space["base_train_config"], dict):
        raise ValueError(f"Model space {path}: base_train_config must be an object")
    supported = space.get("supported_losses", [])
    if not isinstance(supported, list) or not supported:
        raise ValueError(f"Model space {path}: supported_losses must be a non-empty list")
    mapping = space.get("study_loss_mapping", {})
    if not isinstance(mapping, dict):
        raise ValueError(f"Model space {path}: study_loss_mapping must be an object")
    for study_loss, model_loss in mapping.items():
        if str(study_loss) not in {str(value) for value in supported}:
            raise ValueError(
                f"Model space {path}: study_loss_mapping contains unsupported loss {study_loss!r}"
            )
        if not isinstance(model_loss, dict) or not str(model_loss.get("type", "")).strip():
            raise ValueError(
                f"Model space {path}: study_loss_mapping[{study_loss!r}] must define a model loss type"
            )
    space = copy.deepcopy(space)
    space["_path"] = str(path.resolve())
    space["_hash"] = canonical_hash(space)
    return space


def load_study_config(path: str | Path, project_root: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    root = Path(project_root).resolve()
    config = read_json(source)
    for section in (
        "experiment",
        "data",
        "preprocessing",
        "windows_ns",
        "channel_modes",
        "input_transforms",
        "losses",
        "models",
        "cross_validation",
        "selection",
    ):
        if section not in config:
            raise ValueError(f"Study config requires {section!r}")

    result = copy.deepcopy(config)
    # Migration: CV studies now create a direct development/blind split.  The
    # former preliminary validation fraction is intentionally ignored.
    result.setdefault("split", {}).pop("initial_validation_fraction", None)
    result["data"]["root_folder"] = str(
        _resolve(root, result["data"]["root_folder"])
    )
    output = result["experiment"].get(
        "output_dir", root / "results" / "studies" / source.stem
    )
    result["experiment"]["output_dir"] = str(_resolve(root, output))
    model_dir = result.get("model_spaces_dir", "config/model_spaces")
    model_dir = _resolve(root, model_dir)
    result["model_spaces_dir"] = str(model_dir)

    windows: list[dict[str, float | str]] = []
    for index, raw in enumerate(result["windows_ns"], start=1):
        if not isinstance(raw, dict):
            raise ValueError("Each windows_ns entry must be an object")
        start_ns = float(raw["start_ns"])
        end_ns = float(raw["end_ns"])
        if not start_ns < end_ns or start_ns > 0.0 or end_ns < 0.0:
            raise ValueError(
                "Each window must satisfy start_ns <= 0 <= end_ns and start_ns < end_ns"
            )
        windows.append(
            {
                "id": str(raw.get("id", f"w{index:02d}")),
                "start_ns": start_ns,
                "end_ns": end_ns,
                "before_ns": -start_ns,
                "after_ns": end_ns,
            }
        )
    result["windows_ns"] = windows

    modes = [str(value) for value in result["channel_modes"]]
    unknown_modes = sorted(set(modes) - set(CHANNEL_MODES))
    if unknown_modes:
        raise ValueError(f"Unsupported channel modes: {unknown_modes}")
    result["channel_modes"] = modes
    result["input_transforms"] = [
        normalize_input_transform(value) for value in result["input_transforms"]
    ]

    preprocessing = result.setdefault("preprocessing", {})
    raw_factors = preprocessing.get(
        "subsampling_factors",
        [preprocessing.pop("subsampling_factor", 1)],
    )
    if not isinstance(raw_factors, list) or not raw_factors:
        raise ValueError("preprocessing.subsampling_factors must be a non-empty list")
    factors: list[int] = []
    for value in raw_factors:
        factor = normalize_subsampling_factor(value)
        if factor not in factors:
            factors.append(factor)
    preprocessing["subsampling_factors"] = factors

    losses = result["losses"]
    if not isinstance(losses, list) or not losses:
        raise ValueError("losses must be a non-empty list")
    for loss in losses:
        if not isinstance(loss, dict) or not str(loss.get("id", "")).strip():
            raise ValueError("Each loss requires an id")
        loss_type = str(loss.get("type", ""))
        if loss_type not in {"mse", "var_bias"}:
            raise ValueError("Study losses must be mse or var_bias")
        if loss_type == "var_bias" and float(loss.get("bias_weight", 0.0)) < 0.0:
            raise ValueError("var_bias.bias_weight must be non-negative")

    model_ids = [str(value) for value in result["models"]]
    spaces = {}
    for model_id in model_ids:
        space_path = model_dir / f"{model_id}.json"
        if not space_path.is_file():
            raise FileNotFoundError(f"Model-space config not found: {space_path}")
        space = load_model_space(space_path)
        if str(space["id"]) != model_id:
            raise ValueError(
                f"Model-space id mismatch: requested {model_id!r}, file contains {space['id']!r}"
            )
        spaces[model_id] = space
    result["model_spaces"] = spaces

    strict_losses = bool(result.get("strict_loss_compatibility", True))
    if strict_losses:
        requested = {str(loss["type"]) for loss in losses}
        for model_id, space in spaces.items():
            missing = requested - {str(value) for value in space["supported_losses"]}
            if missing:
                raise ValueError(
                    f"Model {model_id!r} does not support the requested common losses {sorted(missing)}"
                )

    selection = result["selection"]
    metric_aliases = {
        "ctr": "ctr_ps",
        "ctr_ps": "ctr_ps",
        "validation_ctr": "ctr_ps",
        "validation_ctr_ps": "ctr_ps",
        "rmse": "rmse_ps",
        "rmse_ps": "rmse_ps",
        "validation_rmse": "rmse_ps",
        "validation_rmse_ps": "rmse_ps",
        "loss": "loss",
        "objective": "loss",
        "validation_loss": "loss",
        "bias": "bias_ps",
        "bias_ps": "bias_ps",
        "validation_bias": "bias_ps",
        "validation_bias_ps": "bias_ps",
    }
    for key, default in (("hyperparameter_metric", "ctr_ps"), ("window_metric", "ctr_ps")):
        raw_metric = str(selection.get(key, default)).strip().lower()
        if raw_metric not in metric_aliases:
            raise ValueError(
                f"selection.{key}={raw_metric!r} is unsupported; "
                f"choose one of {sorted(metric_aliases)}"
            )
        selection[key] = metric_aliases[raw_metric]
    if str(selection.get("method", "median_mad_z")) != "median_mad_z":
        raise ValueError("The study runner supports only median_mad_z outlier rejection")
    z_threshold = float(selection.get("z_threshold", 4.0))
    if not z_threshold > 0.0:
        raise ValueError("selection.z_threshold must be positive")



    raw_standard_methods = result.get("standard_methods", ["led", "cfd"])
    if isinstance(raw_standard_methods, dict):
        if not bool(raw_standard_methods.get("enabled", True)):
            standard_methods = []
        else:
            standard_methods = raw_standard_methods.get("methods", ["led", "cfd"])
    else:
        standard_methods = raw_standard_methods
    if standard_methods is None:
        standard_methods = ["led", "cfd"]
    if not isinstance(standard_methods, list):
        raise ValueError("standard_methods must be a list or an object with a methods list")
    normalized_standard_methods: list[str] = []
    aliases = {"led": "led", "leading_edge": "led", "leading-edge": "led",
               "cfd": "cfd", "constant_fraction": "cfd", "constant-fraction": "cfd"}
    for value in standard_methods:
        key = str(value).strip().lower()
        if key not in aliases:
            raise ValueError("Study standard_methods supports only LED and CFD")
        method = aliases[key]
        if method not in normalized_standard_methods:
            normalized_standard_methods.append(method)
    result["standard_methods"] = normalized_standard_methods

    reporting = result.setdefault("reporting", {})
    reporting.setdefault("plot_best_gaussian_fits", True)
    reporting.setdefault("dpi", 180)
    voltage = reporting.setdefault("voltage_from_filename", {})
    voltage.setdefault("enabled", False)
    voltage.setdefault("pattern", r"^(?P<voltage>\d+(?:\.\d+)?)V")
    voltage.setdefault("group", "voltage")
    voltage.setdefault("plot_ctr_vs_voltage", True)
    if bool(voltage.get("enabled", False)):
        import re
        pattern = str(voltage.get("pattern", ""))
        try:
            compiled = re.compile(pattern, flags=re.IGNORECASE)
        except re.error as exc:
            raise ValueError(f"Invalid reporting.voltage_from_filename.pattern: {exc}") from exc
        group = voltage.get("group", "voltage")
        if isinstance(group, str) and group not in compiled.groupindex:
            raise ValueError(
                "reporting.voltage_from_filename.group must name a capture group in the pattern"
            )

    result["_config_path"] = str(source)
    result["_project_root"] = str(root)
    result["_config_hash"] = canonical_hash(config)
    result["root_files"] = [str(path) for path in discover_root_files(result)]
    return result
