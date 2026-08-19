from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path
from typing import Any

from .common import canonical_hash
from .study_config import CHANNEL_MODES, MLConfigError, load_study_config


_MODULAR_KEYS = {"extends", "data_config", "preprocessing_config"}


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except FileNotFoundError as exc:
        raise MLConfigError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise MLConfigError(f"Invalid JSON in {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MLConfigError(f"{label} {path} must contain a JSON object")
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge dictionaries; lists/scalars are replaced, never appended."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _module_path(owner: Path, value: str | Path, *, key: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = owner.parent / path
    path = path.resolve()
    if not path.is_file():
        raise MLConfigError(f"{key} referenced by {owner} does not exist: {path}")
    return path


def _include_list(value: Any, *, owner: Path) -> list[str | Path]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [value]
    if isinstance(value, list) and all(isinstance(item, (str, Path)) for item in value):
        return list(value)
    raise MLConfigError(f"extends in {owner} must be a path or list of paths")


def _section_from_module(
    module_path: Path,
    *,
    section: str,
    key: str,
) -> dict[str, Any]:
    value = _read_json_object(module_path, label=key)
    if section in value:
        section_value = value[section]
        if not isinstance(section_value, dict):
            raise MLConfigError(
                f"{key} {module_path}: top-level {section!r} must be an object"
            )
        return copy.deepcopy(section_value)
    # Compact module form: the whole file is the section itself.
    if any(name in value for name in _MODULAR_KEYS):
        raise MLConfigError(
            f"{key} {module_path} looks like a full experiment config. "
            f"Use a raw {section} object or wrap it as {{\"{section}\": {{...}}}}."
        )
    return copy.deepcopy(value)


def _source_record(path: Path, role: str, value: dict[str, Any]) -> dict[str, str]:
    return {
        "role": role,
        "path": str(path),
        "content_hash": canonical_hash(value),
    }


def _resolve_modular_file(
    source: Path,
    *,
    stack: tuple[Path, ...],
    role: str,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    source = source.resolve()
    if source in stack:
        cycle = " -> ".join(str(path) for path in (*stack, source))
        raise MLConfigError(f"Configuration include cycle detected: {cycle}")

    raw = _read_json_object(source, label="configuration")
    next_stack = (*stack, source)
    resolved: dict[str, Any] = {}
    sources: list[dict[str, str]] = []

    # Generic shared defaults. Earlier entries are lower precedence; later
    # entries and finally the current file override them.
    for reference in _include_list(raw.get("extends"), owner=source):
        base_path = _module_path(source, reference, key="extends")
        base, base_sources = _resolve_modular_file(
            base_path, stack=next_stack, role="extends"
        )
        resolved = _deep_merge(resolved, base)
        sources.extend(base_sources)

    # Dedicated data/preprocessing modules are resolved relative to the file
    # that references them. Local `data` / `preprocessing` blocks below remain
    # valid and act as deep overrides.
    for key, section in (
        ("data_config", "data"),
        ("preprocessing_config", "preprocessing"),
    ):
        reference = raw.get(key)
        if reference is None:
            continue
        if not isinstance(reference, (str, Path)):
            raise MLConfigError(f"{key} in {source} must be a JSON file path")
        module_path = _module_path(source, reference, key=key)
        module_raw = _read_json_object(module_path, label=key)
        section_value = _section_from_module(module_path, section=section, key=key)
        resolved = _deep_merge(resolved, {section: section_value})
        sources.append(_source_record(module_path, key, module_raw))

    local = {
        key: copy.deepcopy(value)
        for key, value in raw.items()
        if key not in _MODULAR_KEYS
    }
    resolved = _deep_merge(resolved, local)
    sources.append(_source_record(source, role, raw))

    # Deduplicate provenance records while preserving first occurrence/order.
    unique: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for record in sources:
        token = (record["role"], record["path"])
        if token not in seen:
            seen.add(token)
            unique.append(record)
    return resolved, unique


def resolve_modular_config(path: str | Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Resolve `extends`, `data_config` and `preprocessing_config`.

    Merge rules:
      * dictionaries are deep-merged;
      * arrays/scalars replace inherited values;
      * local experiment values always win;
      * include paths are relative to the JSON file that contains the reference.

    The returned dictionary contains no modular loader keys, so it can be passed
    directly to the canonical study validator.
    """
    source = Path(path).expanduser().resolve()
    return _resolve_modular_file(source, stack=(), role="experiment")


def _normalise_modes(raw: dict[str, Any]) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Resolve dictionary mode syntax while retaining legacy channel_modes lists."""
    mode_block = raw.get("modes")
    settings: dict[str, dict[str, Any]] = {}

    if mode_block is None:
        legacy = raw.get("channel_modes", list(CHANNEL_MODES))
        if isinstance(legacy, dict):
            mode_block = legacy
        else:
            enabled = [str(value) for value in legacy]
            unknown = sorted(set(enabled) - set(CHANNEL_MODES))
            if unknown:
                raise MLConfigError(
                    f"Unsupported channel modes: {unknown}; available: {sorted(CHANNEL_MODES)}"
                )
            for name in CHANNEL_MODES:
                settings[name] = {
                    "enabled": name in enabled,
                    "cfd": name in enabled,
                }
            return enabled, settings

    if not isinstance(mode_block, dict):
        raise MLConfigError("modes must be an object keyed by channel mode")

    unknown = sorted(set(map(str, mode_block)) - set(CHANNEL_MODES))
    if unknown:
        raise MLConfigError(
            f"Unsupported modes: {unknown}; available: {sorted(CHANNEL_MODES)}"
        )

    enabled: list[str] = []
    for name in CHANNEL_MODES:
        value = mode_block.get(name, {"enabled": False})
        if isinstance(value, bool):
            cfg = {"enabled": bool(value)}
        elif isinstance(value, dict):
            cfg = copy.deepcopy(value)
        else:
            raise MLConfigError(f"modes.{name} must be a boolean or object")
        cfg.setdefault("enabled", False)
        cfg.setdefault("cfd", bool(cfg["enabled"]))
        cfg["enabled"] = bool(cfg["enabled"])
        cfg["cfd"] = bool(cfg["cfd"])
        settings[name] = cfg
        if cfg["enabled"]:
            enabled.append(name)

    if not enabled:
        raise MLConfigError("At least one modes.<name>.enabled must be true")
    return enabled, settings


def _hash_payload(cfg: dict[str, Any], *, include_reporting: bool) -> dict[str, Any]:
    payload = copy.deepcopy(cfg)
    for key in list(payload):
        if str(key).startswith("_"):
            payload.pop(key, None)
    payload.pop("logging", None)

    experiment = payload.get("experiment")
    if isinstance(experiment, dict):
        experiment.pop("output_dir", None)
        experiment.pop("name", None)

    mode_settings = payload.pop("modes", {})
    payload["channel_modes"] = list(cfg["channel_modes"])
    if include_reporting:
        payload["mode_reporting"] = {
            name: {"cfd": bool(values.get("cfd", True))}
            for name, values in mode_settings.items()
            if bool(values.get("enabled", False))
        }
    else:
        payload.pop("reporting", None)
    return payload


def load_experiment_config(path: str | Path, project_root: str | Path) -> dict[str, Any]:
    """Load, resolve and validate a modular experiment configuration.

    Modular references are fully expanded *before* canonical validation and
    fingerprinting. Consequently changing photopeak/RMSE settings in a shared
    preprocessing JSON changes the scientific/core fingerprint even though the
    small experiment JSON itself is unchanged.
    """
    source = Path(path).expanduser().resolve()
    raw, sources = resolve_modular_config(source)

    enabled_modes, settings = _normalise_modes(raw)
    normalised = copy.deepcopy(raw)
    normalised.pop("modes", None)
    normalised["channel_modes"] = enabled_modes

    # Reuse the canonical validator/default resolver instead of duplicating its
    # semantics. Paths inside the resolved config remain project-root-relative,
    # exactly as in the non-modular format.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", encoding="utf-8", delete=False
    ) as handle:
        json.dump(normalised, handle)
        temp_path = Path(handle.name)
    try:
        cfg = load_study_config(temp_path, project_root)
    finally:
        temp_path.unlink(missing_ok=True)

    cfg["_config_path"] = str(source)
    cfg["_config_sources"] = sources
    cfg["modes"] = settings
    cfg["channel_modes"] = enabled_modes

    # Provenance is path-independent for reuse: hashes are based on the fully
    # resolved effective configuration, not on where the modules are stored.
    cfg["_core_hash"] = canonical_hash(_hash_payload(cfg, include_reporting=False))
    cfg["_artifact_hash"] = canonical_hash(_hash_payload(cfg, include_reporting=True))
    cfg["_config_hash"] = cfg["_artifact_hash"]
    return cfg


def public_resolved_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the serialisable effective configuration without runtime internals."""
    return {
        key: copy.deepcopy(value)
        for key, value in config.items()
        if not str(key).startswith("_")
    }


def mode_settings(config: dict[str, Any], mode: str) -> dict[str, Any]:
    block = config.get("modes", {})
    if isinstance(block, dict) and isinstance(block.get(mode), dict):
        return block[mode]
    return {"enabled": mode in config.get("channel_modes", []), "cfd": True}


def cfd_enabled(config: dict[str, Any], mode: str) -> bool:
    return bool(mode_settings(config, mode).get("cfd", True))
