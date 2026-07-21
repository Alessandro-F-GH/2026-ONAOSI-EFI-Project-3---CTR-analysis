from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

CACHE_SCHEMA_VERSION = 6


def load_state(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "stages": {},
            "metadata": {},
        }
    with path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    state.setdefault("cache_schema_version", 1)
    state.setdefault("stages", {})
    state.setdefault("metadata", {})
    return state


def save_state(path: str | Path, state: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def signature(inputs: Any, config: Any) -> str:
    payload = json.dumps(
        {"inputs": inputs, "config": config},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def outputs_exist(outputs: list[str | Path]) -> bool:
    return all(Path(output).exists() for output in outputs)


def stage_valid(
    state: dict[str, Any],
    stage: str,
    expected_signature: str,
    outputs: list[str | Path],
) -> bool:
    return stage_valid_any(state, stage, [expected_signature], outputs)


def stage_valid_any(
    state: dict[str, Any],
    stage: str,
    expected_signatures: list[str],
    outputs: list[str | Path],
) -> bool:
    record = state.get("stages", {}).get(stage)
    return bool(
        record
        and record.get("completed") is True
        and record.get("signature") in expected_signatures
        and outputs_exist(outputs)
    )


def stage_migratable(
    state: dict[str, Any],
    stage: str,
    outputs: list[str | Path],
    migration_mode: bool,
) -> bool:
    record = state.get("stages", {}).get(stage)
    return bool(
        migration_mode
        and record
        and record.get("completed") is True
        and outputs_exist(outputs)
    )


def mark_stage(
    state: dict[str, Any],
    stage: str,
    stage_signature: str,
    outputs: list[str | Path],
    metadata: dict[str, Any] | None = None,
) -> None:
    state.setdefault("stages", {})[stage] = {
        "completed": True,
        "signature": stage_signature,
        "outputs": [str(Path(output)) for output in outputs],
        "metadata": metadata or {},
    }
