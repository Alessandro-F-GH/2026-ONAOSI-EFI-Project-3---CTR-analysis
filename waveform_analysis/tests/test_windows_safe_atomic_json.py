from __future__ import annotations

import json
from pathlib import Path

from ml_pipeline import common


def test_atomic_json_skips_identical_metadata(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "results_metadata.json"
    value = {"schema_version": 2, "models": ["mlp"]}
    common.atomic_json(path, value)

    def unexpected_replace(_source, _destination):
        raise AssertionError("byte-identical metadata should not be replaced")

    monkeypatch.setattr(common.os, "replace", unexpected_replace)
    common.atomic_json(path, value)
    assert json.loads(path.read_text(encoding="utf-8")) == value


def test_atomic_json_retries_transient_windows_replace_lock(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "results_metadata.json"
    original_replace = common.os.replace
    attempts = 0

    def flaky_replace(source, destination):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("transient Windows lock")
        return original_replace(source, destination)

    monkeypatch.setattr(common.os, "replace", flaky_replace)
    monkeypatch.setattr(common.time, "sleep", lambda _seconds: None)

    common.atomic_json(path, {"value": 7})
    assert attempts == 3
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 7}


def test_atomic_json_falls_back_to_in_place_rewrite(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "results_metadata.json"
    common.atomic_json(path, {"value": 1})

    monkeypatch.setattr(
        common.os,
        "replace",
        lambda _source, _destination: (_ for _ in ()).throw(
            PermissionError("destination denies delete sharing")
        ),
    )
    monkeypatch.setattr(common.time, "sleep", lambda _seconds: None)

    common.atomic_json(path, {"value": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 2}
    assert not list(tmp_path.glob(".*.tmp"))
