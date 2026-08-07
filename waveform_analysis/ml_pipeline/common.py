from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import random
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def canonical_hash(value: Any) -> str:
    payload = json.dumps(json_safe(value), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def source_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def atomic_json(path: Path, value: Any) -> None:
    """Write JSON safely, avoiding redundant Windows file replacements.

    Windows can temporarily deny ``os.replace`` when antivirus, indexing, or a
    viewer has the destination open without delete sharing.  Study metadata is
    requested after every fold even though it usually changes only when a new
    trial/model/codebook is introduced.  Skipping byte-identical writes avoids
    nearly all of those unnecessary rename operations.

    For genuine updates we first retry the atomic replacement.  If Windows
    continues to deny rename/delete sharing, we fall back to truncating and
    rewriting the destination in place, which is allowed by many readers that
    prohibit replacement.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        json_safe(value), indent=2, allow_nan=False
    ).encode("utf-8")

    # Most calls made while a fold is running produce identical metadata.
    # Avoid touching the file at all in that common case.
    try:
        if path.is_file() and path.read_bytes() == payload:
            return
    except OSError:
        # A transient read lock should not prevent us from attempting the write.
        pass

    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

        delay_seconds = 0.05
        last_error: PermissionError | None = None
        for _attempt in range(8):
            try:
                os.replace(temporary, path)
                return
            except PermissionError as exc:
                last_error = exc
                time.sleep(delay_seconds)
                delay_seconds = min(delay_seconds * 2.0, 0.8)

        # Some Windows programs allow writing an open file but deny replacing
        # it.  Preserve progress with a direct rewrite after atomic retries fail.
        try:
            with path.open("wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            return
        except PermissionError as exc:
            raise PermissionError(
                f"Unable to update {path}. Close any program viewing this file "
                "and resume the study; previously persisted fold rows remain safe."
            ) from (last_error or exc)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


class _StudyProgressFilter(logging.Filter):
    """Keep study progress plus warnings/errors, hiding verbose training INFO logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.WARNING or bool(
            getattr(record, "study_progress", False)
        )


def restrict_to_study_progress(logger: logging.Logger) -> logging.Logger:
    """Apply concise study logging without changing standalone training logging."""

    for handler in logger.handlers:
        # Avoid stacking equivalent filters when the runner is resumed/reconfigured.
        if not any(isinstance(item, _StudyProgressFilter) for item in handler.filters):
            handler.addFilter(_StudyProgressFilter())
    return logger


def setup_logging(log_path: Path, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("energy_ml_pipeline")
    logger.handlers.clear()
    logger.propagate = False
    numeric_level = getattr(logging, str(level).upper(), logging.INFO)
    logger.setLevel(numeric_level)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(numeric_level)
    logger.addHandler(console)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(numeric_level)
    logger.addHandler(file_handler)
    return logger


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows([json_safe(row) for row in rows])
