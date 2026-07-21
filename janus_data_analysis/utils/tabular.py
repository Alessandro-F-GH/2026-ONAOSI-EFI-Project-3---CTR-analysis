from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Iterable

from .binary_io import DataError, atomic_write_csv, read_csv

SIDE_TO_CODE = {"a": 0, "b": 1}
CODE_TO_SIDE = {value: key for key, value in SIDE_TO_CODE.items()}

STATUS_TO_CODE = {
    "accepted": 0,
    "no_timing_candidate": 1,
    "no_overlapping_timing_pair": 2,
    "deviation_too_large": 3,
    "missing_energy_pulse": 4,
    "write_failed": 5,
    "other_side_missing_candidate": 6,
    "other_side_deviation_too_large": 7,
    "unknown": 255,
}
CODE_TO_STATUS = {value: key for key, value in STATUS_TO_CODE.items()}


def side_code(value: Any) -> int:
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in SIDE_TO_CODE:
            return SIDE_TO_CODE[stripped]
        try:
            value = int(stripped)
        except ValueError as exc:
            raise DataError(f"Unknown matching side: {value!r}") from exc
    code = int(value)
    if code not in CODE_TO_SIDE:
        raise DataError(f"Unknown matching side code: {code}")
    return code


def side_name(value: Any) -> str:
    return CODE_TO_SIDE[side_code(value)]


def status_code(value: Any) -> int:
    if value is None or value == "":
        return STATUS_TO_CODE["unknown"]
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in STATUS_TO_CODE:
            return STATUS_TO_CODE[stripped]
        try:
            value = int(stripped)
        except ValueError:
            return STATUS_TO_CODE["unknown"]
    return int(value)


def status_name(value: Any) -> str:
    return CODE_TO_STATUS.get(status_code(value), "unknown")


def table_format(cfg: dict[str, Any]) -> str:
    output = cfg.get("analysis_output", {})
    value = str(output.get("large_table_format", "csv")).strip().lower()
    if value not in {"csv", "parquet"}:
        raise DataError("analysis_output.large_table_format must be csv or parquet")
    return value


def table_path(directory: str | Path, stem: str, cfg: dict[str, Any]) -> Path:
    return Path(directory) / f"{stem}.{table_format(cfg)}"


def _normalise_scalar(value: Any) -> Any:
    # Keep numeric values numeric. Only missing/non-finite values become None in
    # parquet; CSV continues to use an empty field for backward compatibility.
    if hasattr(value, "item"):
        try:
            value = value.item()
        except ValueError:
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_table(
    path: str | Path,
    fieldnames: Iterable[str],
    rows: Iterable[dict[str, Any]],
) -> None:
    path = Path(path)
    fields = list(fieldnames)
    materialized = [
        {field: _normalise_scalar(row.get(field, "")) for field in fields}
        for row in rows
    ]
    if path.suffix.lower() == ".csv":
        atomic_write_csv(path, fields, materialized)
        return
    if path.suffix.lower() != ".parquet":
        raise DataError(f"Unsupported table format: {path.suffix}")
    try:
        import pandas as pd
    except ImportError as exc:
        raise DataError(
            "Parquet output requires pandas and pyarrow. Install the optional "
            "tabular dependencies from requirements.txt."
        ) from exc
    try:
        frame = pd.DataFrame.from_records(materialized, columns=fields)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        frame.to_parquet(temporary, index=False, engine="pyarrow", compression="zstd")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(path)
    except ImportError as exc:
        raise DataError(
            "Parquet output requires pyarrow. Install the optional tabular "
            "dependencies from requirements.txt."
        ) from exc


def read_table(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return read_csv(path)
    if path.suffix.lower() != ".parquet":
        raise DataError(f"Unsupported table format: {path.suffix}")
    try:
        import pandas as pd
    except ImportError as exc:
        raise DataError("Reading parquet requires pandas and pyarrow") from exc
    frame = pd.read_parquet(path, engine="pyarrow")
    rows: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        rows.append(
            {
                key: ("" if value is None or (isinstance(value, float) and math.isnan(value)) else value)
                for key, value in record.items()
            }
        )
    return rows
