from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

from .gaussian import FitResult

FIT_FIELDS = [
    "method", "parameter", "success", "n_total", "n_selected", "n_valid", "n_fit",
    "crossing_efficiency", "mean_ps", "mean_error_ps", "sigma_ps", "sigma_error_ps",
    "ctr_ps", "ctr_error_ps", "chi2", "ndof", "fit_low_ps", "fit_high_ps",
    "iterations", "message", "bin_width_ps", "bin_phase_ps", "phase_ctr_std_ps",
    "edges_ps", "counts", "expected",
]


def write_fit_csv(path: str | Path, fit: FitResult, diagnostic_mode: str = "compact") -> None:
    del diagnostic_mode
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    row = fit.as_dict()
    row.update({
        "success": int(fit.success),
        "edges_ps": json.dumps(fit.edges_ps.tolist(), separators=(",", ":")),
        "counts": json.dumps(fit.counts.tolist(), separators=(",", ":")),
        "expected": json.dumps(fit.expected.tolist(), separators=(",", ":")),
    })
    row.pop("n_rejected", None)
    row.pop("chi2_ndof", None)
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIT_FIELDS)
        writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in FIT_FIELDS})
    os.replace(temporary, output)


def load_fit_csv(path: str | Path) -> FitResult:
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise RuntimeError(f"Expected one fit row in {path}")
    row = rows[0]

    def f(name: str) -> float:
        value = row.get(name, "")
        return float(value) if value not in ("", None) else float("nan")

    return FitResult(
        method=str(row["method"]),
        parameter=f("parameter"),
        success=bool(int(row["success"])),
        n_total=int(row["n_total"]),
        n_selected=int(row["n_selected"]),
        n_valid=int(row["n_valid"]),
        n_fit=int(row["n_fit"]),
        crossing_efficiency=f("crossing_efficiency"),
        mean_ps=f("mean_ps"),
        mean_error_ps=f("mean_error_ps"),
        sigma_ps=f("sigma_ps"),
        sigma_error_ps=f("sigma_error_ps"),
        ctr_ps=f("ctr_ps"),
        ctr_error_ps=f("ctr_error_ps"),
        chi2=f("chi2"),
        ndof=int(row["ndof"]),
        fit_low_ps=f("fit_low_ps"),
        fit_high_ps=f("fit_high_ps"),
        iterations=int(row["iterations"]),
        message=str(row.get("message", "")),
        bin_width_ps=f("bin_width_ps"),
        bin_phase_ps=f("bin_phase_ps"),
        phase_ctr_std_ps=f("phase_ctr_std_ps"),
        edges_ps=np.asarray(json.loads(row["edges_ps"]), dtype=np.float64),
        counts=np.asarray(json.loads(row["counts"]), dtype=np.int64),
        expected=np.asarray(json.loads(row["expected"]), dtype=np.float64),
    )
