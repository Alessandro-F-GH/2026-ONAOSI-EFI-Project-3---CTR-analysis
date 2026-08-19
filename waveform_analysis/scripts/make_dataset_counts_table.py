from __future__ import annotations

"""Write the LaTeX dataset-count table used by the CTR presentation.

Intended location in the repository:
    waveform_analysis/scripts/make_dataset_counts_table.py

Default output:
    waveform_analysis/results/presentation/tables/dataset_counts_table.tex

The table reports, for each voltage/file:
  * events before the photopeak cut, as recorded by the permanent prepared manifest;
  * events retained in the physical photopeak cohort;
  * final prepared events after timing-validity / optional LED-mismatch cuts;
  * development and blind/test event counts from a specific study run's results.csv.

Usage example:
    python scripts/make_dataset_counts_table.py ^
      --run results/YOUR_RUN ^
      --prepared processed_data/ml_prepared
"""

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    candidates = [here.parent, *here.parents, Path.cwd().resolve()]
    for candidate in candidates:
        if (candidate / "ml_pipeline").is_dir():
            return candidate
        if (candidate / "waveform_analysis" / "ml_pipeline").is_dir():
            return candidate / "waveform_analysis"
    return here.parents[1]


PROJECT = _find_project_root()
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from ml_pipeline.dataset import PreparedDataset, load_prepared_dataset


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _as_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _fmt_int(value: int | None) -> str:
    if value is None:
        return "--"
    return f"{int(value):,}"


def _fmt_voltage(value: float) -> str:
    if not math.isfinite(value):
        return "--"
    if abs(value - round(value)) < 1e-6:
        return f"{int(round(value))}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _latex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def _prepared_candidates(prepared: Path) -> list[Path]:
    prepared = prepared.resolve()
    if (prepared / "manifest.json").is_file():
        return [prepared]
    if not prepared.is_dir():
        raise FileNotFoundError(f"Prepared path does not exist: {prepared}")
    candidates = [p for p in prepared.iterdir() if p.is_dir() and (p / "manifest.json").is_file()]
    return sorted(candidates, key=lambda p: p.name)


def _load_prepared_all(prepared: Path) -> list[PreparedDataset]:
    datasets: list[PreparedDataset] = []
    for path in _prepared_candidates(prepared):
        try:
            datasets.append(load_prepared_dataset(path))
        except Exception as exc:
            print(f"warning: skipping {path}: {exc}", file=sys.stderr)
    if not datasets:
        raise RuntimeError(f"No loadable prepared datasets found under {prepared}")
    return datasets


def _source_name(dataset: PreparedDataset) -> str:
    source = str(dataset.manifest.get("source_root", ""))
    if source:
        return Path(source).name
    return dataset.directory.name


def _source_stem(dataset: PreparedDataset) -> str:
    return Path(_source_name(dataset)).stem


def _median_voltage(dataset: PreparedDataset) -> float:
    values = np.asarray(dataset.bias_voltage_V, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def _selection_counts(dataset: PreparedDataset) -> dict[str, int | None]:
    manifest = dataset.manifest
    selection = manifest.get("selection", {}) or {}
    physical = selection.get("physical_selection", {}) or {}

    # Best interpretation of "before photopeak" in this repository: the finite,
    # physically valid population immediately before the energy-photopeak cut.
    before_photopeak = _as_int(physical.get("valid_before_photopeak"))

    # Events after the physical/photopeak cohort, before timing-validity and
    # optional LED-mismatch rejection are applied.
    after_photopeak = _as_int(physical.get("selected_events"))

    # Final reusable prepared cohort: after timing-validity and dataset-level cuts.
    prepared = _as_int(selection.get("selected_events"))
    if prepared is None:
        prepared = _as_int(manifest.get("event_count"))
    if prepared is None:
        prepared = int(dataset.event_id.size)

    # Optional raw total fallback, useful for older manifests.
    raw_cache = manifest.get("raw_cache_manifest", {}) or {}
    raw_total = _as_int(raw_cache.get("event_count"))
    if before_photopeak is None:
        before_photopeak = raw_total

    return {
        "before_photopeak": before_photopeak,
        "after_photopeak": after_photopeak,
        "prepared": prepared,
    }


def _run_file_codebook(run_manifest: dict[str, Any]) -> dict[str, int]:
    return {str(k): int(v) for k, v in ((run_manifest.get("codebooks", {}) or {}).get("file", {}) or {}).items()}


def _file_id_for_dataset(dataset: PreparedDataset, run_manifest: dict[str, Any]) -> int | None:
    mapping = _run_file_codebook(run_manifest)
    if not mapping:
        return None
    names = {
        _source_name(dataset),
        _source_stem(dataset),
        dataset.directory.name,
        f"{dataset.directory.name}.root",
    }
    for name in names:
        if name in mapping:
            return int(mapping[name])
    # Fallback: compare stems in case codebook uses full filenames while the
    # prepared directory uses the ROOT stem.
    stem = _source_stem(dataset)
    for name, value in mapping.items():
        if Path(name).stem == stem:
            return int(value)
    return None


def _most_common_n(values: list[int]) -> int | None:
    values = [int(v) for v in values if v is not None and int(v) >= 0]
    if not values:
        return None
    return int(Counter(values).most_common(1)[0][0])


def _stage_count_from_results(
    rows: list[dict[str, str]],
    *,
    stage: int,
    file_id: int | None,
    voltage: float,
) -> int | None:
    ns: list[int] = []
    for row in rows:
        if _as_int(row.get("stage"), -999) != stage:
            continue
        if file_id is not None and _as_int(row.get("file_id"), -999) != file_id:
            continue
        if file_id is None and math.isfinite(voltage):
            row_v = _as_float(row.get("voltage_V"))
            if not math.isfinite(row_v) or abs(row_v - voltage) > 0.25:
                continue
        # The compact results contain many candidates; selected rows are enough
        # and avoid rare rejected/diagnostic rows when present. If selected is
        # absent in an older file, keep the row.
        selected = row.get("selected", "1")
        if selected not in ("", None) and _as_int(selected, 1) != 1:
            continue
        n = _as_int(row.get("n"))
        if n is not None:
            ns.append(n)
    return _most_common_n(ns)


def _counts_from_materialized_splits(dataset: PreparedDataset) -> tuple[int | None, int | None]:
    # Older prepared datasets could materialize train/validation/test. Current
    # permanent datasets normally do not, and the study runner creates splits in memory.
    train = np.asarray(dataset.train, dtype=np.int64)
    validation = np.asarray(dataset.validation, dtype=np.int64)
    test = np.asarray(dataset.test, dtype=np.int64)
    if test.size > 0:
        return int(np.unique(np.concatenate([train, validation])).size), int(test.size)
    return None, None


def _build_rows(
    datasets: list[PreparedDataset],
    run_manifest: dict[str, Any] | None,
    result_rows: list[dict[str, str]] | None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for dataset in datasets:
        voltage = _median_voltage(dataset)
        selection = _selection_counts(dataset)
        file_id = _file_id_for_dataset(dataset, run_manifest or {}) if run_manifest else None

        development_n: int | None = None
        blind_n: int | None = None
        if result_rows is not None:
            development_n = _stage_count_from_results(result_rows, stage=0, file_id=file_id, voltage=voltage)
            blind_n = _stage_count_from_results(result_rows, stage=1, file_id=file_id, voltage=voltage)
        if development_n is None or blind_n is None:
            split_dev, split_test = _counts_from_materialized_splits(dataset)
            development_n = development_n if development_n is not None else split_dev
            blind_n = blind_n if blind_n is not None else split_test

        output.append({
            "voltage_V": voltage,
            "file": _source_name(dataset),
            "before_photopeak": selection["before_photopeak"],
            "photopeak": selection["after_photopeak"],
            "prepared": selection["prepared"],
            "development": development_n,
            "blind_test": blind_n,
        })
    return sorted(output, key=lambda r: (float("inf") if not math.isfinite(r["voltage_V"]) else r["voltage_V"], str(r["file"])))


def _write_latex_table(rows: list[dict[str, Any]], output: Path, include_file: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    colspec = "l" + ("l" if include_file else "") + "rrrrr"
    header_cells = [r"Voltage [V]"]
    if include_file:
        header_cells.append(r"File")
    header_cells.extend([
        r"Before photopeak",
        r"Photopeak cohort",
        r"Prepared",
        r"Development",
        r"Blind/test",
    ])
    lines = [
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        " & ".join(header_cells) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        cells = [_fmt_voltage(float(row["voltage_V"]))]
        if include_file:
            cells.append(_latex_escape(str(row["file"])))
        cells.extend([
            _fmt_int(row["before_photopeak"]),
            _fmt_int(row["photopeak"]),
            _fmt_int(row["prepared"]),
            _fmt_int(row["development"]),
            _fmt_int(row["blind_test"]),
        ])
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\endgroup",
        "",
    ])
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {output}")


def _write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "voltage_V", "file", "before_photopeak", "photopeak", "prepared",
        "development", "blind_test",
    ]
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})
    print(f"wrote {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the dataset-count LaTeX table for the CTR presentation.")
    parser.add_argument(
        "--run",
        type=Path,
        default=None,
        help="Study run directory containing results.csv and manifest.json. Used for development/blind counts.",
    )
    parser.add_argument(
        "--prepared",
        type=Path,
        default=None,
        help="Prepared dataset directory, or parent containing one subdirectory per voltage/file. Default: prepared_dir from run manifest.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "results" / "presentation" / "tables" / "dataset_counts_table.tex",
        help="Output LaTeX table path.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=PROJECT / "results" / "presentation" / "tables" / "dataset_counts_table.csv",
        help="Optional CSV copy of the same counts.",
    )
    parser.add_argument(
        "--include-file",
        action="store_true",
        help="Include the ROOT filename as a second table column. Usually not needed if voltage identifies the file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_manifest: dict[str, Any] | None = None
    result_rows: list[dict[str, str]] | None = None

    if args.run is not None:
        run_dir = args.run.resolve()
        if run_dir.is_file():
            run_dir = run_dir.parent
        manifest_path = run_dir / "manifest.json"
        results_path = run_dir / "results.csv"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Run manifest.json not found: {manifest_path}")
        if not results_path.is_file():
            raise FileNotFoundError(f"Run results.csv not found: {results_path}")
        run_manifest = _read_json(manifest_path)
        result_rows = _read_csv(results_path)

    prepared = args.prepared
    if prepared is None:
        if not run_manifest or not run_manifest.get("prepared_dir"):
            raise ValueError("Pass --prepared, or pass --run with a manifest containing prepared_dir")
        prepared = Path(str(run_manifest["prepared_dir"]))
    if not prepared.is_absolute():
        prepared = (PROJECT / prepared).resolve()

    datasets = _load_prepared_all(prepared)
    rows = _build_rows(datasets, run_manifest, result_rows)
    _write_latex_table(rows, args.output.resolve(), include_file=bool(args.include_file))
    if args.csv_output:
        _write_csv(rows, args.csv_output.resolve())

    print("\nLaTeX include line:")
    print(r"\input{results/presentation/tables/dataset_counts_table.tex}")


if __name__ == "__main__":
    main()
