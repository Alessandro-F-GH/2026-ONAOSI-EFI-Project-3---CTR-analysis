from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from ml_pipeline.config import load_json
from ml_pipeline.dataset import load_prepared_dataset, prepared_dataset_view, window_slice_indices
from ml_pipeline.standard_methods import fit_linear_spline
from ml_pipeline.standard_methods.linear_spline import save_linear_spline_artifact


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit the linear-spline standard correction method."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()

    cfg = load_json(args.config)
    dataset_path = _resolve(PROJECT, str(cfg["dataset"]))
    output = _resolve(PROJECT, str(cfg["output_dir"]))
    if output.exists() and not args.restart:
        raise FileExistsError(f"Output already exists: {output}; use --restart")
    if output.exists():
        import shutil
        shutil.rmtree(output)

    dataset = load_prepared_dataset(dataset_path)
    window = cfg.get("window")
    if isinstance(window, dict):
        start, stop = window_slice_indices(
            dataset,
            float(window["before_ns"]),
            float(window["after_ns"]),
        )
        dataset = prepared_dataset_view(dataset, window_start=start, window_stop=stop)
    artifact = fit_linear_spline(dataset, dataset.train, dict(cfg.get("parameters", {})))
    save_linear_spline_artifact(artifact, output)
    with (output / "source.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["dataset", "dataset_fingerprint", "train_events"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "dataset": str(dataset_path),
                "dataset_fingerprint": dataset.manifest["fingerprint"],
                "train_events": int(dataset.train.size),
            }
        )


if __name__ == "__main__":
    main()
