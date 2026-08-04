from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from ml_pipeline.common import setup_logging
from ml_pipeline.correction_analysis import (
    analyze_right_corrections,
    save_correction_analysis,
)
from ml_pipeline.dataset import load_prepared_dataset_spec
from ml_pipeline.evaluation import evaluate_trained_model, load_trained_model
from ml_pipeline.training_utils import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rank evaluation events by how much an ML correction moves the timing "
            "estimate closer to the known TOF, then plot their waveform pairs."
        )
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Training run directory, training_summary.json, or .pt checkpoint",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--minimum-improvement-ps", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Write CSV/JSON ranking without waveform PNG files",
    )
    args = parser.parse_args()

    output = args.output.resolve()
    logger = setup_logging(output / "top_corrections.log", "INFO")
    dataset = load_prepared_dataset_spec(args.dataset.resolve())
    if dataset.evaluation.size == 0:
        raise RuntimeError(f"Dataset has no evaluation events: {dataset.directory}")
    trained = load_trained_model(args.model.resolve())
    device = resolve_device(args.device)
    prediction = evaluate_trained_model(
        trained,
        dataset,
        {
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "pin_memory": device.type == "cuda",
        },
        device,
    )
    analysis = analyze_right_corrections(
        prediction.dataset_view,
        prediction.dataset_view.evaluation,
        raw_ps=prediction.raw_ps,
        corrected_ps=prediction.corrected_ps,
        predicted_correction_ps=prediction.predicted_correction_ps,
        true_tof_ps=prediction.true_tof_ps,
        top_n=args.top_n,
        minimum_improvement_ps=args.minimum_improvement_ps,
    )
    save_correction_analysis(
        analysis,
        prediction.dataset_view,
        output_dir=output,
        input_transform=prediction.input_transform,
        input_waveform_source=prediction.input_waveform_source,
        prediction_target=prediction.prediction_target,
        model_name=trained.model_name,
        dpi=args.dpi,
        save_waveform_plots=not args.metadata_only,
    )

    top = analysis.summary["top_right_correction_ps"]
    if top is None:
        logger.info(
            "Top right correction: none above %.3f ps",
            args.minimum_improvement_ps,
        )
    else:
        logger.info(
            "Top right correction: %.3f ps | event_id=%s | row=%s",
            float(top),
            analysis.summary["top_right_correction_event_id"],
            analysis.summary["top_right_correction_dataset_index"],
        )
    logger.info(
        "Useful corrections: %d/%d (%.2f%%)",
        analysis.summary["right_correction_count"],
        analysis.summary["event_count"],
        100.0 * analysis.summary["right_correction_fraction"],
    )
    logger.info("Saved correction analysis to %s", output)


if __name__ == "__main__":
    main()
