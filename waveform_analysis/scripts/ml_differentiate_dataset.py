from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from ml_pipeline.common import setup_logging
from ml_pipeline.dataset import load_prepared_dataset
from ml_pipeline.input_transform import materialize_training_input_cache


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build an optional first-difference model-input cache from a canonical "
            "prepared dataset. Training normally creates this cache automatically; "
            "this command is retained only for inspection/migration workflows."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Canonical prepared dataset produced by scripts/ml_preprocess.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=(
            "Directory under a training/model run in which input_cache/ will be "
            "created. The result is not a second prepared dataset."
        ),
    )
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument(
        "--dtype",
        default=None,
        help=(
            "Deprecated compatibility option. The cache now preserves the "
            "canonical dataset dtype, so this value is ignored."
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be >= 1")

    dataset = load_prepared_dataset(args.input.resolve())
    output = args.output.resolve()
    logger = setup_logging(output / "differentiate_input_cache.log", "INFO")
    if args.dtype is not None:
        logger.warning(
            "--dtype is deprecated and ignored; transformed input caches preserve "
            "the canonical dataset dtype (%s)",
            dataset.windows_mV.dtype,
        )
    _, cache_dir = materialize_training_input_cache(
        dataset,
        "differentiate",
        output / "input_cache",
        chunk_size=args.chunk_size,
        rebuild=args.force,
        logger=logger,
    )
    logger.info(
        "Differentiated input cache ready at %s. It intentionally contains no "
        "event metadata or split files and cannot be used as a canonical dataset.",
        cache_dir,
    )


if __name__ == "__main__":
    main()
