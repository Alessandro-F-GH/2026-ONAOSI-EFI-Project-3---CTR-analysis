#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from utils.config import load_config
from utils.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Binary Janus timing-analysis pipeline")
    parser.add_argument("-c", "--config", default="config/janus_pipeline.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    run_pipeline(load_config(root / args.config, root))


if __name__ == "__main__":
    main()
