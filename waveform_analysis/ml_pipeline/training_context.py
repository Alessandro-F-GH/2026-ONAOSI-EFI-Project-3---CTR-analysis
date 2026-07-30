from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dataset import PreparedDataset
from .torch_data import Normalization


@dataclass(frozen=True)
class TrainingContext:
    """Common data prepared once by the unified training entry point."""

    config: dict[str, Any]
    model_type: str
    model_name: str
    model_config: dict[str, Any]
    datasets: list[PreparedDataset]
    input_length: int
    normalization: Normalization
    output_dir: Path
    plot_dir: Path
    checkpoint_dir: Path
    logger: Any
    data_view: dict[str, Any]
