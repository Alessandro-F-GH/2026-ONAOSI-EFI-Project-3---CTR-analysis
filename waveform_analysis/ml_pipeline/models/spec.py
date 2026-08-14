from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from torch import nn

from ..training_context import TrainingContext

ModelBuilder = Callable[[dict[str, Any], int], nn.Module]
ModelValidator = Callable[[dict[str, Any]], None]
TrainingValidator = Callable[[dict[str, Any]], None]
ModelTrainer = Callable[[TrainingContext], dict[str, Any]]
ModelComplexityCounter = Callable[[dict[str, Any], int], int]
# Optional non-Torch inference hook. It receives the loaded checkpoint payload,
# the already-resolved/materialized dataset view and the evaluation config, and
# returns the learned anchor-relative correction in ps for dataset.evaluation.
CheckpointPredictor = Callable[[dict[str, Any], Any, dict[str, Any]], np.ndarray]


@dataclass(frozen=True)
class ModelSpec:
    """Complete plug-in contract for one trainable model.

    A new model is added by placing a module in ``ml_pipeline/models`` that
    exports ``MODEL_SPEC``. No experiment, training or registry edit is needed.

    Most models use the Torch ``builder`` path. Models backed by another
    library can additionally provide ``checkpoint_predictor`` so evaluation can
    replay their native sparse/non-Torch representation without wrapping it in
    an artificial ``torch.nn.Module``.
    """

    name: str
    builder: ModelBuilder
    validator: ModelValidator
    training_validator: TrainingValidator
    trainer: ModelTrainer
    complexity_counter: ModelComplexityCounter | None = None
    checkpoint_predictor: CheckpointPredictor | None = None
