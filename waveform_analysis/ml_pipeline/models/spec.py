from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from torch import nn

from ..training_context import TrainingContext

ModelBuilder = Callable[[dict[str, Any], int], nn.Module]
ModelValidator = Callable[[dict[str, Any]], None]
TrainingValidator = Callable[[dict[str, Any]], None]
ModelTrainer = Callable[[TrainingContext], dict[str, Any]]
ModelComplexityCounter = Callable[[dict[str, Any], int], int]


@dataclass(frozen=True)
class ModelSpec:
    """Complete plug-in contract for one trainable model.

    A new model is added by placing a module in ``ml_pipeline/models`` that
    exports ``MODEL_SPEC``. No experiment, training or registry edit is needed.
    """

    name: str
    builder: ModelBuilder
    validator: ModelValidator
    training_validator: TrainingValidator
    trainer: ModelTrainer
    complexity_counter: ModelComplexityCounter | None = None
