from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import Any

from torch import nn

from ..training_context import TrainingContext
from .spec import ModelSpec

_MODEL_REGISTRY: dict[str, ModelSpec] = {}
_DISCOVERED = False
_EXCLUDED_MODULES = {"__init__", "registry", "spec"}


def register_model(spec: ModelSpec) -> None:
    name = str(spec.name).strip()
    if not name:
        raise ValueError("ModelSpec.name must be non-empty")
    if name in _MODEL_REGISTRY:
        raise ValueError(f"Duplicate model name: {name!r}")
    _MODEL_REGISTRY[name] = spec


def _discover_models() -> None:
    global _DISCOVERED
    if _DISCOVERED:
        return
    package_name = __package__
    package_path = Path(__file__).resolve().parent
    for module_info in pkgutil.iter_modules([str(package_path)]):
        if module_info.name in _EXCLUDED_MODULES or module_info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{package_name}.{module_info.name}")
        spec = getattr(module, "MODEL_SPEC", None)
        if spec is None:
            continue
        if not isinstance(spec, ModelSpec):
            raise TypeError(
                f"{module.__name__}.MODEL_SPEC must be a ModelSpec, got {type(spec).__name__}"
            )
        register_model(spec)
    _DISCOVERED = True


def model_registry() -> dict[str, ModelSpec]:
    _discover_models()
    return dict(_MODEL_REGISTRY)


def _spec(model_type: str) -> ModelSpec:
    _discover_models()
    key = str(model_type)
    try:
        return _MODEL_REGISTRY[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model type {key!r}; available: {sorted(_MODEL_REGISTRY)}"
        ) from exc


def build_model(model_type: str, config: dict[str, Any], input_length: int) -> nn.Module:
    return _spec(model_type).builder(config, int(input_length))


def count_model_parameters(model_type: str, config: dict[str, Any], input_length: int) -> int:
    spec = _spec(model_type)
    if spec.complexity_counter is not None:
        return int(spec.complexity_counter(config, int(input_length)))
    model = spec.builder(config, int(input_length))
    return int(sum(parameter.numel() for parameter in model.parameters()))


def validate_model(model_type: str, config: dict[str, Any]) -> None:
    _spec(model_type).validator(config)


def validate_model_training(model_type: str, config: dict[str, Any]) -> None:
    _spec(model_type).training_validator(config)


def train_registered_model(model_type: str, context: TrainingContext) -> dict[str, Any]:
    return _spec(model_type).trainer(context)
