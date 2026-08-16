from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SelectionSplit:
    train: np.ndarray
    score: np.ndarray
    split_index: int


def random_dev_blind(
    n: int,
    *,
    blind_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if n < 5:
        raise RuntimeError("Need at least five prepared events")
    if not 0.0 < blind_fraction < 0.5:
        raise ValueError("blind_fraction must be in (0, 0.5)")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n).astype(np.int64)
    n_blind = int(round(n * blind_fraction))
    n_blind = min(max(1, n_blind), n - 2)
    blind = np.sort(order[:n_blind])
    development = np.sort(order[n_blind:])
    return development, blind


def holdout_split(
    indices: np.ndarray,
    *,
    fraction: float,
    seed: int,
) -> SelectionSplit:
    values = np.asarray(indices, dtype=np.int64).reshape(-1)
    if values.size < 3:
        raise RuntimeError("Need at least three events for holdout validation")
    if not 0.0 < fraction < 0.5:
        raise ValueError("holdout fraction must be in (0, 0.5)")
    order = np.random.default_rng(seed).permutation(values)
    n_score = int(round(values.size * fraction))
    n_score = min(max(1, n_score), values.size - 2)
    score = np.sort(order[:n_score])
    train = np.sort(order[n_score:])
    return SelectionSplit(train=train, score=score, split_index=0)


def kfold_splits(
    indices: np.ndarray,
    *,
    n_splits: int,
    seed: int,
) -> list[SelectionSplit]:
    values = np.asarray(indices, dtype=np.int64).reshape(-1)
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if values.size < n_splits:
        raise RuntimeError(f"Only {values.size} events for {n_splits}-fold validation")
    order = np.random.default_rng(seed).permutation(values)
    score_folds = [np.sort(v.astype(np.int64)) for v in np.array_split(order, n_splits)]
    output: list[SelectionSplit] = []
    for i, score in enumerate(score_folds):
        train = np.sort(np.concatenate([v for j, v in enumerate(score_folds) if j != i]))
        output.append(SelectionSplit(train=train, score=score, split_index=i))
    return output


def effective_selection_strategy(validation: dict[str, Any]) -> str:
    strategy = str(validation.get("strategy", "cv")).strip().lower()
    if strategy == "nested":
        return str(validation.get("nested", {}).get("inner_strategy", "holdout")).strip().lower()
    return strategy


def selection_splits(
    indices: np.ndarray,
    validation: dict[str, Any],
    *,
    seed: int,
    strategy: str | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    selected_strategy = (strategy or effective_selection_strategy(validation)).strip().lower()
    if selected_strategy == "holdout":
        split = holdout_split(
            indices,
            fraction=float(validation.get("holdout_fraction", 0.2)),
            seed=seed,
        )
        return [(split.train, split.score)]
    if selected_strategy == "cv":
        n_splits = int(validation.get("n_splits", 5))
        return [
            (split.train, split.score)
            for split in kfold_splits(indices, n_splits=n_splits, seed=seed)
        ]
    raise ValueError(f"Unsupported selection strategy: {selected_strategy}")


def outer_splits(
    development: np.ndarray,
    validation: dict[str, Any],
    *,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    nested = validation.get("nested", {}) or {}
    n_splits = int(nested.get("outer_folds", 5))
    return [
        (split.train, split.score)
        for split in kfold_splits(development, n_splits=n_splits, seed=seed)
    ]


def nested_inner_validation(validation: dict[str, Any]) -> dict[str, Any]:
    """Return a normal selection config representing one nested inner procedure."""
    nested = validation.get("nested", {}) or {}
    strategy = str(nested.get("inner_strategy", "holdout")).strip().lower()
    cfg = dict(validation)
    cfg["strategy"] = strategy
    if strategy == "holdout":
        cfg["holdout_fraction"] = float(
            nested.get("inner_holdout_fraction", validation.get("holdout_fraction", 0.2))
        )
    else:
        cfg["n_splits"] = int(nested.get("inner_folds", validation.get("n_splits", 5)))
    return cfg
