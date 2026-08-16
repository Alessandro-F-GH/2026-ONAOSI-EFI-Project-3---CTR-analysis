from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations, product
from typing import Any, Iterable
import pickle
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from .signal import threshold_crossing_relative_ps


def _crossing_table(X: np.ndarray, time_ps: np.ndarray, thresholds_mV: Iterable[float]) -> dict[float, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    t = np.asarray(time_ps, dtype=np.float64)
    output: dict[float, np.ndarray] = {}
    for threshold in thresholds_mV:
        th = float(threshold)
        values = np.full((X.shape[0], 2), np.nan, dtype=np.float64)
        for i in range(X.shape[0]):
            for c in range(2):
                values[i, c] = threshold_crossing_relative_ps(X[i, c], t, th)
        output[th] = values
    return output


def feature_matrix(table: dict[float, np.ndarray], thresholds: Iterable[float]) -> np.ndarray:
    """Translation-invariant multithreshold features.

    For every selected threshold we use only detector-to-detector relative crossing
    differences. No absolute/global LED timestamp enters the feature matrix.
    """
    cols = []
    for th in thresholds:
        pair = np.asarray(table[float(th)], dtype=np.float64)
        cols.append(pair[:, 0] - pair[:, 1])
    return np.column_stack(cols) if cols else np.empty((len(next(iter(table.values()))), 0), dtype=np.float64)


@dataclass
class MultithresholdModel:
    estimator: Any
    thresholds_mV: tuple[float, ...]
    time_ps: np.ndarray
    def predict(self, X: np.ndarray) -> np.ndarray:
        table = _crossing_table(X, self.time_ps, self.thresholds_mV)
        features = feature_matrix(table, self.thresholds_mV)
        if np.any(~np.isfinite(features)):
            raise RuntimeError("Multithreshold prediction contains missing threshold crossings")
        return np.asarray(self.estimator.predict(features), dtype=np.float64)
    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle)


def candidate_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    thresholds = sorted(set(float(x) for x in config.get("thresholds_mV", [5, 10, 25, 50, 75, 100])))
    lo = max(1, int(config.get("min_thresholds", 1)))
    hi = min(len(thresholds), int(config.get("max_thresholds", min(4, len(thresholds)))))
    subsets = [tuple(v) for k in range(lo, hi + 1) for v in combinations(thresholds, k)]
    kernels = [str(x) for x in config.get("kernels", ["linear", "rbf"])]
    C_values = [float(x) for x in config.get("C_values", [1.0, 10.0, 100.0])]
    eps_values = [float(x) for x in config.get("epsilon_values_ps", [0.0, 10.0, 30.0])]
    gamma_values = list(config.get("gamma_values", ["scale"]))
    output = []
    for subset, kernel, C, eps in product(subsets, kernels, C_values, eps_values):
        gammas = gamma_values if kernel == "rbf" else ["scale"]
        for gamma in gammas:
            output.append({"thresholds_mV": list(subset), "kernel": kernel, "C": C, "epsilon_ps": eps, "gamma": gamma})
    return output


def valid_mask(table: dict[float, np.ndarray], thresholds: Iterable[float]) -> np.ndarray:
    mask = None
    for th in thresholds:
        finite = np.all(np.isfinite(table[float(th)]), axis=1)
        mask = finite if mask is None else (mask & finite)
    return np.asarray(mask, dtype=bool) if mask is not None else np.zeros(0, dtype=bool)


def fit_from_table(table: dict[float, np.ndarray], y: np.ndarray, train_indices: np.ndarray, spec: dict[str, Any], time_ps: np.ndarray) -> MultithresholdModel:
    thresholds = tuple(float(x) for x in spec["thresholds_mV"])
    features = feature_matrix(table, thresholds)
    idx = np.asarray(train_indices, dtype=np.int64)
    if idx.size < 3 or np.any(~np.isfinite(features[idx])):
        raise RuntimeError("Insufficient finite multithreshold training events")
    estimator = make_pipeline(
        StandardScaler(),
        SVR(
            kernel=str(spec.get("kernel", "linear")),
            C=float(spec.get("C", 1.0)),
            epsilon=float(spec.get("epsilon_ps", 10.0)),
            gamma=spec.get("gamma", "scale"),
            max_iter=int(spec.get("max_iterations", 30000)),
            tol=float(spec.get("tolerance", 1e-4)),
        ),
    )
    estimator.fit(features[idx], np.asarray(y, dtype=np.float64)[idx])
    return MultithresholdModel(estimator, thresholds, np.asarray(time_ps, dtype=np.float64))


def crossing_table(X: np.ndarray, time_ps: np.ndarray, thresholds_mV: Iterable[float]) -> dict[float, np.ndarray]:
    return _crossing_table(X, time_ps, thresholds_mV)
