from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .binary_io import DataError, LEAD_TRAIL, STREAMING, iter_events, read_header
from .pulses import earliest_energy_pair, leading_hits_before, timing_overlap_candidates
from .tabular import (
    SIDE_TO_CODE,
    STATUS_TO_CODE,
    read_table,
    side_name,
    status_code,
    status_name,
    write_table,
)

PAIR_KEYS = ("a", "b")

TRAINING_FIELDS = [
    "event_index",
    "side_code",
    "energy_duration_lsb",
    "delay_lsb",
]
TRAINING_DEBUG_FIELDS = [
    *TRAINING_FIELDS,
    "energy_channel",
    "timing_channel",
    "energy_leading_lsb",
    "timing_leading_lsb",
]
MODEL_METRIC_FIELDS = [
    "side_code",
    "average_delay_lsb",
    "delay_std_lsb",
    "robust_center_lsb",
    "robust_scale_lsb",
    "input_samples",
    "training_samples",
    "outliers_rejected",
    "outlier_iterations",
    "outlier_z_threshold",
]
TOTAL_COMPACT_FIELDS = [
    "event_index",
    "duration_a_lsb",
    "duration_b_lsb",
    "average_delay_a_lsb",
    "average_delay_b_lsb",
    "selected_delay_a_lsb",
    "selected_delay_b_lsb",
    "baseline_delay_a_lsb",
    "baseline_delay_b_lsb",
    "deviation_a_lsb",
    "deviation_b_lsb",
    "maximum_deviation_lsb",
    "candidate_count",
    "candidate_changed",
    "status_code",
]
TOTAL_DEBUG_FIELDS = [
    "event_index",
    "side_code",
    "energy_channel",
    "timing_channel",
    "energy_duration_lsb",
    "average_delay_lsb",
    "selected_delay_lsb",
    "baseline_delay_lsb",
    "candidate_changed",
    "deviation_lsb",
    "maximum_deviation_lsb",
    "candidate_count",
    "event_accepted",
    "status_code",
]
TOTAL_SUMMARY_FIELDS = [
    "events",
    "accepted_events",
    "rejected_events",
    "changed_events",
    "mean_deviation_a_lsb",
    "mean_deviation_b_lsb",
]


@dataclass(slots=True, frozen=True)
class MatchingSamples:
    event_index: np.ndarray
    energy_duration_lsb: np.ndarray
    delay_lsb: np.ndarray

    @property
    def size(self) -> int:
        return int(self.event_index.size)

    def subset(self, mask: np.ndarray) -> "MatchingSamples":
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != self.event_index.shape:
            raise ValueError("Matching-sample mask has an invalid shape")
        return MatchingSamples(
            event_index=self.event_index[mask],
            energy_duration_lsb=self.energy_duration_lsb[mask],
            delay_lsb=self.delay_lsb[mask],
        )


@dataclass(slots=True, frozen=True)
class AverageDelayModel:
    pair: str
    energy_channel: int
    timing_channel: int
    average_delay_lsb: float
    delay_std_lsb: float
    robust_center_lsb: float
    robust_scale_lsb: float
    input_samples: int
    training_samples: int
    outliers_rejected: int
    outlier_iterations: int
    outlier_z_threshold: float

    @property
    def residual_center_lsb(self) -> float:
        """Compatibility alias for older plotting/consumer code."""
        return self.robust_center_lsb - self.average_delay_lsb

    @property
    def residual_scale_lsb(self) -> float:
        """Compatibility alias for older matching diagnostics."""
        return self.robust_scale_lsb

    def predict(self, energy_duration_lsb: np.ndarray | float | int) -> np.ndarray | float:
        values = np.asarray(energy_duration_lsb)
        if values.ndim == 0:
            return float(self.average_delay_lsb)
        return np.full(values.shape, self.average_delay_lsb, dtype=float)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "average_delay_v1",
            "pair": self.pair,
            "side_code": SIDE_TO_CODE[self.pair],
            "energy_channel": self.energy_channel,
            "timing_channel": self.timing_channel,
            "average_delay_lsb": self.average_delay_lsb,
            "delay_std_lsb": self.delay_std_lsb,
            "robust_center_lsb": self.robust_center_lsb,
            "robust_scale_lsb": self.robust_scale_lsb,
            "input_samples": self.input_samples,
            "training_samples": self.training_samples,
            "outliers_rejected": self.outliers_rejected,
            "outlier_iterations": self.outlier_iterations,
            "outlier_z_threshold": self.outlier_z_threshold,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AverageDelayModel":
        model_type = str(data.get("model_type", ""))
        if model_type == "average_delay_v1":
            pair = str(data.get("pair", side_name(data.get("side_code", 0))))
            return cls(
                pair=pair,
                energy_channel=int(data["energy_channel"]),
                timing_channel=int(data["timing_channel"]),
                average_delay_lsb=float(data["average_delay_lsb"]),
                delay_std_lsb=float(data.get("delay_std_lsb", data.get("robust_scale_lsb", 0.0))),
                robust_center_lsb=float(data.get("robust_center_lsb", data["average_delay_lsb"])),
                robust_scale_lsb=float(data.get("robust_scale_lsb", data.get("delay_std_lsb", 1.0))),
                input_samples=int(data.get("input_samples", data.get("training_samples", 0))),
                training_samples=int(data.get("training_samples", 0)),
                outliers_rejected=int(data.get("outliers_rejected", 0)),
                outlier_iterations=int(data.get("outlier_iterations", 0)),
                outlier_z_threshold=float(data.get("outlier_z_threshold", math.nan)),
            )
        # Backward-compatible migration from the former ridge-polynomial JSON.
        if model_type in {"ridge_polynomial_v1", "ridge_polynomial_v2"} or "y_mean_lsb" in data:
            pair = str(data.get("pair", "a"))
            average = float(data.get("y_mean_lsb", 0.0))
            scale = float(data.get("residual_scale_lsb", data.get("y_scale_lsb", 1.0)))
            return cls(
                pair=pair,
                energy_channel=int(data["energy_channel"]),
                timing_channel=int(data["timing_channel"]),
                average_delay_lsb=average,
                delay_std_lsb=max(scale, 0.0),
                robust_center_lsb=average + float(data.get("residual_center_lsb", 0.0)),
                robust_scale_lsb=max(scale, 1.0),
                input_samples=int(data.get("input_samples", data.get("training_samples", 0))),
                training_samples=int(data.get("training_samples", 0)),
                outliers_rejected=int(data.get("outliers_rejected", 0)),
                outlier_iterations=int(data.get("outlier_iterations", 0)),
                outlier_z_threshold=float(data.get("outlier_z_threshold", math.nan)),
            )
        raise DataError("Unsupported matching-model file")


# Compatibility alias: external code importing the old class keeps working, but
# the object is now a constant average-delay model.
RidgePolynomialModel = AverageDelayModel


def _channel_pairs(cfg: dict) -> dict[str, tuple[int, int]]:
    channels = cfg["channels"]
    return {
        "a": (int(channels["signal_a"]), int(channels["time_a"])),
        "b": (int(channels["signal_b"]), int(channels["time_b"])),
    }


def window_ns_to_lsb(window_ns: float, toa_lsb_ps: float) -> int:
    if not math.isfinite(toa_lsb_ps) or toa_lsb_ps <= 0:
        raise DataError("ToA LSB must be positive")
    return max(1, int(math.ceil(float(window_ns) * 1000.0 / toa_lsb_ps)))


def scan_matching_training(
    input_path: str | Path,
    acquisition_mode: str,
    cfg: dict,
    selected_event_indices: set[int],
) -> tuple[dict[str, MatchingSamples], dict[str, Any]]:
    """Build unambiguous delay calibration samples after energy selection."""
    input_path = Path(input_path)
    pairs = _channel_pairs(cfg)
    rows: dict[str, list[dict[str, int | str]]] = {key: [] for key in PAIR_KEYS}
    events_read = 0
    selected_events_seen = 0

    with input_path.open("rb") as handle:
        meta = read_header(handle, acquisition_mode)
        if meta.measurement_mode != LEAD_TRAIL:
            raise DataError("Average-delay calibration requires LEAD_TRAIL measurement mode")
        training_cfg = cfg["matching_model"]["training"]
        window_lsb = window_ns_to_lsb(training_cfg["window_ns"], meta.toa_lsb_ps)
        for event in iter_events(handle, meta):
            events_read += 1
            if event.event_index not in selected_event_indices:
                continue
            selected_events_seen += 1
            energy_pair_a = earliest_energy_pair(event.hits, pairs["a"][0])
            energy_pair_b = earliest_energy_pair(event.hits, pairs["b"][0])
            if energy_pair_a is None or energy_pair_b is None:
                continue

            if meta.acquisition_mode == STREAMING:
                timing_candidates = timing_overlap_candidates(
                    event.hits,
                    pairs["a"][1],
                    pairs["b"][1],
                    energy_pair_a,
                    energy_pair_b,
                    window_lsb,
                )
                if len(timing_candidates) != 1:
                    continue
                timing_pair_a, timing_pair_b = timing_candidates[0]
                selected = {
                    "a": (energy_pair_a, timing_pair_a[0]),
                    "b": (energy_pair_b, timing_pair_b[0]),
                }
                for pair_key, (energy_channel, timing_channel) in pairs.items():
                    energy_pair, timing_lead = selected[pair_key]
                    rows[pair_key].append(
                        {
                            "event_index": event.event_index,
                            "pair": pair_key,
                            "energy_channel": energy_channel,
                            "timing_channel": timing_channel,
                            "energy_duration_lsb": energy_pair[1].toa_lsb - energy_pair[0].toa_lsb,
                            "delay_lsb": energy_pair[0].toa_lsb - timing_lead.toa_lsb,
                            "energy_leading_lsb": energy_pair[0].toa_lsb,
                            "timing_leading_lsb": timing_lead.toa_lsb,
                        }
                    )
                continue

            for pair_key, (energy_channel, timing_channel) in pairs.items():
                energy_pair = energy_pair_a if pair_key == "a" else energy_pair_b
                timing_candidates = leading_hits_before(
                    event.hits, timing_channel, energy_pair[0].toa_lsb, window_lsb
                )
                if len(timing_candidates) != 1:
                    continue
                timing_lead = timing_candidates[0]
                rows[pair_key].append(
                    {
                        "event_index": event.event_index,
                        "pair": pair_key,
                        "energy_channel": energy_channel,
                        "timing_channel": timing_channel,
                        "energy_duration_lsb": energy_pair[1].toa_lsb - energy_pair[0].toa_lsb,
                        "delay_lsb": energy_pair[0].toa_lsb - timing_lead.toa_lsb,
                        "energy_leading_lsb": energy_pair[0].toa_lsb,
                        "timing_leading_lsb": timing_lead.toa_lsb,
                    }
                )

    samples = {
        pair_key: MatchingSamples(
            event_index=np.asarray([int(row["event_index"]) for row in rows[pair_key]], dtype=np.int64),
            energy_duration_lsb=np.asarray([int(row["energy_duration_lsb"]) for row in rows[pair_key]], dtype=np.int64),
            delay_lsb=np.asarray([int(row["delay_lsb"]) for row in rows[pair_key]], dtype=np.int64),
        )
        for pair_key in PAIR_KEYS
    }
    metadata = {
        "events_read": events_read,
        "energy_selected_events": selected_events_seen,
        "training_samples_a": samples["a"].size,
        "training_samples_b": samples["b"].size,
        "ambiguous_or_missing_a": selected_events_seen - samples["a"].size,
        "ambiguous_or_missing_b": selected_events_seen - samples["b"].size,
        "toa_lsb_ps": meta.toa_lsb_ps,
        "training_window_ns": float(training_cfg["window_ns"]),
        "training_window_lsb": window_lsb,
        "physical_time_source": "streaming_hit_toa_uint64" if meta.acquisition_mode == STREAMING else "trigger_relative_toa",
        "streaming_requires_unique_overlapping_timing_pair": meta.acquisition_mode == STREAMING,
        "rows": [row for pair_key in PAIR_KEYS for row in rows[pair_key]],
    }
    return samples, metadata


def _training_rows_from_samples(
    samples: dict[str, MatchingSamples],
    cfg: dict | None = None,
) -> list[dict[str, Any]]:
    channels = _channel_pairs(cfg) if cfg is not None else {"a": (0, 0), "b": (0, 0)}
    rows: list[dict[str, Any]] = []
    for pair in PAIR_KEYS:
        energy_channel, timing_channel = channels[pair]
        local = samples[pair]
        rows.extend(
            {
                "event_index": int(event_index),
                "side_code": SIDE_TO_CODE[pair],
                "pair": pair,
                "energy_channel": energy_channel,
                "timing_channel": timing_channel,
                "energy_duration_lsb": int(duration),
                "delay_lsb": int(delay),
            }
            for event_index, duration, delay in zip(
                local.event_index, local.energy_duration_lsb, local.delay_lsb
            )
        )
    return rows


def write_training_csv(
    path: str | Path,
    rows: list[dict[str, Any]],
    diagnostic_mode: str = "compact",
) -> None:
    debug = str(diagnostic_mode).lower() == "debug"
    encoded = []
    for row in rows:
        pair = side_name(row.get("side_code", row.get("pair", 0)))
        item = {
            "event_index": int(row["event_index"]),
            "side_code": SIDE_TO_CODE[pair],
            "energy_duration_lsb": int(row.get("energy_duration_lsb", 0)),
            "delay_lsb": int(row["delay_lsb"]),
        }
        if debug:
            item.update(
                {
                    "energy_channel": int(row.get("energy_channel", 0)),
                    "timing_channel": int(row.get("timing_channel", 0)),
                    "energy_leading_lsb": row.get("energy_leading_lsb", ""),
                    "timing_leading_lsb": row.get("timing_leading_lsb", ""),
                }
            )
        encoded.append(item)
    write_table(path, TRAINING_DEBUG_FIELDS if debug else TRAINING_FIELDS, encoded)


def load_training_csv(path: str | Path) -> dict[str, MatchingSamples]:
    grouped: dict[str, list[dict[str, Any]]] = {key: [] for key in PAIR_KEYS}
    for row in read_table(path):
        pair = side_name(row.get("side_code", row.get("pair", 0)))
        grouped[pair].append(row)
    return {
        pair: MatchingSamples(
            event_index=np.asarray([int(row["event_index"]) for row in rows], dtype=np.int64),
            energy_duration_lsb=np.asarray([int(row.get("energy_duration_lsb", 0)) for row in rows], dtype=np.int64),
            delay_lsb=np.asarray([int(row["delay_lsb"]) for row in rows], dtype=np.int64),
        )
        for pair, rows in grouped.items()
    }


def _robust_location_scale(values: np.ndarray, minimum_scale_lsb: float) -> tuple[float, float]:
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    scale = 1.4826 * mad
    if not math.isfinite(scale) or scale <= 0:
        scale = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    return center, max(scale, float(minimum_scale_lsb))


def fit_average_delay(
    pair: str,
    samples: MatchingSamples,
    energy_channel: int,
    timing_channel: int,
    cfg: dict,
) -> tuple[AverageDelayModel, MatchingSamples]:
    minimum_samples = int(cfg["matching_model"]["training"]["minimum_samples"])
    if samples.size < minimum_samples:
        raise DataError(
            f"Side {pair}: {samples.size} calibration samples; at least {minimum_samples} are required"
        )
    average_cfg = cfg["matching_model"]["average_delay"]
    filter_cfg = average_cfg["outlier_filter"]
    enabled = bool(filter_cfg["enabled"])
    z_threshold = float(filter_cfg["z_threshold"])
    maximum_iterations = int(filter_cfg["max_iterations"])
    minimum_scale = float(filter_cfg["minimum_scale_lsb"])

    delays = samples.delay_lsb.astype(float)
    mask = np.ones(samples.size, dtype=bool)
    iterations = 0
    if enabled:
        for _ in range(maximum_iterations):
            current = delays[mask]
            center, scale = _robust_location_scale(current, minimum_scale)
            keep = np.abs(delays - center) <= z_threshold * scale
            if np.array_equal(keep, mask):
                break
            if int(np.count_nonzero(keep)) < minimum_samples:
                break
            mask = keep
            iterations += 1

    filtered = samples.subset(mask)
    retained = filtered.delay_lsb.astype(float)
    robust_center, robust_scale = _robust_location_scale(retained, minimum_scale)
    average_delay = float(np.mean(retained))
    delay_std = float(np.std(retained, ddof=1)) if retained.size > 1 else 0.0
    model = AverageDelayModel(
        pair=pair,
        energy_channel=energy_channel,
        timing_channel=timing_channel,
        average_delay_lsb=average_delay,
        delay_std_lsb=delay_std,
        robust_center_lsb=robust_center,
        robust_scale_lsb=robust_scale,
        input_samples=samples.size,
        training_samples=filtered.size,
        outliers_rejected=samples.size - filtered.size,
        outlier_iterations=iterations,
        outlier_z_threshold=z_threshold,
    )
    return model, filtered


def fit_ridge_polynomial(
    pair: str,
    samples: MatchingSamples,
    energy_channel: int,
    timing_channel: int,
    cfg: dict,
) -> tuple[AverageDelayModel, MatchingSamples]:
    """Deprecated compatibility wrapper for the former ridge fitter."""
    return fit_average_delay(pair, samples, energy_channel, timing_channel, cfg)


def train_matching_models(
    samples: dict[str, MatchingSamples],
    cfg: dict,
) -> tuple[dict[str, AverageDelayModel], dict[str, MatchingSamples]]:
    models: dict[str, AverageDelayModel] = {}
    filtered_samples: dict[str, MatchingSamples] = {}
    for pair, (energy_channel, timing_channel) in _channel_pairs(cfg).items():
        model, filtered = fit_average_delay(
            pair, samples[pair], energy_channel, timing_channel, cfg
        )
        models[pair] = model
        filtered_samples[pair] = filtered
    return models, filtered_samples


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_model(path: str | Path, model: AverageDelayModel) -> None:
    _atomic_write_json(Path(path), model.to_dict())


def load_model(path: str | Path) -> AverageDelayModel:
    with Path(path).open("r", encoding="utf-8") as handle:
        return AverageDelayModel.from_dict(json.load(handle))


def write_model_metrics(path: str | Path, models: dict[str, AverageDelayModel]) -> None:
    rows = []
    for pair in PAIR_KEYS:
        model = models[pair]
        rows.append(
            {
                "side_code": SIDE_TO_CODE[pair],
                "average_delay_lsb": model.average_delay_lsb,
                "delay_std_lsb": model.delay_std_lsb,
                "robust_center_lsb": model.robust_center_lsb,
                "robust_scale_lsb": model.robust_scale_lsb,
                "input_samples": model.input_samples,
                "training_samples": model.training_samples,
                "outliers_rejected": model.outliers_rejected,
                "outlier_iterations": model.outlier_iterations,
                "outlier_z_threshold": model.outlier_z_threshold,
            }
        )
    write_table(path, MODEL_METRIC_FIELDS, rows)


def write_filtered_training_csv(
    path: str | Path,
    samples: dict[str, MatchingSamples],
    models: dict[str, AverageDelayModel],
    diagnostic_mode: str = "compact",
) -> None:
    # Kept only for external callers. The main pipeline no longer stores this
    # redundant second event table.
    rows = _training_rows_from_samples(samples)
    for row in rows:
        pair = side_name(row["side_code"])
        row["energy_channel"] = models[pair].energy_channel
        row["timing_channel"] = models[pair].timing_channel
    write_training_csv(path, rows, diagnostic_mode)


def _compact_total_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        event_index = int(row["event_index"])
        pair = side_name(row.get("side_code", row.get("pair", 0)))
        grouped.setdefault(event_index, {})[pair] = row
    compact: list[dict[str, Any]] = []
    for event_index in sorted(grouped):
        pair_rows = grouped[event_index]
        row_a = pair_rows.get("a", {})
        row_b = pair_rows.get("b", {})
        statuses = [status_name(item.get("status_code", item.get("status", "unknown"))) for item in pair_rows.values()]
        event_accepted = bool(pair_rows) and all(
            int(item.get("event_accepted", item.get("accepted", 0))) == 1
            or status_name(item.get("status_code", item.get("status", "unknown"))) == "accepted"
            for item in pair_rows.values()
        ) and len(pair_rows) == 2
        if event_accepted:
            event_status = "accepted"
        else:
            event_status = next((status for status in statuses if status != "accepted"), "unknown")
        compact.append(
            {
                "event_index": event_index,
                "duration_a_lsb": row_a.get("energy_duration_lsb", ""),
                "duration_b_lsb": row_b.get("energy_duration_lsb", ""),
                "average_delay_a_lsb": row_a.get("average_delay_lsb", row_a.get("predicted_delay_lsb", "")),
                "average_delay_b_lsb": row_b.get("average_delay_lsb", row_b.get("predicted_delay_lsb", "")),
                "selected_delay_a_lsb": row_a.get("selected_delay_lsb", ""),
                "selected_delay_b_lsb": row_b.get("selected_delay_lsb", ""),
                "baseline_delay_a_lsb": row_a.get("baseline_delay_lsb", ""),
                "baseline_delay_b_lsb": row_b.get("baseline_delay_lsb", ""),
                "deviation_a_lsb": row_a.get("deviation_lsb", row_a.get("prediction_error_lsb", "")),
                "deviation_b_lsb": row_b.get("deviation_lsb", row_b.get("prediction_error_lsb", "")),
                "maximum_deviation_lsb": row_a.get("maximum_deviation_lsb", row_b.get("maximum_deviation_lsb", "")),
                "candidate_count": max(int(row_a.get("candidate_count", 0) or 0), int(row_b.get("candidate_count", 0) or 0)),
                "candidate_changed": int(
                    bool(int(row_a.get("candidate_changed", row_a.get("model_changed_candidate", 0)) or 0))
                    or bool(int(row_b.get("candidate_changed", row_b.get("model_changed_candidate", 0)) or 0))
                ),
                "status_code": STATUS_TO_CODE[event_status],
            }
        )
    return compact


def _summary_total_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = _compact_total_rows(rows)
    accepted = [row for row in compact if int(row["status_code"]) == STATUS_TO_CODE["accepted"]]
    changed = [row for row in compact if int(row["candidate_changed"]) == 1]
    def mean_field(name: str) -> float | str:
        values = [float(row[name]) for row in accepted if row.get(name, "") not in {"", None}]
        return float(np.mean(values)) if values else ""
    return [{
        "events": len(compact),
        "accepted_events": len(accepted),
        "rejected_events": len(compact) - len(accepted),
        "changed_events": len(changed),
        "mean_deviation_a_lsb": mean_field("deviation_a_lsb"),
        "mean_deviation_b_lsb": mean_field("deviation_b_lsb"),
    }]


def write_total_csv(
    path: str | Path,
    rows: list[dict[str, Any]],
    diagnostic_mode: str = "compact",
) -> None:
    mode = str(diagnostic_mode).lower()
    if mode == "debug":
        encoded = []
        for row in rows:
            pair = side_name(row.get("side_code", row.get("pair", 0)))
            encoded.append(
                {
                    "event_index": int(row["event_index"]),
                    "side_code": SIDE_TO_CODE[pair],
                    "energy_channel": int(row.get("energy_channel", 0)),
                    "timing_channel": int(row.get("timing_channel", 0)),
                    "energy_duration_lsb": row.get("energy_duration_lsb", ""),
                    "average_delay_lsb": row.get("average_delay_lsb", row.get("predicted_delay_lsb", "")),
                    "selected_delay_lsb": row.get("selected_delay_lsb", ""),
                    "baseline_delay_lsb": row.get("baseline_delay_lsb", ""),
                    "candidate_changed": int(row.get("candidate_changed", row.get("model_changed_candidate", 0)) or 0),
                    "deviation_lsb": row.get("deviation_lsb", row.get("prediction_error_lsb", "")),
                    "maximum_deviation_lsb": row.get("maximum_deviation_lsb", row.get("maximum_prediction_error_lsb", "")),
                    "candidate_count": int(row.get("candidate_count", 0) or 0),
                    "event_accepted": int(row.get("event_accepted", row.get("accepted", 0)) or 0),
                    "status_code": status_code(row.get("status_code", row.get("status", "unknown"))),
                }
            )
        write_table(path, TOTAL_DEBUG_FIELDS, encoded)
        return
    if mode == "summary":
        write_table(path, TOTAL_SUMMARY_FIELDS, _summary_total_rows(rows))
        return
    write_table(path, TOTAL_COMPACT_FIELDS, _compact_total_rows(rows))


def _decode_compact_total_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    decoded: list[dict[str, Any]] = []
    for compact in rows:
        accepted = int(status_code(compact.get("status_code"))) == STATUS_TO_CODE["accepted"]
        for pair in PAIR_KEYS:
            suffix = pair
            decoded.append(
                {
                    "event_index": compact["event_index"],
                    "pair": pair,
                    "side_code": SIDE_TO_CODE[pair],
                    "energy_duration_lsb": compact.get(f"duration_{suffix}_lsb", ""),
                    "average_delay_lsb": compact.get(f"average_delay_{suffix}_lsb", ""),
                    "predicted_delay_lsb": compact.get(f"average_delay_{suffix}_lsb", ""),
                    "selected_delay_lsb": compact.get(f"selected_delay_{suffix}_lsb", ""),
                    "baseline_delay_lsb": compact.get(f"baseline_delay_{suffix}_lsb", ""),
                    "candidate_changed": compact.get("candidate_changed", 0),
                    "model_changed_candidate": compact.get("candidate_changed", 0),
                    "deviation_lsb": compact.get(f"deviation_{suffix}_lsb", ""),
                    "prediction_error_lsb": compact.get(f"deviation_{suffix}_lsb", ""),
                    "maximum_deviation_lsb": compact.get("maximum_deviation_lsb", ""),
                    "candidate_count": compact.get("candidate_count", 0),
                    "accepted": int(accepted),
                    "event_accepted": int(accepted),
                    "status": status_name(compact.get("status_code")),
                    "status_code": status_code(compact.get("status_code")),
                }
            )
    return decoded


def write_total_cache(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write the minimal per-event matching state as a compressed numeric NPZ.

    This cache is intentionally separate from user-facing diagnostics. It lets
    ``diagnostic_mode=summary`` retain all state required by later cached stages
    without storing a verbose event CSV.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    compact = _compact_total_rows(rows)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")

    def float_array(field: str) -> np.ndarray:
        values = []
        for row in compact:
            value = row.get(field, "")
            try:
                values.append(float(value) if value not in {"", None} else math.nan)
            except (TypeError, ValueError):
                values.append(math.nan)
        return np.asarray(values, dtype=np.float64)

    payload: dict[str, np.ndarray] = {
        "event_index": np.asarray([int(row["event_index"]) for row in compact], dtype=np.int64),
        "candidate_count": np.asarray([int(row.get("candidate_count", 0) or 0) for row in compact], dtype=np.int32),
        "candidate_changed": np.asarray([int(row.get("candidate_changed", 0) or 0) for row in compact], dtype=np.uint8),
        "status_code": np.asarray([status_code(row.get("status_code")) for row in compact], dtype=np.uint8),
    }
    for field in TOTAL_COMPACT_FIELDS:
        if field not in payload and field not in {"event_index", "candidate_count", "candidate_changed", "status_code"}:
            payload[field] = float_array(field)
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def load_total_cache(path: str | Path) -> list[dict[str, Any]]:
    """Load the compressed matching cache and decode it to compatibility rows."""
    with np.load(Path(path), allow_pickle=False) as data:
        size = int(data["event_index"].size)
        compact: list[dict[str, Any]] = []
        for index in range(size):
            row: dict[str, Any] = {
                "event_index": int(data["event_index"][index]),
                "candidate_count": int(data["candidate_count"][index]),
                "candidate_changed": int(data["candidate_changed"][index]),
                "status_code": int(data["status_code"][index]),
            }
            for field in TOTAL_COMPACT_FIELDS:
                if field in row or field not in data:
                    continue
                value = float(data[field][index])
                row[field] = "" if math.isnan(value) else value
            compact.append(row)
    return _decode_compact_total_rows(compact)


def load_total_csv(path: str | Path) -> list[dict[str, Any]]:
    rows = read_table(path)
    if not rows:
        return []
    first = rows[0]
    if "events" in first and "accepted_events" in first:
        return []
    if "selected_delay_a_lsb" in first:
        return _decode_compact_total_rows(rows)
    decoded = []
    for row in rows:
        pair = side_name(row.get("side_code", row.get("pair", 0)))
        item = dict(row)
        item["pair"] = pair
        item["side_code"] = SIDE_TO_CODE[pair]
        item["status"] = status_name(row.get("status_code", row.get("status", "unknown")))
        item["status_code"] = status_code(row.get("status_code", row.get("status", "unknown")))
        item["accepted"] = int(row.get("accepted", row.get("event_accepted", item["status"] == "accepted")) or 0)
        item["event_accepted"] = int(row.get("event_accepted", item["accepted"]) or 0)
        item["model_changed_candidate"] = int(row.get("model_changed_candidate", row.get("candidate_changed", 0)) or 0)
        item["candidate_changed"] = item["model_changed_candidate"]
        item["predicted_delay_lsb"] = row.get("predicted_delay_lsb", row.get("average_delay_lsb", ""))
        item["prediction_error_lsb"] = row.get("prediction_error_lsb", row.get("deviation_lsb", ""))
        decoded.append(item)
    return decoded


def scan_streaming_matching_training(
    pulse_cache_dir: str | Path,
    candidate_index_dir: str | Path,
    cfg: dict,
    selected_event_indices: set[int],
) -> tuple[dict[str, MatchingSamples], dict[str, Any]]:
    """Build average-delay calibration samples from the compact STREAMING cache.

    Timing candidates are evaluated independently for each detector side.  A
    side contributes a calibration sample only when exactly one preceding
    timing leading edge lies inside the configured training window.
    """
    del candidate_index_dir
    from .streaming_cache import load_streaming_event_cache

    arrays, metadata = load_streaming_event_cache(pulse_cache_dir)
    pairs = _channel_pairs(cfg)
    toa_lsb_ps = float(metadata["toa_lsb_ps"])
    window_lsb = window_ns_to_lsb(
        float(cfg["matching_model"]["training"]["window_ns"]), toa_lsb_ps
    )
    event_count = int(arrays["energy_lead_a"].size)
    selected_sorted = sorted(
        index for index in selected_event_indices if 0 <= index < event_count
    )

    rows: dict[str, list[dict[str, int | str]]] = {key: [] for key in PAIR_KEYS}
    ambiguous_or_missing = {"a": 0, "b": 0}
    for event_index in selected_sorted:
        for side in PAIR_KEYS:
            energy_lead = int(arrays[f"energy_lead_{side}"][event_index])
            energy_trail = int(arrays[f"energy_trail_{side}"][event_index])
            offsets = arrays[f"timing_offsets_{side}"]
            candidates = arrays[f"timing_leads_{side}"][
                int(offsets[event_index]) : int(offsets[event_index + 1])
            ]
            delays = energy_lead - candidates.astype(np.int64)
            valid = candidates[(delays > 0) & (delays < window_lsb)]
            if valid.size != 1:
                ambiguous_or_missing[side] += 1
                continue
            timing_lead = int(valid[0])
            energy_channel, timing_channel = pairs[side]
            rows[side].append(
                {
                    "event_index": event_index,
                    "pair": side,
                    "energy_channel": energy_channel,
                    "timing_channel": timing_channel,
                    "energy_duration_lsb": energy_trail - energy_lead,
                    "delay_lsb": energy_lead - timing_lead,
                    "energy_leading_lsb": energy_lead,
                    "timing_leading_lsb": timing_lead,
                }
            )

    samples = {
        side: MatchingSamples(
            event_index=np.asarray(
                [int(row["event_index"]) for row in rows[side]], dtype=np.int64
            ),
            energy_duration_lsb=np.asarray(
                [int(row["energy_duration_lsb"]) for row in rows[side]],
                dtype=np.int64,
            ),
            delay_lsb=np.asarray(
                [int(row["delay_lsb"]) for row in rows[side]], dtype=np.int64
            ),
        )
        for side in PAIR_KEYS
    }
    result_metadata = {
        "events_read": event_count,
        "energy_selected_events": len(selected_sorted),
        "training_samples_a": samples["a"].size,
        "training_samples_b": samples["b"].size,
        "ambiguous_or_missing_a": ambiguous_or_missing["a"],
        "ambiguous_or_missing_b": ambiguous_or_missing["b"],
        "toa_lsb_ps": toa_lsb_ps,
        "training_window_ns": float(cfg["matching_model"]["training"]["window_ns"]),
        "training_window_lsb": window_lsb,
        "physical_time_source": "streaming_hit_toa_uint64",
        "streaming_candidate_sides_independent": True,
        "candidate_storage": "single_streaming_events_npz",
        "rows": [row for side in PAIR_KEYS for row in rows[side]],
    }
    return samples, result_metadata


def count_model_corrected_alignments(
    rows: list[dict[str, Any]],
    *,
    center_a_lsb: float,
    scale_a_lsb: float,
    center_b_lsb: float,
    scale_b_lsb: float,
    z_threshold: float,
) -> int:
    """Count events where average-delay matching repairs joint alignment.

    The baseline is the latest preceding timing candidate. This remains an
    operational data-driven comparison, not external ground truth.
    """
    grouped: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        try:
            if int(row.get("event_accepted", 0)) != 1:
                continue
            event_index = int(row["event_index"])
            pair = side_name(row.get("side_code", row.get("pair", 0)))
        except (KeyError, TypeError, ValueError, DataError):
            continue
        grouped.setdefault(event_index, {})[pair] = row

    def passes(delay: float, center: float, scale: float) -> bool:
        if not all(math.isfinite(value) for value in (delay, center, scale)) or scale <= 0:
            return False
        alignment = -delay
        return abs(alignment - center) <= z_threshold * scale

    corrected = 0
    for pair_rows in grouped.values():
        if "a" not in pair_rows or "b" not in pair_rows:
            continue
        row_a = pair_rows["a"]
        row_b = pair_rows["b"]
        try:
            changed = bool(int(row_a.get("candidate_changed", row_a.get("model_changed_candidate", 0)))) or bool(
                int(row_b.get("candidate_changed", row_b.get("model_changed_candidate", 0)))
            )
            if not changed:
                continue
            selected_a = float(row_a["selected_delay_lsb"])
            selected_b = float(row_b["selected_delay_lsb"])
            baseline_a = float(row_a["baseline_delay_lsb"])
            baseline_b = float(row_b["baseline_delay_lsb"])
        except (KeyError, TypeError, ValueError):
            continue
        selected_pass = passes(selected_a, center_a_lsb, scale_a_lsb) and passes(selected_b, center_b_lsb, scale_b_lsb)
        baseline_pass = passes(baseline_a, center_a_lsb, scale_a_lsb) and passes(baseline_b, center_b_lsb, scale_b_lsb)
        if selected_pass and not baseline_pass:
            corrected += 1
    return corrected
