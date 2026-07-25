from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Iterable

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

from .config import config_copy, load_config
from .io import decode_voltage_mV, get_event, iterate_chunks
from .pipeline import build_selection, extract_features, load_features, save_features
from .signal import baseline_and_basic_features

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BaseEventInfo:
    entry: int
    event_id: int
    source_file_id: int
    t3_ns: float
    t4_ns: float
    led_delta_ps: float


@dataclass(frozen=True)
class DatasetSummary:
    selected_before_waveform_validation: int
    valid_crossings_and_crop: int
    mad_rejected: int
    retained_base_events: int
    train_base_events: int
    validation_base_events: int
    test_base_events: int
    discrete_positions: list[int]
    uniform_positions: list[int]
    crop_samples: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_before_waveform_validation": self.selected_before_waveform_validation,
            "valid_crossings_and_crop": self.valid_crossings_and_crop,
            "mad_rejected": self.mad_rejected,
            "retained_base_events": self.retained_base_events,
            "train_base_events": self.train_base_events,
            "validation_base_events": self.validation_base_events,
            "test_base_events": self.test_base_events,
            "discrete_positions": self.discrete_positions,
            "uniform_positions": self.uniform_positions,
            "crop_samples": self.crop_samples,
        }


from .cnn_preprocessing import (
    direct_pair_crop,
    first_rising_crossing_ns,
    invariant_pair_crop,
    relative_grid_ns,
    prepare_signal_sampler,
)

def _source_id(value: np.ndarray) -> int:
    flat = np.asarray(value).reshape(-1)
    return int(flat[0]) if flat.size else -1


def _decode_timing_pair(
    chunk: Any,
    row: int,
    *,
    timing_channels_zero_based: tuple[int, int],
    polarities: np.ndarray,
    baseline_samples: int,
    trigger_threshold_mV: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    decoded: list[tuple[np.ndarray, np.ndarray]] = []
    for channel in timing_channels_zero_based:
        raw = np.asarray(ak.to_numpy(chunk.samples[channel][row]), dtype=np.int16)
        voltage = decode_voltage_mV(
            raw,
            float(chunk.vertical_gain_v_per_count[row, channel]),
            float(chunk.vertical_offset_v[row, channel]),
        )
        basic = baseline_and_basic_features(
            voltage,
            baseline_samples=baseline_samples,
            polarity=int(polarities[channel]),
            trigger_threshold_mV=trigger_threshold_mV,
            horizontal_interval_s=float(chunk.horizontal_interval_s[row, channel]),
            horizontal_offset_s=float(chunk.horizontal_offset_s[row, channel]),
        )
        time_ns = (
            float(chunk.horizontal_offset_s[row, channel])
            + np.arange(voltage.size, dtype=np.float64)
            * float(chunk.horizontal_interval_s[row, channel])
        ) * 1.0e9
        decoded.append((time_ns, basic.corrected_signal_mV))
    return decoded[0][0], decoded[0][1], decoded[1][0], decoded[1][1]


def _crop_bounds_valid(
    *,
    time3_min: float,
    time3_max: float,
    time4_min: float,
    time4_max: float,
    t3: float,
    t4: float,
    shifts_ps: Iterable[int],
    shifted_timing_channel: int,
    relative_grid: np.ndarray,
) -> bool:
    # Invariant local crops.
    if t3 + relative_grid[0] < time3_min or t3 + relative_grid[-1] > time3_max:
        return False
    if t4 + relative_grid[0] < time4_min or t4 + relative_grid[-1] > time4_max:
        return False
    for shift_ps in shifts_ps:
        shift_ns = float(shift_ps) / 1000.0
        shifted_t3 = t3 + (shift_ns if shifted_timing_channel == 3 else 0.0)
        shifted_t4 = t4 + (shift_ns if shifted_timing_channel == 4 else 0.0)
        center = 0.5 * (shifted_t3 + shifted_t4)
        output_low = center + relative_grid[0]
        output_high = center + relative_grid[-1]
        query3_low = output_low - (shift_ns if shifted_timing_channel == 3 else 0.0)
        query3_high = output_high - (shift_ns if shifted_timing_channel == 3 else 0.0)
        query4_low = output_low - (shift_ns if shifted_timing_channel == 4 else 0.0)
        query4_high = output_high - (shift_ns if shifted_timing_channel == 4 else 0.0)
        if query3_low < time3_min or query3_high > time3_max:
            return False
        if query4_low < time4_min or query4_high > time4_max:
            return False
    return True


def _mad_mask(values: np.ndarray, threshold: float | None) -> tuple[np.ndarray, dict[str, float]]:
    data = np.asarray(values, dtype=np.float64)
    if threshold is None:
        return np.ones(data.size, dtype=bool), {
            "median_ps": float(np.median(data)),
            "mad_ps": float(np.median(np.abs(data - np.median(data)))),
            "robust_scale_ps": np.nan,
            "threshold": np.nan,
        }
    median = float(np.median(data))
    mad = float(np.median(np.abs(data - median)))
    robust_scale = 1.4826 * mad
    if not np.isfinite(robust_scale) or robust_scale <= 0:
        return np.ones(data.size, dtype=bool), {
            "median_ps": median,
            "mad_ps": mad,
            "robust_scale_ps": robust_scale,
            "threshold": float(threshold),
        }
    distance = np.abs(data - median) / robust_scale
    return distance <= float(threshold), {
        "median_ps": median,
        "mad_ps": mad,
        "robust_scale_ps": robust_scale,
        "threshold": float(threshold),
        "max_distance": float(np.max(distance)),
        "worst_index": int(np.argmax(distance)),
    }


def _plot_largest_mad_outlier(
    input_path: Path,
    event_entry: int,
    output_path: Path,
    *,
    baseline_samples: int,
    polarities: np.ndarray,
    led_delta_ps: float,
    mad_distance: float,
    dpi: int = 180,
) -> None:
    event = get_event(input_path, event_entry)
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for channel, (time_ns, voltage_mV) in enumerate(event["waveforms"]):
        count = min(int(baseline_samples), voltage_mV.size)
        baseline = float(np.mean(voltage_mV[:count]))
        corrected = int(polarities[channel]) * (voltage_mV - baseline)
        axes[channel].plot(time_ns, corrected, linewidth=1.0)
        axes[channel].set_ylabel(f"ch{channel + 1} [mV]")
        axes[channel].grid(alpha=0.25)
    axes[-1].set_xlabel("Time [ns]")
    fig.suptitle(
        "Largest LED MAD outlier — "
        f"entry={event_entry}, LED Δt={led_delta_ps:.1f} ps, MAD distance={mad_distance:.2f}"
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _selection_mask(
    input_path: Path,
    analysis_config_path: Path,
    cache_dir: Path,
    *,
    reuse: bool,
    max_events: int,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    analysis_config = config_copy(load_config(analysis_config_path))
    if max_events > 0:
        analysis_config["io"]["max_events"] = int(max_events)
    cache_path = cache_dir / "cnn_selection_features.npz"
    if reuse and cache_path.is_file():
        try:
            features = load_features(cache_path, analysis_config, input_path)
            LOGGER.info("Loaded selection feature cache: %s", cache_path)
        except Exception as exc:
            LOGGER.warning("Selection cache rejected (%s); regenerating", exc)
            features = extract_features(input_path, analysis_config)
            save_features(cache_path, features)
    else:
        features = extract_features(input_path, analysis_config)
        save_features(cache_path, features)
        LOGGER.info("Selection feature cache written: %s", cache_path)
    selection = build_selection(features, analysis_config)
    return selection.selected, analysis_config, selection.cutflow


def build_cnn_dataset_cache(
    *,
    input_path: Path,
    analysis_config_path: Path,
    preprocessing_config: dict[str, Any],
    cache_dir: Path,
    max_events: int = 0,
    reuse_selection_cache: bool = True,
) -> tuple[Path, DatasetSummary]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = cache_dir / str(preprocessing_config["cache"]["filename"])
    selected, analysis_config, selection_cutflow = _selection_mask(
        input_path,
        analysis_config_path,
        cache_dir,
        reuse=reuse_selection_cache,
        max_events=max_events,
    )

    crop_config = preprocessing_config["crop"]
    augmentation = preprocessing_config["augmentation"]
    split_config = preprocessing_config["split"]
    selection_config = preprocessing_config["selection"]
    relative_grid = relative_grid_ns(
        float(crop_config["width_ns"]),
        float(crop_config["resample_step_ps"]),
    )
    discrete_shifts = sorted(int(item) for item in augmentation["discrete_shifts_ps"])
    uniform_cfg = augmentation["uniform_test"]
    uniform_shifts = list(
        range(
            int(uniform_cfg["min_ps"]),
            int(uniform_cfg["max_ps"]) + 1,
            int(uniform_cfg["step_ps"]),
        )
    )
    all_shifts = sorted(set(discrete_shifts + uniform_shifts))
    shifted_timing_channel = int(augmentation["shifted_timing_channel"])
    timing_channels = tuple(int(item) - 1 for item in analysis_config["channels"]["timing"])
    if timing_channels != (2, 3):
        raise ValueError("CNN experiment currently expects timing channels [3, 4]")
    polarities = np.asarray(analysis_config["channels"]["polarities"], dtype=np.int8)
    baseline_samples = int(analysis_config["waveform"]["baseline_samples"])
    trigger_threshold = float(analysis_config["waveform"]["trigger_threshold_mV"])
    threshold_mV = float(crop_config["threshold_mV"])
    step_size = analysis_config["io"].get("step_size", "128 MB")
    entry_stop = max_events if max_events > 0 else None
    progress_every = max(1, int(preprocessing_config.get("progress_every", 500)))

    # Pass 1: obtain threshold crossings and identify base events valid for every
    # requested position before splitting, preventing position-dependent losses.
    base_events: list[BaseEventInfo] = []
    global_entry = 0
    selected_count = int(np.count_nonzero(selected))
    for chunk in iterate_chunks(input_path, step_size=step_size, entry_stop=entry_stop):
        for row in range(chunk.event_id.size):
            if global_entry >= selected.size:
                break
            if selected[global_entry]:
                time3, signal3, time4, signal4 = _decode_timing_pair(
                    chunk,
                    row,
                    timing_channels_zero_based=timing_channels,
                    polarities=polarities,
                    baseline_samples=baseline_samples,
                    trigger_threshold_mV=trigger_threshold,
                )
                t3 = first_rising_crossing_ns(time3, signal3, threshold_mV)
                t4 = first_rising_crossing_ns(time4, signal4, threshold_mV)
                if np.isfinite(t3) and np.isfinite(t4) and _crop_bounds_valid(
                    time3_min=float(time3[0]),
                    time3_max=float(time3[-1]),
                    time4_min=float(time4[0]),
                    time4_max=float(time4[-1]),
                    t3=t3,
                    t4=t4,
                    shifts_ps=all_shifts,
                    shifted_timing_channel=shifted_timing_channel,
                    relative_grid=relative_grid,
                ):
                    base_events.append(
                        BaseEventInfo(
                            entry=global_entry,
                            event_id=int(chunk.event_id[row]),
                            source_file_id=_source_id(chunk.source_file_id[row]),
                            t3_ns=t3,
                            t4_ns=t4,
                            led_delta_ps=(t3 - t4) * 1000.0,
                        )
                    )
            global_entry += 1
            if global_entry % progress_every == 0:
                LOGGER.info(
                    "CNN preprocessing pass 1: %d events scanned; valid=%d",
                    global_entry,
                    len(base_events),
                )

    if len(base_events) < 30:
        raise RuntimeError(f"Only {len(base_events)} valid base events remain")

    led_values = np.asarray([item.led_delta_ps for item in base_events], dtype=np.float64)
    mad_threshold = selection_config.get("led_mad_threshold")
    mad_keep, mad_summary = _mad_mask(
        led_values,
        None if mad_threshold is None else float(mad_threshold),
    )
    mad_rejected = int(np.count_nonzero(~mad_keep))
    if mad_threshold is not None and mad_summary.get("robust_scale_ps", 0) > 0:
        distances = np.abs(led_values - mad_summary["median_ps"]) / mad_summary["robust_scale_ps"]
        worst = int(np.argmax(distances))
        plot_name = str(selection_config.get("largest_outlier_plot", "largest_led_mad_outlier.png"))
        _plot_largest_mad_outlier(
            input_path,
            base_events[worst].entry,
            cache_dir / plot_name,
            baseline_samples=baseline_samples,
            polarities=polarities,
            led_delta_ps=base_events[worst].led_delta_ps,
            mad_distance=float(distances[worst]),
            dpi=int(preprocessing_config.get("plot_dpi", 180)),
        )
    base_events = [item for item, keep in zip(base_events, mad_keep, strict=True) if keep]
    LOGGER.info(
        "LED MAD filter: retained %d/%d valid base events; rejected=%d",
        len(base_events),
        len(mad_keep),
        mad_rejected,
    )

    rng = np.random.default_rng(int(split_config["random_seed"]))
    order = rng.permutation(len(base_events))
    n_train = int(round(len(order) * float(split_config["train_fraction"])))
    n_validation = int(round(len(order) * float(split_config["validation_fraction"])))
    n_train = min(max(1, n_train), len(order) - 2)
    n_validation = min(max(1, n_validation), len(order) - n_train - 1)
    train_entries = {base_events[int(index)].entry for index in order[:n_train]}
    validation_entries = {
        base_events[int(index)].entry for index in order[n_train : n_train + n_validation]
    }
    test_entries = {
        base_events[int(index)].entry for index in order[n_train + n_validation :]
    }
    valid_by_entry = {item.entry: item for item in base_events}

    samples = relative_grid.size
    counts = {
        "train": len(train_entries),
        "validation": len(validation_entries),
        "test": len(test_entries),
    }

    def allocate_direct(base_count: int, positions: list[int]) -> dict[str, np.ndarray]:
        total = base_count * len(positions)
        return {
            "x": np.empty((total, 2, samples), dtype=np.float32),
            "y": np.empty(total, dtype=np.float32),
            "base_entry": np.empty(total, dtype=np.int64),
            "event_id": np.empty(total, dtype=np.int64),
        }

    direct_train = allocate_direct(counts["train"], discrete_shifts)
    direct_validation = allocate_direct(counts["validation"], discrete_shifts)
    direct_test_discrete = allocate_direct(counts["test"], discrete_shifts)
    direct_test_uniform = allocate_direct(counts["test"], uniform_shifts)

    def allocate_correction(base_count: int) -> dict[str, np.ndarray]:
        return {
            "x": np.empty((base_count, 2, samples), dtype=np.float32),
            "led_delta_ps": np.empty(base_count, dtype=np.float32),
            "base_entry": np.empty(base_count, dtype=np.int64),
            "event_id": np.empty(base_count, dtype=np.int64),
        }

    correction_train = allocate_correction(counts["train"])
    correction_validation = allocate_correction(counts["validation"])
    correction_test = allocate_correction(counts["test"])
    direct_counters = {"train": 0, "validation": 0, "test_discrete": 0, "test_uniform": 0}
    correction_counters = {"train": 0, "validation": 0, "test": 0}
    interpolation = str(crop_config.get("interpolation", "cubic"))

    # Pass 2: waveform translation first, then paper-like crop generation.
    global_entry = 0
    for chunk in iterate_chunks(input_path, step_size=step_size, entry_stop=entry_stop):
        for row in range(chunk.event_id.size):
            info = valid_by_entry.get(global_entry)
            if info is not None:
                time3, signal3, time4, signal4 = _decode_timing_pair(
                    chunk,
                    row,
                    timing_channels_zero_based=timing_channels,
                    polarities=polarities,
                    baseline_samples=baseline_samples,
                    trigger_threshold_mV=trigger_threshold,
                )
                sampler3 = prepare_signal_sampler(
                    time3, signal3, interpolation=interpolation
                )
                sampler4 = prepare_signal_sampler(
                    time4, signal4, interpolation=interpolation
                )
                invariant = invariant_pair_crop(
                    time3,
                    signal3,
                    time4,
                    signal4,
                    t3_cross_ns=info.t3_ns,
                    t4_cross_ns=info.t4_ns,
                    relative_grid=relative_grid,
                    interpolation=interpolation,
                    sampler3=sampler3,
                    sampler4=sampler4,
                )
                if invariant is None:
                    raise RuntimeError(f"Invariant crop unexpectedly failed for entry {global_entry}")

                if global_entry in train_entries:
                    split_name = "train"
                    direct_target = direct_train
                    positions = discrete_shifts
                    correction_target = correction_train
                elif global_entry in validation_entries:
                    split_name = "validation"
                    direct_target = direct_validation
                    positions = discrete_shifts
                    correction_target = correction_validation
                else:
                    split_name = "test"
                    direct_target = direct_test_discrete
                    positions = discrete_shifts
                    correction_target = correction_test

                correction_index = correction_counters[split_name]
                correction_target["x"][correction_index] = invariant
                correction_target["led_delta_ps"][correction_index] = info.led_delta_ps
                correction_target["base_entry"][correction_index] = global_entry
                correction_target["event_id"][correction_index] = info.event_id
                correction_counters[split_name] += 1

                direct_key = "test_discrete" if split_name == "test" else split_name
                for shift in positions:
                    crop = direct_pair_crop(
                        time3,
                        signal3,
                        time4,
                        signal4,
                        t3_cross_ns=info.t3_ns,
                        t4_cross_ns=info.t4_ns,
                        shift_ps=shift,
                        shifted_timing_channel=shifted_timing_channel,
                        relative_grid=relative_grid,
                        interpolation=interpolation,
                        sampler3=sampler3,
                        sampler4=sampler4,
                    )
                    if crop is None:
                        raise RuntimeError(
                            f"Direct crop unexpectedly failed for entry {global_entry}, shift {shift} ps"
                        )
                    index = direct_counters[direct_key]
                    direct_target["x"][index] = crop
                    direct_target["y"][index] = shift
                    direct_target["base_entry"][index] = global_entry
                    direct_target["event_id"][index] = info.event_id
                    direct_counters[direct_key] += 1

                if split_name == "test":
                    for shift in uniform_shifts:
                        crop = direct_pair_crop(
                            time3,
                            signal3,
                            time4,
                            signal4,
                            t3_cross_ns=info.t3_ns,
                            t4_cross_ns=info.t4_ns,
                            shift_ps=shift,
                            shifted_timing_channel=shifted_timing_channel,
                            relative_grid=relative_grid,
                            interpolation=interpolation,
                            sampler3=sampler3,
                            sampler4=sampler4,
                        )
                        if crop is None:
                            raise RuntimeError(
                                f"Uniform crop unexpectedly failed for entry {global_entry}, shift {shift} ps"
                            )
                        index = direct_counters["test_uniform"]
                        direct_test_uniform["x"][index] = crop
                        direct_test_uniform["y"][index] = shift
                        direct_test_uniform["base_entry"][index] = global_entry
                        direct_test_uniform["event_id"][index] = info.event_id
                        direct_counters["test_uniform"] += 1
            global_entry += 1
            if global_entry % progress_every == 0:
                LOGGER.info(
                    "CNN preprocessing pass 2: %d events scanned; train crops=%d; test uniform crops=%d",
                    global_entry,
                    direct_counters["train"],
                    direct_counters["test_uniform"],
                )

    expected_direct = {
        "train": direct_train["x"].shape[0],
        "validation": direct_validation["x"].shape[0],
        "test_discrete": direct_test_discrete["x"].shape[0],
        "test_uniform": direct_test_uniform["x"].shape[0],
    }
    if direct_counters != expected_direct:
        raise RuntimeError(f"direct dataset counters do not match allocations: {direct_counters} vs {expected_direct}")
    expected_correction = {
        "train": correction_train["x"].shape[0],
        "validation": correction_validation["x"].shape[0],
        "test": correction_test["x"].shape[0],
    }
    if correction_counters != expected_correction:
        raise RuntimeError(
            f"correction dataset counters do not match allocations: {correction_counters} vs {expected_correction}"
        )

    summary = DatasetSummary(
        selected_before_waveform_validation=selected_count,
        valid_crossings_and_crop=len(mad_keep),
        mad_rejected=mad_rejected,
        retained_base_events=len(base_events),
        train_base_events=counts["train"],
        validation_base_events=counts["validation"],
        test_base_events=counts["test"],
        discrete_positions=discrete_shifts,
        uniform_positions=uniform_shifts,
        crop_samples=samples,
    )
    metadata = {
        "summary": summary.as_dict(),
        "selection_cutflow": selection_cutflow,
        "mad": mad_summary,
        "input_root": str(input_path.resolve()),
        "analysis_config": str(analysis_config_path.resolve()),
        "preprocessing_config": preprocessing_config,
        "relative_time_ns": relative_grid.tolist(),
    }
    arrays: dict[str, np.ndarray] = {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        "relative_time_ns": relative_grid.astype(np.float32),
        "discrete_positions_ps": np.asarray(discrete_shifts, dtype=np.int32),
        "uniform_positions_ps": np.asarray(uniform_shifts, dtype=np.int32),
    }
    for prefix, block in (
        ("direct_train", direct_train),
        ("direct_validation", direct_validation),
        ("direct_test_discrete", direct_test_discrete),
        ("direct_test_uniform", direct_test_uniform),
        ("correction_train", correction_train),
        ("correction_validation", correction_validation),
        ("correction_test", correction_test),
    ):
        for key, value in block.items():
            arrays[f"{prefix}_{key}"] = value

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    if bool(preprocessing_config["cache"].get("compressed", True)):
        np.savez_compressed(dataset_path, **arrays)
    else:
        np.savez(dataset_path, **arrays)
    with (cache_dir / "cnn_dataset_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    LOGGER.info("CNN dataset cache written: %s", dataset_path)
    return dataset_path, summary

