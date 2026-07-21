from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np

from .config import extraction_fingerprint, grid_from_config
from .io import decode_voltage_mV, iterate_chunks
from .photopeak import PhotopeakResult, fit_photopeak, photopeak_mask
from .signal import (
    INVALID_TIME_FS,
    baseline_and_basic_features,
    prepare_timing_features,
)

CACHE_FORMAT_VERSION = 4


@dataclass(frozen=True)
class SelectionResult:
    selected: np.ndarray
    photopeak_results: tuple[PhotopeakResult, PhotopeakResult]
    cutflow: dict[str, int]


def extract_features(
    input_path: Path,
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    channels = config["channels"]
    waveform_config = config["waveform"]
    io_config = config["io"]
    timing_scan = config["timing_scan"]

    energy_channels = np.asarray(channels["energy"], dtype=np.int64) - 1
    timing_channels = np.asarray(channels["timing"], dtype=np.int64) - 1
    polarities = np.asarray(channels["polarities"], dtype=np.int8)
    led_thresholds = np.asarray(
        grid_from_config(timing_scan["led_thresholds_mV"]), dtype=np.float64
    )
    cfd_fractions = np.asarray(
        grid_from_config(timing_scan["cfd_fractions"]), dtype=np.float64
    )

    event_ids: list[int] = []
    amplitudes: list[np.ndarray] = []
    baselines: list[np.ndarray] = []
    noises: list[np.ndarray] = []
    trigger_indices: list[np.ndarray] = []
    trigger_times_fs: list[np.ndarray] = []
    crop_peaks: list[np.ndarray] = []
    led_a: list[np.ndarray] = []
    led_b: list[np.ndarray] = []
    cfd_a: list[np.ndarray] = []
    cfd_b: list[np.ndarray] = []

    max_events = int(io_config.get("max_events", 0))
    entry_stop = max_events if max_events > 0 else None
    progress_every = max(1, int(io_config.get("progress_every", 500)))
    processed = 0

    for chunk in iterate_chunks(
        input_path,
        step_size=io_config.get("step_size", "128 MB"),
        entry_stop=entry_stop,
    ):
        for row in range(chunk.event_id.size):
            amplitude = np.full(4, np.nan, dtype=np.float64)
            baseline = np.full(4, np.nan, dtype=np.float64)
            noise = np.full(4, np.nan, dtype=np.float64)
            trigger_index = np.full(4, -1, dtype=np.int32)
            trigger_time = np.full(4, INVALID_TIME_FS, dtype=np.int64)
            event_crop_peaks = np.full(2, np.nan, dtype=np.float64)
            event_led: dict[int, np.ndarray] = {}
            event_cfd: dict[int, np.ndarray] = {}

            for channel in range(4):
                raw = np.asarray(ak.to_numpy(chunk.samples[channel][row]), dtype=np.int16)
                voltage_mV = decode_voltage_mV(
                    raw,
                    float(chunk.vertical_gain_v_per_count[row, channel]),
                    float(chunk.vertical_offset_v[row, channel]),
                )
                basic = baseline_and_basic_features(
                    voltage_mV,
                    baseline_samples=int(waveform_config["baseline_samples"]),
                    polarity=int(polarities[channel]),
                    trigger_threshold_mV=float(waveform_config["trigger_threshold_mV"]),
                    horizontal_interval_s=float(chunk.horizontal_interval_s[row, channel]),
                    horizontal_offset_s=float(chunk.horizontal_offset_s[row, channel]),
                )
                amplitude[channel] = basic.amplitude_mV
                baseline[channel] = basic.baseline_mV
                noise[channel] = basic.noise_rms_mV
                trigger_index[channel] = basic.trigger_index
                trigger_time[channel] = basic.trigger_time_fs

                if channel in timing_channels:
                    position = int(np.flatnonzero(timing_channels == channel)[0])
                    timing = prepare_timing_features(
                        basic.corrected_signal_mV,
                        trigger_index=basic.trigger_index,
                        horizontal_interval_s=float(chunk.horizontal_interval_s[row, channel]),
                        horizontal_offset_s=float(chunk.horizontal_offset_s[row, channel]),
                        crop_before_ns=float(waveform_config["timing_crop_ns"]["before"]),
                        crop_after_ns=float(waveform_config["timing_crop_ns"]["after"]),
                        upsample_step_ps=float(waveform_config["upsample_step_ps"]),
                        led_thresholds_mV=led_thresholds,
                        cfd_fractions=cfd_fractions,
                    )
                    event_crop_peaks[position] = timing.cropped_peak_mV
                    event_led[channel] = timing.led_times_fs
                    event_cfd[channel] = timing.cfd_times_fs

            event_ids.append(int(chunk.event_id[row]))
            amplitudes.append(amplitude)
            baselines.append(baseline)
            noises.append(noise)
            trigger_indices.append(trigger_index)
            trigger_times_fs.append(trigger_time)
            crop_peaks.append(event_crop_peaks)
            led_a.append(event_led[int(timing_channels[0])])
            led_b.append(event_led[int(timing_channels[1])])
            cfd_a.append(event_cfd[int(timing_channels[0])])
            cfd_b.append(event_cfd[int(timing_channels[1])])

            processed += 1
            if processed % progress_every == 0:
                print(f"Extracted waveform features for {processed} events")

    source_stat = input_path.stat()
    return {
        "cache_format_version": np.asarray(CACHE_FORMAT_VERSION, dtype=np.int32),
        "extraction_fingerprint": np.asarray(extraction_fingerprint(config)),
        "source_path": np.asarray(str(input_path.resolve())),
        "source_size_bytes": np.asarray(source_stat.st_size, dtype=np.int64),
        "source_mtime_ns": np.asarray(source_stat.st_mtime_ns, dtype=np.int64),
        "event_id": np.asarray(event_ids, dtype=np.int64),
        "amplitude_mV": np.asarray(amplitudes, dtype=np.float64),
        "baseline_mV": np.asarray(baselines, dtype=np.float64),
        "noise_rms_mV": np.asarray(noises, dtype=np.float64),
        "trigger_index": np.asarray(trigger_indices, dtype=np.int32),
        "trigger_time_fs": np.asarray(trigger_times_fs, dtype=np.int64),
        "timing_crop_peak_mV": np.asarray(crop_peaks, dtype=np.float64),
        "t_led_a_fs": np.asarray(led_a, dtype=np.int64),
        "t_led_b_fs": np.asarray(led_b, dtype=np.int64),
        "t_cfd_a_fs": np.asarray(cfd_a, dtype=np.int64),
        "t_cfd_b_fs": np.asarray(cfd_b, dtype=np.int64),
        "led_thresholds_mV": led_thresholds,
        "cfd_fractions": cfd_fractions,
        "energy_channels_zero_based": energy_channels,
        "timing_channels_zero_based": timing_channels,
        "polarities": polarities,
    }


def save_features(path: Path, features: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **features)


def load_features(path: Path, config: dict[str, Any], input_path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        features = {key: loaded[key] for key in loaded.files}
    required = {
        "cache_format_version",
        "extraction_fingerprint",
        "source_path",
        "source_size_bytes",
        "source_mtime_ns",
        "event_id",
        "amplitude_mV",
        "baseline_mV",
        "noise_rms_mV",
        "trigger_index",
        "trigger_time_fs",
        "timing_crop_peak_mV",
        "t_led_a_fs",
        "t_led_b_fs",
        "t_cfd_a_fs",
        "t_cfd_b_fs",
        "led_thresholds_mV",
        "cfd_fractions",
        "energy_channels_zero_based",
        "timing_channels_zero_based",
        "polarities",
    }
    missing = sorted(required.difference(features))
    if missing:
        raise ValueError(f"feature cache is incomplete; missing: {', '.join(missing)}")
    if int(features["cache_format_version"]) != CACHE_FORMAT_VERSION:
        raise ValueError("feature cache version differs; regenerate it")
    if str(features["extraction_fingerprint"].item()) != extraction_fingerprint(config):
        raise ValueError(
            "waveform/timing configuration or max_events differs from the cache; regenerate features"
        )
    source = input_path.resolve()
    stat = source.stat()
    if str(features["source_path"].item()) != str(source):
        raise ValueError("feature cache belongs to a different input ROOT file")
    if int(features["source_size_bytes"]) != stat.st_size or int(features["source_mtime_ns"]) != stat.st_mtime_ns:
        raise ValueError("input ROOT file changed after the feature cache was created")
    return features


def build_selection(features: dict[str, np.ndarray], config: dict[str, Any]) -> SelectionResult:
    amplitude = features["amplitude_mV"]
    noise = features["noise_rms_mV"]
    trigger_index = features["trigger_index"]
    energy_channels = features["energy_channels_zero_based"].astype(np.int64)
    timing_channels = features["timing_channels_zero_based"].astype(np.int64)
    total = int(amplitude.shape[0])

    photopeak_results = tuple(
        fit_photopeak(
            amplitude[:, int(channel)],
            channel=int(channel) + 1,
            config=config["photopeak"],
        )
        for channel in energy_channels
    )
    if not all(item.success for item in photopeak_results):
        messages = "; ".join(item.message for item in photopeak_results if not item.success)
        raise RuntimeError(f"photopeak fit failed: {messages}")

    mask_e0 = photopeak_mask(amplitude[:, int(energy_channels[0])], photopeak_results[0])
    mask_e1 = photopeak_mask(amplitude[:, int(energy_channels[1])], photopeak_results[1])
    photopeak = mask_e0 & mask_e1

    trigger_range = config["selection"].get("energy_trigger_index_range")
    if trigger_range is None:
        energy_trigger = np.ones(total, dtype=bool)
    else:
        low, high = float(trigger_range[0]), float(trigger_range[1])
        energy_trigger = np.ones(total, dtype=bool)
        for channel in energy_channels:
            values = trigger_index[:, int(channel)]
            energy_trigger &= (values > low) & (values < high)

    noise_limit = config["selection"].get("timing_noise_max_mV")
    if noise_limit is None:
        timing_noise = np.ones(total, dtype=bool)
    else:
        timing_noise = np.ones(total, dtype=bool)
        for channel in timing_channels:
            timing_noise &= noise[:, int(channel)] < float(noise_limit)

    if bool(config["selection"].get("require_valid_timing_trigger", True)):
        timing_trigger = np.ones(total, dtype=bool)
        for channel in timing_channels:
            timing_trigger &= trigger_index[:, int(channel)] >= 0
    else:
        timing_trigger = np.ones(total, dtype=bool)

    selected = photopeak & energy_trigger & timing_noise & timing_trigger
    cutflow = {
        "total": total,
        "photopeak_ch_a": int(np.count_nonzero(mask_e0)),
        "photopeak_ch_b": int(np.count_nonzero(mask_e1)),
        "photopeak_and": int(np.count_nonzero(photopeak)),
        "energy_trigger": int(np.count_nonzero(photopeak & energy_trigger)),
        "timing_noise": int(np.count_nonzero(photopeak & energy_trigger & timing_noise)),
        "timing_trigger": int(
            np.count_nonzero(photopeak & energy_trigger & timing_noise & timing_trigger)
        ),
        "selected": int(np.count_nonzero(selected)),
        "rejected": total - int(np.count_nonzero(selected)),
    }
    return SelectionResult(
        selected=selected,
        photopeak_results=(photopeak_results[0], photopeak_results[1]),
        cutflow=cutflow,
    )
