from __future__ import annotations

from typing import Any

import numpy as np

from utils.photopeak import fit_photopeak, photopeak_mask


def apply_energy_preselection(
    amplitudes_mV: np.ndarray,
    noise_rms_mV: np.ndarray,
    trigger_index: np.ndarray,
    *,
    energy_channels: tuple[int, int],
    selection: dict[str, Any],
    photopeak: dict[str, Any],
    logger: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select events using raw energy-channel information only.

    This first-stage selection is deliberately independent of LED/CFD validity
    and timing channels. It discards non-photopeak events before denoising,
    timing extraction, and waveform window materialization.
    """
    amplitudes = np.asarray(amplitudes_mV, dtype=np.float64)
    noise = np.asarray(noise_rms_mV, dtype=np.float64)
    triggers = np.asarray(trigger_index, dtype=np.int64)
    if amplitudes.ndim != 2 or amplitudes.shape[1] != 2:
        raise ValueError("Energy preselection amplitudes must have shape [event, 2]")
    if noise.shape != amplitudes.shape or triggers.shape != amplitudes.shape:
        raise ValueError("Energy preselection feature arrays must have matching shapes")

    valid = (
        np.all(np.isfinite(amplitudes), axis=1)
        & np.all(np.isfinite(noise), axis=1)
        & np.all(triggers >= 0, axis=1)
    )
    summary: dict[str, Any] = {
        "stage": "raw_energy_first_pass_before_timing_preprocessing",
        "source_signal_variant": "raw_energy",
        "scanned_events": int(amplitudes.shape[0]),
        "valid_basic_energy_features": int(np.count_nonzero(valid)),
    }

    trigger_range = selection.get("energy_trigger_index_range")
    if trigger_range is not None:
        low, high = int(trigger_range[0]), int(trigger_range[1])
        valid &= np.all((triggers > low) & (triggers < high), axis=1)
        summary["energy_trigger_index_range"] = [low, high]

    noise_limit = selection.get("energy_noise_max_mV")
    if noise_limit is not None:
        if isinstance(noise_limit, (list, tuple)):
            limits = np.asarray(noise_limit, dtype=np.float64).reshape(-1)
            if limits.size != 2:
                raise ValueError("preprocessing.selection.energy_noise_max_mV must be scalar or length 2")
        else:
            limits = np.asarray([float(noise_limit), float(noise_limit)], dtype=np.float64)
        valid &= (noise[:, 0] < limits[0]) & (noise[:, 1] < limits[1])
        summary["energy_noise_max_mV"] = limits.tolist()

    summary["eligible_before_photopeak"] = int(np.count_nonzero(valid))
    photopeak_rows: list[dict[str, Any]] = []
    if bool(photopeak.get("enabled", False)):
        fit_indices = np.flatnonzero(valid)
        if fit_indices.size < 20:
            raise RuntimeError(
                f"Too few energy-selected events ({fit_indices.size}) for photopeak fitting"
            )
        for position, channel_number in enumerate(energy_channels):
            result = fit_photopeak(
                amplitudes[fit_indices, position],
                channel=int(channel_number),
                config=photopeak,
            )
            if not result.success:
                raise RuntimeError(
                    f"Photopeak fit failed for energy channel {channel_number}: {result.message}"
                )
            valid &= photopeak_mask(amplitudes[:, position], result)
            photopeak_rows.append(result.as_dict())
    summary["photopeak"] = photopeak_rows
    summary["photopeak_enabled"] = bool(photopeak.get("enabled", False))
    summary["selected_events"] = int(np.count_nonzero(valid))

    minimum = int(selection.get("minimum_events", selection.get("minimum_events_per_split", 100)))
    if int(np.count_nonzero(valid)) < minimum:
        raise RuntimeError(
            f"Only {int(np.count_nonzero(valid))} events remain after raw-energy/photopeak "
            f"preselection; need at least {minimum}"
        )
    logger.info(
        "Energy/photopeak preselection | retained=%d/%d | expensive timing preprocessing only on retained events",
        int(np.count_nonzero(valid)),
        int(valid.size),
    )
    return valid, summary
