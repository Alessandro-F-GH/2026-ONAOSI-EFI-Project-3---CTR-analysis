from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt


def apply_optional_lowpass_denoising(
    signal_mV: np.ndarray,
    *,
    horizontal_interval_s: float,
    denoising_config: dict[str, Any] | None,
) -> np.ndarray:
    """Apply optional zero-phase low-pass denoising to one waveform.

    The Butterworth filter is evaluated forward and backward. This removes the
    filter phase delay, which is essential because the same waveform is later
    used to estimate LED and CFD crossing times.
    """
    signal = np.asarray(signal_mV, dtype=np.float64)
    config = denoising_config or {}
    if not bool(config.get("enabled", False)):
        return signal

    method = str(config.get("method", "butterworth_lowpass"))
    if method != "butterworth_lowpass":
        raise ValueError(f"Unsupported waveform denoising method: {method}")

    interval_s = float(horizontal_interval_s)
    if not np.isfinite(interval_s) or interval_s <= 0.0:
        raise ValueError("horizontal_interval_s must be positive for denoising")

    cutoff_hz = float(config["cutoff_GHz"]) * 1.0e9
    sampling_frequency_hz = 1.0 / interval_s
    nyquist_hz = 0.5 * sampling_frequency_hz
    if not 0.0 < cutoff_hz < nyquist_hz:
        raise ValueError(
            "waveform.denoising.cutoff_GHz must be below the Nyquist frequency "
            f"({nyquist_hz / 1.0e9:.6g} GHz for this waveform)"
        )

    order = int(config.get("order", 4))
    sos = butter(
        order,
        cutoff_hz,
        btype="lowpass",
        fs=sampling_frequency_hz,
        output="sos",
    )

    # SciPy's default padding can exceed short waveform lengths. Capping it
    # preserves normal behavior for long traces while keeping short tests and
    # diagnostic traces usable.
    zero_count = min(
        int(np.count_nonzero(sos[:, 2] == 0.0)),
        int(np.count_nonzero(sos[:, 5] == 0.0)),
    )
    default_padlen = 3 * (2 * int(sos.shape[0]) + 1 - zero_count)
    padlen = min(default_padlen, max(0, signal.size - 1))
    return np.asarray(sosfiltfilt(sos, signal, padlen=padlen), dtype=np.float64)
