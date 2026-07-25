"""Energy-channel-only machine-learning pipeline for CTR correction.

This package is intentionally separate from the existing classical waveform
analysis pipeline.  It reads only the configured energy-channel branches.
"""

__all__ = ["config", "data", "model", "training", "evaluation"]
