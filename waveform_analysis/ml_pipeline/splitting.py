from __future__ import annotations

import numpy as np


def contiguous_block_split(
    n_events: int,
    fractions: tuple[float, float, float],
    guard_gap_events: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronological split with unused guard gaps at both boundaries."""
    gap = int(guard_gap_events)
    if gap < 0:
        raise ValueError("guard_gap_events must be non-negative")
    assigned = int(n_events) - 2 * gap
    if assigned < 3:
        raise ValueError(
            "Not enough events for contiguous blocks after removing guard gaps"
        )
    n_train = int(np.floor(fractions[0] * assigned))
    n_validation = int(np.floor(fractions[1] * assigned))
    n_test = assigned - n_train - n_validation
    if min(n_train, n_validation, n_test) <= 0:
        raise ValueError("Contiguous split produced an empty partition")

    train_stop = n_train
    validation_start = train_stop + gap
    validation_stop = validation_start + n_validation
    test_start = validation_stop + gap
    test_stop = test_start + n_test
    if test_stop != int(n_events):
        raise RuntimeError("Internal contiguous split accounting error")
    return (
        np.arange(0, train_stop, dtype=np.int64),
        np.arange(validation_start, validation_stop, dtype=np.int64),
        np.arange(test_start, test_stop, dtype=np.int64),
    )


def contiguous_development_blind_split(
    n_events: int,
    blind_fraction: float,
    guard_gap_events: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Chronological development/blind split with one unused guard gap.

    Cross-validation creates its own train/validation partitions inside the
    development block.  A preliminary validation block would therefore be
    redundant and would waste an additional guard gap.
    """

    total = int(n_events)
    gap = int(guard_gap_events)
    fraction = float(blind_fraction)
    if gap < 0:
        raise ValueError("guard_gap_events must be non-negative")
    if not 0.0 < fraction < 1.0:
        raise ValueError("blind_fraction must lie strictly between 0 and 1")

    assigned = total - gap
    if assigned < 2:
        raise ValueError(
            "Not enough events for development/blind blocks after removing the guard gap"
        )

    n_development = int(np.floor((1.0 - fraction) * assigned))
    n_blind = assigned - n_development
    if min(n_development, n_blind) <= 0:
        raise ValueError("Development/blind split produced an empty partition")

    blind_start = n_development + gap
    if blind_start + n_blind != total:
        raise RuntimeError("Internal development/blind split accounting error")

    return (
        np.arange(0, n_development, dtype=np.int64),
        np.arange(blind_start, total, dtype=np.int64),
    )
