from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ml_pipeline.concatenate_energy_runs import (
    _event_amplitudes,
    discover_input_files,
    extract_bias_voltage,
    validate_concatenation_config,
)


def test_event_amplitudes_are_baseline_subtracted_and_polarity_aware() -> None:
    raw_a = np.asarray([10, 10, 10, 15, 20], dtype=np.int16)
    raw_b = np.asarray([20, 20, 20, 15, 10], dtype=np.int16)
    amplitude_a, amplitude_b = _event_amplitudes(
        (
            raw_a,
            raw_b,
            (0.001, 0.001),
            (0.0, 0.0),
            (1, -1),
            3,
        )
    )
    assert amplitude_a == 10.0
    assert amplitude_b == 10.0


def test_bias_voltage_is_read_from_parent_folder() -> None:
    pattern = r"(?:^|[\\/])(?P<bias_voltage>\d+(?:\.\d+)?)V(?:-|[\\/])"
    assert extract_bias_voltage(Path("data/47V-470mV/converted.root"), pattern) == 47.0
    assert np.isnan(extract_bias_voltage(Path("data/unknown/converted.root"), pattern))


def test_discovery_is_recursive_sorted_and_excludes_output(tmp_path: Path) -> None:
    first = tmp_path / "45V" / "converted.root"
    second = tmp_path / "46V" / "converted.root"
    output = tmp_path / "combined" / "converted.root"
    for path in (first, second, output):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    found = discover_input_files(
        tmp_path,
        pattern="converted.root",
        recursive=True,
        excluded_paths=[output],
    )
    assert found == [first.resolve(), second.resolve()]


def test_example_config_validates() -> None:
    path = Path(__file__).resolve().parents[1] / "config" / "concatenate_energy_photopeak_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_concatenation_config(config)
