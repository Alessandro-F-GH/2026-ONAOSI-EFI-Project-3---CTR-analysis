from __future__ import annotations

from pathlib import Path

PATH = Path("utils/cnn_evaluation.py")

if not PATH.is_file():
    raise FileNotFoundError(
        f"{PATH} not found. Run this script from the waveform_analysis repository root."
    )

text = PATH.read_text(encoding="utf-8")

old_helper = '''def _safe_nanmean(values: list[float] | np.ndarray) -> float:\n    array = np.asarray(values, dtype=np.float64)\n    finite = array[np.isfinite(array)]\n    return float(np.mean(finite)) if finite.size else float("nan")\n'''

new_helper = '''def _safe_nanmean(values: list[float] | np.ndarray) -> float:\n    array = np.asarray(values, dtype=np.float64)\n    finite = array[np.isfinite(array)]\n    return float(np.mean(finite)) if finite.size else float("nan")\n\n\ndef _safe_nanstd(\n    values: list[float] | np.ndarray,\n    *,\n    ddof: int = 1,\n) -> float:\n    \"\"\"Return a finite-only standard deviation without NumPy warnings.\n\n    No finite values -> NaN. One finite realization with ddof=1 -> 0, because\n    there is no between-realization spread to estimate.\n    \"\"\"\n    array = np.asarray(values, dtype=np.float64)\n    finite = array[np.isfinite(array)]\n    if finite.size == 0:\n        return float("nan")\n    if finite.size <= ddof:\n        return 0.0\n    return float(np.std(finite, ddof=ddof))\n'''

old_aggregate = '''            row[f"{key}_mean"] = float(np.mean(values))\n            row[f"{key}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0\n'''

new_aggregate = '''            row[f"{key}_mean"] = _safe_nanmean(values)\n            row[f"{key}_std"] = _safe_nanstd(values)\n            row[f"{key}_valid_realizations"] = int(np.isfinite(values).sum())\n'''

old_position = '''            row[f"{key}_mean"] = _safe_nanmean(values)\n            row[f"{key}_std"] = float(np.nanstd(values, ddof=1)) if values.size > 1 else 0.0\n'''

new_position = '''            row[f"{key}_mean"] = _safe_nanmean(values)\n            row[f"{key}_std"] = _safe_nanstd(values)\n            row[f"{key}_valid_realizations"] = int(np.isfinite(values).sum())\n'''

replacements = [
    (old_helper, new_helper, "safe-statistics helper"),
    (old_aggregate, new_aggregate, "global metric aggregation"),
    (old_position, new_position, "per-position aggregation"),
]

for old, new, label in replacements:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"Could not patch {label}: expected exactly one match, found {count}. "
            "The file may already be patched or may be a different version."
        )
    text = text.replace(old, new, 1)

backup = PATH.with_suffix(".py.before_nan_patch")
if not backup.exists():
    backup.write_text(PATH.read_text(encoding="utf-8"), encoding="utf-8")

PATH.write_text(text, encoding="utf-8")
print(f"Patched: {PATH}")
print(f"Backup:  {backup}")
