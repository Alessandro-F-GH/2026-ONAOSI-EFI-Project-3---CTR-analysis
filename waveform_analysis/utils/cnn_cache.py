from __future__ import annotations

from pathlib import Path

import numpy as np


def load_cnn_dataset_cache(path: str | Path) -> dict[str, np.ndarray]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}
