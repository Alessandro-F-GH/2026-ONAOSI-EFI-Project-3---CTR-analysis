from __future__ import annotations

import argparse
import shutil
from pathlib import Path


BUNDLE_ROOT = Path(__file__).resolve().parents[1]


def _replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text:
        print(f"[already] {label}: {path}")
        return
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"Cannot apply {label}: expected exactly one current-main anchor in {path}, found {count}. "
            "The repository may have changed; update/rebase the patch manually instead of guessing."
        )
    path.write_text(text.replace(old, new), encoding="utf-8")
    print(f"[patched] {label}: {path}")


def _copy(source_relative: str, repo: Path) -> None:
    source = BUNDLE_ROOT / source_relative
    destination = repo / source_relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    print(f"[copied] {source_relative}")


def patch_study_config(repo: Path) -> None:
    path = repo / "waveform_analysis/ml_pipeline/study_config.py"
    old = '''    preprocessing = cfg.setdefault("preprocessing", {})\n    preprocessing["prepared_dir"] = str(_resolve(root, preprocessing.get("prepared_dir", "processed_data/ml_prepared")))\n    preprocessing.setdefault("materialization_chunk_size", 2048)\n'''
    new = '''    preprocessing = cfg.setdefault("preprocessing", {})\n    preprocessing["prepared_dir"] = str(_resolve(root, preprocessing.get("prepared_dir", "processed_data/ml_prepared")))\n    preprocessing["selection_store_dir"] = str(\n        _resolve(root, preprocessing.get("selection_store_dir", "processed_data/selected_events"))\n    )\n    materialized = preprocessing.setdefault(\n        "materialized_window_ns", {"before": 5.0, "after": 60.0}\n    )\n    if not isinstance(materialized, dict):\n        raise MLConfigError("preprocessing.materialized_window_ns must be an object")\n    materialized_before = _finite(\n        materialized.get("before", materialized.get("before_ns", 5.0)),\n        "preprocessing.materialized_window_ns.before",\n        positive=True,\n    )\n    materialized_after = _finite(\n        materialized.get("after", materialized.get("after_ns", 60.0)),\n        "preprocessing.materialized_window_ns.after",\n        positive=True,\n    )\n    preprocessing["materialized_window_ns"] = {\n        "before": materialized_before, "after": materialized_after\n    }\n    for window in cfg["windows_ns"]:\n        if (\n            float(window["before_ns"]) > materialized_before + 1e-12\n            or float(window["after_ns"]) > materialized_after + 1e-12\n        ):\n            raise MLConfigError(\n                f"Experiment window {window['id']!r} [-{window['before_ns']}, +{window['after_ns']}] ns "\n                f"exceeds preprocessing.materialized_window_ns "\n                f"[-{materialized_before}, +{materialized_after}] ns. Increase the permanent "\n                "materialized window once instead of tying preprocessing to each window study."\n            )\n    preprocessing.setdefault("materialization_chunk_size", 2048)\n'''
    _replace_once(path, old, new, "persistent selection + canonical window config")

    old = '''    cv = cfg.setdefault("cross_validation", {})\n    cv.setdefault("n_splits", 5)\n    cv.setdefault("seed", 20260813)\n    cv.setdefault("blind_fraction", 0.2)\n    cv.setdefault("early_stop_fraction", 0.15)\n    if int(cv["n_splits"]) < 2:\n        raise MLConfigError("cross_validation.n_splits must be >= 2")\n    if not 0.0 < float(cv["blind_fraction"]) < 0.5:\n        raise MLConfigError("cross_validation.blind_fraction must be in (0, 0.5)")\n    if not 0.0 < float(cv["early_stop_fraction"]) < 0.5:\n        raise MLConfigError("cross_validation.early_stop_fraction must be in (0, 0.5)")\n'''
    new = '''    # ``validation`` is authoritative.  The legacy cross_validation block is\n    # retained as a compatibility mirror because existing training primitives\n    # still read seed/early-stop values from it.\n    cv = cfg.setdefault("cross_validation", {})\n    cv.setdefault("n_splits", 5)\n    cv.setdefault("seed", 20260813)\n    cv.setdefault("blind_fraction", 0.2)\n    cv.setdefault("early_stop_fraction", 0.15)\n\n    validation = cfg.setdefault("validation", {})\n    if not isinstance(validation, dict):\n        raise MLConfigError("validation must be an object")\n    validation.setdefault("strategy", "cv")\n    strategy = str(validation["strategy"]).strip().lower()\n    if strategy not in {"holdout", "cv", "nested"}:\n        raise MLConfigError("validation.strategy must be holdout, cv, or nested")\n    validation["strategy"] = strategy\n    validation.setdefault("seed", int(cv["seed"]))\n    validation.setdefault("blind_fraction", float(cv["blind_fraction"]))\n    validation.setdefault("early_stop_fraction", float(cv["early_stop_fraction"]))\n    validation.setdefault("holdout_fraction", 0.2)\n    validation.setdefault("n_splits", int(cv["n_splits"]))\n    if int(validation["n_splits"]) < 2:\n        raise MLConfigError("validation.n_splits must be >= 2")\n    if not 0.0 < float(validation["blind_fraction"]) < 0.5:\n        raise MLConfigError("validation.blind_fraction must be in (0, 0.5)")\n    if not 0.0 < float(validation["early_stop_fraction"]) < 0.5:\n        raise MLConfigError("validation.early_stop_fraction must be in (0, 0.5)")\n    if not 0.0 < float(validation["holdout_fraction"]) < 0.5:\n        raise MLConfigError("validation.holdout_fraction must be in (0, 0.5)")\n\n    nested = validation.setdefault("nested", {})\n    if not isinstance(nested, dict):\n        raise MLConfigError("validation.nested must be an object")\n    nested.setdefault("outer_folds", 5)\n    nested.setdefault("inner_strategy", "holdout")\n    nested.setdefault("inner_holdout_fraction", float(validation["holdout_fraction"]))\n    nested.setdefault("inner_folds", int(validation["n_splits"]))\n    inner_strategy = str(nested["inner_strategy"]).strip().lower()\n    if inner_strategy not in {"holdout", "cv"}:\n        raise MLConfigError("validation.nested.inner_strategy must be holdout or cv")\n    nested["inner_strategy"] = inner_strategy\n    if int(nested["outer_folds"]) < 2:\n        raise MLConfigError("validation.nested.outer_folds must be >= 2")\n    if int(nested["inner_folds"]) < 2:\n        raise MLConfigError("validation.nested.inner_folds must be >= 2")\n    if not 0.0 < float(nested["inner_holdout_fraction"]) < 0.5:\n        raise MLConfigError("validation.nested.inner_holdout_fraction must be in (0, 0.5)")\n\n    # Backward-compatible mirror for the existing fit/final-training helpers.\n    cv["n_splits"] = int(validation["n_splits"])\n    cv["seed"] = int(validation["seed"])\n    cv["blind_fraction"] = float(validation["blind_fraction"])\n    cv["early_stop_fraction"] = float(validation["early_stop_fraction"])\n'''
    _replace_once(path, old, new, "holdout/CV/nested validation config")

    old = '''    reporting.setdefault("voltage_pattern", r"(?P<voltage>\\d+(?:\\.\\d+)?)V")\n    reporting.setdefault("save_final_fit_plots", True)\n'''
    new = '''    reporting.setdefault("voltage_pattern", r"(?P<voltage>\\d+(?:\\.\\d+)?)V")\n    reporting.setdefault("save_final_fit_plots", True)\n    reporting.setdefault("max_ctr_to_led_ratio", 2.0)\n    reporting.setdefault("top_corrections_k", 3)\n    reporting.setdefault("ctr_uncertainty_bootstrap_samples", 1000)\n    _finite(\n        reporting["max_ctr_to_led_ratio"],\n        "reporting.max_ctr_to_led_ratio",\n        positive=True,\n    )\n    if int(reporting["top_corrections_k"]) < 1:\n        raise MLConfigError("reporting.top_corrections_k must be >= 1")\n    if int(reporting["ctr_uncertainty_bootstrap_samples"]) < 2:\n        raise MLConfigError("reporting.ctr_uncertainty_bootstrap_samples must be >= 2")\n'''
    _replace_once(path, old, new, "reporting defaults")


def patch_prepared_data(repo: Path) -> None:
    path = repo / "waveform_analysis/ml_pipeline/prepared_data.py"
    _replace_once(
        path,
        "from .dataset import DATASET_FORMAT_VERSION, PreparedDataset, load_prepared_dataset\nPREPARED_SELECTION_VERSION = 5\n",
        "from .dataset import DATASET_FORMAT_VERSION, PreparedDataset, load_prepared_dataset\nfrom .selection_store import SELECTION_STORE_VERSION\nPREPARED_SELECTION_VERSION = 6\n",
        "prepared/selection protocol version",
    )
    old = '''    for key in ("prepared_dir", "cleanup_raw_cache", "materialization_chunk_size", "parallelization"):\n        preprocessing.pop(key, None)\n'''
    new = '''    for key in (\n        "prepared_dir", "selection_store_dir", "cleanup_raw_cache",\n        "materialization_chunk_size", "parallelization"\n    ):\n        preprocessing.pop(key, None)\n'''
    _replace_once(path, old, new, "prepared fingerprint path independence")

    _replace_once(
        path,
        '        "selection_version": PREPARED_SELECTION_VERSION,\n        "source": source_signature(root_file),\n',
        '        "selection_version": PREPARED_SELECTION_VERSION,\n        "permanent_selection_store_version": SELECTION_STORE_VERSION,\n        "source": source_signature(root_file),\n',
        "selection-store version in prepared fingerprint",
    )

    old = '''        "true_tof_ps": float(study["data"].get("true_tof_ps", 0.0)),\n        "windows_ns": study["windows_ns"],\n        "preprocessing": preprocessing,\n'''
    new = '''        "true_tof_ps": float(study["data"].get("true_tof_ps", 0.0)),\n        # Experiment windows are cheap runtime slices.  Only the canonical\n        # materialized window in preprocessing can invalidate permanent data.\n        "preprocessing": preprocessing,\n'''
    _replace_once(path, old, new, "decouple prepared data from experiment windows")

    old = '''    max_before = max(float(window["before_ns"]) for window in study["windows_ns"])\n    max_after = max(float(window["after_ns"]) for window in study["windows_ns"])\n    energy["ml_window_ns"] = {"before": max_before, "after": max_after}\n    timing["ml_window_ns"] = {"before": max_before, "after": max_after}\n'''
    new = '''    materialized = preprocessing["materialized_window_ns"]\n    max_before = float(materialized["before"])\n    max_after = float(materialized["after"])\n    energy["ml_window_ns"] = {"before": max_before, "after": max_after}\n    timing["ml_window_ns"] = {"before": max_before, "after": max_after}\n'''
    _replace_once(path, old, new, "canonical permanent waveform extent")

    old = '''        "parallelization": copy.deepcopy(preprocessing.get("parallelization", {"preprocessing_backend": "process", "preprocessing_workers": 0, "preprocessing_chunksize": 8})),\n        "cache": {"raw_cache_dir": str(cache_dir)},\n'''
    new = '''        "parallelization": copy.deepcopy(preprocessing.get("parallelization", {"preprocessing_backend": "process", "preprocessing_workers": 0, "preprocessing_chunksize": 8})),\n        "selection_store_dir": str(preprocessing["selection_store_dir"]),\n        "cache": {"raw_cache_dir": str(cache_dir)},\n'''
    _replace_once(path, old, new, "pass permanent selection store")

    old = '''        "parallelization": raw_cfg["parallelization"],\n    }\n'''
    new = '''        "parallelization": raw_cfg["parallelization"],\n        "selection_store_dir": raw_cfg["selection_store_dir"],\n    }\n'''
    _replace_once(path, old, new, "selection store cache config")


def patch_data(repo: Path) -> None:
    path = repo / "waveform_analysis/ml_pipeline/data.py"
    old = '''from .event_selection import apply_energy_preselection\nfrom .signal import (\n'''
    new = '''from .event_selection import apply_energy_preselection\nfrom .selection_store import load_or_compute_selection\nfrom .signal import (\n'''
    _replace_once(path, old, new, "selection store import")

    old = '''    # First pass: select the photopeak population from raw energy information\n    # before allocating waveform caches or performing any denoising/timing work.\n    selected_entries, preselection_summary = _scan_energy_preselection(\n        input_path,\n        n_events=n_events,\n        energy_channels=energy_channels,\n        energy_polarities=energy_polarities,\n        config=config,\n        logger=logger,\n    )\n'''
    new = '''    # First pass: select the physical/photopeak population once.  The selected\n    # ROOT-entry indices are a permanent upstream artifact and survive transient\n    # waveform-cache deletion or unrelated model/window studies.\n    selection_store = config.get("selection_store_dir")\n    if selection_store:\n        selected_entries, preselection_summary = load_or_compute_selection(\n            input_path,\n            n_events=n_events,\n            config=config,\n            store_root=selection_store,\n            compute=lambda: _scan_energy_preselection(\n                input_path,\n                n_events=n_events,\n                energy_channels=energy_channels,\n                energy_polarities=energy_polarities,\n                config=config,\n                logger=logger,\n            ),\n            logger=logger,\n        )\n    else:\n        selected_entries, preselection_summary = _scan_energy_preselection(\n            input_path,\n            n_events=n_events,\n            energy_channels=energy_channels,\n            energy_polarities=energy_polarities,\n            config=config,\n            logger=logger,\n        )\n'''
    _replace_once(path, old, new, "reuse permanent photopeak selection")


def patch_ml_experiment(repo: Path) -> None:
    path = repo / "waveform_analysis/scripts/ml_experiment.py"
    old = '''from ml_pipeline.study import run_study\n'''
    new = '''from ml_pipeline.study_runner import run_study\n'''
    _replace_once(path, old, new, "unified experiment runner")

    old = '''            "Run the compact CTR study: permanent dataset preparation, random "\n            "development/blind split, fold-wise CV model selection, early stopping "\n            "on a train-only subset, one blind evaluation, standards, XAI and "\n            "raw-only multithreshold SVR."\n'''
    new = '''            "Run the CTR study with permanent event/prepared-data reuse, configurable "\n            "holdout/CV/nested selection, one untouched blind evaluation, integrated "\n            "standards/multithreshold models, and centralized reporting."\n'''
    _replace_once(path, old, new, "CLI description")

    old = '''    parser.add_argument("--rebuild-preprocessing", action="store_true")\n    args = parser.parse_args()\n'''
    new = '''    parser.add_argument("--rebuild-preprocessing", action="store_true")\n    parser.add_argument(\n        "--prepare-only", action="store_true",\n        help="Prepare/reuse permanent selected-event and waveform datasets, then stop",\n    )\n    args = parser.parse_args()\n'''
    _replace_once(path, old, new, "prepare-only CLI")

    old = '''        rebuild_preprocessing=args.rebuild_preprocessing,\n        logger=logger,\n    )\n'''
    new = '''        rebuild_preprocessing=args.rebuild_preprocessing,\n        logger=logger,\n        prepare_only=args.prepare_only,\n    )\n'''
    _replace_once(path, old, new, "prepare-only dispatch")


def patch_compatibility_scripts(repo: Path) -> None:
    # Keep old entry points as thin compatibility wrappers; normal use now goes
    # through ml_experiment.py.  This avoids breaking existing commands while
    # removing their separate scientific implementation paths.
    preprocess = repo / "waveform_analysis/scripts/ml_preprocess.py"
    if preprocess.is_file():
        text = preprocess.read_text(encoding="utf-8")
        if "from ml_pipeline.study_runner import run_study" not in text:
            replacement = '''from __future__ import annotations\n\nimport argparse\nfrom pathlib import Path\nimport sys\n\nPROJECT = Path(__file__).resolve().parents[1]\nif str(PROJECT) not in sys.path:\n    sys.path.insert(0, str(PROJECT))\n\nfrom ml_pipeline.common import setup_logging\nfrom ml_pipeline.study_config import load_study_config\nfrom ml_pipeline.study_runner import run_study\n\n\ndef main() -> None:\n    parser = argparse.ArgumentParser(\n        description="Compatibility wrapper. Prefer ml_experiment.py --prepare-only."\n    )\n    parser.add_argument("--config", type=Path, required=True)\n    parser.add_argument("--rebuild", action="store_true")\n    args = parser.parse_args()\n    config = load_study_config(args.config, PROJECT)\n    output = Path(config["experiment"]["output_dir"])\n    logger = setup_logging(output / "preprocess.log", config.get("logging", {}).get("level", "INFO"))\n    run_study(\n        config, dry_run=False, resume=False, restart=False,\n        rebuild_preprocessing=args.rebuild, logger=logger, prepare_only=True,\n    )\n\n\nif __name__ == "__main__":\n    main()\n'''
            preprocess.write_text(replacement, encoding="utf-8")
            print(f"[replaced] compatibility wrapper: {preprocess}")

    multithreshold = repo / "waveform_analysis/scripts/ml_multithreshold.py"
    if multithreshold.is_file():
        text = multithreshold.read_text(encoding="utf-8")
        if "from ml_pipeline.study_runner import run_study" not in text:
            if "from ml_pipeline.study import run_study" not in text:
                raise RuntimeError(f"Unexpected ml_multithreshold.py structure: {multithreshold}")
            multithreshold.write_text(
                text.replace("from ml_pipeline.study import run_study", "from ml_pipeline.study_runner import run_study"),
                encoding="utf-8",
            )
            print(f"[patched] compatibility wrapper: {multithreshold}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the CTR pipeline refactor to the current GitHub main checkout."
    )
    parser.add_argument("repo", nargs="?", type=Path, default=Path.cwd())
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    marker = repo / "waveform_analysis/ml_pipeline/study.py"
    if not marker.is_file():
        raise SystemExit(f"Not a compatible repository root: {repo}")

    targets = [
        repo / "waveform_analysis/ml_pipeline/study_config.py",
        repo / "waveform_analysis/ml_pipeline/prepared_data.py",
        repo / "waveform_analysis/ml_pipeline/data.py",
        repo / "waveform_analysis/scripts/ml_experiment.py",
        repo / "waveform_analysis/scripts/ml_preprocess.py",
        repo / "waveform_analysis/scripts/ml_multithreshold.py",
        repo / "waveform_analysis/scripts/analyze_energy_timing_led_correlation.py",
    ]
    backup = repo / ".ctr_pipeline_update_backup"
    if not args.no_backup and not backup.exists():
        for path in targets:
            if path.is_file():
                relative = path.relative_to(repo)
                destination = backup / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
        print(f"[backup] {backup}")

    # Copy additive/refactored modules first; patched imports become valid only
    # after these files exist.
    for relative in (
        "waveform_analysis/ml_pipeline/selection_store.py",
        "waveform_analysis/ml_pipeline/validation.py",
        "waveform_analysis/ml_pipeline/reporting.py",
        "waveform_analysis/ml_pipeline/nested_evaluation.py",
        "waveform_analysis/ml_pipeline/study_runner.py",
        "waveform_analysis/scripts/analyze_energy_timing_led_correlation.py",
    ):
        _copy(relative, repo)

    patch_study_config(repo)
    patch_prepared_data(repo)
    patch_data(repo)
    patch_ml_experiment(repo)
    patch_compatibility_scripts(repo)

    print("\nUpdate applied.")
    print("Validate syntax with:")
    print("  python -m compileall waveform_analysis/ml_pipeline waveform_analysis/scripts")
    print("Prepare only with:")
    print("  python waveform_analysis/scripts/ml_experiment.py --config <study.json> --prepare-only")
    print("Run with:")
    print("  python waveform_analysis/scripts/ml_experiment.py --config <study.json>")


if __name__ == "__main__":
    main()
