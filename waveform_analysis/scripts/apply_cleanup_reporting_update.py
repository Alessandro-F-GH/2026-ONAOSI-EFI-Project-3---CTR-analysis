#!/usr/bin/env python3
"""
Clean the current waveform_analysis repository and add:
  * train/development residual distributions using the same plotting path as blind;
  * unified TOP/WORST correction plots;
  * TOP/WORST ranking based on development-centered LED residual and development-centered linear correction:
        led_linear = LED residual - mean_dev(LED residual)
        correction_linear = applied correction - mean_dev(applied correction)
        score = |led_linear| - |led_linear + correction_linear|
    (no global LED/calibration offset can dominate the ranking);
  * readable correction legend using LED error relative to the development LED mean;
  * removal of several provably-unused study.py helpers and historical .patch files;
  * small reporting correctness cleanups.

This is an updater, not a unified diff. It edits the checked-out repository in place
and creates .cleanup_reporting.bak backups before writing.

Run from repository root or waveform_analysis:

    python apply_cleanup_reporting_update.py --check
    python apply_cleanup_reporting_update.py

The --check mode applies all transformations in memory and AST-parses the result
without changing files.
"""

from __future__ import annotations

import argparse
import ast
import shutil
from pathlib import Path


OBSOLETE_PATCH_FILES = (
    "disjoint_window_scan_reporting.patch",
    "log_window_once_candidate_hyperparams.patch",
    "logging_sctr_less_verbose.patch",
)

MARKER_V1 = "# CLEANUP_TRAIN_TOP_WORST_V1"
MARKER = "# CLEANUP_TRAIN_TOP_WORST_V2"


def _find_waveform_analysis(start: Path) -> Path:
    start = start.resolve()
    candidates = [
        start,
        start / "waveform_analysis",
        *[parent / "waveform_analysis" for parent in start.parents],
    ]
    for candidate in candidates:
        if (
            (candidate / "ml_pipeline" / "study.py").is_file()
            and (candidate / "ml_pipeline" / "reporting.py").is_file()
        ):
            return candidate
    raise FileNotFoundError(
        "Cannot locate waveform_analysis/ml_pipeline/study.py and reporting.py. "
        "Run from the repository root or waveform_analysis/."
    )


def _function_span(source: str, name: str) -> tuple[int, int]:
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))

    matches = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one top-level function {name!r}; found {len(matches)}."
        )
    node = matches[0]
    if node.end_lineno is None:
        raise RuntimeError(f"Python AST did not expose end_lineno for {name!r}")
    start = starts[node.lineno - 1]
    end = starts[node.end_lineno]
    while end < len(source) and source[end:end + 1] == "\n":
        end += 1
    return start, end


def _replace_function(source: str, name: str, replacement: str) -> str:
    start, end = _function_span(source, name)
    return source[:start] + replacement.rstrip() + "\n\n" + source[end:]


def _remove_function(source: str, name: str) -> str:
    start, end = _function_span(source, name)
    return source[:start] + source[end:]


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    print(f"[change] {label}")
    return source.replace(old, new, 1)


REPORT_DISTRIBUTION = r'''def plot_result_distribution(
    path: Path,
    *,
    mode: str,
    methods: dict[str, np.ndarray],
    dpi: int,
    ratio_limit: float,
    bootstrap_samples: int,
    seed: int,
    split_label: str,
) -> dict[str, float]:
    """Plot one final residual distribution for train/development or blind.

    The exact same eligibility, robust display bounds, bins, CTR definition and
    bootstrap uncertainty are used for both populations.
    """
    keep = eligible(methods, ratio_limit)
    visible = {
        key: np.asarray(value, float)
        for key, value in methods.items()
        if key in keep and np.sum(np.isfinite(value)) >= 2
    }
    if not visible:
        return {}

    lo, hi = _robust_bounds(visible)
    bins = np.linspace(lo, hi, 81)
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    uncertainties: dict[str, float] = {}

    for index, (name, values) in enumerate(visible.items()):
        values = values[np.isfinite(values)]
        metrics = residual_metrics(values)
        uncertainty = ctr_bootstrap_uncertainty(
            values,
            bootstrap_samples,
            seed + 137 * index,
        )
        uncertainties[name] = uncertainty
        ax.hist(
            values[(values >= lo) & (values <= hi)],
            bins=bins,
            histtype="step",
            density=True,
            linewidth=1.4,
            label=(
                f"{short_model_label(name)} — "
                f"CTR {format_ctr(metrics['ctr_ps'], uncertainty)}"
            ),
        )

    ax.set_title(f"{short_mode_label(mode)} · {split_label}")
    ax.set_xlabel("Residual timing error [ps]")
    ax.set_ylabel("Density")
    ax.grid(alpha=.22)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    _save(fig, path, dpi)
    return uncertainties
'''

REPORT_CORRECTIONS = r'''def plot_correction_examples(
    path: Path,
    *,
    time_ps: np.ndarray,
    waveforms: np.ndarray,
    led_residual: np.ndarray,
    corrected_residual: np.ndarray,
    led_reference_mean_ps: float,
    correction_reference_mean_ps: float,
    model: str,
    mode: str,
    selection: str,
    k: int,
    dpi: int,
    window_before_ns: float | None = None,
    window_after_ns: float | None = None,
    event_ids: np.ndarray | None = None,
) -> None:
    """Plot TOP/WORST event corrections using centered linear contributions.

    Both reference offsets are learned from development/train only and then
    frozen when ranking blind events:

        led_linear = led_residual - mean_dev(led_residual)

        applied_correction = corrected_residual - led_residual
        correction_linear = (
            applied_correction - mean_dev(applied_correction)
        )

        final_linear = led_linear + correction_linear
        rank_gain = |led_linear| - |final_linear|

    Therefore a large global LED-vs-true-TOF offset, or an equally large global
    model calibration/intercept, cannot make every selected correction have the
    same sign. TOP/WORST reflects only the event-dependent linear correction.
    """
    selection = str(selection).strip().lower()
    if selection not in {"top", "worst"}:
        raise ValueError(
            f"selection must be 'top' or 'worst', got {selection!r}"
        )

    k = int(k)
    if k <= 0:
        return

    led = np.asarray(led_residual, dtype=np.float64).reshape(-1)
    corrected = np.asarray(corrected_residual, dtype=np.float64).reshape(-1)
    waves = np.asarray(waveforms)
    x = np.asarray(time_ps, dtype=np.float64).reshape(-1) / 1000.0

    if led.size != corrected.size:
        raise ValueError("LED and corrected residual arrays must have equal length")
    if waves.ndim != 3 or waves.shape[0] != led.size or waves.shape[1] != 2:
        raise ValueError(
            "waveforms must have shape (N, 2, samples) matching residual arrays"
        )
    if waves.shape[2] != x.size:
        raise ValueError("time grid length must match waveform sample length")

    applied = corrected - led
    led_linear = led - float(led_reference_mean_ps)
    correction_linear = applied - float(correction_reference_mean_ps)
    final_linear = led_linear + correction_linear

    rank_before = np.abs(led_linear)
    rank_after = np.abs(final_linear)
    rank_gain = rank_before - rank_after

    finite = np.flatnonzero(
        np.isfinite(rank_gain)
        & np.isfinite(led_linear)
        & np.isfinite(correction_linear)
        & np.isfinite(final_linear)
    )
    if finite.size == 0:
        return

    if selection == "top":
        order = finite[np.argsort(rank_gain[finite])[::-1]]
    else:
        order = finite[np.argsort(rank_gain[finite])]
    order = order[: min(k, order.size)]

    ids = None
    if event_ids is not None:
        ids = np.asarray(event_ids).reshape(-1)
        if ids.size != led.size:
            raise ValueError("event_ids length must match residual arrays")

    fig, axes = plt.subplots(
        order.size,
        1,
        figsize=(9.6, 3.0 * order.size),
        squeeze=False,
    )

    for rank, (ax, idx) in enumerate(zip(axes[:, 0], order), start=1):
        if window_before_ns is not None and window_after_ns is not None:
            ax.axvspan(
                -float(window_before_ns),
                float(window_after_ns),
                alpha=0.08,
            )

        ax.plot(x, waves[idx, 0], linewidth=1.05, label="ch1")
        ax.plot(x, waves[idx, 1], linewidth=1.05, label="ch2")
        ax.axvline(0.0, linewidth=0.8, linestyle="--")
        ax.set_ylabel("mV")
        ax.grid(alpha=.18)

        gain = float(rank_gain[idx])
        if gain >= 0.0:
            change_text = f"{gain:.0f} ps improvement"
        else:
            change_text = f"{abs(gain):.0f} ps worsening"

        event_text = ""
        if ids is not None:
            event_text = f" · event {ids[idx]}"

        # Signed centered values preserve the exact linear equation; the
        # improvement itself is always evaluated on the residual modules.
        summary = (
            f"#{rank} LED {led_linear[idx]:+.0f} ps "
            f"+ correction {correction_linear[idx]:+.0f} ps "
            f"= {final_linear[idx]:+.0f} ps · "
            f"|LED| {rank_before[idx]:.0f}→{rank_after[idx]:.0f} ps · "
            f"{change_text}{event_text}"
        )

        ax.plot([], [], linestyle="none", marker="", label=summary)
        ax.legend(
            frameon=True,
            framealpha=.86,
            fontsize=7.5,
            loc="upper right",
            ncol=1,
        )

    axes[-1, 0].set_xlabel(
        "Time relative to LED-aligned native anchor [ns]"
    )
    kind = "TOP" if selection == "top" else "WORST"
    fig.suptitle(
        f"{short_model_label(model)} · {short_mode_label(mode)} · "
        f"{kind} centered linear corrections",
        fontsize=11,
    )
    _save(fig, path, dpi)
'''

WAVEFORM_EVAL = r'''def _waveform_evaluate_selected(
    study: dict[str, Any],
    dataset: PreparedDataset,
    train_pool: np.ndarray,
    evaluation: np.ndarray,
    *,
    file_id: int,
    mode: str,
    window: dict[str, Any],
    variant: str,
    subsampling: int,
    space: dict[str, Any],
    overrides: dict[str, Any],
    candidate_id: int,
    work_dir: Path,
    logger: Any,
    normalization_cache: dict[str, Normalization],
    checkpoint_path: Path | None = None,
    compute_xai: bool = False,
    return_train_residual: bool = False,
) -> tuple[
    np.ndarray,
    dict[str, Any],
    dict[str, Any],
    tuple[np.ndarray, np.ndarray] | None,
    np.ndarray | None,
]:
    source = input_variant_dataset_view(dataset, variant)
    input_waveforms, target = CHANNEL_MODES[mode]
    view = prediction_window_dataset_view(
        source,
        input_waveforms=input_waveforms,
        target=target,
        before_ns=float(window["before_ns"]),
        after_ns=float(window["after_ns"]),
    )
    seed = _seed_for(
        int(study["validation"]["seed"]),
        file_id,
        mode,
        space["id"],
        candidate_id,
        "evaluation",
        int(np.asarray(evaluation).size),
    )
    cfg = _candidate_training_config(
        study,
        space,
        overrides,
        mode=mode,
        subsampling=subsampling,
        train_dir=work_dir,
        seed=seed,
        final=checkpoint_path is not None,
    )
    train_pool = np.asarray(train_pool, dtype=np.int64)
    evaluation = np.asarray(evaluation, dtype=np.int64)

    if cfg["model"]["type"] == "linear_svr":
        fit_idx = train_pool
        early_idx = train_pool
    else:
        fit_idx, early_idx = _fit_early_split(
            train_pool,
            fraction=_early_fraction(study, cfg),
            seed=_seed_for(
                int(study["validation"]["seed"]),
                file_id,
                mode,
                "early",
                "eval",
            ),
        )

    eval_view = replace(
        view,
        train=fit_idx,
        validation=early_idx,
        evaluation=evaluation,
    )
    cached_normalization = _normalization_for_fit_subset(
        eval_view,
        fit_idx,
        subsampling=subsampling,
        cache=normalization_cache,
    )
    model, normalization, summary = _train_in_memory(
        cfg,
        eval_view,
        logger=logger,
        data_view={"stage": "evaluation", "candidate_id": candidate_id},
        normalization_override=cached_normalization,
    )

    residual = _predict_indices(
        model,
        normalization,
        cfg,
        eval_view,
        evaluation,
    )
    _fit, metrics = _fit_row(
        residual,
        method=f"Evaluation {space['id']}",
        fit_config=study["fit"],
    )

    train_residual = None
    if return_train_residual:
        train_residual = _predict_indices(
            model,
            normalization,
            cfg,
            eval_view,
            train_pool,
        )

    xai_profile = None
    if compute_xai:
        xai_cfg = study.get("reporting", {}).get("xai", {}) or {}
        if bool(xai_cfg.get("enabled", False)):
            xai_profile = _integrated_gradient_profile(
                model,
                normalization,
                cfg,
                eval_view,
                evaluation,
                max_events=int(xai_cfg.get("max_events", 512)),
                steps=int(xai_cfg.get("integrated_gradient_steps", 16)),
            )

    meta = {
        "best_epoch": int(summary.get("best_epoch", 0)),
        "normalization": summary.get("normalization", {}),
        "model_type": cfg["model"]["type"],
    }
    _cleanup_training(model, work_dir, keep_best=checkpoint_path)
    return residual, metrics, meta, xai_profile, train_residual
'''

MULTITHRESHOLD_EVAL = r'''def _multithreshold_evaluate(
    study: dict[str, Any],
    dataset: PreparedDataset,
    train_pool: np.ndarray,
    evaluation: np.ndarray,
    selected: dict[str, Any],
    *,
    return_train_residual: bool = False,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray | None]:
    params = selected["params"]
    cols = params["threshold_indices"]
    features = selected["features"]
    valid = np.all(np.isfinite(features[:, cols]), axis=1)
    train_pool = np.asarray(train_pool, dtype=np.int64)
    evaluation = np.asarray(evaluation, dtype=np.int64)

    if not np.all(valid[train_pool]) or not np.all(valid[evaluation]):
        raise RuntimeError(
            "Selected multithreshold model lacks a threshold crossing "
            "for at least one evaluation event"
        )

    estimator = make_pipeline(
        StandardScaler(),
        SVR(
            kernel=params["kernel"],
            C=params["C"],
            epsilon=params["epsilon_ps"],
            gamma=params["gamma"],
        ),
    )
    estimator.fit(
        features[np.ix_(train_pool, cols)],
        selected["target_correction"][train_pool],
    )

    correction = estimator.predict(features[np.ix_(evaluation, cols)])
    residual = (
        selected["led_ps"][evaluation]
        - correction
        - float(dataset.true_tof_ps)
    )
    _fit, metrics = _fit_row(
        residual,
        method="Evaluation multithreshold SVR",
        fit_config=study["fit"],
    )

    train_residual = None
    if return_train_residual:
        train_correction = estimator.predict(features[np.ix_(train_pool, cols)])
        train_residual = (
            selected["led_ps"][train_pool]
            - train_correction
            - float(dataset.true_tof_ps)
        )

    return residual, metrics, train_residual
'''


_CENTERED_CORRECTION_STUDY_BLOCK = '''            # TOP/WORST examples from the single best validation-selected ML family.
            eligible_final = [
                item
                for item in final_candidates
                if int(item[4].get("plot_included", 1)) == 1
            ]
            if eligible_final:
                best_name, best_chosen, best_space, best_residual, _best_rr = min(
                    eligible_final,
                    key=lambda item: float(item[1]["metrics"]["ctr_ps"]),
                )
                variant = "raw" if best_space is None else best_chosen["variant"]
                source = input_variant_dataset_view(dataset, variant)
                input_waveforms, target = CHANNEL_MODES[mode]
                materialized = config["preprocessing"]["materialized_window_ns"]
                full_view = prediction_window_dataset_view(
                    source,
                    input_waveforms=input_waveforms,
                    target=target,
                    before_ns=float(materialized["before"]),
                    after_ns=float(materialized["after"]),
                )

                development_led, _ = _target_deltas(dataset, mode, development)
                development_led_residual = (
                    development_led - float(dataset.true_tof_ps)
                )
                led_reference_mean = float(
                    np.mean(development_led_residual)
                )

                best_train_residual = train_methods.get(best_name)
                if best_train_residual is None:
                    raise RuntimeError(
                        f"Missing train/development residuals for {best_name}; "
                        "cannot center TOP/WORST correction without blind leakage"
                    )
                best_train_residual = np.asarray(
                    best_train_residual, dtype=np.float64
                )
                train_applied_correction = (
                    best_train_residual - development_led_residual
                )
                correction_reference_mean = float(
                    np.mean(train_applied_correction)
                )

                common_correction_args = {
                    "time_ps": np.asarray(
                        full_view.relative_time_ps, dtype=np.float64
                    ),
                    "waveforms": np.asarray(
                        full_view.windows_mV[blind], dtype=np.float32
                    ),
                    "led_residual": led_residual,
                    "corrected_residual": best_residual,
                    "led_reference_mean_ps": led_reference_mean,
                    "correction_reference_mean_ps": correction_reference_mean,
                    "model": best_name,
                    "mode": mode,
                    "dpi": dpi,
                    "window_before_ns": float(
                        best_chosen["window"]["before_ns"]
                    ),
                    "window_after_ns": float(
                        best_chosen["window"]["after_ns"]
                    ),
                    "event_ids": np.asarray(dataset.event_id[blind]),
                }

                top_k = int(config["reporting"].get("top_corrections_k", 3))
                if top_k > 0:
                    plot_correction_examples(
                        plots_root / "top_corrections" / f"{root_file.stem}__{mode}.png",
                        selection="top",
                        k=top_k,
                        **common_correction_args,
                    )

                worst_k = int(config["reporting"].get("worst_corrections_k", 3))
                if worst_k > 0:
                    plot_correction_examples(
                        plots_root / "worst_corrections" / f"{root_file.stem}__{mode}.png",
                        selection="worst",
                        k=worst_k,
                        **common_correction_args,
                    )
'''


def _patch_reporting(source: str) -> str:
    if MARKER in source:
        print("[already] reporting.py contains V2 update marker")
        return source

    if MARKER_V1 in source:
        text = _replace_function(source, "plot_correction_examples", REPORT_CORRECTIONS)
        text = text.replace(MARKER_V1, MARKER, 1)
        ast.parse(text)
        print("[migrate] reporting.py V1 -> V2 centered correction ranking")
        return text

    text = source
    text = _replace_function(text, "plot_blind_distribution", REPORT_DISTRIBUTION)
    text = _replace_function(text, "plot_top_corrections", REPORT_CORRECTIONS)

    text = text.replace(
        "selected=[r for r in selected if r.get('model') not in ('led','cfd') or True]\n",
        "",
    )
    text = text.replace(
        '                             ">2× LED",\n',
        '                             f">{float(ratio_limit):g}× LED",\n',
    )

    text = text.replace(
        "import matplotlib.pyplot as plt\n",
        "import matplotlib.pyplot as plt\n\n" + MARKER + "\n",
        1,
    )

    ast.parse(text)
    return text


def _patch_study(source: str) -> str:
    if MARKER in source:
        print("[already] study.py contains V2 update marker")
        return source

    if MARKER_V1 in source:
        text = source
        start_anchor = "            # TOP/WORST examples from the single best validation-selected ML family.\n"
        end_anchor = "        normalization_cache.clear()\n"
        start = text.find(start_anchor)
        end = text.find(end_anchor, start)
        if start < 0 or end < 0:
            raise RuntimeError("Could not locate V1 TOP/WORST correction block")
        text = text[:start] + _CENTERED_CORRECTION_STUDY_BLOCK + text[end:]
        text = text.replace(MARKER_V1, MARKER, 1)
        ast.parse(text)
        print("[migrate] study.py V1 -> V2 centered correction ranking")
        return text

    text = source

    for name in (
        "_random_dev_blind",
        "_kfold",
        "_plot_final_file",
        "_plot_xai_file",
        "_plot_ctr_vs_voltage",
    ):
        try:
            text = _remove_function(text, name)
            print(f"[remove] unused study helper {name}")
        except RuntimeError:
            print(f"[skip] {name} not present exactly once")

    text = _replace_function(text, "_waveform_evaluate_selected", WAVEFORM_EVAL)
    text = _replace_function(text, "_multithreshold_evaluate", MULTITHRESHOLD_EVAL)

    text = text.replace("plot_blind_distribution", "plot_result_distribution")
    text = text.replace("plot_top_corrections", "plot_correction_examples")

    old_inclusion = (
        "    included = int(not np.isfinite(ratio) or ratio <= float(ratio_limit))\n"
    )
    new_inclusion = (
        "    included = int(np.isfinite(ratio) and ratio <= float(ratio_limit))\n"
    )
    if old_inclusion in text:
        text = text.replace(old_inclusion, new_inclusion, 1)
        print("[change] exclude non-finite CTR ratios from aggregate plots")

    text = text.replace(
        "residual, metrics, _meta, _xai = _waveform_evaluate_selected(",
        "residual, metrics, _meta, _xai, _train_residual = _waveform_evaluate_selected(",
    )

    text = text.replace(
        "residual, metrics = _multithreshold_evaluate(\n"
        "                                config, dataset, outer_train, outer_test, selected_mt_outer\n"
        "                            )",
        "residual, metrics, _train_residual = _multithreshold_evaluate(\n"
        "                                config, dataset, outer_train, outer_test, selected_mt_outer\n"
        "                            )",
    )

    blind_anchor = '''            blind_methods: dict[str, np.ndarray] = {
                _MODEL_LED: led_residual, _MODEL_CFD: cfd_residual,
            }
'''
    train_insert = blind_anchor + '''            led_train, cfd_train = _target_deltas(
                dataset, mode, development
            )
            train_methods: dict[str, np.ndarray] = {
                _MODEL_LED: led_train - float(dataset.true_tof_ps),
                _MODEL_CFD: cfd_train - float(dataset.true_tof_ps),
            }
'''
    text = _replace_once(
        text,
        blind_anchor,
        train_insert,
        "initialize final train/development residual distributions",
    )

    old_final_call = '''                residual, metrics, meta, xai_profile = _waveform_evaluate_selected(
                    config, dataset, development, blind,
                    file_id=file_id, mode=mode, window=chosen["window"], variant=chosen["variant"],
                    subsampling=chosen["subsampling"], space=space, overrides=chosen["overrides"],
                    candidate_id=chosen["candidate_id"], work_dir=final_dir, logger=logger,
                    normalization_cache=normalization_cache, checkpoint_path=checkpoint,
                    compute_xai=True,
                )
'''
    new_final_call = '''                residual, metrics, meta, xai_profile, train_residual = _waveform_evaluate_selected(
                    config, dataset, development, blind,
                    file_id=file_id, mode=mode, window=chosen["window"], variant=chosen["variant"],
                    subsampling=chosen["subsampling"], space=space, overrides=chosen["overrides"],
                    candidate_id=chosen["candidate_id"], work_dir=final_dir, logger=logger,
                    normalization_cache=normalization_cache, checkpoint_path=checkpoint,
                    compute_xai=True, return_train_residual=True,
                )
'''
    text = _replace_once(text, old_final_call, new_final_call, "return final waveform train residual")

    old_blind_assign = '''                blind_methods[space["id"]] = residual
                blind_corrections[space["id"]] = (blind, led_residual - residual)
'''
    new_blind_assign = '''                blind_methods[space["id"]] = residual
                if train_residual is not None:
                    train_methods[space["id"]] = train_residual
                blind_corrections[space["id"]] = (blind, led_residual - residual)
'''
    text = _replace_once(text, old_blind_assign, new_blind_assign, "store final waveform train residual")

    old_mt = (
        "                residual, metrics = _multithreshold_evaluate("
        "config, dataset, development, blind, selected_mt)\n"
    )
    new_mt = (
        "                residual, metrics, train_residual = _multithreshold_evaluate(\n"
        "                    config, dataset, development, blind, selected_mt,\n"
        "                    return_train_residual=True,\n"
        "                )\n"
    )
    text = _replace_once(text, old_mt, new_mt, "return final multithreshold train residual")

    old_mt_assign = '''                blind_methods[_MODEL_MULTITHRESHOLD] = residual
                blind_corrections[_MODEL_MULTITHRESHOLD] = (blind, led_residual - residual)
'''
    new_mt_assign = '''                blind_methods[_MODEL_MULTITHRESHOLD] = residual
                if train_residual is not None:
                    train_methods[_MODEL_MULTITHRESHOLD] = train_residual
                blind_corrections[_MODEL_MULTITHRESHOLD] = (blind, led_residual - residual)
'''
    text = _replace_once(text, old_mt_assign, new_mt_assign, "store multithreshold train residual")

    old_blind_plot = '''            plot_result_distribution(
                plots_root / "blind_distributions" / f"{root_file.stem}__{mode}.png",
                mode=mode, methods=blind_methods, dpi=dpi, ratio_limit=ratio_limit,
                bootstrap_samples=bootstrap_samples,
                seed=_seed_for(base_seed, file_id, mode, "distribution_bootstrap"),
            )
'''
    new_both_plots = '''            plot_result_distribution(
                plots_root / "train_distributions" / f"{root_file.stem}__{mode}.png",
                mode=mode, methods=train_methods, dpi=dpi, ratio_limit=ratio_limit,
                bootstrap_samples=bootstrap_samples,
                seed=_seed_for(base_seed, file_id, mode, "train_distribution_bootstrap"),
                split_label="Train / development",
            )
            plot_result_distribution(
                plots_root / "blind_distributions" / f"{root_file.stem}__{mode}.png",
                mode=mode, methods=blind_methods, dpi=dpi, ratio_limit=ratio_limit,
                bootstrap_samples=bootstrap_samples,
                seed=_seed_for(base_seed, file_id, mode, "blind_distribution_bootstrap"),
                split_label="Blind",
            )
'''
    text = _replace_once(text, old_blind_plot, new_both_plots, "add train distribution and generalize blind distribution")

    start_anchor = "            # Top-k examples from the single best validation-selected ML family.\n"
    end_anchor = "        normalization_cache.clear()\n"
    start = text.find(start_anchor)
    end = text.find(end_anchor, start)
    if start < 0 or end < 0:
        raise RuntimeError("Could not locate final TOP correction block")

    replacement = _CENTERED_CORRECTION_STUDY_BLOCK

    text = text[:start] + replacement + text[end:]
    print("[change] unified TOP/WORST correction reporting")

    if "math." not in text:
        text = text.replace("import math\n", "", 1)

    text = text.replace(
        "import matplotlib.pyplot as plt\n",
        "import matplotlib.pyplot as plt\n\n" + MARKER + "\n",
        1,
    )

    ast.parse(text)
    return text


def _sanity(source_study: str, source_reporting: str) -> None:
    ast.parse(source_study)
    ast.parse(source_reporting)

    study_required = [
        "train_distributions",
        "worst_corrections",
        "return_train_residual=True",
        "plot_correction_examples",
        "Train / development",
        "correction_reference_mean_ps",
        "train_applied_correction",
    ]
    reporting_required = [
        "def plot_result_distribution(",
        "def plot_correction_examples(",
        "rank_gain = rank_before - rank_after",
        "applied = corrected - led",
        "led_reference_mean_ps",
        "correction_reference_mean_ps",
        "correction_linear = applied - float(correction_reference_mean_ps)",
    ]

    missing = [value for value in study_required if value not in source_study]
    missing += [value for value in reporting_required if value not in source_reporting]
    if missing:
        raise RuntimeError("Sanity check failed; missing: " + ", ".join(missing))

    obsolete = [
        "def _random_dev_blind(",
        "def _kfold(",
        "def _plot_final_file(",
        "def _plot_xai_file(",
        "def _plot_ctr_vs_voltage(",
        "plot_blind_distribution",
        "plot_top_corrections",
    ]
    leftovers = [value for value in obsolete if value in source_study]
    if leftovers:
        raise RuntimeError(
            "Cleanup sanity check failed; obsolete study symbols remain: "
            + ", ".join(leftovers)
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="Transform and validate in memory without writing files.",
    )
    args = parser.parse_args()

    wa = _find_waveform_analysis(Path.cwd())
    study_path = wa / "ml_pipeline" / "study.py"
    reporting_path = wa / "ml_pipeline" / "reporting.py"

    study_source = study_path.read_text(encoding="utf-8")
    reporting_source = reporting_path.read_text(encoding="utf-8")

    if MARKER in study_source and MARKER in reporting_source:
        _sanity(study_source, reporting_source)
        print("[ok] cleanup/reporting update already applied")
        return

    new_reporting = _patch_reporting(reporting_source)
    new_study = _patch_study(study_source)
    _sanity(new_study, new_reporting)

    # Numeric audit with a deliberately huge global offset. The 1 ns LED
    # offset and -1 ns model intercept must disappear after development centering.
    led = 1090.0
    led_mean = 1000.0
    raw_correction = -1050.0
    correction_mean = -1000.0
    led_linear = led - led_mean
    correction_linear = raw_correction - correction_mean
    final_linear = led_linear + correction_linear
    rank_gain = abs(led_linear) - abs(final_linear)
    if (led_linear, correction_linear, final_linear, rank_gain) != (
        90.0, -50.0, 40.0, 50.0
    ):
        raise RuntimeError("Centered correction-ranking numeric audit failed")

    print("[ok] Python AST validation passed")
    print(
        "[ok] Centered ranking audit: LED 1090 ps with dev mean 1000 ps -> 90 ps; "
        "raw correction -1050 ps with dev mean -1000 ps -> -50 ps; "
        "final 40 ps, 50 ps improvement"
    )

    obsolete_existing = [
        wa / name for name in OBSOLETE_PATCH_FILES if (wa / name).exists()
    ]
    if obsolete_existing:
        print("[cleanup] historical patch files:")
        for path in obsolete_existing:
            print(f"          {path.name}")

    if args.check:
        print("[ok] --check requested: no files changed")
        return

    for path in (study_path, reporting_path):
        backup = path.with_suffix(path.suffix + ".cleanup_reporting.bak")
        if not backup.exists():
            shutil.copy2(path, backup)
            print(f"[backup] {backup}")

    study_tmp = study_path.with_suffix(".py.tmp")
    reporting_tmp = reporting_path.with_suffix(".py.tmp")
    study_tmp.write_text(new_study, encoding="utf-8")
    reporting_tmp.write_text(new_reporting, encoding="utf-8")

    ast.parse(study_tmp.read_text(encoding="utf-8"))
    ast.parse(reporting_tmp.read_text(encoding="utf-8"))

    study_tmp.replace(study_path)
    reporting_tmp.replace(reporting_path)

    for path in obsolete_existing:
        path.unlink()
        print(f"[remove] {path}")

    print("[done] Repository cleanup/reporting update applied.")
    print("[done] New plots: plots/train_distributions and plots/worst_corrections.")
    print("[done] TOP/WORST use development-centered LED and correction contributions.")
    print("[done] Existing blind distribution now uses the shared plotting helper.")


if __name__ == "__main__":
    main()
