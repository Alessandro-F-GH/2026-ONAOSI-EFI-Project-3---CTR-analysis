# CTR pipeline update — configuration additions

The normal entry point is now:

```bash
python waveform_analysis/scripts/ml_experiment.py --config config/experiments/<study>.json
```

To prepare/reuse the permanent selected-event and waveform datasets without training:

```bash
python waveform_analysis/scripts/ml_experiment.py --config config/experiments/<study>.json --prepare-only
```

## Permanent preprocessing

Add this under `preprocessing` (the defaults are shown):

```json
{
  "preprocessing": {
    "selection_store_dir": "processed_data/selected_events",
    "prepared_dir": "processed_data/ml_prepared",
    "materialized_window_ns": {
      "before": 5.0,
      "after": 60.0
    }
  }
}
```

`selection_store_dir` contains immutable, fingerprinted `selected_entries.npy` artifacts. The photopeak scan is reused across waveform preparation and experiments as long as the raw file and physical-selection settings are unchanged.

`materialized_window_ns` is the permanent waveform envelope. Every entry in top-level `windows_ns` must lie inside it. Changing only `windows_ns` no longer invalidates the prepared dataset.

## Quick holdout validation

```json
{
  "validation": {
    "strategy": "holdout",
    "seed": 20260813,
    "blind_fraction": 0.20,
    "early_stop_fraction": 0.15,
    "holdout_fraction": 0.20
  }
}
```

All candidates for the same file use the same deterministic development holdout.

## Standard K-fold CV

```json
{
  "validation": {
    "strategy": "cv",
    "seed": 20260813,
    "blind_fraction": 0.20,
    "early_stop_fraction": 0.15,
    "n_splits": 5
  }
}
```

Candidate CTR is the arithmetic mean of the independent score-fold CTR values; residuals are not pooled to compute the selection CTR.

## Nested evaluation with inner holdout

```json
{
  "validation": {
    "strategy": "nested",
    "seed": 20260813,
    "blind_fraction": 0.20,
    "early_stop_fraction": 0.15,
    "holdout_fraction": 0.20,
    "n_splits": 5,
    "nested": {
      "outer_folds": 5,
      "inner_strategy": "holdout",
      "inner_holdout_fraction": 0.20,
      "inner_folds": 4
    }
  }
}
```

Each outer training fold repeats the full candidate/model/window selection using one internal holdout. The outer test fold estimates selection-pipeline performance. The final blind set remains untouched and is evaluated only after a final selection on the complete development set.

## Nested evaluation with inner CV

Change only:

```json
{
  "validation": {
    "strategy": "nested",
    "nested": {
      "outer_folds": 5,
      "inner_strategy": "cv",
      "inner_folds": 4
    }
  }
}
```

## Reporting

```json
{
  "reporting": {
    "dpi": 180,
    "max_ctr_to_led_ratio": 2.0,
    "top_corrections_k": 3,
    "ctr_uncertainty_bootstrap_samples": 1000
  }
}
```

`max_ctr_to_led_ratio` is a reporting-only filter. Pathological methods remain in `results.csv` and `summary_results.csv`; they are simply omitted from plots when their blind CTR exceeds the configured multiple of the same file/mode LED baseline.

The blind-distribution CTR uncertainty is a non-refit event bootstrap of the fixed blind residual vector. It is distinct from any optional model-refit/bootstrap study.
