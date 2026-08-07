# Results-only study storage

The experiment runner now treats model checkpoints as temporary working files by default.

```json
"storage": {
  "cleanup_raw_cache_after_materialization": true,
  "cleanup_after_completed_file": true,
  "keep_checkpoints": false
}
```

With `keep_checkpoints: false`:

- only the currently best CV trial is retained while a window is being optimized;
- losing trial directories are deleted as soon as they cannot win;
- after window selection, checkpoints for non-selected windows are deleted;
- before compact reporting, only the overall CV winner for each channel mode remains;
- after `summary_results.csv`, `model_loss_results.csv`, and plots are produced, all remaining run/checkpoint directories for that ROOT file are deleted.

The durable experiment state is numeric: `_state/all_results.csv`, `_state/results_metadata.json`, compact CSV reports, selection JSON files, folds, and plots. If an interrupted results-only resume has CV rows but is missing the selected checkpoint needed for a blind audit, only that selected fold is retrained temporarily.

Set `keep_checkpoints: true` only when model replay or later inference is required.
