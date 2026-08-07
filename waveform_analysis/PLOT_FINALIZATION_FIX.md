# Study plot finalization fix

The compact numeric CSV refactor removed the legacy `_normalize_row` helper,
but `_plot_results` still called it after every experiment block had completed.
This caused finalization to fail with `NameError` even though all training results
had already been saved.

The plotting layer now builds its DataFrame directly from the decoded internal
result schema. It also assigns window labels to both mean and SEM rows, avoiding
a second plotting failure when indexing SEM values by window.

No preprocessing, model training, standard analysis, metric, selection, or CSV
schema logic was changed.

Recovery: install this patch and run the same study with `--resume`. Completed
file markers are reused, so training is skipped and only summaries/plots are
finalized.
