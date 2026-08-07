# Compact study reporting

The folder study now keeps fold-level rows only in the private `_state/` directory for resume support.
The final user-facing outputs are:

- `summary_results.csv`: one row for the overall CV-selected best configuration per ROOT file and channel mode;
- `model_loss_results.csv`: one row per ROOT file, channel mode, model, and loss, with the best transform/window chosen by mean CV CTR;
- `summary_plots/<file>_best.png`: validation and blind Gaussian fits for the overall winner;
- `summary_plots/<file>_best_ctr_vs_window.png`: best mean CV CTR versus window size, one line per model and one panel per channel mode;
- `summary_plots/ctr_vs_voltage.png`: optional voltage scan plot.

Model, transform, loss, hyperparameters, and window are selected using cross-validation only. Blind data are used only for reporting.

Voltage extraction is optional and uses a configurable regex. The supplied example recognizes names such as `45V_400mV.root`.

For a previously completed study that failed during old plot finalization, run:

```powershell
python scripts\ml_experiment.py `
  --config config\experiments\folder_window_channel_study.json `
  --resume
```

Existing training is reused. The model/loss table and window-size plots are
backfilled directly from `_state/all_results.csv` when the original compact
summary already exists. Preprocessing is rebuilt only when the main compact
summary itself is missing; completed training blocks are not retrained.
