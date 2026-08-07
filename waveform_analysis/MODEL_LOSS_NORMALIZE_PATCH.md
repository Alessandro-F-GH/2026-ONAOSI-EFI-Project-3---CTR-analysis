# Model/loss reporting, window scan, and feature normalization patch

## New compact table

`model_loss_results.csv` contains one row for every completed:

```text
ROOT file × channel mode × model × loss
```

For each row, the input transform and window are selected using the lowest mean
cross-validation metric. Blind metrics are copied only after the CV choice is
fixed.

## New plot

Each ROOT file receives:

```text
summary_plots/<file>_best_ctr_vs_window.png
```

Each panel is a channel mode and each line is a model. At a fixed model and
window, the plotted CTR is the lowest completed mean CV CTR over compatible
losses and input transforms.

## New `normalize` input transform

For each time position `t`, training-fold statistics are computed by pooling all
training events and both detector channels:

```text
z[event, channel, t] = (x[event, channel, t] - mean_train[t]) / std_train[t]
```

The statistics are saved in the checkpoint and reused for validation and blind
data. They are never fitted from the complete prepared dataset, so the transform
is cross-validation safe. Zero-variance features use a scale of one.

## Resume/backfill

Report markers now contain a schema version. `--resume` can backfill the new
model/loss table and window plot from the existing private result state without
retraining completed blocks. A completed-file marker is also checked against the
current experiment grid, so adding `normalize` runs only the missing normalize
blocks while reusing all previously completed transforms.
