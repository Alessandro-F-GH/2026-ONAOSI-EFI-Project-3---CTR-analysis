# Folder-driven ML experiment study

The ML experiment interface is now intentionally narrow. One command expands a
folder of ROOT files into a reproducible study over channel modes, LED-relative
windows, input representations, common losses and model-specific hyperparameter
spaces.

```bash
python scripts/ml_experiment.py \
  --config config/experiments/folder_window_channel_study.json \
  --restart \
  --rebuild-preprocessing
```

Use `--dry-run` to validate the configuration and print the number of planned
contexts without opening a ROOT file. Use `--resume` to continue an interrupted
study.

## Configuration responsibilities

The study configuration contains only scientific axes and common protocol:

- `data.root_folder`, `root_glob`, `recursive`;
- LED-relative windows as `start_ns` and `end_ns`;
- channel modes;
- input transformations;
- common losses;
- selected model-space IDs;
- standard methods (`LED` and `CFD`, both enabled by default);
- modality-specific preprocessing;
- cross-validation and robust selection settings;
- common CTR-fit settings.

Model architecture and optimization ranges live independently in:

```text
config/model_spaces/mlp.json
config/model_spaces/light_cnn.json
```

A model-space file cannot contain ROOT paths, prepared-dataset paths, windows,
channel modes or input transformations.

## Channel modes

The mode registry is fixed:

| Mode | Input | Target and LED baseline |
|---|---|---|
| `energy_to_energy` | energy waveforms | energy LED |
| `energy_to_timing` | energy waveforms | timing LED |
| `timing_to_timing` | timing waveforms | timing LED |
| `energy_timing_to_timing` | energy then timing per detector | timing LED |

Combined inputs preserve modality boundaries. Transformations are applied to
each modality separately. For `concatenate_diff`, the order is:

```text
energy raw, energy derivative, timing raw, timing derivative
```

No derivative is formed across the energy/timing boundary.

The `normalize` transform keeps the original input length and applies a
featurewise z-score. For time sample `t`, statistics are fitted only on the
training partition of the current CV fold:

```text
z[event, channel, t] = (x[event, channel, t] - mean_train[t]) / std_train[t]
```

The two detector channels share `mean_train[t]` and `std_train[t]`, preserving
pair-swap symmetry. Validation and blind events never contribute to these
statistics.

## Modality-specific preprocessing

The largest requested window is materialized once for each ROOT file. Smaller
windows are zero-copy slices.

Energy waveforms are cached with both relevant LED alignments when timing
channels are available:

- `energy_to_energy` uses energy waveforms aligned to the energy-channel LED;
- `energy_to_timing` and `energy_timing_to_timing` use energy waveforms aligned
  to the timing-channel LED;
- timing waveforms are always aligned to the timing-channel LED.

Thus every configured window is relative to the LED estimator defining that
channel mode's target, rather than to one global alignment chosen at
preprocessing time.

Energy and timing preprocessing profiles are resolved separately. The supplied
configuration enables an energy low-pass filter. Timing low-pass filtering is
disabled as a fixed study invariant, not as a hyperparameter:

```json
"energy": {
  "denoising": {
    "enabled": true,
    "method": "butterworth_lowpass",
    "cutoff_GHz": 0.5,
    "order": 4
  }
},
"timing": {
  "denoising": {"enabled": false}
}
```

The standalone analysis pipeline under `scripts/analyze_ctr.py` and `utils/` is
not used or modified by this study runner.

## Development and blind split for cross-validation

The study runner does not create a preliminary fixed validation partition.
Each ROOT file is split directly as:

```text
development block | guard gap | blind block
```

The full development block is then divided by `cross_validation.n_splits`.
Internally, the prepared-dataset compatibility structure stores development
events in `train`, leaves `validation` empty, and stores the held-out block in
the physically separate blind dataset.

For a contiguous split, only one `guard_gap_events` interval is excluded: the
one separating development from blind data. The old
`initial_validation_fraction` setting is ignored and has been removed from the
example study configuration.

Because this changes the event set, checkpoints created by the former
train/initial-validation/blind preprocessing protocol cannot be resumed. The
runner detects that mismatch and requires `--restart` rather than silently
mixing results from the two protocols.


## Standard-method comparison

The study evaluates LED and CFD as non-trainable competitors by default:

```json
"standard_methods": ["led", "cfd"]
```

They use the exact validation and blind masks defined for each CV fold. LED is the
channel-mode target estimator. CFD is selected from the same waveform family as
the target: energy CFD for `energy_to_energy`, timing CFD for the three timing-LED
target modes. Preprocessing stores both energy and timing CFD timestamps so this
choice is explicit rather than inferred from one generic array.

Standard rows report MSE, RMSE, bias, CTR, Gaussian-fit statistics and improvement
relative to LED. They have `model_type=standard_method`,
`loss_id=evaluation_mse`, and no input transform, trial or waveform window. They
are included in `model_loss_results.csv` and may become the overall winner in
`summary_results.csv`. They are excluded from CTR-versus-window plots because no
window is optimized.

## Robust outlier rejection

Outlier rejection is fitted independently inside each CV fold using only its
training events. For the LED estimator defining the channel-mode target:

\[
 m=\operatorname{median}(\Delta t_{\rm LED}),
 \qquad
 s=1.4826\,\operatorname{MAD}(\Delta t_{\rm LED}),
\]

and an event is retained when

\[
 \frac{|\Delta t_{\rm LED}-m|}{s}\le z_{\max}.
\]

If MAD is zero, the implementation falls back to the Gaussian-consistent IQR
scale. The same fold-training center and scale are applied to that fold's
validation events and to the blind dataset. The mask is shared by every model,
loss, transformation and window inside the same ROOT file, channel mode and
fold.

## Selection and blind audit

Hyperparameters and windows are selected exclusively from cross-validation.
The default selection metric is mean validation CTR.

The blind dataset is evaluated for each CV-selected hyperparameter set at every
window only to assess validation quality. It never changes the selected window.
The study reports:

- Pearson and Spearman agreement between CV and blind CTR across windows;
- mean CV-to-blind CTR gap;
- blind rank of the CV-selected window;
- diagnostic blind regret of the CV-selected window.

Every selected configuration file contains:

```json
"selection_source": "cross_validation_only",
"blind_used_for_selection": false
```

## Results

All fold-level, aggregate and blind-audit metrics are stored in one compact,
numeric long-format file:

```text
results/studies/<study>/_state/all_results.csv
```

The CSV contains only numeric values: categorical fields are integer-coded,
row identifiers are split into two exact 48-bit integers, and paths, labels,
hyperparameter dictionaries, and error messages are not repeated in every row.
They are stored once in:

```text
results/studies/<study>/_state/results_metadata.json
```

The runner decodes this sidecar transparently when a study is resumed. Record
codes distinguish trial definitions, validation folds, blind folds, aggregate
statistics, and CV-versus-blind diagnostics. Hyperparameters are stored once per
trial in the metadata rather than as repeated JSON strings in the CSV.

The reported uncertainty is fold-to-fold variability, not the covariance of the
Gaussian fit. Relative CTR improvement is computed on the same events as:

\[
100\,\frac{\mathrm{CTR}_{\rm LED}-\mathrm{CTR}_{\rm model}}
{\mathrm{CTR}_{\rm LED}}.
\]

The human-readable compact tables are `summary_results.csv` and
`model_loss_results.csv`. The former stores one overall CV-selected winner per
file/mode; the latter retains every model/loss winner plus LED and CFD. Final plots
include the selected Gaussian fits, the best trainable-model CTR versus window
size, and the optional best-result CTR versus voltage.
