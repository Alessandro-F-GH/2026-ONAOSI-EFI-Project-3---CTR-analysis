# ML pipeline architecture

## Optional timing-channel LED preprocessing

The preprocessing layer supports two explicitly recorded conventions:

- `energy_channel_led`: channels 1 and 2 provide LED timestamps and window alignment;
- `timing_channel_led`: channels 3 and 4 provide LED timestamps and window alignment,
  while channels 1 and 2 remain the only ML inputs.

In timing-channel mode the event flow is:

```text
channel 3/4 waveform -> timing LED timestamp only
channel 1/2 waveform + timing LED timestamp -> aligned energy ML window
```

The timing waveform is never written to the prepared dataset. Energy-channel
CFD, amplitude, noise and trigger metadata remain unchanged. The prepared
dataset manifest records `led_timestamp_source`, `cfd_timestamp_source`,
`ml_window_alignment_source`, and `timing_channel_waveforms_saved` so downstream
runs cannot silently confuse the two preprocessing conventions.

## Trainable models

Every model is self-contained in `ml_pipeline/models/<model>.py` and exposes a
`MODEL_SPEC`. Automatic discovery is implemented by
`ml_pipeline/models/registry.py`.

A model module owns architecture construction, validation, training and
checkpoint-compatible configuration. The shared training entry point prepares
datasets, normalization, output paths and the `TrainingContext`.

## Shared position-aware shapelet regressor

The model type `shapelet_regressor` implements:

```text
correction(s1, s2) = g(s1) - g(s2)
```

The same shapelet bank and regression head are applied to both channels. This
enforces exact ordered-pair antisymmetry by construction.

### Why shapelets

The previous CART strategy summarized fixed waveform partitions and then fitted
a non-differentiable tree. The shapelet model instead learns localized pulse
motifs and their useful temporal scale directly from data. It preserves local
waveform morphology without treating more than 20,000 adjacent samples as
independent tabular columns.

### Nanosecond-based temporal controls

Temporal settings are defined in physical units:

```json
"shapelets": {
  "lengths_ns": [0.08, 0.16, 0.32, 0.64],
  "count_per_length": [8, 8, 8, 8],
  "search_region_ns": {"start": -1.0, "stop": 3.0},
  "stride_ns": 0.02,
  "softmin_temperature": 0.05
}
```

The trainer infers the sampling step from the prepared dataset's
`relative_time_ps` grid. It verifies uniform sampling, converts each requested
length and stride to samples, and rejects configurations that do not fit in the
requested search region. The resolved values are embedded in the checkpoint.

This means `0.16 ns` keeps the same physical interpretation when the acquisition
sampling step changes, while the corresponding number of samples changes
automatically.

### Differentiable shapelet matching

For each normalized shapelet `q_k` and each candidate waveform window, the model
computes normalized correlation using `conv1d`. The normalized squared distance
is:

```text
distance = 2 - 2 * correlation
```

A soft minimum over candidate positions provides differentiable match weights.
Each shapelet can emit:

- `distance`;
- `position_ns`;
- `local_mean`;
- `local_std`;
- `correlation`.

`position_ns` is the physical starting time of the matched subsequence, not a
sample index or normalized fraction.

### Pairwise objective

The model is optimized directly with:

```text
MSE(g(s1) - g(s2), LED_delta - true_TOF)
```

The optional bias-aware objective remains available through
`model.loss.type = "mse_bias"`. A shapelet-diversity penalty can be enabled with
`shapelets.diversity_weight_ps2` to discourage duplicate motifs.

### Initialization

The default initialization copies normalized subsequences from actual training
waveforms inside the configured search region. This avoids beginning with
patterns unrelated to detector pulses. `random_normal` initialization remains
available for controlled comparisons.

### Head

The default head is linear and maps shapelet features to a single-channel
correction `g(s)`. This makes coefficients directly inspectable in
`shapelet_head_features.csv`. An MLP head can be selected with:

```json
"head": {
  "type": "mlp",
  "hidden_units": [64, 16],
  "activation": "relu",
  "dropout": 0.0
}
```

### Scaling with long waveforms

The memory-heavy object is the batched distance map, whose approximate size is
proportional to:

```text
batch_size * shapelet_count * search_positions
```

It is not proportional to a materialized `events x 20000` feature table. Reduce
`batch_size`, increase `stride_ns`, narrow `search_region_ns`, or reduce the
shapelet count when GPU/CPU memory is limiting.

### Artifacts

- `checkpoints/best.pt`: common evaluator-compatible checkpoint;
- `training_metrics.csv`: pairwise train/validation history;
- `learned_shapelets.npz`: learned normalized motifs;
- `shapelet_metadata.csv`: requested/actual lengths and search timing;
- `shapelet_head_features.csv`: physical feature names and linear coefficients;
- `plots/shapelets_bank_*.png`: learned motifs by physical length.

## Random ordered-pair swap augmentation

Trainable gradient-based models can randomly reverse the two time-channel
signals for each training event:

```json
"training": {
  "random_pair_swap": true
}
```

A swap negates the target, LED difference, CFD difference and true TOF so the
ordered-pair convention remains physically consistent.

## Optional waveform denoising

High-frequency components can be attenuated during preprocessing with a
zero-phase Butterworth low-pass filter:

```json
"waveform": {
  "denoising": {
    "enabled": true,
    "method": "butterworth_lowpass",
    "cutoff_GHz": 1.0,
    "order": 4
  }
}
```

The filter is applied before trigger, LED/CFD and ML-window extraction.

## Experiments

`ml_pipeline/experiments.py` remains model-neutral and supports grid, random and
Optuna-TPE search, cross-validation, repeated seeds, waveform-window views and
final refitting. The shapelet experiment searches physical lengths, shapelet
counts, scan stride and soft-min temperature.

## Standard methods

LED, CFD and linear spline remain under `ml_pipeline/standard_methods/` and are
excluded from the trainable-model registry.

## Constructive identity MLP encoder

`constructive_mlp_encoder` grows a shared single-channel encoder one scalar unit
at a time. Unit `k` receives the normalized raw waveform and the outputs of all
previous frozen units:

```text
h_k(s) = identity(W_raw,k s + W_hidden,k [h_1(s), ..., h_{k-1}(s)] + b_k)
```

The pair correction remains antisymmetric:

```text
prediction = g(s1) - g(s2)
g(s) = sum_k output_weight_k * h_k(s)
```

Training begins with one unit. After the best checkpoint for that unit is found,
its input weights and output coefficient are frozen. A new unit is appended and
only that unit plus its new output coefficient are optimized. Growth stops when
the validation-RMSE improvement is below both the configured absolute and
relative thresholds.

Train with:

```bash
python scripts/ml_train.py \
  --config config/ml_train_constructive_encoder.json \
  --restart
```

Important mathematical limitation: because every activation is identity, the
complete cascade is affine in the normalized waveform. More units create a
supervised low-dimensional basis and a greedy optimization path, but they do not
add nonlinear expressive power over a single linear regressor.

### Constructive artifacts

- `constructive_growth.csv`: accepted/rejected unit history;
- `plots/constructive_growth.png`: train/validation RMSE versus accepted units;
- `encoder_effective_weights.npy`: equivalent raw-waveform projection for each unit;
- `encoder_effective_bias.npy`: equivalent affine bias for each unit;
- `encoder_output_weights.npy`: frozen scalar readout coefficients;
- `overall_effective_weight.npy`: the complete predictor collapsed to one raw-input vector;
- `encoder_units.csv`: unit norms and readout weights.

### Exporting compressed datasets

Use the frozen hidden-unit values as a reduced representation:

```bash
python scripts/ml_encode_constructive.py \
  --checkpoint results/47V/train/constructive_identity_encoder/checkpoints/best.pt \
  --dataset datasets/central_source_47V/train_validation \
  --dataset datasets/central_source_47V/blind_test \
  --output-root datasets_encoded/constructive_identity_encoder \
  --batch-size 256 \
  --overwrite
```

For `K` accepted units, each output dataset contains:

- `encoded_channels.npy` with shape `[events, 2, K]`;
- `encoded_pair_difference.npy` with shape `[events, K]`;
- `predicted_led_correction_ps.npy`;
- `target_led_correction_ps.npy`;
- original event identifiers and split indices;
- a manifest tying the representation to the exact checkpoint and normalization.

## Configurable Gaussian-fit cadence during training

The iterative Gaussian fit used to estimate CTR can be skipped on most epochs when checkpoint selection uses RMSE, loss, or arithmetic bias. Configure it under `training`:

```json
{
  "fit_interval_epochs": 10,
  "fit_train_during_training": false,
  "fit_validation_during_training": true
}
```

- `fit_interval_epochs: 1` preserves the original every-epoch behavior.
- `fit_interval_epochs: 10` fits only on epochs 10, 20, 30, ... .
- `fit_interval_epochs: 0` disables Gaussian fits during iterative training.
- Skipped epochs still compute predictions, RMSE, arithmetic bias, and the configured loss; CTR and Gaussian bias are stored as `NaN`.
- Final train and validation fits are always run after restoring the best checkpoint.
- If `selection_metric` is `validation_ctr`, validation fitting is forced every epoch because CTR is required for checkpoint selection. Train fitting remains configurable.

`training_metrics.csv` includes `train_fit_performed` and `validation_fit_performed` flags.
