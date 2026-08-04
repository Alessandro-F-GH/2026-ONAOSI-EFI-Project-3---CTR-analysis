# Waveform timing-analysis pipeline

## Main commands

Build a joint canonical dataset containing both energy and timing waveforms:

```bash
python scripts/ml_preprocess.py --config config/ml_preprocess_timing_led.json --rebuild
```

Train separate MLP corrections from the same prepared dataset:

```bash
# Energy waveforms -> energy-channel LED correction
python scripts/ml_train.py --config config/ml_train_mlp.json --restart

# Timing waveforms -> timing-channel LED correction
python scripts/ml_train.py --config config/ml_train_mlp_timing.json --restart

# Optional cross-channel task: energy waveforms -> timing-channel LED correction
python scripts/ml_train.py --config config/ml_train_mlp_energy_to_timing.json --restart

# Linear SVR with an epsilon scan (energy waveforms -> energy LED)
python scripts/ml_train.py --config config/ml_train_linear_svr.json --restart

# Lightweight shared-branch 1-D CNN on differentiated waveforms
python scripts/ml_train.py --config config/ml_train_cnn.json --restart

# Matching raw-waveform CNN for a controlled raw-vs-difference comparison
python scripts/ml_train.py --config config/ml_train_cnn_raw.json --restart

python scripts/ml_evaluate.py --config config/ml_evaluate_cnn.json
```

`ml_evaluate` can also rank the events for which the ML model makes the largest
useful correction. A correction is useful when it reduces the absolute distance
from the known TOF:

```text
improvement_ps = |raw - true_TOF| - |corrected - true_TOF|
```

Enable it in `config/ml_evaluate.json`:

```json
"correction_analysis": {
  "enabled": true,
  "top_n": 10,
  "minimum_improvement_ps": 0.0,
  "save_waveform_plots": true
}
```

For each model and blind test, evaluation writes `top_right_corrections.csv`, a
JSON summary, and one waveform diagnostic per ranked event under
`top_corrections/`. The evaluation log and metrics CSV also report
`top_right_correction_ps`.

The same analysis can be run independently:

```bash
python scripts/plot_top_corrections.py \
  --dataset datasets/central_source_all/blind_test_timing_led \
  --model results/all/train/<model-run> \
  --output results/top_corrections \
  --top-n 10
```

## Energy and timing waveform inputs

When `channels.timing` is configured and
`waveform.timing_channel_led.enabled` is true, preprocessing writes one canonical
prepared dataset with:

- `windows_mV.npy`: standard energy-channel windows;
- `timing_windows_mV.npy`: standard timing-channel windows;
- `energy_led_time_fs.npy`: locally interpolated energy LED timestamps;
- `timing_led_time_fs.npy`: locally interpolated timing LED timestamps;
- `led_time_fs.npy`: the legacy/default LED target, retained for compatibility.

Both waveform families remain on their original acquisition grids. No waveform
upsampling is materialized. Energy windows keep the existing alignment behavior:
in timing-reference mode they are aligned to the timing LED. Timing windows are
aligned to their own timing LED.

Training selects a view from that single dataset:

```json
"prediction": {
  "input_waveforms": "energy",
  "target": "energy_led"
}
```

or:

```json
"prediction": {
  "input_waveforms": "timing",
  "target": "timing_led"
}
```

The original cross-channel experiment remains available with
`input_waveforms = "energy"` and `target = "timing_led"`. Old configs without a
`prediction` section remain compatible and resolve to energy waveforms with the
legacy `prepared_led` target.

## Optional differentiated model input

Set `input_transform` to `none`, `differentiate`, or `concatenate_diff`.
`concatenate_diff` stores the raw waveform samples first and then appends the
first differences, producing `2L-1` input values from a waveform of length `L`.
Differentiation is applied
after selecting the energy/timing waveform family and is cached inside the model
run directory. Evaluation reads both the waveform source and transform from the
checkpoint.

## Enforced final train-bias calibration

Every MLP, lightweight CNN, and linear SVR checkpoint is calibrated after model selection.
The arithmetic mean of the corrected residual on the training split is measured
and removed through `pair_output_bias_ps`. The calibrated state overwrites the
best checkpoint and is the state used during evaluation.

The final calibration always uses residual semantics:

```text
mean(target - calibrated_prediction) = 0
```

For the MLP, an optional `training.zero_bias_constraint` can still perform
additional per-epoch calibration. Final train-bias removal remains mandatory for
both model families.


## Lightweight shared-branch 1-D CNN

`cnn_regressor` preserves the same detector-pair structure used by the MLP:

```text
correction(s1, s2) = g_cnn(s1) - g_cnn(s2) + pair_output_bias
```

The two detector waveforms always pass through the same CNN weights. The default
configuration is designed for long, strongly autocorrelated signals and limited
compute: three small strided convolutions reduce the temporal length by a factor
of 64 before a single linear output layer. No dense layer is applied directly to
the original waveform, so the parameter count remains small even for 20k-sample
inputs.

The first configuration uses differentiated input and a linear head:

```json
"channels": [8, 16, 24],
"kernel_sizes": [17, 9, 5],
"strides": [8, 4, 2],
"adaptive_pool_length": null,
"dense_units": []
```

`adaptive_pool_length = null` preserves every downsampled temporal position.
`dense_units = []` means the head is one simple `Linear` layer. Use
`config/ml_train_cnn_raw.json` as the otherwise matched raw-waveform control.

## Linear SVR epsilon scan

`linear_svr` trains on the normalized pair difference `s1 - s2`, preserving the
same shared-branch logic as the MLP. A true SVR is fitted with an
epsilon-insensitive objective for every value in `model.epsilon_values`. The best
epsilon is selected with `model.loss.type`: `variance` (default), `rmse`, or
`variance_bias`, where the latter is `variance + bias_weight * bias^2`.

The scan is written to `epsilon_scan.csv`; the selected checkpoint remains fully
compatible with `ml_evaluate.py`.

## Position-aware shapelet regressor

`shapelet_regressor` replaces the previous CART/interval-feature strategy. It
uses one shared differentiable model `g` for both time channels:

```text
correction(s1, s2) = g(s1) - g(s2)
```

The model is trained directly against the complete LED correction target:

```text
y = LED_delta - true_TOF
```

No artificial single-channel targets or iterative tree refits are required.

### Physical time configuration

All temporal controls are written in nanoseconds:

```json
"shapelets": {
  "lengths_ns": [0.08, 0.16, 0.32, 0.64],
  "count_per_length": [8, 8, 8, 8],
  "search_region_ns": {
    "start": -1.0,
    "stop": 3.0
  },
  "stride_ns": 0.02
}
```

At training time, the pipeline reads `relative_time_ps.npy`, verifies a uniform
time grid and converts each nanosecond value into an integer sample count. The
requested and actual physical values are saved in the checkpoint and
`training_summary.json`. Evaluation therefore rebuilds exactly the same model
without relying on an implicit hard-coded sampling frequency.

### Features produced by each shapelet

Each learnable shapelet scans only the configured physical search region. A
soft minimum over normalized sliding-window distances produces:

- normalized shape distance;
- soft match position in ns;
- local waveform mean;
- local waveform standard deviation;
- optional signed correlation.

The default linear head keeps the model interpretable. A small MLP head can be
selected through `model.head.type`.

### Efficient matching

Sliding distances are computed with batched one-dimensional convolutions. The
model never constructs a tabular matrix with one column for every one of the
20,000 waveform samples. Computation is controlled mainly by:

- number of shapelets;
- physical shapelet lengths;
- search-region duration;
- `stride_ns`;
- batch size.

Train with:

```bash
python scripts/ml_train.py \
  --config config/ml_train_shapelet.json \
  --restart
```

Main artifacts include:

- `checkpoints/best.pt`;
- `training_metrics.csv`;
- `learned_shapelets.npz`;
- `shapelet_metadata.csv`;
- `shapelet_head_features.csv`;
- one plot for each shapelet-length bank.

## Pipeline separation

- `ml_pipeline/models/`: trainable ML models only.
- `ml_pipeline/standard_methods/`: LED, CFD and linear-spline estimators.
- `ml_pipeline/experiments.py`: model-independent search/CV orchestration.
- `ml_pipeline/evaluation.py`: blind comparison of models and standard methods.

## Adding a trainable model

Create one file under `ml_pipeline/models/` and export:

```python
MODEL_SPEC = ModelSpec(
    name="new_model",
    builder=build,
    validator=validate_config,
    training_validator=validate_training_config,
    trainer=train,
)
```

The registry discovers the file automatically.

## Constructive encoder

A new `constructive_mlp_encoder` model grows identity-activated units
sequentially. Previous units are frozen, each new unit receives the raw waveform
plus all frozen hidden values, and growth stops when validation-RMSE improvement
is marginal. The accepted hidden values can be exported as a compressed dataset
with `scripts/ml_encode_constructive.py`.

```bash
python scripts/ml_train.py --config config/ml_train_constructive_encoder.json --restart
```

Because the activation is identity, the encoder is a learned affine projection;
it is intended as supervised dimensionality reduction rather than a nonlinear
network.
