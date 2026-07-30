# Waveform timing-analysis pipeline

## Main commands

```bash
python scripts/ml_preprocess.py --config config/ml_preprocess.json
python scripts/ml_train.py --config config/ml_train_mlp.json --restart
python scripts/ml_train.py --config config/ml_train_shapelet.json --restart
python scripts/ml_experiment.py --config config/experiments/model_experiment.json --restart
python scripts/ml_experiment.py --config config/experiments/shapelet_experiment.json --restart
python scripts/fit_linear_spline.py --config config/linear_spline.json --restart
python scripts/ml_evaluate.py --config config/ml_evaluate.json
```

Optional high-frequency denoising is configured in
`config/ml_preprocess.json` under `waveform.denoising`. It is disabled by
default.

## Optional timing-channel LED reference

Preprocessing can read channels 3 and 4 only to estimate LED timestamps and
align the energy-channel ML windows. The ML arrays still contain channels 1 and
2 exclusively.

Enable the supplied example with:

```bash
python scripts/ml_preprocess.py --config config/ml_preprocess_timing_led.json --rebuild
```

Equivalent settings in a custom preprocessing configuration are:

```json
"channels": {
  "energy": [1, 2],
  "polarities": [1, 1],
  "timing": [3, 4],
  "timing_polarities": [1, 1]
},
"waveform": {
  "timing_channel_led": {
    "enabled": true,
    "baseline_samples": 500,
    "search_trigger_threshold_mV": 50.0,
    "analysis_crop_ns": {"before": 5.0, "after": 80.0},
    "upsample_step_ps": 2.5,
    "led_threshold_mV": 7.0,
    "denoising": {"enabled": false}
  }
}
```

With this mode:

- `led_time_fs.npy` comes from timing channels 3 and 4;
- each energy waveform window is centered on that timing-channel LED crossing;
- `cfd_time_fs.npy`, amplitudes, noise and trigger indices remain energy-channel quantities;
- `windows_mV.npy` contains only energy-channel samples;
- timing-channel waveforms are discarded after LED extraction.

The raw-cache fingerprint includes this choice, so switching LED source requires
and automatically identifies a distinct preprocessing cache.

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
