# Lightweight shared-branch CNN implementation

## Pair model

The new model type is `cnn_regressor`. It preserves the same single-model,
shared-branch convention used by the MLP:

```text
g1 = g_cnn(signal_1)
g2 = g_cnn(signal_2)
predicted_correction = g1 - g2 + pair_output_bias_ps
```

Both detector signals always use exactly the same convolutional and dense
parameters. The final scalar pair bias is reserved for the existing mandatory
training-residual bias calibration.

## Default low-compute architecture

The supplied configurations use:

```json
"channels": [8, 16, 24],
"kernel_sizes": [17, 9, 5],
"strides": [8, 4, 2],
"dilations": [1, 1, 1],
"adaptive_pool_length": null,
"dense_units": []
```

The total temporal stride is 64. Each convolution sees a local temporal region,
while the large stride removes the heavy redundancy caused by waveform
autocorrelation. `dense_units = []` means there is no hidden dense network: the
flattened downsampled CNN representation is connected directly to one scalar
output layer.

Typical model sizes are:

| Input length | Encoded length | Parameters |
|---:|---:|---:|
| 5,119 | 80 | 5,226 |
| 21,601 | 338 | 11,418 |

The parameter count therefore remains small even when the waveform contains more
than twenty thousand samples.

## Configurations

- `config/ml_train_cnn.json`: differentiated energy waveform input.
- `config/ml_train_cnn_raw.json`: matched raw-waveform control.
- `config/ml_evaluate_cnn.json`: evaluates only `cnn_regressor` checkpoints and
  skips incompatible legacy runs.

The raw and differentiated training configs intentionally use the same model,
optimizer, seed, loss, batch size, and stopping criteria. This makes the input
representation the controlled experimental variable.

## Training and evaluation

```bash
python scripts/ml_train.py --config config/ml_train_cnn.json --restart
python scripts/ml_train.py --config config/ml_train_cnn_raw.json --restart
python scripts/ml_evaluate.py --config config/ml_evaluate_cnn.json
```

The model uses the existing neural-model training logic: Adam, optional mixed
precision, gradient clipping, early stopping, optional bias-aware MSE, baseline
guard, checkpoint replay, and mandatory final zero training-residual-bias
calibration.

## Validation

The test suite checks model discovery, parameter count, pair antisymmetry,
checkpoint reconstruction, end-to-end training, final bias calibration, and
blind-evaluation replay.
