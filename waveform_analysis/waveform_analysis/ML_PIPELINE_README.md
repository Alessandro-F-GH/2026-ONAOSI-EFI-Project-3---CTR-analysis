# Energy-only CNN correction pipeline

This is a **new and separate pipeline**. The existing `scripts/analyze_ctr.py` and the existing `utils/` classical CTR pipeline are unchanged.

The new pipeline reads only the two configured energy waveform branches. With the default configuration these are:

```text
samples_ch1
samples_ch2
```

It does not request `samples_ch3` or `samples_ch4`, and its event selection uses only energy-channel quantities.

## Estimators

For every selected event, the pipeline calculates three energy-channel TOF estimates on the same frozen test events:

1. **Energy LED standard**
2. **Energy CFD standard**
3. **Energy LED + CNN correction**

The correction model is constrained to

```text
y_theta(s1, s2) = g_theta(s1) - g_theta(s2)
```

where `g_theta` is one shared 1D CNN. The model receives local energy-waveform windows centered on each energy-channel LED crossing. Absolute acquisition time is therefore not part of the CNN input.

All reported CTR values use the repository's existing iterative Gaussian-fit implementation. The final output reports **bias** and **CTR** for all three methods and produces one Gaussian-fit plot for each method.

## Data separation

The default split is:

```text
80% training
10% validation
10% blind final test
```

The final test indices are stored in the frozen split cache, but the training code never creates a test `Dataset` and never evaluates it. The test set is opened only by `scripts/ml_evaluate.py` after checkpoint selection.

Photopeak parameters, waveform normalization, LED centering, and timing calibration are obtained from the training split only. The same selected blind-test events are used for LED, CFD, and corrected LED.

## Installation

From the repository root:

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements_ml.txt
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements_ml.txt
```

For a CUDA build of PyTorch, install the appropriate PyTorch wheel for the machine before running `pip install -r requirements_ml.txt`.

## Configuration

Edit:

```text
config/ml_pipeline_config.json
config/cnn_config.json
```

The most important pipeline settings are:

```json
{
  "data": {
    "input_root": "data/45V-400mV/converted.root",
    "true_tof_ps": 0.0
  },
  "waveform": {
    "led_threshold_mV": 7.0,
    "cfd_fraction": 0.045,
    "upsample_step_ps": 2.5,
    "subsample_factor": 2
  }
}
```

`subsample_factor: 2` means that the extracted local window keeps one point every two points of the configured upsampled grid.

Parallel execution is configured in one place:

```json
{
  "parallelization": {
    "preprocessing_backend": "process",
    "preprocessing_workers": 4,
    "preprocessing_chunksize": 8,
    "training_num_workers": 4,
    "prefetch_factor": 2,
    "persistent_workers": true,
    "pin_memory": true,
    "torch_num_threads": 0,
    "torch_num_interop_threads": 0
  }
}
```

Use `preprocessing_workers: 0` and `training_num_workers: 0` for serial debugging. On a memory-constrained computer, reduce worker counts and `prefetch_factor`.

## Execution from terminal

### Recommended staged execution

Prepare the energy-only cache and freeze the split:

```bash
python scripts/ml_prepare.py \
  --pipeline-config config/ml_pipeline_config.json
```

Train the CNN:

```bash
python scripts/ml_train_cnn.py \
  --pipeline-config config/ml_pipeline_config.json \
  --cnn-config config/cnn_config.json
```

Run the final blind-test comparison:

```bash
python scripts/ml_evaluate.py \
  --pipeline-config config/ml_pipeline_config.json \
  --cnn-config config/cnn_config.json
```

### One-command execution

```bash
python scripts/run_energy_ml_pipeline.py \
  --pipeline-config config/ml_pipeline_config.json \
  --cnn-config config/cnn_config.json
```

### Resume interrupted training

The trainer writes `last.pt` every configured number of batches and also writes `interrupted.pt` after `Ctrl+C`.

```bash
python scripts/ml_train_cnn.py \
  --pipeline-config config/ml_pipeline_config.json \
  --cnn-config config/cnn_config.json \
  --resume
```

The per-epoch shuffle is deterministic, and the checkpoint stores the next batch index, optimizer, scheduler, mixed-precision scaler, random states, history, and data/configuration fingerprints.

### Start training from scratch

```bash
python scripts/ml_train_cnn.py \
  --pipeline-config config/ml_pipeline_config.json \
  --cnn-config config/cnn_config.json \
  --restart
```

### Rebuild preprocessing or split caches

```bash
python scripts/ml_prepare.py \
  --pipeline-config config/ml_pipeline_config.json \
  --rebuild-cache \
  --rebuild-split
```

A changed input file, waveform setting, subsampling factor, or channel definition invalidates the preprocessing fingerprint. A changed split, photopeak, or energy-selection setting invalidates the split fingerprint.

## Main outputs

Default output root:

```text
results/energy_ml_cnn/
```

Important files:

```text
cache/energy_dataset/manifest.json
cache/split/manifest.json
checkpoints/last.pt
checkpoints/best_validation.pt
checkpoints/training_history.csv
plots/training/loss_curves.png
plots/training/corrected_std_curves.png
plots/training/ctr_curves.png
plots/final_evaluation/energy_led_standard_gaussian_fit.png
plots/final_evaluation/energy_cfd_standard_gaussian_fit.png
plots/final_evaluation/energy_led_plus_cnn_correction_gaussian_fit.png
plots/final_evaluation/method_comparison.png
plots/final_evaluation/swap_test_gaussian_fit.png
final_metrics.csv
final_evaluation.json
```

The energy-cache manifest explicitly records `branches_read` and an empty `timing_channel_branches_read` list so the no-timing-channel condition can be audited.

## Calibration and bias

The standard LED and standard CFD offsets are estimated from their respective training distributions with the same iterative Gaussian strategy used for final CTR:

```text
calibration offset = fitted training mean - configured true TOF
```

The corrected estimator reuses the **same frozen LED offset** as the uncorrected LED estimator, matching `TOF_LED - C_LED - y_theta`. This prevents a non-zero-mean learned correction from being hidden by a second calibration. No test recentering is performed. Final bias is

```text
bias = fitted blind-test mean - true TOF
```

This gives the requested comparison:

```text
Energy LED standard vs Energy CFD standard vs Energy LED corrected
```

## Swap test

There is no mixed-channel training mode because it is mathematically redundant for the shared antisymmetric model. Final evaluation still runs the network on swapped test pairs and reports:

- correction antisymmetry error;
- corrected-estimator sign error;
- canonical versus swapped CTR difference;
- canonical versus swapped bias.

A failed swap test indicates an implementation or sign-convention problem, not useful channel-order information.
