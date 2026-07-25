# Energy-only antisymmetric ML correction pipeline

This is a **new and separate pipeline**. The existing classical pipeline (`scripts/analyze_ctr.py` and the existing `utils/` analysis code) is unchanged.

The ML pipeline reads only the two configured energy waveform branches (by default `samples_ch1` and `samples_ch2`). It never requests timing-channel waveforms, and its selection uses only energy-channel quantities.

## Selectable models

Three correction models are implemented:

1. **1D CNN** — `config/cnn_config.json`
2. **Fixed-window time-series MLP** — `config/time_series_regressor_config.json`
3. **Catch22 random forest** — `config/catch22_random_forest_config.json`

All three preserve the shared antisymmetric estimator

```text
y_theta(s1, s2) = g_theta(s1) - g_theta(s2)
```

so that

```text
y_theta(s2, s1) = -y_theta(s1, s2)
y_theta(s, s) = 0
```

The CNN and MLP learn `g_theta` directly from normalized waveform samples. The random-forest model first converts each single-channel waveform into Catch22 features, then learns the same shared single-channel map in feature space:

```text
g_theta(s) = g_theta(catch22(s))
```

## Common target and loss

The training target is computed only from the training split:

```text
target = TOF_LED - mean_train(TOF_LED)
```

Every model is selected using the same calibration-invariant standard-deviation
loss, expressed directly in ps:

```text
residual = y_theta(s1, s2) - target
loss_ps = sqrt(mean((residual - mean(residual))^2))
```

Subtracting the residual mean makes the objective insensitive to a constant
calibration offset. The log therefore reports values such as `val std loss
62.3 ps`, rather than an MSE in ps².

Because the optimized quantity and checkpoint-selection metric changed, do not
resume checkpoints produced by the older MSE version. Start the model again with
`--restart`.

The corrected estimator is

```text
TOF_corrected = TOF_LED - C_LED - y_theta(s1, s2)
```

where `C_LED` is obtained only from the training split.

### How the random forest fits the shared map

A standard random forest predicts one target from one feature vector. To retain the exact shared form `g(s1)-g(s2)`, the implementation uses staged residual fitting.

At a given stage, with pair residual

```text
residual = target - (g(s1) - g(s2))
```

the same forest is trained on both channels with pseudo-targets

```text
channel 1 -> +residual / 2
channel 2 -> -residual / 2
```

The stage is added to the shared map `g`. Validation and checkpoint selection
always use the actual pairwise standard-deviation loss above, not the internal
random-forest split criterion. Before fitting each residual stage, its mean is
removed so the forest learns event-wise variation rather than calibration bias.

## Catch22 feature behavior

The default configuration uses Catch24:

```json
"catch24": true
```

This includes the 22 Catch22 dynamical features plus waveform mean and standard
deviation. The two additional scale features are useful here because LED time
walk can depend on pulse amplitude. A strict morphology-only Catch22 ablation is
still available with

```json
"catch24": false
```

Catch22/Catch24 extraction is performed only for the frozen selected events in
the train, validation, and blind-test splits, after the energy-only quality cuts
and the training-fitted photopeak selection. Rejected events in the raw waveform
cache are not transformed. Features are cached in resumable chunks under the
energy dataset cache; changing the split, selection, or feature settings creates
a new fingerprinted cache.

## Final comparison

On the untouched blind test set, the pipeline compares:

1. Energy LED standard
2. Energy CFD standard
3. Energy LED corrected by the selected model

Every CTR uses the same iterative Gaussian-fit method. Final outputs include bias, CTR, fit plots, comparison plots, and the channel-swap test.

## Data separation

Default split:

```text
80% training
10% validation
10% blind final test
```

Photopeak parameters, LED centering, timing calibration, neural input normalization, and model fitting use training data only. Catch22 extraction is deterministic and may be cached for all events, but test features and labels are not used to fit or select the random forest.

## Installation

From the repository root:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements_ml.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements_ml.txt
```

The Catch22 implementation uses `aeon`, avoiding the need to compile the `pycatch22` C extension on Python 3.11 Windows.

## Shared configuration

Data, selection, waveform processing, parallelization, output paths, logging, and Gaussian fitting are configured in:

```text
config/ml_pipeline_config.json
```

`waveform.subsample_factor` affects only model inputs. LED and CFD crossings are calculated first at full configured interpolation resolution.

## Catch22 random-forest configuration

```json
{
  "model_type": "catch22_random_forest",
  "features": {
    "implementation": "aeon",
    "catch24": true,
    "outlier_norm": true,
    "replace_non_finite": true,
    "chunk_events": 512,
    "n_jobs": 4,
    "parallel_backend": "threading"
  },
  "random_forest": {
    "n_estimators": 200,
    "max_depth": null,
    "min_samples_split": 4,
    "min_samples_leaf": 2,
    "max_features": 0.75,
    "bootstrap": true,
    "max_samples": 0.8,
    "n_jobs": -1
  },
  "training": {
    "stages": 3,
    "stage_learning_rate": 0.5,
    "monitor": "validation_loss"
  },
  "checkpointing": {
    "every_trees": 50
  }
}
```

Parallelization is independently configurable for:

- Catch22 feature extraction: `features.n_jobs` and `features.parallel_backend`
- Random-forest construction: `random_forest.n_jobs`
- ROOT waveform preprocessing: `ml_pipeline_config.json`

On Windows, `parallel_backend: "threading"` is usually safer than process spawning.

## Symmetric channel-swap training augmentation

The pipeline configuration contains:

```json
"channel_swap_augmentation": {
  "enabled": true,
  "paired_batches": true
}
```

When enabled, each **training** event `(s1, s2)` is accompanied by a virtual
copy `(s2, s1)`.  The signed quantities are reversed without recomputing LED:

```text
(s1, s2),  LED,  target   ->   (s2, s1),  -LED,  -target
```

For CNN and MLP training, `paired_batches=true` places every canonical event and
its swapped copy in the same optimization batch.  Validation metrics, training
metrics, and blind-test metrics are still evaluated exactly once on the
canonical channel order.  The existing final swap test remains a separate
diagnostic.  Catch24 features are not extracted again: the cached feature pair
is only reversed in memory.

This augmentation is training-only and does not change the frozen split, the
photopeak selection, or the physical calibration convention.

For a canonical residual `e`, the swapped copy has residual `-e`.  Therefore the
standard deviation over the paired symmetric batch is

```text
std([e, -e]) = sqrt(mean(e^2))
```

which is the canonical residual RMSE in ps.  Unlike the canonical-only standard
deviation, this includes the squared residual mean and therefore penalizes a
learned constant bias.  Epoch metrics and checkpoint selection compute this
same value from the canonical predictions only, so validation/test inference is
not duplicated.

## Terminal execution

### 1. Prepare the shared energy-only waveform cache and frozen split

```bash
python scripts/ml_prepare.py --pipeline-config config/ml_pipeline_config.json
```

### 2. Train a model

CNN:

```bash
python scripts/ml_train.py --pipeline-config config/ml_pipeline_config.json --model-config config/cnn_config.json
```

Time-series MLP:

```bash
python scripts/ml_train.py --pipeline-config config/ml_pipeline_config.json --model-config config/time_series_regressor_config.json
```

Catch22 random forest:

```bash
python scripts/ml_train.py --pipeline-config config/ml_pipeline_config.json --model-config config/catch22_random_forest_config.json
```

The first random-forest run builds the Catch22 cache. Later runs with the same feature configuration reuse it.

### 3. Final blind-test evaluation

```bash
python scripts/ml_evaluate.py --pipeline-config config/ml_pipeline_config.json --model-config config/catch22_random_forest_config.json
```

### One-command execution

```bash
python scripts/run_energy_ml_pipeline.py --pipeline-config config/ml_pipeline_config.json --model-config config/catch22_random_forest_config.json
```

### Resume or restart random-forest training

```bash
python scripts/ml_train.py --pipeline-config config/ml_pipeline_config.json --model-config config/catch22_random_forest_config.json --resume
```

```bash
python scripts/ml_train.py --pipeline-config config/ml_pipeline_config.json --model-config config/catch22_random_forest_config.json --restart
```

Feature extraction resumes at the last completed chunk. Forest fitting checkpoints every configured number of trees and after every residual stage.

## Model-separated outputs

The waveform dataset and split caches remain shared. Alternative models are written to separate directories, for example:

```text
results/energy_ml_cnn/checkpoints/time_series_mlp/
results/energy_ml_cnn/checkpoints/catch22_random_forest/
results/energy_ml_cnn/plots/catch22_random_forest/
results/energy_ml_cnn/catch22_random_forest/final_metrics.csv
```

The Catch22 model additionally writes:

```text
feature_importance.csv
feature_importance.png
training_history.csv
best_validation.joblib
last.joblib
```

## Swap test

Mixed-channel training is not implemented. Final evaluation explicitly swaps the test channels and verifies correction antisymmetry, corrected-estimator sign consistency, CTR consistency, and bias consistency.

## Faster Catch24 extraction on Windows

The default Catch24 feature configuration now uses the compiled `pycatch22`
implementation and process-based parallelism:

```json
"features": {
  "implementation": "aeon",
  "catch24": true,
  "outlier_norm": true,
  "use_pycatch22": true,
  "replace_non_finite": true,
  "chunk_events": 2048,
  "checkpoint_every_chunks": 4,
  "n_jobs": -1,
  "parallel_backend": "loky"
}
```

`threading` can under-use the CPU when feature calculations spend time in
Python code because threads contend for the GIL. `loky` uses separate worker
processes, while `pycatch22` routes feature calculations through the compiled C
implementation. The transformer is now created and fitted once per extraction
run rather than once per chunk. Feature-cache flushing is batched every four
chunks to reduce disk synchronization overhead.

Install the accelerated dependency with:

```cmd
python -m pip install -r requirements_ml.txt
```

The extraction log reports per-chunk throughput, average throughput, ETA, and
whether a resumable checkpoint was written. On memory-constrained machines,
replace `n_jobs: -1` with a fixed value such as `4` or `6`. Reducing
`chunk_events` to `1024` also reduces the amount of data resident in each
worker batch.

Changing `use_pycatch22`, `catch24`, or `outlier_norm` creates a new feature
cache because these options can alter the extracted values. Changing only
`n_jobs`, `parallel_backend`, `chunk_events`, or `checkpoint_every_chunks`
reuses the same feature-value fingerprint.
