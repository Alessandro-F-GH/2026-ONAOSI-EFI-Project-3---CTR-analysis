# Patch summary: native canonical waveforms and model-input transforms

## Architecture

- Preprocessing always produces one canonical standard waveform dataset.
- Canonical windows are direct samples from the acquisition grid; no dense
  interpolated waveform is materialized.
- LED and CFD crossing times are linearly interpolated only between the two
  native samples surrounding the threshold.
- First differences are a model-input transform, selected with
  `input_transform: "differentiate"`.
- Differentiated arrays are cached below the training run directory and are not
  loadable as prepared datasets.
- Training persists the transform in checkpoints and summaries; evaluation and
  constructive-encoder inference replay it automatically.

## Configuration

```json
"input_transform": "none"
```

or:

```json
"input_transform": "differentiate"
```

Use standard prepared dataset paths in both training and evaluation configs.
`model.input_transform` remains a compatibility alias. Old ML preprocessing
keys `upsample_step_ps` and `subsample_factor` are accepted but ignored.

## Tests

`pytest -q` covers:

- standard and differentiated training input paths;
- cache isolation from the canonical prepared-dataset format;
- checkpoint-driven transform replay in evaluation;
- end-to-end tiny train/evaluate runs for both input representations;
- LED and CFD local crossing interpolation;
- native-grid window extraction and ignored legacy upsampling arguments.

## Energy/timing MLP extension

- Joint preprocessing can now persist both energy and timing native-grid windows
  in one canonical dataset.
- Separate `energy_led_time_fs` and `timing_led_time_fs` arrays make the target
  source explicit.
- Training config adds `prediction.input_waveforms` and `prediction.target`.
- Evaluation replays waveform source, target source and input transform from the
  checkpoint.
- MLP training now always applies a final analytic
  residual-bias calibration on the training split and saves only the calibrated
  best checkpoint. Optional epoch-end calibration remains available.
- Existing configs remain compatible through the default
  `energy + prepared_led` prediction view.

The expanded test suite covers energy/timing views, automatic timing-model
evaluation, both zero-bias modes, standard/differentiated inputs, and local LED/CFD
interpolation.

## Top useful correction diagnostics

- `ml_evaluate` can rank ML-corrected evaluation events by the reduction in
  absolute error with respect to the known TOF, rather than by raw prediction
  magnitude.
- The evaluation log and metrics CSV report the largest useful correction in ps,
  its event identifier and dataset row, plus useful/wrong correction counts.
- Per-model artifacts include a ranked CSV, JSON summary, and waveform plots
  showing the exact input pair, input-pair asymmetry, and raw-to-corrected timing
  movement.
- `scripts/plot_top_corrections.py` runs the same event-level analysis directly
  from one prepared dataset and one checkpoint or training run.
- Tests cover ranking semantics (including the 50 ps versus 2 ps case) and plot
  artifact generation.

## Linear SVR and epsilon scanning

- Added `linear_svr` as an automatically discovered trainable model.
- The model uses one shared linear score and the pair difference
  `w^T(s1 - s2)`, matching the ordered-pair convention of the MLP.
- Every training run scans all configured epsilon values with scikit-learn
  `LinearSVR` and saves `epsilon_scan.csv`.
- Candidate selection supports residual variance, RMSE, or
  `variance + lambda * bias^2`; variance is the default.
- Final arithmetic train-bias calibration is enforced through the same explicit
  pair-level output offset used by the MLP.
- Added energy-to-energy, timing-to-timing, and energy-to-timing example configs.
- Removed the former waveform grouping modules, scripts, sidecars, configs,
  loader branches, checkpoint contracts, and tests.

## Evaluation differentiation compatibility

- Blind evaluation now materializes/reuses the same model-input transform cache used by training.
- Differentiated checkpoints therefore receive arrays with shape `[events, 2, L-1]` and are loaded by `CorrectionDataset` with `input_transform="none"`, preventing double differentiation and dtype/shape drift.
- The raw canonical dataset view is retained for correction-analysis plots.
- Optional evaluation keys: `input_transform_cache_dir`, `input_transform_chunk_size`, and `rebuild_input_transform_cache`.


## Model-output correlation and variance-bias loss

- `ml_evaluate` now writes `model_output_correlation.csv` and
  `plots/model_output_correlation.png` for the per-event predicted corrections
  of all compatible evaluated models. This uses model corrections rather than
  corrected timestamps, avoiding an artificially high correlation from the
  common raw LED term.
- The former `mse_bias` loss is replaced by `var_bias`:
  population residual variance plus the existing normalized squared-bias
  penalty. Included training and experiment configs now use `var_bias`.
