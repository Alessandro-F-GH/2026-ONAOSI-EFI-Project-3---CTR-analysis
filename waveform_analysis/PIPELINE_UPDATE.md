# CTR pipeline update — restored working ML implementation

This tree uses the uploaded working repository as the source of truth for model and ML behavior.

## Preserved unchanged

The following scientific/training implementation is copied unchanged from the working version:

- `ml_pipeline/models/*`
- `ml_pipeline/torch_data.py`
- `ml_pipeline/training.py`
- `ml_pipeline/training_utils.py`
- `ml_pipeline/training_context.py`
- `ml_pipeline/losses.py`

Therefore waveform target construction, antisymmetric model definitions, normalization, LinearSVR fitting, constructive MLP training and CNN training are not replaced by the newer simplified runner implementation.

## New experiment infrastructure

- Permanent physical/photopeak cohort cache under `processed_data/selected_events`.
- Canonical permanent prepared waveform dataset under `processed_data/ml_prepared`.
- Experiment windows are runtime slices of `preprocessing.materialized_window_ns`; changing only study windows no longer invalidates the permanent dataset.
- `true_tof_ps`, validation strategy, models and reporting settings do not invalidate preprocessing.
- LED mismatch rejection remains upstream of ML splitting and supports either `max_distance_ps` or robust `zscore_limit`.
- Energy denoising is materialized only for energy-channel waveforms. Timing waveforms remain raw.
- `preprocessing.input_variant_by_channel` can select energy `denoised` and timing `raw` without making these separate experiments.

## Validation

`validation.strategy` supports:

- `holdout`
- `cv`
- `nested`

Nested evaluation uses outer K-fold evaluation and may use either `holdout` or `cv` internally. After nested performance estimation, the final candidate selection is rerun on the complete development population and blind is opened once.

The old `cross_validation` block is retained as a compatibility mirror for unchanged model-training helpers.

## Results

- `results.csv`: compact machine-facing candidate/blind table, same numeric stage/codebook layout.
- `summary_results.csv`: selected blind results only, with resolved hyperparameters and window margins.
- `nested_results.csv`: outer-fold results when nested validation is enabled.
- `report_results.csv`: expanded reporting table used to build figures.

Final plots include clean blind distributions with integer-ps CTR ± statistical uncertainty, validation/blind CTR vs voltage, grouped blind bars, validation-vs-blind correlation, optional nested-vs-blind plots, validation/blind correction-correlation matrices, and top-k correction examples.

Models with CTR above `reporting.max_ctr_to_led_ratio * LED CTR` remain in CSV outputs but are omitted from reporting figures.
