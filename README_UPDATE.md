# CTR analysis pipeline update

This bundle targets the current `main` layout of `2026-ONAOSI-EFI-Project-3---CTR-analysis` as inspected on 2026-08-16.

## Apply to your checkout

From anywhere:

```bash
python <bundle>/tools/apply_ctr_pipeline_update.py C:\path\to\2026-ONAOSI-EFI-Project-3---CTR-analysis
```

The installer creates `.ctr_pipeline_update_backup/` before modifying existing files unless `--no-backup` is supplied. It fails rather than guessing if the expected current-main source anchors have changed.

Then run:

```bash
cd C:\path\to\2026-ONAOSI-EFI-Project-3---CTR-analysis
python -m compileall waveform_analysis/ml_pipeline waveform_analysis/scripts
python waveform_analysis/scripts/ml_experiment.py --config waveform_analysis/config/experiments/<study>.json --prepare-only
python waveform_analysis/scripts/ml_experiment.py --config waveform_analysis/config/experiments/<study>.json
```

See `docs/CONFIGURATION.md` for holdout, CV, nested-inner-holdout, nested-inner-CV, permanent preprocessing, and reporting settings.

## Main additions

- `ml_pipeline/selection_store.py`: permanent fingerprinted physical/photopeak event selection (`selected_entries.npy` + manifest), upstream of ML.
- `ml_pipeline/validation.py`: common holdout/CV/nested split primitives.
- `ml_pipeline/nested_evaluation.py`: development-only outer-fold pipeline evaluation with inner holdout or CV.
- `ml_pipeline/reporting.py`: centralized clean plots, blind CTR uncertainty, `summary_results.csv`, grouped bars, validation/blind comparison, correction matrices, top-k correction examples.
- `ml_pipeline/study_runner.py`: one normal experiment entry point wrapping the proven current training engine.
- `scripts/analyze_energy_timing_led_correlation.py`: rewritten to consume the permanent prepared cohort; it no longer performs its own photopeak selection.

## Output additions

The original `results.csv` schema is not modified. New outputs include:

```text
summary_results.csv
nested_results.csv                 # nested strategy only
nested_manifest.json               # nested strategy only
report_data/blind_residuals/*.npz
plots/validation_ctr_vs_voltage.png
plots/blind_ctr_vs_voltage.png
plots/blind_ctr_bar_by_voltage.png
plots/validation_vs_blind_ctr.png
plots/nested_ctr_vs_voltage.png     # nested strategy only
plots/nested_pipeline_ctr_vs_voltage.png
plots/nested_vs_blind_ctr.png       # nested strategy only
plots/correlations/*__blind_corrections.png
plots/top_corrections/*.png
```

Final blind distribution legends use compact labels such as `SVR — CTR 60 ± 1 ps`. The uncertainty is rounded to 1 ps and computed by event bootstrap on the fixed blind residuals.

## Scientific invariants kept

- Photopeak/physical selection is upstream of model training and is reused permanently.
- Experiment windows are runtime slices of a canonical materialized waveform envelope.
- The blind split is not used in nested selection or nested pipeline evaluation.
- In nested mode, the outer fold evaluates the *selection procedure*; final blind evaluation still performs a fresh final selection on all development data. Nested-vs-blind reporting is generated only after selection is complete.
- Models with blind CTR above the configured LED ratio are filtered from figures only, never deleted from result tables.
- Top-correction event ranking uses a development-derived LED calibration offset and cannot influence model/window selection.
