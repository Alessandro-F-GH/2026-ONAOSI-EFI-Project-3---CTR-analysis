# CTR waveform-ML analysis

This repository is intentionally narrow: it searches the best CTR obtainable from antisymmetric waveform corrections, compares them with LED/CFD, explains the learned waveform information, and compares the result with a reduced-data multithreshold SVR.

## Supported models

- **Linear SVR** — full-waveform linear baseline; exact pair correction `g(s1) - g(s2)`.
- **Constructive MLP** — nonlinear units added progressively; every new unit sees the raw waveform plus all frozen accepted units.
- **CNN** — nonlinear 1-D convolutional shared scorer; exact pair correction `g(s1) - g(s2)`.
- **Multithreshold SVR** — separate reduced-data study using raw threshold crossings only, with linear/RBF kernels.

Obsolete threshold regressors, shapelets/RDST/BORF, autoencoders, feature regressions and the old generic MLP experiment paths were removed.

## Scientific protocol

`ROOT -> permanent prepared dataset -> random development/blind split`

Waveform-ML candidate evaluation uses:

`development -> K-fold CV -> (K-1 folds -> fit + early-stop) -> untouched score fold`

The score fold is never used for early stopping. Pooled out-of-fold predictions from all K score folds are fitted once with the global CTR fitter and used to rank a candidate. After selecting a candidate for each model family, the model is trained from scratch on the complete development population split into fit/early-stop with the same candidate fraction, then evaluated once on blind data.

Linear SVR has no early stopping and uses the complete K-1 fold training pool (or all development in the final fit).

The multithreshold study uses the same random development/blind population and pooled OOF selection, but no early-stop split because SVR has no iterative stopping criterion.

## Permanent preprocessing

Photopeak selection and optional gross LED mismatch rejection are dataset-preparation operations. They happen once per ROOT file, before any ML split. The retained events are written as NumPy `.npy` arrays and loaded with memory mapping, so large waveform matrices do not need to be duplicated in RAM.

LED/CFD timestamps are extracted from the **raw** baseline-corrected signals and are frozen. Optional Butterworth denoising is materialized as a separate permanent waveform representation and can be searched only by the regular waveform-ML pipeline. Multithreshold SVR always reads the raw representation and has no denoising option.

Prepared datasets contain no train/CV/blind split. Splits are deterministic in-memory random index arrays generated from the experiment seed.

Run preprocessing alone with:

```bash
python scripts/ml_preprocess.py --config config/experiments/ctr_ml_search.json
```

One example plot per ROOT file is produced with the first retained energy/timing waveforms on a fine major/minor grid.

## Run the experiment

Validate configuration/model availability first:

```bash
python scripts/ml_experiment.py --config config/experiments/ctr_ml_search.json --check
```

Run everything:

```bash
python scripts/ml_experiment.py --config config/experiments/ctr_ml_search.json
```

Run only the multithreshold comparison using the same preparation/evaluation implementation:

```bash
python scripts/ml_multithreshold.py --config config/experiments/ctr_ml_search.json
```

## One global CTR fit

`fit` is configured once at experiment level and is used identically for LED, CFD, pooled OOF predictions, final ML predictions and multithreshold predictions.

The fitter:

1. uses **all prepared evaluation events** (no fit-time outlier rejection),
2. estimates a robust preliminary width,
3. sets histogram width proportional to preliminary FWHM,
4. scans several bin-origin phases at fixed width,
5. fits a bin-integrated Gaussian likelihood,
6. selects the phase with minimum reduced Poisson deviance (the count-data analogue of chi-square),
7. records phase-to-phase CTR spread as a stability diagnostic.

If an event is invalid for a requested final method, evaluation fails rather than silently removing it. Dataset-level filtering must be fixed upstream.

## Compact outputs

A normal study keeps only final-level information:

- `results.csv` — numeric/coded rows for every pooled-OOF candidate and final blind LED/CFD/selected-model result;
- `manifest.json` — codebooks, candidate parameter dictionaries, protocol and the single global fit configuration;
- `models/` — only final development-trained waveform models (no CV-fold checkpoints);
- `ctr_vs_voltage.png` — final blind CTR versus voltage, where voltage is parsed from filenames such as `45V-400mV.root -> 45 V`;
- `preprocessing_examples/` — one signal example figure per input file;
- final-fit/XAI figures when enabled.

Temporary fold directories are removed immediately after OOF prediction. The data cache is memory-mapped and reused between candidates; only tiny fit-subset normalization statistics are cached across candidates that use the identical data view, while model objects are released between folds/candidates.

## Explainability

The retained waveform models expose a shared single-channel scorer. Final selected models can therefore be compared through temporal Integrated Gradients and through blind correction-output correlations. Linear SVR weights provide the direct linear reference.

## Tests

```bash
python -m unittest discover -s tests -v
```

The protocol tests cover partition isolation, all-event Gaussian fitting, filename voltage parsing and exact antisymmetry of the retained waveform models.
