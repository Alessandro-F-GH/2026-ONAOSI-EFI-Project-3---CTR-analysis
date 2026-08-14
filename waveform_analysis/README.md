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

The score fold is never used for early stopping. Each score fold is evaluated independently with the model trained for that fold. CTR is computed directly from that fold's ordinary sample standard deviation, and candidate selection uses the arithmetic mean of the fold CTR values. Predictions from different fitted models are never concatenated to estimate CTR. After selecting a candidate for each model family, the model is trained from scratch on the complete development population split into fit/early-stop with the same candidate fraction, then evaluated once on blind data.

Linear SVR has no early stopping and uses the complete K-1 fold training pool (or all development in the final fit).

The multithreshold study uses the same random development/blind population and fold-wise CTR selection, but no early-stop split because SVR has no iterative stopping criterion.

## Permanent preprocessing

Preprocessing uses two ROOT passes. The **first pass reads only the two raw energy channels** and computes the cheap quantities needed for event selection (baseline, amplitude, noise RMS and trigger). Trigger/noise cuts and photopeak fitting are applied here. No denoising, timing-channel processing, LED/CFD extraction or waveform windowing is performed for events that fail this first-stage selection. The transient waveform cache is therefore allocated only for the retained photopeak population.

The **second pass performs the expensive timing preprocessing only on those retained events**. Channel preprocessing is applied before timing extraction, so LED/CFD, anchor and ML window all come from the same signal representation later consumed by ML (for the default study: denoised energy and raw timing). A dedicated raw extraction is retained only where needed by multithreshold SVR, which is always raw. Optional gross LED mismatch rejection is then applied once, before any ML split, using `abs(delta_LED - median) / (1.4826 * MAD)` with standard-deviation fallback only for degenerate MAD. The final `ml_prepared` dataset therefore contains only first-pass-selected photopeak events that also pass the configured timing-validity/mismatch requirements. Arrays are stored as NumPy `.npy` files and loaded with memory mapping.

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

## CTR estimation

The experiment runner does not perform a Gaussian fit for model selection or final reporting. For every individual evaluation population it keeps all prepared events and computes the ordinary sample mean and sample standard deviation (`ddof=1`). CTR is the Gaussian-equivalent FWHM

`CTR = 2 * sqrt(2 * ln(2)) * sample_std`.

During cross-validation this calculation is done independently for each untouched score fold. The compact candidate result stores the arithmetic mean fold CTR and the standard deviation of the fold CTR values. Model outputs from different folds are never pooled. Final blind CTR is computed once from the single final model's blind residual distribution. Bias is not part of candidate selection.

## Compact outputs

A normal study keeps only final-level information:

- `results.csv` — numeric/coded rows for every CV candidate summary and final blind LED/CFD/selected-model result;
- `results.csv` is updated atomically after every completed CV candidate and blind result; rerun with `--resume` after interruption to skip completed candidates;
- `manifest.json` — codebooks, candidate parameter dictionaries, protocol and CTR-estimator definition;
- `models/` — only final development-trained waveform models (no CV-fold checkpoints);
- `ctr_vs_voltage.png` — final blind CTR versus voltage, where voltage is parsed from filenames such as `45V-400mV.root -> 45 V`;
- `preprocessing_examples/` — one signal example figure per input file;
- final-distribution/XAI figures when enabled.

Temporary fold directories are removed immediately after score-fold prediction. The data cache is memory-mapped and reused between candidates; only tiny fit-subset normalization statistics are cached across candidates that use the identical data view, while model objects are released between folds/candidates.

## Explainability

The retained waveform models expose a shared single-channel scorer. Final selected models can therefore be compared through temporal Integrated Gradients and through blind correction-output correlations. Linear SVR weights provide the direct linear reference.

## Tests

```bash
python -m unittest discover -s tests -v
```

The protocol tests cover partition isolation, channel-specific raw/denoised routing, classical all-event CTR statistics, filename voltage parsing and exact antisymmetry of the retained waveform models.

## Waveform-ML target and resume contract

Waveform models operate on native-sample windows translated to the nearest LED anchor. For each detector the alignment residual is `delta = t_LED - t_anchor`; the supervised target removes the pairwise residual still encoded as sub-sample phase:

`g(s1)-g(s2) = Delta t_LED - Delta(delta) - true_TOF`.

No anchor/alignment term is added analytically at inference; the learned output itself is the correction applied to the measured LED difference. Multithreshold SVR remains raw-only and keeps its own LED-relative threshold feature formulation.

Study progress is persisted after every completed CV candidate. `--resume` reuses compatible completed candidates and completed files; an interrupted in-progress candidate is retrained from the beginning and no fold checkpoint is retained.
