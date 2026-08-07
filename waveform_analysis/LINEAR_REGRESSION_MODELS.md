# Linear regression, Ridge and Lasso study models

The model plug-in `linear_regression` implements a shared linear score

\[
g(s)=w^T s,
\qquad
\hat c_{\mathrm{model}}=g(s_1)-g(s_2)+b
=w^T(s_1-s_2)+b.
\]

The exact interpolated-LED/native-anchor shift remains added analytically by the
common prediction path, so the final correction still applies to the
interpolated LED timestamp.

## Model-space IDs

Three model-space configurations are supplied:

- `linear_regression`: ordinary least squares, one fixed trial;
- `ridge_regression`: L2-regularized least squares with `model.alpha` searched;
- `lasso_regression`: L1-regularized least squares with `model.alpha` searched.

They all use the same `model_type = linear_regression`, so checkpoint loading and
comparison tables remain uniform. The default experiment configuration includes
all three model IDs.

Ridge solves

\[
\min_w \frac{1}{2N}\lVert y-Xw\rVert_2^2 + \alpha\lVert w\rVert_2^2,
\]

while Lasso solves the scikit-learn convention

\[
\min_w \frac{1}{2N}\lVert y-Xw\rVert_2^2 + \alpha\lVert w\rVert_1.
\]

A scalar pair-output bias is calibrated from the training residual after the
coefficient fit. Feature normalization is always learned from the fold-training
events only.

## Results-only coefficient storage

With `storage.keep_checkpoints = false`, checkpoints and run directories are
still temporary. For each physical window, only the CV-selected hyperparameter
trial is exported to:

```text
results/studies/<study>/linear_model_weights.csv
```

The table stores one row per fold and transformed feature. It includes:

- model ID, regularization and selected alpha;
- physical window and CV metrics;
- component (`energy` or `timing`) and feature kind (`raw` or
  `first_difference`);
- relative feature time;
- coefficient in normalized model space;
- coefficient converted to physical feature units (`ps/mV`);
- whether the window was selected by CV.

This CSV survives checkpoint cleanup and is sufficient for coefficient plots.

## Weight-norm plot

Example:

```powershell
python scripts\plot_linear_weights_vs_time.py `
  --study-dir results\studies\folder_window_channel_study `
  --file 47V-470mV.root `
  --channel-mode timing_to_timing `
  --model-id ridge_regression `
  --loss-id mse `
  --transform normalize
```

The default curve is the fold-RMS coefficient magnitude

\[
\lVert w(t)\rVert_{\mathrm{fold,RMS}}
=\sqrt{\frac{1}{K}\sum_{k=1}^{K} w_k(t)^2}.
\]

One line is drawn for every tested window. Combined energy/timing inputs and
`concatenate_diff` are separated into distinct component/feature-kind plots so
coefficients with different meanings are not mixed.
