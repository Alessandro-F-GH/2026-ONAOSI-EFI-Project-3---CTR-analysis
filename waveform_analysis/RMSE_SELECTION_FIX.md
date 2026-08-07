# RMSE study-selection fix

The study runner now treats the common reporting/selection metrics separately:

- `loss`: the configured training/study objective (`mse` or `var_bias`),
- `rmse_ps`: residual root-mean-square error in ps,
- `ctr_ps`: Gaussian-fit CTR in ps,
- `bias_ps`: residual mean bias in ps.

Accepted aliases for `selection.hyperparameter_metric` and
`selection.window_metric` include `rmse`, `rmse_ps`, `validation_rmse`, and
`validation_rmse_ps`; these are canonicalized to `rmse_ps`.

Old cached MSE fold rows can reconstruct RMSE as `sqrt(loss)`. Old cached
`var_bias` fold rows cannot be converted and are retrained automatically when
RMSE is the requested selection metric.

Compact reports now also expose validation/blind RMSE mean and SEM columns.
