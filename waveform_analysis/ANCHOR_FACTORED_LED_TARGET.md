# Anchor-factored interpolated LED target

## Purpose

The waveform stays on the native acquisition grid, but the correction target is
referenced to the interpolated LED without asking the model to learn the known
nearest-sample alignment offset.

For detector `j`:

- `t_LED,j`: linearly interpolated LED crossing;
- `t_anchor,j`: exact native sample used as waveform-window anchor;
- `delta_j = t_LED,j - t_anchor,j`.

For the ordered pair:

```text
c_LED = (t_LED,1 - t_LED,2) - TOF_true
delta_pair = delta_1 - delta_2
c_model = c_LED - delta_pair
```

The model learns `c_model`. Evaluation reconstructs

```text
c_LED_hat = c_model_hat + delta_pair
Delta t_corrected = Delta t_LED - c_LED_hat.
```

## Storage

Preprocessing now writes native-anchor timestamps for every available alignment:

- `energy_window_anchor_time_fs.npy`;
- `timing_aligned_energy_window_anchor_time_fs.npy`;
- `timing_window_anchor_time_fs.npy`.

Cache and prepared-dataset format versions were incremented. Existing studies
must be restarted because old targets and numeric rows are not comparable.

## Compatibility

Synthetic or legacy in-memory datasets without anchor arrays fall back to zero
anchor shift for unit-test and migration compatibility. New preprocessing always
materializes the anchor arrays.
