# Baseline RMSE / no-shift update

Replace the files in this archive at the same paths in the repository.

## Changed behavior

- The waveform is **not baseline-subtracted** anymore. Only the configured polarity is applied.
- The baseline interval is used only to compute the per-event baseline RMSE around its own mean.
- Photopeak is the first physical event-selection cut.
- After photopeak, the baseline RMSE upper limit is derived independently for each energy channel from the photopeak population:

  `upper = median(RMSE) + baseline_rmse_robust_z * (1.4826 * MAD(RMSE))`

  with a standard-deviation fallback if MAD is degenerate.
- Trigger validity / trigger-index range are applied after photopeak and baseline-RMSE quality filtering.
- The old absolute `preprocessing.selection.energy_noise_max_mV` is ignored with a warning.
- `noise_rms_mV` is retained as the cached field name for compatibility; it now explicitly represents the baseline RMSE used for quality filtering.

## Configuration

The default robust multiplier is `5.0`, so no config change is required.

To configure it explicitly:

```json
"selection": {
  "baseline_rmse_robust_z": 5.0,
  "minimum_events": 100
}
```

Set `baseline_rmse_robust_z` to `null` to disable the RMSE quality cut.

Remove obsolete `energy_noise_max_mV` entries from configs when convenient.

## Cache compatibility

- `PREPARED_SELECTION_VERSION` is bumped to 3.
- `SELECTION_STORE_VERSION` is bumped to 2.
- Raw-cache preprocessing receives `baseline_handling = quality_only_no_shift_v1`, which changes the existing raw-cache fingerprint without requiring a broad cache-format migration.

Old baseline-shifted raw caches and old physical-selection stores therefore are not silently reused.
