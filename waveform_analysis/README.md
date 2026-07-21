# Python waveform analysis — C-compatible pipeline

This repository analyzes the **single ROOT file per run** produced by the Ubuntu/WSL `trc_to_root` converter. It does not read `.trc` files and CERN ROOT is not required on Windows; `uproot` reads the converted file.

The waveform treatment follows the useful physical steps of the original `07-08-code` coincidence pipeline:

1. decode raw ADC samples with each channel's calibration;
2. baseline and RMS noise from the first 500 samples;
3. first corrected sample above 50 mV as trigger;
4. energy amplitudes from channels 1 and 2;
5. timing crop of ±2.5 ns around the trigger on channels 3 and 4;
6. cubic-spline resampling every 2.5 ps;
7. LED and fractional-threshold CFD crossings on the prepared timing waveform;
8. integer `int64` femtosecond timestamps and differences;
9. iterative Gaussian CTR fit;
10. convert to ps only for tables and plots.

The best parameter is the smallest CTR among fits that mathematically converged.

## Photopeak selection

Each energy channel is fit automatically with an iterative Gaussian. The accepted interval is asymmetric:

```text
mean - 2 sigma  <= amplitude <=  mean + 4 sigma
```

The final event selection is the AND of both photopeak windows plus the configured trigger/noise conditions.

## Installation on Windows

Open PowerShell in this repository:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Configuration

All analysis choices are in:

```text
config\analysis.json
```

The command line only supplies paths. Important sections:

- `channels`: energy/timing roles and signal polarities;
- `waveform`: baseline, trigger, crop, and spline spacing;
- `timing_scan`: LED thresholds and CFD fractions;
- `photopeak`: iterative fit and asymmetric selection window;
- `selection`: noise and trigger-position conditions;
- `fit`: integer-fs histogram and iterative Gaussian settings;
- `plot`: output resolution and bin counts;
- `cache`: feature-cache name and reuse preference.

For the current `45V-400mV` run, the default energy-trigger interval is `[2000, 3500]`, matching the original C pipeline. Set it to `null` to disable that cut:

```json
"energy_trigger_index_range": null
```

Likewise, set `timing_noise_max_mV` to `null` to disable the timing-noise cut.

## Run the analysis

```powershell
python scripts\analyze_ctr.py `
  --input "C:\Users\aless\Desktop\UChicago\Prj_3\oscilloscope\converted_runs\45V-400mV.root" `
  --config "config\analysis.json" `
  --output "results\45V-400mV"
```

### Quick test on 500 events

```powershell
python scripts\analyze_ctr.py `
  --input "C:\Users\aless\Desktop\UChicago\Prj_3\oscilloscope\converted_runs\45V-400mV.root" `
  --config "config\analysis.json" `
  --output "results\45V-400mV_test" `
  --max-events 500
```

### Refit without reprocessing waveforms

Changing photopeak, event-selection, fit, or plotting settings does not require rebuilding LED/CFD timestamps:

```powershell
python scripts\analyze_ctr.py `
  --input "C:\Users\aless\Desktop\UChicago\Prj_3\oscilloscope\converted_runs\45V-400mV.root" `
  --config "config\analysis.json" `
  --output "results\45V-400mV" `
  --reuse-features
```

The cache is rejected automatically when channel roles, polarities, baseline settings, trigger/crop settings, upsampling, LED grid, or CFD grid change.

## Outputs

```text
results\45V-400mV\
├── config_used.json
├── waveform_features.npz
├── cutflow.json
├── summary.json
├── led_scan.csv
├── cfd_scan.csv
├── energy_photopeak_selection.png
├── energy_correlation.png
├── timing_noise.png
├── timing_trigger_toa.png
├── ctr_vs_led_threshold.png
├── ctr_vs_cfd_fraction.png
├── best_led_fit.png
├── best_cfd_fit.png
├── best_led_toa.png
└── best_cfd_toa.png
```

The best-fit plot uses two short legends in opposite corners:

- CTR, mean, sigma to `0.1 ps`, and chi-square/ndof;
- selected/rejected/valid event counts and displayed fit interval.

Labels use at most three significant digits where practical to avoid overlap.

## Integer timing and fitting

Crossings are stored as integer femtoseconds:

```text
t_led_a_fs, t_led_b_fs, t_cfd_a_fs, t_cfd_b_fs : int64
```

Time differences and histogram edges remain integer femtoseconds. A Gaussian has continuous mean and width, so SciPy's optimizer necessarily represents its parameters as floating point, but those values remain in **femtosecond units** throughout the fit. Conversion is performed only for output:

```text
sigma_ps = sigma_fs / 1000
CTR_ps   = 2.354820045 × sigma_fs / 1000
```

The fit is a weighted iterative Gaussian fit to integer histogram counts. The reported goodness of fit is Pearson chi-square divided by the number of degrees of freedom.

## Tests

From the repository root:

```powershell
$env:PYTHONPATH = "."
python tests\test_signal.py
python tests\test_fit.py
python tests\test_photopeak.py
```

## Notes

- The CFD implementation matches the original digital method: each fraction is converted to an absolute threshold using the maximum of the cropped, spline-resampled pulse, then the same leading-edge crossing routine is used.
- No SNR/reliability filter changes which parameter is selected. Crossing efficiency and fit diagnostics are still written to CSV so pathological results remain visible.
- The photopeak is fitted automatically; there are no hardcoded voltage-specific peak centers.
