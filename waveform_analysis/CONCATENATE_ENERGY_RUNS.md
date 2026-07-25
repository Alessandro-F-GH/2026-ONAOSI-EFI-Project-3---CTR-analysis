# Concatenating energy-only runs across bias voltages

The script scans a configured folder for processed ROOT files and produces one energy-only ROOT file containing only photopeak events. The photopeak fit is repeated independently for every input run, so voltage-dependent amplitude regimes are not forced into one global amplitude window.

## Selection

For run \(k\), the two fitted windows are

\[
A_{1,k}\in[L_{1,k},H_{1,k}],\qquad
A_{2,k}\in[L_{2,k},H_{2,k}].
\]

An event is copied only when both conditions are true:

\[
M_k=M_{1,k}\land M_{2,k}.
\]

No timing-channel branch, LED time, CFD time, or timing-channel quality cut is used.

## Configuration

Edit:

```text
config/concatenate_energy_photopeak_config.json
```

Important fields:

- `input.folder`: top-level folder to scan;
- `input.pattern`: file pattern such as `converted.root` or `*.root`;
- `input.recursive`: include subfolders;
- `output.root_file`: concatenated ROOT file;
- `channels.energy`: two input energy channels;
- `channels.polarities`: pulse polarities;
- `photopeak`: independent Gaussian-fit and selection parameters;
- `metadata.bias_voltage_regex`: extracts the bias voltage from paths such as `45V-400mV`;
- `parallelization`: amplitude-extraction backend and worker count.

## Run

```cmd
python scripts\concatenate_energy_photopeak_runs.py --config config\concatenate_energy_photopeak_config.json
```

Override paths from the terminal:

```cmd
python scripts\concatenate_energy_photopeak_runs.py --config config\concatenate_energy_photopeak_config.json --input-folder data --output-root data\combined\energy_all_bias.root --overwrite
```

## Output branches

The `events` tree contains only energy information:

- globally unique `event_index` and `event_id`;
- `source_run_index` and parsed `bias_voltage_V`;
- original event identifiers;
- a run-unique `source_file_id` pair and preserved `original_source_file_id`;
- sample counts, ADC calibration and time-axis calibration for the two energy channels;
- fitted amplitudes in mV;
- `samples_ch1` and `samples_ch2`.

The script also saves one photopeak-selection plot per accepted run and writes a JSON manifest with all source paths, fitted means, fitted sigmas, selection windows, fit quality, and event counts.

## Use in the ML pipeline

Point `data.input_root` to the concatenated ROOT file and disable the second photopeak fit:

```json
"photopeak": {
  "enabled": false
}
```

Choose the split according to the experiment:

- `"strategy": "event"` mixes events from all voltage regimes across train/validation/test;
- `"strategy": "source_file"` assigns complete input runs to one split, because the concatenation script writes a unique `source_file_id` pair for every run.

A ready-to-edit example is included at:

```text
config/ml_pipeline_all_bias_example.json
```

It disables the global photopeak refit and the global LED-median outlier cut. The latter is deliberately disabled because a voltage-dependent timing offset could otherwise make an entire bias-voltage regime look like an outlier. Re-enable it only after checking the LED distribution by run or after implementing a per-run timing cut.
