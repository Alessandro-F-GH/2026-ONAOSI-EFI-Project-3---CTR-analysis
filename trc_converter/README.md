# Linux/WSL `.trc` → one ROOT file per run

This program has one job only: convert the many oscilloscope `.trc` files of one run into one compact file that Python can read efficiently.

It does not calculate baselines, amplitudes, LED times, CFD times, energy cuts, or CTR. No run-specific JSON configuration is used.

## What is preserved

For every event and every channel C1–C4, the ROOT file stores:

- the original signed 16-bit ADC samples;
- sample count;
- vertical gain and offset;
- horizontal sampling interval and time offset;
- event ID and original file ID.

The Python conversion is therefore exact:

```text
voltage_mV[i] = (raw_adc[i] * vertical_gain - vertical_offset) * 1000
time_ns[i]    = (i * horizontal_interval + horizontal_offset) * 1e9
```

The raw samples remain `int16`, so the intermediate file is much smaller than storing four full `float64` voltage/time arrays.

## Requirements

- Ubuntu/WSL
- C++17 compiler
- CERN ROOT with `root-config`

Activate the environment you already use:

```bash
conda activate rootenv
```

## Build

```bash
cd ~/trc_singlefile_pipeline/cpp_converter
make -j"$(nproc)"
```

The executable is:

```text
bin/trc_to_root
```

## Locate one run

Your current data root is:

```text
/home/afalcetta/ctr_trc_pipeline/ctr_trc_pipeline/data
```

Check whether the `.trc` files are directly there or inside a run subdirectory:

```bash
find /home/afalcetta/ctr_trc_pipeline/ctr_trc_pipeline/data \
  -type f -iname '*.trc' | head
```

If the path contains `.../data/44V-250mV/C1--...trc`, use the run directory `.../data/44V-250mV` as `--input`.

## Test conversion: 100 events

```bash
mkdir -p ~/converted_runs

./bin/trc_to_root \
  --input /home/afalcetta/ctr_trc_pipeline/ctr_trc_pipeline/data/44V-250mV \
  --output ~/converted_runs/44V-250mV_test.root \
  --max-events 100
```

If the `.trc` files are directly in `data`, use:

```bash
./bin/trc_to_root \
  --input /home/afalcetta/ctr_trc_pipeline/ctr_trc_pipeline/data \
  --output ~/converted_runs/44V-250mV_test.root \
  --run-name 44V-250mV \
  --max-events 100
```

## Full conversion

```bash
./bin/trc_to_root \
  --input /home/afalcetta/ctr_trc_pipeline/ctr_trc_pipeline/data/44V-250mV \
  --output ~/converted_runs/44V-250mV.root \
  --max-events 0
```

`--max-events 0` means all events.

For maximum speed, write the ROOT file into the Linux filesystem first, then copy the single result to Windows:

```bash
mkdir -p /mnt/c/Users/aless/Desktop/UChicago/Prj_3/converted_runs
cp ~/converted_runs/44V-250mV.root \
   /mnt/c/Users/aless/Desktop/UChicago/Prj_3/converted_runs/
```

You may also write directly to `/mnt/c/...`, but sequential output is usually faster in the Linux filesystem.

## Convert all run subdirectories

```bash
./scripts/convert_all_runs.sh \
  /home/afalcetta/ctr_trc_pipeline/ctr_trc_pipeline/data \
  ~/converted_runs \
  0
```

The script creates one `.root` file for every direct run subdirectory.

## File association

The default `--pairing auto` mode:

1. pairs C1–C4 by the final numeric ID when at least 80% of IDs are common;
2. otherwise pairs the sorted files by rank;
3. always stores the four original source IDs for auditing.

Override only for diagnostics:

```bash
--pairing id
--pairing rank
```

If duplicate IDs are detected in one channel, conversion stops. This usually means that `--input` points to a directory containing multiple runs.

## ROOT structure

`events` TTree:

```text
event_index
event_id
source_file_id[4]
sample_count[4]
vertical_gain_v_per_count[4]
vertical_offset_v[4]
horizontal_interval_s[4]
horizontal_offset_s[4]
samples_ch1
samples_ch2
samples_ch3
samples_ch4
```

`metadata` TTree records the run name, source directory, pairing method, counts, format version, and fixed `.trc` header offsets.

## Synthetic test

```bash
python3 tests/generate_synthetic_trc.py /tmp/synthetic_trc --events 10
./bin/trc_to_root --input /tmp/synthetic_trc --output /tmp/synthetic.root
```
