# TOF-PET Timing Analysis

Analysis tools developed for the ONAOSI/UChicago project on timing resolution in TOF-PET detector systems.

The repository contains complementary pipelines for oscilloscope waveform data and Janus/Pico-TDC data, together with tools for converting raw oscilloscope traces to a compact ROOT representation.

## Project overview

The main goal is to study the timing information available in SiPM detector signals and evaluate methods for improving coincidence time resolution (CTR).

The repository includes:

- conventional timing estimators based on LED and CFD;
- machine-learning corrections using full oscilloscope waveforms;
- physics-constrained pair models of the form

  \[
  y(s_1,s_2)=g(s_1)-g(s_2),
  \]

  which enforce antisymmetry and use the same single-channel model for both detector signals;
- reduced multithreshold models intended to approximate the information available from practical TDC-based readout;
- Janus/Pico-TDC event matching and timing analysis.

Experimental data are not included in the repository.

## Repository structure

```text
.
├── trc_converter/
│   └── C++ converter from oscilloscope .trc files to ROOT
│
├── waveform_analysis/
│   ├── config/          experiment and model configurations
│   ├── ml_pipeline/     waveform preprocessing, training and evaluation
│   ├── scripts/         analysis and experiment entry points
│   └── tests/           pipeline tests
│
├── janus_data_analysis/
│   ├── config/          Janus analysis configuration
│   ├── utils/           event matching and analysis modules
│   └── main.py          main Janus pipeline
│
├── notebooks/           optional exploratory/report notebooks
└── docs/                additional documentation
```

## Oscilloscope waveform workflow

Raw oscilloscope `.trc` files can first be converted to ROOT using the C++ converter:

```bash
cd trc_converter
make -j"$(nproc)"
```

See `trc_converter/README.md` for converter usage and file-format details.

The converted ROOT files can then be processed by the waveform-analysis pipeline:

```bash
cd waveform_analysis
pip install -r requirements.txt

python scripts/ml_experiment.py \
    --config config/experiments/ctr_ml_search.json \
    --check

python scripts/ml_experiment.py \
    --config config/experiments/ctr_ml_search.json
```

The waveform pipeline compares LED/CFD baselines with machine-learning timing corrections and supports full-waveform and multithreshold studies.

See `waveform_analysis/README.md` for the scientific protocol, available models, preprocessing strategy and generated outputs.

## Janus / Pico-TDC analysis

The Janus pipeline processes binary timing data using a configurable event-matching and analysis workflow.

```bash
cd janus_data_analysis

python main.py \
    --config config/janus_pipeline.json
```

Input data and generated analysis outputs are kept outside version control.

## Waveform models

The current waveform-analysis pipeline includes:

- Linear SVR
- Constructive MLP
- 1-D CNN
- Multithreshold SVR

For the full-waveform models, the pair correction is constructed from a shared single-channel scorer:

\[
y(s_1,s_2)=g(s_1)-g(s_2).
\]

This guarantees exact antisymmetry under detector exchange while avoiding independent pair-specific models.

## Reproducibility

Experiment settings are stored in version-controlled configuration files.

Large experimental datasets, prepared-data caches, trained outputs and generated result directories are intentionally excluded from Git.

Tests for the waveform-analysis pipeline can be run with:

```bash
cd waveform_analysis
python -m unittest discover -s tests -v
```

## License

This project is released under the MIT License. See `LICENSE`.