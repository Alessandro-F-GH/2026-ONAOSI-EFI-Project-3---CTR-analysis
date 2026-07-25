# Verification

The CNN experiment was checked with:

```text
python -m compileall -q .
pytest -q
```

Result:

```text
5 passed
```

A synthetic end-to-end smoke run was also completed for:

- direct CNN training;
- invariant correction CNN training;
- checkpoint loading;
- discrete and uniform evaluation;
- iterative Gaussian CTR evaluation;
- CSV, NPZ, and plot generation.

The existing CTR implementation was restored from the clean repository and was
not modified. SHA-256 values match the original for:

```text
scripts/analyze_ctr.py
utils/fit.py
utils/pipeline.py
utils/signal.py
config/analysis.json
```

A complete ROOT-to-training run could not be executed in the build environment
because the real converted waveform ROOT file was not included with this task.
