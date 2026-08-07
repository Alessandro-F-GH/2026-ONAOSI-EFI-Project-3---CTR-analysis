# Windows results metadata write fix

The study writes `all_results.csv` after each fold. Its sidecar
`results_metadata.json` usually does not change between folds, but the previous
implementation atomically replaced it every time. On Windows, antivirus,
indexing, or an open viewer can briefly deny delete/rename sharing and make
`os.replace` fail with `WinError 5`.

The ML result persistence layer now:

1. serializes metadata once and skips the write when the existing bytes are
   identical;
2. uses a unique temporary file rather than a shared `.tmp` name;
3. flushes and fsyncs before replacement;
4. retries transient `PermissionError` failures with bounded exponential
   backoff;
5. falls back to an in-place rewrite if Windows allows writing but denies file
   replacement;
6. removes stale temporary files in `finally`.

No standard-analysis code was changed.
