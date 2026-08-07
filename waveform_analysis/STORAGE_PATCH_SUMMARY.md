# Storage-efficiency patch

This patch changes only the ML experiment pipeline. The standalone standard analysis pipeline is untouched.

## Changes

- Adds a disk-space preflight check before allocating preprocessing memmaps.
- Removes the raw preprocessing cache after development/blind materialization.
- Uses one shared transformed-input cache per ROOT file, channel mode, window, and transform, reused across models, losses, trials, and CV folds.
- Removes prepared datasets and shared transform caches after a ROOT file completes successfully.
- Writes a completed-file marker so `--resume` skips fully completed ROOT files without rebuilding deleted temporary caches.
- Keeps caches when a file has failed folds, allowing a resume attempt.

## Configuration

```json
"storage": {
  "cleanup_raw_cache_after_materialization": true,
  "cleanup_after_completed_file": true
}
```
