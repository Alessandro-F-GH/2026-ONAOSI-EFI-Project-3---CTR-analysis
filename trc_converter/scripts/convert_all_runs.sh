#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 INPUT_ROOT OUTPUT_DIR [MAX_EVENTS]" >&2
  echo "Example: $0 /home/afalcetta/ctr_trc_pipeline/ctr_trc_pipeline/data /mnt/c/Users/aless/Desktop/UChicago/Prj_3/converted 0" >&2
  exit 2
fi

INPUT_ROOT=$(realpath "$1")
OUTPUT_DIR=$2
MAX_EVENTS=${3:-0}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
CONVERTER="$PROJECT_DIR/bin/trc_to_root"

if [[ ! -x "$CONVERTER" ]]; then
  make -C "$PROJECT_DIR" -j"$(nproc)"
fi
mkdir -p "$OUTPUT_DIR"

has_direct_trc() {
  find "$1" -maxdepth 1 -type f -iname '*.trc' -print -quit | grep -q .
}

converted=0
if has_direct_trc "$INPUT_ROOT"; then
  run_name=$(basename "$INPUT_ROOT")
  "$CONVERTER" --input "$INPUT_ROOT" --output "$OUTPUT_DIR/$run_name.root" \
    --run-name "$run_name" --max-events "$MAX_EVENTS"
  converted=$((converted + 1))
else
  while IFS= read -r -d '' run_dir; do
    if find "$run_dir" -type f -iname '*.trc' -print -quit | grep -q .; then
      run_name=$(basename "$run_dir")
      echo "=== Converting $run_name ==="
      "$CONVERTER" --input "$run_dir" --output "$OUTPUT_DIR/$run_name.root" \
        --run-name "$run_name" --max-events "$MAX_EVENTS"
      converted=$((converted + 1))
    fi
  done < <(find "$INPUT_ROOT" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
fi

if [[ $converted -eq 0 ]]; then
  echo "Error: no run directory containing .trc files was found under $INPUT_ROOT" >&2
  exit 1
fi

echo "Converted $converted run(s) into $OUTPUT_DIR"
