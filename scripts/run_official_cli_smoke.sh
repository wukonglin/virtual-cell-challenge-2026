#!/usr/bin/env bash
set -euo pipefail

readonly PROJECT_DIR="${VCC_PROJECT_DIR:-$(git rev-parse --show-toplevel)}"
readonly VCC_BIN="${VCC_CLI:-$PROJECT_DIR/.venv-vcc/bin/vcc}"
readonly DATA_DIR="$PROJECT_DIR/dataset/controls"
readonly OUTPUT="$PROJECT_DIR/artifacts/vcc_official_random_smoke.vcc"
readonly JSON_LOG="$PROJECT_DIR/artifacts/vcc_official_random_smoke.json"

cd "$PROJECT_DIR"

"$VCC_BIN" --version
"$VCC_BIN" sample \
  -g "$DATA_DIR/gene_names.csv" \
  -p "$DATA_DIR/pert_counts.csv" \
  -o "$OUTPUT" \
  --full \
  --genes-per-cell 300 \
  --seed 20260825 \
  --force \
  --json > "$JSON_LOG"

sha256sum "$OUTPUT"
stat --format='bytes=%s path=%n' "$OUTPUT"
