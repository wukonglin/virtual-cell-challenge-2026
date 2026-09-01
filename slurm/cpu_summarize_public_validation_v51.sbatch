#!/usr/bin/env bash
#SBATCH --job-name=vcc-public-v51-summary
#SBATCH --account=allaccess
#SBATCH --partition=full
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/cpu_public_v51_summary_%j.out
#SBATCH --error=logs/cpu_public_v51_summary_%j.err

set -euo pipefail

readonly PROJECT_DIR="${VCC_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
readonly PYTHON_BIN="$PROJECT_DIR/.venv-state/bin/python"
readonly SCORE_ROOT="$PROJECT_DIR/artifacts/public_v51/scoring"
readonly OUTPUT_JSON="$SCORE_ROOT/v51_comparison.json"

cd "$PROJECT_DIR"
export PYTHONUNBUFFERED=1

for required_path in \
  "$PYTHON_BIN" \
  scripts/validate_public_v51_scoring_receipts.py \
  scripts/summarize_public_validation.py \
  dataset/public_v51/split_manifest.csv \
  "$SCORE_ROOT/hepg2/scoring_receipt.json" \
  "$SCORE_ROOT/jurkat/scoring_receipt.json" \
  "$SCORE_ROOT/hepg2/anchor_cell_eval2/results.csv" \
  "$SCORE_ROOT/hepg2/v51_alpha010_cell_eval2/results.csv" \
  "$SCORE_ROOT/jurkat/anchor_cell_eval2/results.csv" \
  "$SCORE_ROOT/jurkat/v51_alpha010_cell_eval2/results.csv"; do
  if [[ ! -r "$required_path" ]]; then
    echo "Missing required score input: $required_path" >&2
    exit 2
  fi
done
if [[ -e "$OUTPUT_JSON" ]]; then
  echo "Refusing to overwrite public comparison: $OUTPUT_JSON" >&2
  exit 3
fi

"$PYTHON_BIN" scripts/validate_public_v51_scoring_receipts.py \
  --receipt "$SCORE_ROOT/hepg2/scoring_receipt.json" HepG2 \
  --receipt "$SCORE_ROOT/jurkat/scoring_receipt.json" Jurkat

date --iso-8601=seconds
"$PYTHON_BIN" scripts/summarize_public_validation.py \
  --manifest dataset/public_v51/split_manifest.csv \
  --baseline-label anchor \
  --primary-context HepG2 \
  --non-harm-context Jurkat \
  --entry anchor HepG2 "$SCORE_ROOT/hepg2/anchor_cell_eval2/results.csv" \
  --entry v51_alpha010 HepG2 "$SCORE_ROOT/hepg2/v51_alpha010_cell_eval2/results.csv" \
  --entry anchor Jurkat "$SCORE_ROOT/jurkat/anchor_cell_eval2/results.csv" \
  --entry v51_alpha010 Jurkat "$SCORE_ROOT/jurkat/v51_alpha010_cell_eval2/results.csv" \
  --output-json "$OUTPUT_JSON"
date --iso-8601=seconds
