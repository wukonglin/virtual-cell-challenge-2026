#!/usr/bin/env bash
#SBATCH --job-name=vcc-state-hepg2-ce2
#SBATCH --account=allaccess
#SBATCH --partition=full
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=01:00:00
#SBATCH --output=logs/cpu_state_hepg2_ce2_v0_%j.out
#SBATCH --error=logs/cpu_state_hepg2_ce2_v0_%j.err

set -euo pipefail

readonly PROJECT_DIR="${VCC_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
readonly PYTHON_BIN="$PROJECT_DIR/.venv-state/bin/python"
readonly REAL_H5AD="$PROJECT_DIR/dataset/state_support/hepg2_holdout.h5ad"
readonly PRED_H5AD="$PROJECT_DIR/artifacts/holdout/hepg2_state_sm_step16000_direct.h5ad"
readonly OUTPUT_DIR="$PROJECT_DIR/artifacts/holdout/cell_eval2_direct_lognorm_v0"

cd "$PROJECT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR/external/cell-eval2/src${PYTHONPATH:+:$PYTHONPATH}"

for required_path in "$PYTHON_BIN" "$REAL_H5AD" "$PRED_H5AD"; do
  if [[ ! -r "$required_path" ]]; then
    echo "Missing required input: $required_path" >&2
    exit 2
  fi
done
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite holdout score directory: $OUTPUT_DIR" >&2
  exit 3
fi

date --iso-8601=seconds
# The released HepG2 support matrix is normalized log1p rather than raw counts.
# We therefore keep the current VCC metric membership but consistently override
# the input representation to lognorm.  This is a ranking proxy, not an exact
# reproduction of the counts-space leaderboard score.
"$PYTHON_BIN" -c 'from cell_eval2.cli import main; main()' run \
  --adata-pred "$PRED_H5AD" \
  --adata-real "$REAL_H5AD" \
  --preset vcc2026 \
  --input-type lognorm \
  --pert-col target_gene \
  --control non-targeting \
  --outdir "$OUTPUT_DIR"
date --iso-8601=seconds
