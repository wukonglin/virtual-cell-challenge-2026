#!/usr/bin/env bash
#SBATCH --job-name=vcc-state-hepg2-eval
#SBATCH --account=allaccess
#SBATCH --partition=full
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --constraint=gpu-h100
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/h100_state_hepg2_eval_v0_%j.out
#SBATCH --error=logs/h100_state_hepg2_eval_v0_%j.err

set -euo pipefail

readonly PROJECT_DIR="${VCC_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
readonly STATE_BIN="$PROJECT_DIR/.venv-state/bin/state"
readonly MODEL_DIR="$PROJECT_DIR/artifacts/state_runs/state_sm_20k_v0"
readonly CHECKPOINT="$MODEL_DIR/checkpoints/selected.ckpt"
readonly INPUT_H5AD="$PROJECT_DIR/dataset/state_support/hepg2_holdout.h5ad"
readonly OUTPUT_H5AD="$PROJECT_DIR/artifacts/holdout/hepg2_state_sm_step16000_direct.h5ad"

cd "$PROJECT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export CPATH="$PROJECT_DIR/.venv-state-headers/usr/include/python3.11"

for required_path in "$STATE_BIN" "$CHECKPOINT" "$INPUT_H5AD"; do
  if [[ ! -r "$required_path" ]]; then
    echo "Missing required input: $required_path" >&2
    exit 2
  fi
done
if [[ -e "$OUTPUT_H5AD" ]]; then
  echo "Refusing to overwrite holdout prediction: $OUTPUT_H5AD" >&2
  exit 3
fi
mkdir -p "$(dirname "$OUTPUT_H5AD")"

date --iso-8601=seconds
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader

"$STATE_BIN" tx infer \
  --output "$OUTPUT_H5AD" \
  --model-dir "$MODEL_DIR" \
  --checkpoint "$CHECKPOINT" \
  --adata "$INPUT_H5AD" \
  --pert-col target_gene \
  --celltype-col cell_type \
  --control-pert non-targeting \
  --seed 20260831 \
  --max-set-len 128

test -s "$OUTPUT_H5AD"
date --iso-8601=seconds
