#!/usr/bin/env bash
#SBATCH --job-name=vcc-state-infer-smoke
#SBATCH --account=allaccess
#SBATCH --partition=full
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --constraint=gpu-h100
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/h100_state_infer_smoke_pseudobulk_v3_%j.out
#SBATCH --error=logs/h100_state_infer_smoke_pseudobulk_v3_%j.err

set -euo pipefail

readonly PROJECT_DIR="${VCC_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
readonly PYTHON_BIN="$PROJECT_DIR/.venv-state/bin/python"
readonly MODEL_DIR="$PROJECT_DIR/artifacts/state_runs/state_sm_smoke_v0"
readonly OUTPUT_NPZ="$PROJECT_DIR/artifacts/state_smoke_effect_prior_pseudobulk_v3.npz"
readonly OUTPUT_JSON="$PROJECT_DIR/artifacts/state_smoke_effect_prior_pseudobulk_v3.json"

cd "$PROJECT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export CPATH="$PROJECT_DIR/.venv-state-headers/usr/include/python3.11"

for required_path in \
  "$PYTHON_BIN" \
  "$MODEL_DIR/checkpoints/best.ckpt" \
  "$MODEL_DIR/pert_onehot_map.pt" \
  "$MODEL_DIR/var_dims.pkl" \
  scripts/infer_state_effect_prior.py \
  dataset/state_support/extracted/gene_names.csv \
  dataset/controls/gene_names.csv \
  dataset/controls/pert_counts.csv \
  dataset/controls/context_A.h5ad \
  dataset/controls/context_B.h5ad \
  dataset/controls/context_C.h5ad; do
  if [[ ! -r "$required_path" ]]; then
    echo "Missing required input: $required_path" >&2
    exit 2
  fi
done

for output_path in "$OUTPUT_NPZ" "$OUTPUT_JSON"; do
  if [[ -e "$output_path" ]]; then
    echo "Refusing to overwrite existing smoke artifact: $output_path" >&2
    exit 3
  fi
done

date --iso-8601=seconds
hostname
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader

"$PYTHON_BIN" scripts/infer_state_effect_prior.py \
  --model-dir "$MODEL_DIR" \
  --checkpoint best.ckpt \
  --device cuda \
  --require-cuda \
  --require-h100 \
  --precision float32 \
  --input-normalization log1p-raw \
  --effect-reference model-control \
  --effect-bulk-target-sum 50000 \
  --sets-per-context 1 \
  --top-k 161 \
  --target-strategy blend \
  --target-blend-weight 0.5 \
  --output-npz "$OUTPUT_NPZ" \
  --output-json "$OUTPUT_JSON"

test -s "$OUTPUT_NPZ"
test -s "$OUTPUT_JSON"
date --iso-8601=seconds
