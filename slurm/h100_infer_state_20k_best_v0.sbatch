#!/usr/bin/env bash
#SBATCH --job-name=vcc-state-infer-20k
#SBATCH --account=allaccess
#SBATCH --partition=full
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --constraint=gpu-h100
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/h100_state_infer_20k_best_v0_%j.out
#SBATCH --error=logs/h100_state_infer_20k_best_v0_%j.err

set -euo pipefail

readonly PROJECT_DIR="${VCC_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
readonly PYTHON_BIN="$PROJECT_DIR/.venv-state/bin/python"
readonly MODEL_DIR="$PROJECT_DIR/artifacts/state_runs/state_sm_20k_v0"
readonly OUTPUT_NPZ="$PROJECT_DIR/artifacts/state_effect_prior_20k_best_v0.npz"
readonly OUTPUT_JSON="$PROJECT_DIR/artifacts/state_effect_prior_20k_best_v0.json"

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
  dataset/controls/gene_names.csv \
  dataset/controls/pert_counts.csv; do
  if [[ ! -r "$required_path" ]]; then
    echo "Missing required input: $required_path" >&2
    exit 2
  fi
done

for output_path in "$OUTPUT_NPZ" "$OUTPUT_JSON"; do
  if [[ -e "$output_path" ]]; then
    echo "Refusing to overwrite existing production artifact: $output_path" >&2
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
  --sets-per-context 4 \
  --top-k 161 \
  --target-strategy blend \
  --target-blend-weight 0.5 \
  --output-npz "$OUTPUT_NPZ" \
  --output-json "$OUTPUT_JSON"

test -s "$OUTPUT_NPZ"
test -s "$OUTPUT_JSON"
date --iso-8601=seconds
