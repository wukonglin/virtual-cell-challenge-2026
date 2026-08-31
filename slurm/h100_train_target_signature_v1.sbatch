#!/usr/bin/env bash
#SBATCH --job-name=vcc-signature-v1
#SBATCH --account=allaccess
#SBATCH --partition=full
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --constraint=gpu-h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/h100_signature_v1_%j.out
#SBATCH --error=logs/h100_signature_v1_%j.err

set -euo pipefail

readonly PROJECT_DIR="${VCC_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
readonly PYTHON_BIN="$PROJECT_DIR/.venv-state/bin/python"
readonly EFFECT_ATLAS="$PROJECT_DIR/artifacts/effect_atlas/hepg2_raw_min30_v1.npz"
readonly ESM2="$PROJECT_DIR/dataset/state_support/extracted/ESM2_pert_features.pt"
readonly TARGETS="$PROJECT_DIR/dataset/controls/pert_counts.csv"
readonly GENES="$PROJECT_DIR/dataset/controls/gene_names.csv"
readonly OUTPUT_NPZ="$PROJECT_DIR/artifacts/signature_model/hepg2_esm2_garm_v1.npz"
readonly OUTPUT_CKPT="$PROJECT_DIR/artifacts/signature_model/hepg2_esm2_garm_v1.pt"
readonly OUTPUT_JSON="$PROJECT_DIR/artifacts/signature_model/hepg2_esm2_garm_v1.json"

cd "$PROJECT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

for required_path in "$PYTHON_BIN" "$EFFECT_ATLAS" "$ESM2" "$TARGETS" "$GENES"; do
  if [[ ! -r "$required_path" ]]; then
    echo "Missing required input: $required_path" >&2
    exit 2
  fi
done
for output_path in "$OUTPUT_NPZ" "$OUTPUT_CKPT" "$OUTPUT_JSON"; do
  if [[ -e "$output_path" ]]; then
    echo "Refusing to overwrite output: $output_path" >&2
    exit 3
  fi
done

mkdir -p "$(dirname "$OUTPUT_NPZ")"
date --iso-8601=seconds
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader

"$PYTHON_BIN" scripts/train_target_signature_model.py \
  "$EFFECT_ATLAS" \
  "$ESM2" \
  "$TARGETS" \
  "$OUTPUT_NPZ" \
  "$OUTPUT_CKPT" \
  "$OUTPUT_JSON" \
  --target-column target_gene \
  --challenge-genes "$GENES" \
  --input-rank 256 \
  --output-rank 128 \
  --hidden-dim 512 \
  --clusters 20 \
  --validation-fraction 0.20 \
  --epochs 120 \
  --batch-size 128 \
  --learning-rate 0.001 \
  --weight-decay 0.0001 \
  --device cuda \
  --require-cuda

test -s "$OUTPUT_NPZ"
test -s "$OUTPUT_CKPT"
test -s "$OUTPUT_JSON"
date --iso-8601=seconds
