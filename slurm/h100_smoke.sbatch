#!/usr/bin/env bash
#SBATCH --job-name=vcc-h100-smoke
#SBATCH --account=allaccess
#SBATCH --partition=full
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --constraint=gpu-h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/h100_smoke_%j.out
#SBATCH --error=logs/h100_smoke_%j.err

set -euo pipefail

readonly PROJECT_DIR="${VCC_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
readonly PYTHON_BIN="${VCC_MODEL_PYTHON:-$(command -v python3)}"

cd "$PROJECT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

for required_path in \
  "$PYTHON_BIN" \
  "$PROJECT_DIR/.venv-vcc/bin/vcc" \
  dataset/controls/context_A.h5ad \
  dataset/controls/context_B.h5ad \
  dataset/controls/context_C.h5ad \
  dataset/controls/gene_names.csv \
  dataset/controls/pert_counts.csv; do
  if [[ ! -r "$required_path" ]]; then
    echo "Missing required input: $required_path" >&2
    exit 2
  fi
done

date --iso-8601=seconds
hostname
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader
"$PYTHON_BIN" -c 'import torch; print({"torch": torch.__version__, "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})'

"$PYTHON_BIN" scripts/h100_smoke.py \
  --data-dir dataset/controls \
  --output-json artifacts/h100_smoke_result.json \
  --output-h5ad artifacts/h100_smoke_mini_prediction.h5ad \
  --seed 20260825 \
  --cells-per-context 256 \
  --batch-size 64 \
  --steps 8 \
  --hidden-dim 256 \
  --latent-dim 64 \
  --mini-targets 8 \
  --mini-cells 16

"$PYTHON_BIN" scripts/h100_full_null.py \
  --data-dir dataset/controls \
  --output-h5ad artifacts/h100_context_null_full.h5ad \
  --output-json artifacts/h100_context_null_full.json \
  --seed 20260825 \
  --cells-per-group 400 \
  --genes-per-cell 300 \
  --chunk-size 512 \
  --force

"$PROJECT_DIR/.venv-vcc/bin/vcc" prep \
  artifacts/h100_context_null_full.h5ad \
  -g dataset/controls/gene_names.csv \
  --perts dataset/controls/pert_counts.csv \
  --contexts A,B,C \
  --dry-run \
  --json > artifacts/h100_context_null_vcc_prep.json

date --iso-8601=seconds
