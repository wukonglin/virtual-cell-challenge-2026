#!/usr/bin/env bash
#SBATCH --job-name=vcc-scdfm-contract-smoke
#SBATCH --account=allaccess
#SBATCH --partition=full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --gpus=h100:1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail
unset PYTHONOPTIMIZE

# This job exercises a synthetic training contract only.  It does not read
# challenge expression matrices, evaluate model quality, or submit a model.
readonly PROJECT_DIR="${VCC_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
readonly PYTHON_BIN="$PROJECT_DIR/.venv-state/bin/python"
readonly CONFIG_PATH="$PROJECT_DIR/configs/scdfm/vcc2026_v7_gamma1.toml"
readonly OUTPUT_JSON="${VCC_SCDFM_CONTRACT_RECEIPT:-$PROJECT_DIR/artifacts/scdfm/v7/contracts/contract_smoke_${SLURM_JOB_ID:?SLURM_JOB_ID is required}.json}"

cd "$PROJECT_DIR"
# The pinned PyTorch wheel carries its matching CUDA and cuDNN runtime. Loading
# the cluster CUDA module here can prepend an incompatible system cuDNN to
# LD_LIBRARY_PATH, so this job intentionally uses only the NVIDIA driver
# exposed by Slurm.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for required_path in \
  "$PYTHON_BIN" \
  "$CONFIG_PATH" \
  scripts/smoke_scdfm_contract.py; do
  if [[ ! -r "$required_path" ]]; then
    echo "Missing scDFM contract-smoke input: $required_path" >&2
    exit 2
  fi
done

if [[ -e "$OUTPUT_JSON" ]]; then
  echo "Refusing to overwrite scDFM contract receipt: $OUTPUT_JSON" >&2
  exit 3
fi

date --iso-8601=seconds
hostname
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
nvidia-smi \
  --query-gpu=index,name,uuid,memory.total,driver_version,compute_cap \
  --format=csv,noheader

srun --nodes=1 --ntasks=1 \
  "$PYTHON_BIN" scripts/smoke_scdfm_contract.py \
  --config "$CONFIG_PATH" \
  --launcher slurm/h100_scdfm_contract_smoke.sbatch \
  --output-json "$OUTPUT_JSON" \
  --device cuda \
  --require-cuda \
  --require-h100 \
  --batch-size 8 \
  --synthetic-gene-count 32

test -s "$OUTPUT_JSON"
echo "Synthetic scDFM contract receipt: $OUTPUT_JSON"
date --iso-8601=seconds
