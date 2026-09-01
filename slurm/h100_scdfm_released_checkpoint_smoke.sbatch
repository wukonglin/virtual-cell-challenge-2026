#!/usr/bin/env bash
#SBATCH --job-name=vcc-scdfm-released-smoke
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

# This is a released-checkpoint compatibility test with synthetic expression
# and a dummy all-False graph. It is not a performance test or submission.
readonly PROJECT_DIR="${VCC_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
readonly PYTHON_BIN="$PROJECT_DIR/dataset/scdfm/envs/scdfm/bin/python"
readonly SOURCE_DIR="$PROJECT_DIR/dataset/external_repos/scDFM"
readonly CHECKPOINT_PATH="$PROJECT_DIR/dataset/scdfm/extracted/ckpts/holdout/fold0/checkpoint.pt"
readonly VOCAB_PATH="$SOURCE_DIR/src/tokenizer/norman_5000_highly_vocab.json"
readonly EXPECTED_CHECKPOINT_SHA256="5730a05918e2e727f8a558f21db23ae793a8deb74af5c28b905d852af92354d7"
readonly OUTPUT_JSON="${VCC_SCDFM_RELEASED_SMOKE_RECEIPT:-$PROJECT_DIR/artifacts/scdfm/v7/released_checkpoint_smokes/holdout_fold0_git_snapshot_v2_${SLURM_JOB_ID:?SLURM_JOB_ID is required}.json}"

cd "$PROJECT_DIR"
# The isolated environment supplies its matching CUDA libraries. Intentionally
# do not load a cluster CUDA module, which can introduce an incompatible cuDNN.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for required_path in \
  "$PYTHON_BIN" \
  "$SOURCE_DIR" \
  "$CHECKPOINT_PATH" \
  "$VOCAB_PATH" \
  scripts/smoke_scdfm_released_checkpoint.py; do
  if [[ ! -r "$required_path" ]]; then
    echo "Missing released-checkpoint smoke input: $required_path" >&2
    exit 2
  fi
done

if [[ -e "$OUTPUT_JSON" ]]; then
  echo "Refusing to overwrite released-checkpoint smoke receipt: $OUTPUT_JSON" >&2
  exit 3
fi

date --iso-8601=seconds
hostname
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
nvidia-smi \
  --query-gpu=index,name,uuid,memory.total,driver_version,compute_cap \
  --format=csv,noheader

srun --nodes=1 --ntasks=1 \
  "$PYTHON_BIN" scripts/smoke_scdfm_released_checkpoint.py \
  --source "$SOURCE_DIR" \
  --checkpoint "$CHECKPOINT_PATH" \
  --vocab "$VOCAB_PATH" \
  --expected-checkpoint-sha256 "$EXPECTED_CHECKPOINT_SHA256" \
  --output-json "$OUTPUT_JSON" \
  --batch-size 2 \
  --gene-count 8 \
  --seed 20260901

test -s "$OUTPUT_JSON"
echo "Released-checkpoint compatibility receipt: $OUTPUT_JSON"
date --iso-8601=seconds
