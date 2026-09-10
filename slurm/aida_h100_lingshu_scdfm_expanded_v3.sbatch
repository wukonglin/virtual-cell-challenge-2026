#!/usr/bin/env bash
#SBATCH --job-name=vcc-effect-v3-expanded
#SBATCH --account=allaccess
#SBATCH --partition=full
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=h100:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/aida_effect_v3_expanded_%j.out
#SBATCH --error=logs/aida_effect_v3_expanded_%j.err
#SBATCH --no-requeue
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit through Slurm}"
readonly PROJECT_DIR=/home/fs01/yl4259/yl/virtual-cell-challenge-2026
cd "$PROJECT_DIR"
unset PYTHONHOME PYTHONPATH PYTHONSTARTUP PYTHONINSPECT LD_PRELOAD VCC_TOKEN
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
hostname
date --iso-8601=seconds
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
# Recheck immutable data identities after waiting in the scheduler queue.
printf '%s  %s\n' \
  3be669b7fc0966ec49870ecc5927b606b5f2b2fc352f3d8e46d1b5409e4095b7 artifacts/lingshu_scdfm/public_cache_v1.npz \
  39d04694b579776e8c5fa9d73097287de2927629e1a01e19dbfcb2fc26341e77 artifacts/lingshu_scdfm/public_cache_expanded_train_v3.npz \
  46ccc08d63944da1a000ebcad291ce6e32d77f3ad697c5cc0a8266eb1d40d348 scripts/prepare_lingshu_scdfm_expanded_train.py \
  | sha256sum --check -
srun .venv-lingshu-scdfm/bin/python scripts/prepare_lingshu_scdfm_expanded_train.py \
  --verify-existing
# Each production trainer also requires Slurm, CUDA, and an H100 at runtime.
# Only the public training cache differs from the registered v2 recipe.
for VCC_FEATURE_MODE in true shuffled constant; do
  VCC_ARM_DIR="artifacts/lingshu_scdfm/effect_v3_expanded/$VCC_FEATURE_MODE"
  srun .venv-lingshu-scdfm/bin/python scripts/train_lingshu_scdfm_effect.py \
    --cache artifacts/lingshu_scdfm/public_cache_expanded_train_v3.npz \
    --embeddings artifacts/lingshu_scdfm/v1/lingshu_embeddings.npz \
    --embedding-receipt artifacts/lingshu_scdfm/v1/lingshu_embeddings.json \
    --source external/scdfm_lingshu_2cf6bca1 \
    --output-dir "$VCC_ARM_DIR/fit" --feature-mode "$VCC_FEATURE_MODE" \
    --steps 4000 --validate-every 400 --batch-size 32 --seed 20260909
  if srun .venv-lingshu-scdfm/bin/python scripts/check_lingshu_scdfm_public_quality.py \
    --training-summary "$VCC_ARM_DIR/fit/training_summary.json" \
    --output "$VCC_ARM_DIR/public_quality_preflight.json"; then
    printf '%s: public quality preflight passed\n' "$VCC_FEATURE_MODE"
  else
    VCC_CHECK_STATUS=$?
    if [[ "$VCC_CHECK_STATUS" -ne 1 ]]; then exit "$VCC_CHECK_STATUS"; fi
    printf '%s: public quality preflight blocked generation\n' "$VCC_FEATURE_MODE"
  fi
done
date --iso-8601=seconds
srun .venv-lingshu-scdfm/bin/python scripts/check_lingshu_scdfm_public_quality.py \
  --training-summary artifacts/lingshu_scdfm/effect_v3_expanded/true/fit/training_summary.json \
  --output artifacts/lingshu_scdfm/effect_v3_expanded/true/public_quality_preflight.json
