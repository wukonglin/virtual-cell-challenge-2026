#!/usr/bin/env bash
#SBATCH --job-name=vcc-effect-v2-ablations
#SBATCH --account=allaccess
#SBATCH --partition=full
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=h100:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/aida_effect_v2_ablations_%j.out
#SBATCH --error=logs/aida_effect_v2_ablations_%j.err
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
# Fixed three-arm diagnostic experiment; no challenge generation or upload.
for VCC_FEATURE_MODE in true shuffled constant; do
  VCC_ARM_DIR="artifacts/lingshu_scdfm/effect_v2/$VCC_FEATURE_MODE"
  srun .venv-lingshu-scdfm/bin/python scripts/train_lingshu_scdfm_effect.py \
    --cache artifacts/lingshu_scdfm/public_cache_v1.npz \
    --embeddings artifacts/lingshu_scdfm/v1/lingshu_embeddings.npz \
    --embedding-receipt artifacts/lingshu_scdfm/v1/lingshu_embeddings.json \
    --source external/scdfm_lingshu_2cf6bca1 \
    --output-dir "$VCC_ARM_DIR/fit" --feature-mode "$VCC_FEATURE_MODE" \
    --steps 4000 --validate-every 400 --batch-size 32 --seed 20260909
  # Keep the ablations even when one scientific-quality gate fails. The final
  # primary-arm check determines this job's gate status; execution errors stop it.
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
  --training-summary artifacts/lingshu_scdfm/effect_v2/true/fit/training_summary.json \
  --output artifacts/lingshu_scdfm/effect_v2/true/public_quality_preflight.json
