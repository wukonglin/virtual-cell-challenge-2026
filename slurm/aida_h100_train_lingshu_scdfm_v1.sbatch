#!/usr/bin/env bash
#SBATCH --job-name=vcc-lingshu-scdfm-fit-v1
#SBATCH --account=allaccess
#SBATCH --partition=full
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=h100:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/aida_lingshu_scdfm_fit_%j.out
#SBATCH --error=logs/aida_lingshu_scdfm_fit_%j.err
#SBATCH --no-requeue
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit through Slurm}"
readonly PROJECT_DIR=/home/fs01/yl4259/yl/virtual-cell-challenge-2026
cd "$PROJECT_DIR"
unset PYTHONHOME PYTHONPATH PYTHONSTARTUP PYTHONINSPECT LD_PRELOAD
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
hostname
date --iso-8601=seconds
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
srun .venv-lingshu-scdfm/bin/python scripts/train_lingshu_scdfm.py \
  --cache artifacts/lingshu_scdfm/public_cache_v1.npz \
  --embeddings artifacts/lingshu_scdfm/v1/lingshu_embeddings.npz \
  --embedding-receipt artifacts/lingshu_scdfm/v1/lingshu_embeddings.json \
  --source external/scdfm_lingshu_2cf6bca1 \
  --output-dir artifacts/lingshu_scdfm/v1/fit \
  --steps 2000 --validate-every 200 --batch-size 16 \
  --validation-batch-size 16 --max-validation-groups 32 \
  --hidden-size 128 --heads 4 --layers 2 --dropout 0.1 \
  --learning-rate 0.0001 --mmd-weight 0.5 --noise-std 0.1 --seed 20260909
date --iso-8601=seconds
