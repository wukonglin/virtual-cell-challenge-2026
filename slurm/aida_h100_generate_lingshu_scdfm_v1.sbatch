#!/usr/bin/env bash
#SBATCH --job-name=vcc-lingshu-scdfm-generate-v1
#SBATCH --account=allaccess
#SBATCH --partition=full
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus=h100:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/aida_lingshu_scdfm_generate_%j.out
#SBATCH --error=logs/aida_lingshu_scdfm_generate_%j.err
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
srun .venv-lingshu-scdfm/bin/python scripts/generate_lingshu_scdfm.py \
  --checkpoint artifacts/lingshu_scdfm/v1/fit/best.pt \
  --embeddings artifacts/lingshu_scdfm/v1/lingshu_embeddings.npz \
  --source external/scdfm_lingshu_2cf6bca1 \
  --controls A=dataset/controls/context_A.h5ad \
  --controls B=dataset/controls/context_B.h5ad \
  --controls C=dataset/controls/context_C.h5ad \
  --gene-axis dataset/controls/gene_names.csv \
  --targets dataset/controls/pert_counts.csv \
  --output artifacts/lingshu_scdfm/v1/prediction.h5ad \
  --cells-per-target 400 --batch-size 64 --ode-steps 20 --seed 20260909
date --iso-8601=seconds
