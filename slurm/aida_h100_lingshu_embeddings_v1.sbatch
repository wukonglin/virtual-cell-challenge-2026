#!/usr/bin/env bash
#SBATCH --job-name=vcc-lingshu-embeddings-v1
#SBATCH --account=allaccess
#SBATCH --partition=full
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus=h100:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/aida_lingshu_embeddings_%j.out
#SBATCH --error=logs/aida_lingshu_embeddings_%j.err
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
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p artifacts/lingshu_scdfm/v1
hostname
date --iso-8601=seconds
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
srun .venv-lingshu-scdfm/bin/python scripts/extract_lingshu_gene_embeddings.py \
  --model-dir models/Lingshu-7B \
  --genes-csv artifacts/lingshu_scdfm/embedding_genes_v1.csv \
  --output-npz artifacts/lingshu_scdfm/v1/lingshu_embeddings.npz \
  --output-json artifacts/lingshu_scdfm/v1/lingshu_embeddings.json \
  --batch-size 8 --require-h100
date --iso-8601=seconds
