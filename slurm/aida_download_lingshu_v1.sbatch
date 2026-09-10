#!/usr/bin/env bash
#SBATCH --job-name=vcc-lingshu-download-v1
#SBATCH --account=allaccess
#SBATCH --partition=regular
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=logs/aida_lingshu_download_%j.out
#SBATCH --error=logs/aida_lingshu_download_%j.err
#SBATCH --no-requeue
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit through Slurm}"
readonly PROJECT_DIR=/home/fs01/yl4259/yl/virtual-cell-challenge-2026
cd "$PROJECT_DIR"
unset PYTHONHOME PYTHONPATH PYTHONSTARTUP PYTHONINSPECT LD_PRELOAD
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_XET=1
hostname
date --iso-8601=seconds
.venv-lingshu-scdfm/bin/python scripts/download_lingshu_model.py --model-dir models/Lingshu-7B
date --iso-8601=seconds
