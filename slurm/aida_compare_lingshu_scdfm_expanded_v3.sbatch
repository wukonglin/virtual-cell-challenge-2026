#!/usr/bin/env bash
#SBATCH --job-name=vcc-effect-v3-comparison
#SBATCH --account=allaccess
#SBATCH --partition=regular
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/aida_effect_v3_comparison_%j.out
#SBATCH --error=logs/aida_effect_v3_comparison_%j.err
#SBATCH --no-requeue
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit through Slurm}"
cd /home/fs01/yl4259/yl/virtual-cell-challenge-2026
unset PYTHONHOME PYTHONPATH PYTHONSTARTUP PYTHONINSPECT LD_PRELOAD VCC_TOKEN
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
hostname
srun .venv-lingshu-scdfm/bin/python scripts/compare_lingshu_scdfm_effect_v2.py \
  --root artifacts/lingshu_scdfm/effect_v3_expanded \
  --output artifacts/lingshu_scdfm/effect_v3_expanded/comparison.json
