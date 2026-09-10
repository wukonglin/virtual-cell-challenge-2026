#!/usr/bin/env bash
#SBATCH --job-name=vcc-lingshu-scdfm-test-v1
#SBATCH --account=allaccess
#SBATCH --partition=regular
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/aida_lingshu_scdfm_test_%j.out
#SBATCH --error=logs/aida_lingshu_scdfm_test_%j.err
#SBATCH --no-requeue
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit through Slurm}"
cd /home/fs01/yl4259/yl/virtual-cell-challenge-2026
unset PYTHONHOME PYTHONPATH PYTHONSTARTUP PYTHONINSPECT LD_PRELOAD
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
hostname
srun .venv-lingshu-scdfm/bin/python -m pytest -q -rs \
  tests/test_extract_lingshu_gene_embeddings.py \
  tests/test_prepare_lingshu_scdfm_data.py tests/test_lingshu_scdfm.py \
  tests/test_audit_lingshu_scdfm_submission.py
