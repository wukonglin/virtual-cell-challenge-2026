#!/usr/bin/env bash
#SBATCH --job-name=vcc-lingshu-scdfm-package-v1
#SBATCH --account=allaccess
#SBATCH --partition=regular
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/aida_lingshu_scdfm_package_%j.out
#SBATCH --error=logs/aida_lingshu_scdfm_package_%j.err
#SBATCH --no-requeue
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit through Slurm}"
readonly PROJECT_DIR=/home/fs01/yl4259/yl/virtual-cell-challenge-2026
readonly CLI_ENV="$PROJECT_DIR/.venv-vcc-lingshu"
cd "$PROJECT_DIR"
unset PYTHONHOME PYTHONPATH PYTHONSTARTUP PYTHONINSPECT LD_PRELOAD VCC_TOKEN
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PIP_CACHE_DIR="$PROJECT_DIR/.cache/pip"
hostname
date --iso-8601=seconds
# Packaging is credential-free and does not create a leaderboard submission.
if [[ ! -e "$CLI_ENV" ]]; then
  /usr/bin/python3.11 -m venv "$CLI_ENV"
fi
"$CLI_ENV/bin/python" -m pip install 'vcc-cli==0.2.0'
"$CLI_ENV/bin/python" -m pip freeze > "artifacts/lingshu_scdfm/v1/packaging_environment_${SLURM_JOB_ID}.txt"
test -s artifacts/lingshu_scdfm/v1/prediction.h5ad.receipt.json
test ! -e artifacts/lingshu_scdfm/v1/prediction.vcc
srun "$CLI_ENV/bin/vcc" prep artifacts/lingshu_scdfm/v1/prediction.h5ad \
  -g dataset/controls/gene_names.csv --perts dataset/controls/pert_counts.csv \
  --dry-run --json > artifacts/lingshu_scdfm/v1/prep_validation.json
srun "$CLI_ENV/bin/vcc" prep artifacts/lingshu_scdfm/v1/prediction.h5ad \
  -g dataset/controls/gene_names.csv --perts dataset/controls/pert_counts.csv \
  -o artifacts/lingshu_scdfm/v1/prediction.vcc --json \
  > artifacts/lingshu_scdfm/v1/prep_package.json
sha256sum artifacts/lingshu_scdfm/v1/prediction.vcc > artifacts/lingshu_scdfm/v1/prediction.vcc.sha256
date --iso-8601=seconds
