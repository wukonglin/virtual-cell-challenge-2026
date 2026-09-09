#!/usr/bin/env bash
#SBATCH --job-name=vcc-lingshu-scdfm-setup-v1
#SBATCH --account=allaccess
#SBATCH --partition=regular
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/aida_setup_lingshu_scdfm_%j.out
#SBATCH --error=logs/aida_setup_lingshu_scdfm_%j.err
#SBATCH --no-requeue

set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit this setup through Slurm}"
readonly PROJECT_DIR=/home/fs01/yl4259/yl/virtual-cell-challenge-2026
readonly ENV_DIR="$PROJECT_DIR/.venv-lingshu-scdfm"
readonly SOURCE_DIR="$PROJECT_DIR/external/scdfm_lingshu_2cf6bca1"
readonly SOURCE_COMMIT=2cf6bca1f044e74c4e1dc586892c0495880cf125
cd "$PROJECT_DIR"
unset PYTHONHOME PYTHONPATH PYTHONSTARTUP PYTHONINSPECT LD_PRELOAD
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PIP_CACHE_DIR="$PROJECT_DIR/.cache/pip"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" external artifacts/lingshu_scdfm/v1
hostname
date --iso-8601=seconds
df -h "$PROJECT_DIR"
if [[ ! -e "$ENV_DIR" ]]; then
  /usr/bin/python3.11 -m venv "$ENV_DIR"
fi
"$ENV_DIR/bin/python" -m pip install --upgrade 'pip==25.2'
"$ENV_DIR/bin/python" -m pip install -r requirements/lingshu-scdfm-route.txt
if [[ ! -e "$SOURCE_DIR" ]]; then
  git clone https://github.com/AI4Science-WestlakeU/scDFM.git "$SOURCE_DIR"
fi
git -C "$SOURCE_DIR" checkout --detach "$SOURCE_COMMIT"
test "$(git -C "$SOURCE_DIR" rev-parse HEAD)" = "$SOURCE_COMMIT"
test -z "$(git -C "$SOURCE_DIR" status --porcelain)"
"$ENV_DIR/bin/python" -m pip freeze > "artifacts/lingshu_scdfm/v1/environment_${SLURM_JOB_ID}.txt"
"$ENV_DIR/bin/python" -c 'import torch, transformers, timm, anndata; print({"torch":torch.__version__,"transformers":transformers.__version__,"timm":timm.__version__,"anndata":anndata.__version__})'
date --iso-8601=seconds
