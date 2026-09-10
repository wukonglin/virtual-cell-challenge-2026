#!/usr/bin/env bash
# Fixed 3-arm GO pilot: one allocated H100, 200 updates/arm, no automatic rerun.
#SBATCH --job-name=vcc-go-flow-pilot
#SBATCH --account=bio250247-ai
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --constraint=H100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/go_flow_%j.out
#SBATCH --error=logs/go_flow_%j.err
#SBATCH --no-requeue
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit through Slurm}"
: "${SLURM_JOB_NODELIST:?An allocated node is required}"
: "${VCC_PROJECT_DIR:?Set the isolated pilot release directory}"
: "${VCC_TRAIN_PYTHON:?Set the existing model Python executable}"
readonly VCC_CURRENT_HOST="$(hostname -s)"
VCC_HOST_MATCH=false
while IFS= read -r VCC_ALLOC_HOST; do
  if [[ "$VCC_CURRENT_HOST" = "$VCC_ALLOC_HOST" ]]; then VCC_HOST_MATCH=true; fi
done < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[[ "$VCC_HOST_MATCH" = true ]] || { printf '%s\n' 'Refusing setup outside the allocated compute node.' >&2; exit 2; }
cd "$VCC_PROJECT_DIR"
[[ ! -e .venv-tracking && ! -L .venv-tracking ]] || { printf '%s\n' 'Tracking environment already exists; refusing reuse.' >&2; exit 2; }
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONINSPECT LD_PRELOAD VCC_TOKEN
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PIP_CACHE_DIR="$VCC_PROJECT_DIR/.cache/pip" XDG_CACHE_HOME="$VCC_PROJECT_DIR/.cache"
export PIP_DEFAULT_TIMEOUT=30 PIP_RETRIES=0
srun --ntasks=1 "$VCC_TRAIN_PYTHON" - <<'PY'
import json, os, torch
assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_NODELIST')
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
assert 'H100' in torch.cuda.get_device_name(0)
assert torch.ones(16, device='cuda').sum().item() == 16
torch.cuda.synchronize()
print(json.dumps({'allocated_gpu': torch.cuda.get_device_name(0), 'cuda_probe': 'passed',
                  'slurm_job_id': os.environ['SLURM_JOB_ID'], 'torch_version': torch.__version__}), flush=True)
PY
# All environment setup is inside this allocation. Existing model env is read-only.
"$VCC_TRAIN_PYTHON" -m venv .venv-tracking
.venv-tracking/bin/python -m pip install --disable-pip-version-check --no-input -r requirements/tracking.txt
.venv-tracking/bin/python -m pip check
exec srun --ntasks=1 "$VCC_TRAIN_PYTHON" scripts/run_public_flow_pilot_suite.py \
  --pilot-dir artifacts/public_flow/h1_dev_pilot_v1 --tracking-python "$VCC_PROJECT_DIR/.venv-tracking/bin/python"
