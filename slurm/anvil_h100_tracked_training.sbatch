#!/usr/bin/env bash
# Future pre/post-training entry point; does NOT repeat a historical experiment.
#SBATCH --job-name=vcc-tracked-training
#SBATCH --account=bio250247-ai
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/tracked_training_%j.out
#SBATCH --error=logs/tracked_training_%j.err
#SBATCH --no-requeue

set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit through Slurm; never execute on a head node}"
: "${SLURM_JOB_NODELIST:?An allocated compute node is required}"
[[ "$#" -ge 2 ]] || { printf '%s\n' 'Usage: sbatch script pretrain|posttrain COMMAND [ARGS...]' >&2; exit 2; }
readonly TRAINING_STAGE="$1"
shift
[[ "$TRAINING_STAGE" = pretrain || "$TRAINING_STAGE" = posttrain ]] || exit 2
readonly PROJECT_INPUT="${VCC_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-}}"
: "${PROJECT_INPUT:?Set VCC_PROJECT_DIR or submit from the project root}"
readonly PROJECT_DIR="$(realpath -- "$PROJECT_INPUT")"
readonly TRAIN_PYTHON="${VCC_TRAIN_PYTHON:-$PROJECT_DIR/.venv-lingshu-scdfm/bin/python}"
readonly TRACK_ENV="$PROJECT_DIR/.venv-tracking"
readonly TRACK_PYTHON="$TRACK_ENV/bin/python"
readonly TRACK_ROOT="$PROJECT_DIR/artifacts/tracking/anvil_${SLURM_JOB_ID}_${TRAINING_STAGE}"
readonly TRACK_MODE="${VCC_WANDB_MODE:-offline}"
[[ "$TRACK_MODE" = offline || "$TRACK_MODE" = online ]] || exit 2
if [[ "$TRACK_MODE" = online ]]; then
  : "${WANDB_ENTITY:?Online tracking needs your W&B entity/team}"
  : "${WANDB_API_KEY:?Configure the W&B API key securely in the job environment}"
fi
[[ ! -e "$TRACK_ROOT" ]] || { printf '%s\n' 'Tracking directory exists; refusing to overwrite.' >&2; exit 2; }
cd "$PROJECT_DIR"
unset PYTHONHOME PYTHONPATH PYTHONSTARTUP PYTHONINSPECT LD_PRELOAD VCC_TOKEN
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PIP_CACHE_DIR="$PROJECT_DIR/.cache/pip" XDG_CACHE_HOME="$PROJECT_DIR/.cache"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
# The existing model environment is only READ; the logger gets its own venv.
test -x "$TRAIN_PYTHON"
srun --ntasks=1 "$TRAIN_PYTHON" - <<'PY'
import os
import torch
assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_NODELIST')
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
assert 'H100' in torch.cuda.get_device_name(0), 'The allocation must contain an H100'
assert torch.ones(8, device='cuda').sum().item() == 8
torch.cuda.synchronize()
print('Allocated H100 CUDA probe passed', flush=True)
PY
# Setup stays inside the allocation. Serialize venv changes for concurrent jobs.
(
  flock -w 120 9
  if [[ ! -e "$TRACK_ENV" ]]; then "$TRAIN_PYTHON" -m venv "$TRACK_ENV"; fi
  test -x "$TRACK_PYTHON"
  "$TRACK_PYTHON" -m pip install --disable-pip-version-check --no-input -r requirements/tracking.txt
  "$TRACK_PYTHON" -m pip check
) 9>"$PROJECT_DIR/.venv-tracking.install.lock"

TRACK_ARGS=(--stage "$TRAINING_STAGE" --output-dir "$TRACK_ROOT" --mode "$TRACK_MODE"
  --project "${WANDB_PROJECT:-virtual-cell-challenge-2026}" --group "${VCC_WANDB_GROUP:-lingshu-public}"
  --name "${VCC_WANDB_NAME:-${TRAINING_STAGE}-anvil-${SLURM_JOB_ID}}")
if [[ -n "${WANDB_ENTITY:-}" ]]; then TRACK_ARGS+=(--entity "$WANDB_ENTITY"); fi
if [[ -n "${VCC_TRAINING_SUMMARY:-}" ]]; then TRACK_ARGS+=(--summary-file "$VCC_TRAINING_SUMMARY"); fi
if [[ -n "${VCC_TRAINING_PROVENANCE:-}" ]]; then TRACK_ARGS+=(--summary-file "$VCC_TRAINING_PROVENANCE"); fi
if [[ -n "${VCC_PARENT_CHECKPOINT_SHA256:-}" ]]; then TRACK_ARGS+=(--parent-checkpoint-sha256 "$VCC_PARENT_CHECKPOINT_SHA256"); fi
# Exactly one controller logs; a future torchrun child must emit rank-zero metrics.
# No eval, no embedded credentials, and no automatic rerun of a failed model.
exec srun --ntasks=1 "$TRACK_PYTHON" scripts/run_training_with_wandb.py "${TRACK_ARGS[@]}" -- "$@"
