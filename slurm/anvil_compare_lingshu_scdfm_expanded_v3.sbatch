#!/usr/bin/env bash
# Allocation-internal phase only; never submit this helper separately.
# Called from the H100 launcher's EXIT trap, including scientific fit failures.

set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit through Slurm}"
: "${SLURM_JOB_NODELIST:?An allocated compute node is required}"
test "${VCC_ANVIL_EXPANDED_PIPELINE:-}" = 1
readonly PROJECT_INPUT="${VCC_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-}}"
: "${PROJECT_INPUT:?Set VCC_PROJECT_DIR or submit from the repository root}"
readonly PROJECT_DIR="$(realpath -- "$PROJECT_INPUT")"
readonly PYTHON_BIN="$PROJECT_DIR/.venv-lingshu-scdfm/bin/python"
cd "$PROJECT_DIR"
unset PYTHONHOME PYTHONPATH PYTHONSTARTUP PYTHONINSPECT LD_PRELOAD VCC_TOKEN
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export HF_HOME="$PROJECT_DIR/.cache/huggingface" XDG_CACHE_HOME="$PROJECT_DIR/.cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
hostname
printf '%s  %s\n' \
  676bd900a2795c6689c62e92c139ace8686c4ff9519ca813db1ec77ed43cd828 scripts/check_lingshu_scdfm_public_quality.py \
  497c4d690733c2d0cc0dd6a609929c6618d18640ad1c73f40c9d989854142a1b scripts/compare_lingshu_scdfm_effect_v2.py \
  | sha256sum --check -
srun "$PYTHON_BIN" scripts/compare_lingshu_scdfm_effect_v2.py \
  --root artifacts/lingshu_scdfm/anvil_effect_v3_expanded \
  --output artifacts/lingshu_scdfm/anvil_effect_v3_expanded/comparison.json
