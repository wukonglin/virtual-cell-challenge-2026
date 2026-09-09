#!/usr/bin/env bash
# Allocation-internal phase only; never submit this helper separately.
# The end-to-end H100 launcher runs this with bash on its allocated compute node.

set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit this setup through Slurm}"
: "${SLURM_JOB_NODELIST:?An allocated compute node is required}"
test "${VCC_ANVIL_EXPANDED_PIPELINE:-}" = 1
test "${SLURM_JOB_PARTITION:-}" = ai
test "${SLURM_GPUS_ON_NODE:-}" = 1
test "${SLURM_CPUS_PER_TASK:-}" = 8
readonly PROJECT_INPUT="${VCC_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-}}"
: "${PROJECT_INPUT:?Set VCC_PROJECT_DIR or submit from the repository root}"
readonly PROJECT_DIR="$(realpath -- "$PROJECT_INPUT")"
readonly ENV_DIR="$PROJECT_DIR/.venv-lingshu-scdfm"
readonly SOURCE_DIR="$PROJECT_DIR/external/scdfm_lingshu_2cf6bca1"
readonly SOURCE_COMMIT=2cf6bca1f044e74c4e1dc586892c0495880cf125
readonly SOURCE_TREE=a04130b07020505a609158cbe31e9e65083c0d79
cd "$PROJECT_DIR"
unset PYTHONHOME PYTHONPATH PYTHONSTARTUP PYTHONINSPECT LD_PRELOAD VCC_TOKEN
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export CONDA_PKGS_DIRS="$PROJECT_DIR/.cache/conda_pkgs"
export PIP_CACHE_DIR="$PROJECT_DIR/.cache/pip"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export XDG_CACHE_HOME="$PROJECT_DIR/.cache"
export HF_HUB_DISABLE_TELEMETRY=1 PIP_DISABLE_PIP_VERSION_CHECK=1
hostname
date --iso-8601=seconds
df -h "$PROJECT_DIR"
printf '%s  %s\n' \
  d922c328aee1a114b868f5015494cd08f4a596d0cb922e63f960b5b5e5ffcd91 requirements/lingshu.txt \
  a56aaf5aebac0f8f8e7c11c6b59302bfc7b216dddd324baead6e6c4dd2a9675f requirements/lingshu-scdfm-route.txt \
  | sha256sum --check -
# Download the public pinned source only in this allocation, if not staged.
# Never reset or modify a pre-existing source checkout to conceal differences.
if [[ ! -e "$SOURCE_DIR" ]]; then
  mkdir -p "$PROJECT_DIR/external"
  git clone --no-checkout https://github.com/AI4Science-WestlakeU/scDFM.git "$SOURCE_DIR"
  git -C "$SOURCE_DIR" checkout --detach "$SOURCE_COMMIT"
fi
test "$(git -C "$SOURCE_DIR" rev-parse HEAD)" = "$SOURCE_COMMIT"
test "$(git -C "$SOURCE_DIR" rev-parse 'HEAD^{tree}')" = "$SOURCE_TREE"
test -z "$(git -C "$SOURCE_DIR" status --porcelain --untracked-files=all --ignored=matching)"
module purge
module load conda/2026.03
# Reassert isolation after site module initialization as well as before it.
unset PYTHONHOME PYTHONPATH PYTHONSTARTUP PYTHONINSPECT LD_PRELOAD VCC_TOKEN
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$HF_HOME" \
  artifacts/lingshu_scdfm/anvil_environment_expanded_v3
if [[ ! -e "$ENV_DIR" ]]; then
  conda create -y --prefix "$ENV_DIR" python=3.11
fi
test -x "$ENV_DIR/bin/python"
"$ENV_DIR/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version'
"$ENV_DIR/bin/python" -m pip install --upgrade 'pip==25.2'
"$ENV_DIR/bin/python" -m pip install -r requirements/lingshu-scdfm-route.txt
"$ENV_DIR/bin/python" -m pip check
"$ENV_DIR/bin/python" -m pip freeze > "artifacts/lingshu_scdfm/anvil_environment_expanded_v3/environment_${SLURM_JOB_ID}.txt"
"$ENV_DIR/bin/python" - <<'PY'
import json
import torch
import transformers
import timm
import anndata
versions = {'torch': str(torch.__version__), 'transformers': transformers.__version__,
            'timm': timm.__version__, 'anndata': anndata.__version__}
assert versions['torch'].split('+')[0] == '2.7.1', versions
assert versions['transformers'] == '4.52.1', versions
assert versions['timm'] == '1.0.19' and versions['anndata'] == '0.12.1', versions
assert torch.cuda.is_available() and torch.cuda.device_count() == 1, 'One allocated CUDA GPU required'
assert 'H100' in torch.cuda.get_device_name(0), 'The allocation must provide an H100'
print(json.dumps({'versions': versions, 'torch_cuda': torch.version.cuda,
                  'gpu_name': torch.cuda.get_device_name(0)}, sort_keys=True))
PY
date --iso-8601=seconds
