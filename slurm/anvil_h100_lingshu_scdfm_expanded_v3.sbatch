#!/usr/bin/env bash
# One allocation: setup, integrity tests, three fits, and post-fit comparison.
# No CPU-only partition is available under the current AI allocations.
#SBATCH --job-name=vcc-anvil-effect-v3-expanded
#SBATCH --account=bio250247-ai
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/anvil_effect_v3_expanded_%j.out
#SBATCH --error=logs/anvil_effect_v3_expanded_%j.err
#SBATCH --no-requeue

set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit through Slurm}"
: "${SLURM_JOB_NODELIST:?An allocated compute node is required}"
readonly PROJECT_INPUT="${VCC_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-}}"
: "${PROJECT_INPUT:?Set VCC_PROJECT_DIR or submit from the repository root}"
readonly PROJECT_DIR="$(realpath -- "$PROJECT_INPUT")"
readonly PYTHON_BIN="$PROJECT_DIR/.venv-lingshu-scdfm/bin/python"
readonly RUN_ROOT="$PROJECT_DIR/artifacts/lingshu_scdfm/anvil_effect_v3_expanded"
cd "$PROJECT_DIR"
unset PYTHONHOME PYTHONPATH PYTHONSTARTUP PYTHONINSPECT LD_PRELOAD VCC_TOKEN
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export HF_HOME="$PROJECT_DIR/.cache/huggingface" XDG_CACHE_HOME="$PROJECT_DIR/.cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export VCC_ANVIL_EXPANDED_PIPELINE=1
hostname
date --iso-8601=seconds
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
# These helpers run as batch-shell children, not nested allocations.
# Their srun test/training steps use this same single-GPU allocation.
bash "$PROJECT_DIR/slurm/anvil_setup_lingshu_scdfm_expanded_v3.sbatch"
bash "$PROJECT_DIR/slurm/anvil_test_lingshu_scdfm_expanded_v3.sbatch"
# Reauthenticate the immutable inputs and unchanged training/gate code after queueing.
printf '%s  %s\n' \
  3be669b7fc0966ec49870ecc5927b606b5f2b2fc352f3d8e46d1b5409e4095b7 artifacts/lingshu_scdfm/public_cache_v1.npz \
  39d04694b579776e8c5fa9d73097287de2927629e1a01e19dbfcb2fc26341e77 artifacts/lingshu_scdfm/public_cache_expanded_train_v3.npz \
  615b6867b72d92f7d406b4ae53cfcc3adb86b619cccc88d61bdbde255a2d5e5e artifacts/lingshu_scdfm/public_cache_expanded_train_v3.json \
  66d9f299212472102d292cb4e94c69a39f5ed700cd407b4f0fa5ac51d438f4cb artifacts/lingshu_scdfm/v1/lingshu_embeddings.npz \
  8918387497f21c67de5cb85423a2e3a4000f6a46c14f897b418b013a4c9bf46a artifacts/lingshu_scdfm/v1/lingshu_embeddings.json \
  46ccc08d63944da1a000ebcad291ce6e32d77f3ad697c5cc0a8266eb1d40d348 scripts/prepare_lingshu_scdfm_expanded_train.py \
  fe062caa4f148023e493df52cffd3d6c4b0dc1a42131e62a4f1e81364d35fbb0 scripts/prepare_lingshu_scdfm_data.py \
  5b9e4f5546f123aaa3a58850ae8ed189db39a049e45cae421d4191b163d446be scripts/lingshu_scdfm_effect_model.py \
  55f5f57f20b85fa7c3d7e4898572def86dee5a0b008ef6e6e44c08215acd6ed9 scripts/train_lingshu_scdfm_effect.py \
  6800a3168d3d4cb7d0e3688c2b60e506ba041d4c0aa926f4bd885db53e03202f scripts/lingshu_scdfm_model.py \
  761eaa4975844a114af38bfe4af0bb56605dfadd01977a86bcb3baae1d2a339b scripts/train_lingshu_scdfm.py \
  676bd900a2795c6689c62e92c139ace8686c4ff9519ca813db1ec77ed43cd828 scripts/check_lingshu_scdfm_public_quality.py \
  497c4d690733c2d0cc0dd6a609929c6618d18640ad1c73f40c9d989854142a1b scripts/compare_lingshu_scdfm_effect_v2.py \
  | sha256sum --check -
srun "$PYTHON_BIN" scripts/prepare_lingshu_scdfm_expanded_train.py --verify-existing
# Reserve a new platform-specific run; never resume or overwrite an AIDA/Anvil fit.
mkdir "$RUN_ROOT"
# Run comparison after any fitting/probe termination, preserving prior failures.
# This is the in-allocation equivalent of an afterany comparison dependency.
compare_on_exit() {
  VCC_PIPELINE_STATUS=$?
  trap - EXIT
  set +e
  bash "$PROJECT_DIR/slurm/anvil_compare_lingshu_scdfm_expanded_v3.sbatch"
  VCC_COMPARISON_STATUS=$?
  if [[ "$VCC_PIPELINE_STATUS" -ne 0 ]]; then exit "$VCC_PIPELINE_STATUS"; fi
  exit "$VCC_COMPARISON_STATUS"
}
trap compare_on_exit EXIT
srun "$PYTHON_BIN" - "$RUN_ROOT/runtime.json" <<'PY'
import json
import os
from pathlib import Path
import socket
import sys
import torch
assert torch.cuda.is_available(), 'CUDA is unavailable in the allocation'
assert torch.cuda.device_count() == 1, 'Exactly one visible GPU is required'
name = torch.cuda.get_device_name(0)
assert 'H100' in name, f'Anvil ai allocation is not an H100: {name}'
probe = torch.ones((32, 32), device='cuda')
value = (probe @ probe).sum().item()
torch.cuda.synchronize()
assert value == 32768.0, f'CUDA tensor probe failed: {value}'
record = {'schema': 'vcc-anvil-lingshu-expanded-v3-runtime-v1',
          'hostname': socket.gethostname(), 'slurm_job_id': os.environ['SLURM_JOB_ID'],
          'slurm_job_nodelist': os.environ['SLURM_JOB_NODELIST'],
          'slurm_job_partition': os.environ.get('SLURM_JOB_PARTITION'),
          'gpu_name': name, 'visible_gpu_count': torch.cuda.device_count(),
          'gpu_total_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
          'torch_version': str(torch.__version__), 'torch_cuda_version': torch.version.cuda,
          'cuda_tensor_probe': 'pass', 'challenge_treated_used': False,
          'platform': 'anvil', 'aida_checkpoint_resumed': False}
with Path(sys.argv[1]).open('x') as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
    handle.write('\n')
print(json.dumps(record, sort_keys=True))
PY
# The registered v3 recipe is unchanged: three independent sequential fits.
for VCC_FEATURE_MODE in true shuffled constant; do
  VCC_ARM_DIR="$RUN_ROOT/$VCC_FEATURE_MODE"
  srun "$PYTHON_BIN" scripts/train_lingshu_scdfm_effect.py \
    --cache artifacts/lingshu_scdfm/public_cache_expanded_train_v3.npz \
    --embeddings artifacts/lingshu_scdfm/v1/lingshu_embeddings.npz \
    --embedding-receipt artifacts/lingshu_scdfm/v1/lingshu_embeddings.json \
    --source external/scdfm_lingshu_2cf6bca1 \
    --output-dir "$VCC_ARM_DIR/fit" --feature-mode "$VCC_FEATURE_MODE" \
    --steps 4000 --validate-every 400 --batch-size 32 --seed 20260909
  if srun "$PYTHON_BIN" scripts/check_lingshu_scdfm_public_quality.py \
    --training-summary "$VCC_ARM_DIR/fit/training_summary.json" \
    --output "$VCC_ARM_DIR/public_quality_preflight.json"; then
    printf '%s: public quality preflight passed\n' "$VCC_FEATURE_MODE"
  else
    VCC_CHECK_STATUS=$?
    if [[ "$VCC_CHECK_STATUS" -ne 1 ]]; then exit "$VCC_CHECK_STATUS"; fi
    printf '%s: public quality preflight blocked generation\n' "$VCC_FEATURE_MODE"
  fi
done
date --iso-8601=seconds
# A scientific failure must remain a failure even when every fit completed.
srun "$PYTHON_BIN" scripts/check_lingshu_scdfm_public_quality.py \
  --training-summary "$RUN_ROOT/true/fit/training_summary.json" \
  --output "$RUN_ROOT/true/public_quality_preflight.json"
