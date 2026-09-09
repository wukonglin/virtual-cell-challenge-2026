#!/usr/bin/env bash
# Allocation-internal phase only; never submit this helper separately.
# Called by the end-to-end H100 launcher only after environment setup succeeds.

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
date --iso-8601=seconds
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
mkdir -p artifacts/lingshu_scdfm/anvil_environment_expanded_v3
readonly TEST_REPORT="artifacts/lingshu_scdfm/anvil_environment_expanded_v3/tests_${SLURM_JOB_ID}.xml"
srun "$PYTHON_BIN" -m pytest -q -rs --junitxml="$TEST_REPORT" \
  tests/test_lingshu_scdfm_effect.py \
  tests/test_check_lingshu_scdfm_public_quality.py \
  tests/test_compare_lingshu_scdfm_effect_v2.py \
  tests/test_prepare_lingshu_scdfm_expanded_train.py \
  tests/test_anvil_lingshu_launchers.py
# The genuine installed PAD test is required; pytest success with skips is not enough.
srun "$PYTHON_BIN" - "$TEST_REPORT" <<'PY'
import sys
import xml.etree.ElementTree as ET
root = ET.parse(sys.argv[1]).getroot()
assert not root.findall('.//skipped'), 'Anvil integration tests must have zero skips'
assert any('test_authentic_pad_gradient_and_translation_equivariance' == case.get('name')
           for case in root.findall('.//testcase')), 'Genuine upstream PAD integration test missing'
print({'status': 'pass', 'all_tests_executed': True, 'genuine_pad_integration_required': True})
PY
