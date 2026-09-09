#!/usr/bin/env bash
#SBATCH --job-name=vcc-effect-v3-test
#SBATCH --account=allaccess
#SBATCH --partition=regular
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/aida_effect_v3_test_%j.out
#SBATCH --error=logs/aida_effect_v3_test_%j.err
#SBATCH --no-requeue
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit through Slurm}"
cd /home/fs01/yl4259/yl/virtual-cell-challenge-2026
unset PYTHONHOME PYTHONPATH PYTHONSTARTUP PYTHONINSPECT LD_PRELOAD VCC_TOKEN
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
hostname
printf '%s  %s\n' \
  3be669b7fc0966ec49870ecc5927b606b5f2b2fc352f3d8e46d1b5409e4095b7 artifacts/lingshu_scdfm/public_cache_v1.npz \
  39d04694b579776e8c5fa9d73097287de2927629e1a01e19dbfcb2fc26341e77 artifacts/lingshu_scdfm/public_cache_expanded_train_v3.npz \
  | sha256sum --check -
srun .venv-lingshu-scdfm/bin/python -m pytest -q -rs \
  tests/test_lingshu_scdfm_effect.py \
  tests/test_check_lingshu_scdfm_public_quality.py \
  tests/test_compare_lingshu_scdfm_effect_v2.py \
  tests/test_prepare_lingshu_scdfm_expanded_train.py
srun .venv-lingshu-scdfm/bin/python scripts/prepare_lingshu_scdfm_expanded_train.py \
  --verify-existing
# Authenticate the actual staged implementation and unchanged v2 model recipe.
srun .venv-lingshu-scdfm/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

receipt = json.loads(Path('artifacts/lingshu_scdfm/public_cache_expanded_train_v3.json').read_text())
assert digest('scripts/prepare_lingshu_scdfm_expanded_train.py') == receipt['script_sha256']
assert digest('scripts/prepare_lingshu_scdfm_data.py') == receipt['expansion']['source_reader_sha256']
assert receipt['expansion']['training_target_count'] == 146
assert receipt['expansion']['parent_train_rows'] == 9074
assert receipt['expansion']['expanded_train_rows'] > 9074
assert receipt['expansion']['new_train_cells_per_target_cap'] == 512
assert receipt['expansion']['new_control_pool_cells_per_gem_cap'] == 512
prior = json.loads(Path('artifacts/lingshu_scdfm/effect_v2/true/fit/training_provenance.json').read_text())
for name, expected in prior['code_sha256'].items():
    assert digest(Path('scripts') / name) == expected, f'v2 training code changed: {name}'
assert digest('scripts/check_lingshu_scdfm_public_quality.py') == '676bd900a2795c6689c62e92c139ace8686c4ff9519ca813db1ec77ed43cd828'
print(json.dumps({'status': 'pass', 'expanded_preparation_code_authenticated': True,
                  'v2_model_and_training_code_unchanged': True, 'quality_gate_unchanged': True}))
PY
