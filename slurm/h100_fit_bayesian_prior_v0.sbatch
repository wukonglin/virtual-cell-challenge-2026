#!/usr/bin/env bash
#SBATCH --job-name=vcc-fit-v0
#SBATCH --account=allaccess
#SBATCH --partition=full
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --constraint=gpu-h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/h100_fit_bayesian_v0_%j.out
#SBATCH --error=logs/h100_fit_bayesian_v0_%j.err

set -euo pipefail

readonly PROJECT_DIR="${VCC_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
readonly PYTHON_BIN="${VCC_MODEL_PYTHON:-$(command -v python3)}"
readonly PRIOR_NPZ="$PROJECT_DIR/artifacts/bayesian_prior_h100_v0.npz"
readonly PRIOR_JSON="$PROJECT_DIR/artifacts/bayesian_prior_h100_v0.json"

cd "$PROJECT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

for required_path in \
  "$PYTHON_BIN" \
  artifacts/bayes_public_effects_v1.npz \
  dataset/state_support/extracted/ESM2_pert_features.pt \
  dataset/controls/gene_names.csv \
  dataset/controls/pert_counts.csv; do
  if [[ ! -r "$required_path" ]]; then
    echo "Missing required input: $required_path" >&2
    exit 2
  fi
done

for protected_output in "$PRIOR_NPZ" "$PRIOR_JSON"; do
  if [[ -e "$protected_output" ]]; then
    echo "Refusing to overwrite existing production artifact: $protected_output" >&2
    exit 3
  fi
done

date --iso-8601=seconds
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unknown}"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(); print({"torch": torch.__version__, "torch_cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0)})'

"$PYTHON_BIN" scripts/fit_bayesian_prior_h100.py \
  --device cuda \
  --require-cuda \
  --effect-cache artifacts/bayes_public_effects_v1.npz \
  --esm2 dataset/state_support/extracted/ESM2_pert_features.pt \
  --controls-dir dataset/controls \
  --output-npz "$PRIOR_NPZ" \
  --output-json "$PRIOR_JSON" \
  --rank 64 \
  --context-residual-multiplier 0 \
  --generic-scale 0.30 \
  --coexpression-z-shift 0 \
  --seed 20260826

"$PYTHON_BIN" - "$PRIOR_JSON" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
assert report["device"] == "cuda"
assert "H100" in report["gpu_name"]
assert report["transfer_method"] == "generic_only"
assert report["cross_validation"]["deployed_target_specific_scale"] == 0.0
assert report["direct_target_overlap"] == 17
assert report["generic_context_conditioned"] is True
assert report["selected_effects_per_group"]["max"] <= 161
print("H100 prior assertions passed")
PY

sha256sum \
  artifacts/bayes_public_effects_v1.npz \
  "$PRIOR_NPZ" \
  "$PRIOR_JSON" \
  scripts/build_public_effect_cache.py \
  scripts/fit_bayesian_prior_h100.py \
  | tee artifacts/bayesian_prior_h100_v0_SHA256SUMS

date --iso-8601=seconds
