#!/usr/bin/env bash
#SBATCH --job-name=vcc-bayes-v0
#SBATCH --account=allaccess
#SBATCH --partition=full
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --constraint=gpu-h100
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --output=logs/h100_bayesian_v0_%j.out
#SBATCH --error=logs/h100_bayesian_v0_%j.err

set -euo pipefail

readonly PROJECT_DIR="${VCC_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
readonly PYTHON_BIN="${VCC_MODEL_PYTHON:-$(command -v python3)}"
readonly VCC_BIN="${VCC_CLI:-$PROJECT_DIR/.venv-vcc/bin/vcc}"
readonly PRIOR_NPZ="$PROJECT_DIR/artifacts/bayesian_prior_h100_v0.npz"
readonly PRIOR_JSON="$PROJECT_DIR/artifacts/bayesian_prior_h100_v0.json"
readonly PRED_H5AD="$PROJECT_DIR/artifacts/bayesian_baseline_v0.h5ad"
readonly PRED_JSON="$PROJECT_DIR/artifacts/bayesian_baseline_v0.json"
readonly DRYRUN_JSON="$PROJECT_DIR/artifacts/bayesian_baseline_v0_dryrun.json"

cd "$PROJECT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1

task_tmp="$(mktemp -d /tmp/vcc_bayes_v0_XXXXXX)"
readonly task_tmp
export TMPDIR="$task_tmp"
trap 'rm -rf -- "$task_tmp"' EXIT

for required_path in \
  "$PYTHON_BIN" \
  "$VCC_BIN" \
  artifacts/bayes_public_effects_v1.npz \
  dataset/state_support/extracted/ESM2_pert_features.pt \
  dataset/controls/context_A.h5ad \
  dataset/controls/context_B.h5ad \
  dataset/controls/context_C.h5ad \
  dataset/controls/gene_names.csv \
  dataset/controls/pert_counts.csv; do
  if [[ ! -r "$required_path" ]]; then
    echo "Missing required input: $required_path" >&2
    exit 2
  fi
done

for protected_output in \
  "$PRIOR_NPZ" "$PRIOR_JSON" "$PRED_H5AD" "$PRED_JSON" "$DRYRUN_JSON"; do
  if [[ -e "$protected_output" ]]; then
    echo "Refusing to overwrite existing production artifact: $protected_output" >&2
    exit 3
  fi
done

date --iso-8601=seconds
hostname
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
  --control-sample-cells 2048 \
  --seed 20260826

"$PYTHON_BIN" scripts/generate_bayesian_submission.py \
  --prior "$PRIOR_NPZ" \
  --controls-dir dataset/controls \
  --output-h5ad "$PRED_H5AD" \
  --output-json "$PRED_JSON" \
  --cells-per-group 400 \
  --max-base-genes 5780 \
  --compression lzf \
  --seed 20260826

"$VCC_BIN" prep "$PRED_H5AD" \
  -g dataset/controls/gene_names.csv \
  --perts dataset/controls/pert_counts.csv \
  --pert-col target_gene \
  --context-col context \
  --contexts A,B,C \
  --verify-targets \
  --check-cell-counts \
  --require-counts \
  --reject-controls \
  --encoding 32 \
  --dry-run \
  --json | tee "$DRYRUN_JSON"

sha256sum "$PRIOR_NPZ" "$PRED_H5AD" "$DRYRUN_JSON" \
  | tee artifacts/bayesian_baseline_v0_prepackage_SHA256SUMS

date --iso-8601=seconds
