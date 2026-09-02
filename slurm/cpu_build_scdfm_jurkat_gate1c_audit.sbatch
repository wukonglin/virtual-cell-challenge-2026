#!/usr/bin/env bash
#SBATCH --job-name=vcc-scdfm-jurkat-gate1c
#SBATCH --account=allaccess
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail
umask 077
unset PYTHONOPTIMIZE

# This CPU-only job performs the expensive offline split audit. It may replay
# the registered deterministic KMeans split, but it never starts model
# training and never reads treated expression-matrix values.
readonly PROJECT_DIR="${VCC_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
readonly PYTHON_BIN="$PROJECT_DIR/.venv-state/bin/python"
readonly CONFIG_PATH="$PROJECT_DIR/configs/scdfm/vcc2026_v7_gamma1.toml"
readonly MANIFEST_PATH="$PROJECT_DIR/artifacts/scdfm/v7/splits/jurkat_non_harm.json"
readonly PREFLIGHT_PATH="$PROJECT_DIR/artifacts/scdfm/v7/training/jurkat_training_preflight.json"
readonly AUDIT_DIR="$PROJECT_DIR/artifacts/scdfm/v7/audits"
readonly MANIFEST_AUTH="$AUDIT_DIR/jurkat_manifest_auth_${SLURM_JOB_ID:?SLURM_JOB_ID is required}.json"
readonly PREFLIGHT_AUTH="$AUDIT_DIR/jurkat_preflight_auth_${SLURM_JOB_ID}.json"

cd "$PROJECT_DIR"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

for required_path in \
  "$PYTHON_BIN" \
  "$CONFIG_PATH" \
  scripts/build_scdfm_jurkat_non_harm_manifest.py \
  scripts/authenticate_scdfm_jurkat_non_harm_manifest.py \
  scripts/preflight_scdfm_training_data.py \
  scripts/authenticate_scdfm_training_preflight.py; do
  if [[ ! -r "$required_path" ]]; then
    echo "Missing Gate 1c audit input: $required_path" >&2
    exit 2
  fi
done

for job_receipt in "$MANIFEST_AUTH" "$PREFLIGHT_AUTH"; do
  if [[ -e "$job_receipt" || -L "$job_receipt" ]]; then
    echo "Refusing to overwrite job-specific Gate 1c receipt: $job_receipt" >&2
    exit 3
  fi
done

date --iso-8601=seconds
hostname
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "Offline audit Python: $PYTHON_BIN"

if [[ ! -e "$MANIFEST_PATH" && ! -L "$MANIFEST_PATH" ]]; then
  srun --nodes=1 --ntasks=1 \
    "$PYTHON_BIN" scripts/build_scdfm_jurkat_non_harm_manifest.py \
    --config "$CONFIG_PATH" \
    --output "$MANIFEST_PATH"
else
  echo "Manifest already exists; authenticating without rebuilding: $MANIFEST_PATH"
fi
test -s "$MANIFEST_PATH"

srun --nodes=1 --ntasks=1 \
  "$PYTHON_BIN" scripts/authenticate_scdfm_jurkat_non_harm_manifest.py \
  --config "$CONFIG_PATH" \
  --manifest "$MANIFEST_PATH" \
  --output "$MANIFEST_AUTH"
test -s "$MANIFEST_AUTH"

if [[ ! -e "$PREFLIGHT_PATH" && ! -L "$PREFLIGHT_PATH" ]]; then
  srun --nodes=1 --ntasks=1 \
    "$PYTHON_BIN" scripts/preflight_scdfm_training_data.py \
    --config "$CONFIG_PATH" \
    --manifest "$MANIFEST_PATH" \
    --output "$PREFLIGHT_PATH"
else
  echo "Preflight already exists; authenticating without rebuilding: $PREFLIGHT_PATH"
fi
test -s "$PREFLIGHT_PATH"

srun --nodes=1 --ntasks=1 \
  "$PYTHON_BIN" scripts/authenticate_scdfm_training_preflight.py \
  --config "$CONFIG_PATH" \
  --manifest "$MANIFEST_PATH" \
  --preflight "$PREFLIGHT_PATH" \
  --output "$PREFLIGHT_AUTH"
test -s "$PREFLIGHT_AUTH"

echo "Sealed manifest: $MANIFEST_PATH"
echo "Manifest authentication: $MANIFEST_AUTH"
echo "Sealed preflight: $PREFLIGHT_PATH"
echo "Preflight authentication: $PREFLIGHT_AUTH"
echo "training_ready=false (no portable trainer integration)"
date --iso-8601=seconds
