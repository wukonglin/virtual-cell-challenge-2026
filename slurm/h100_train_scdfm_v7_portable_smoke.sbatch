#!/usr/bin/env bash
#SBATCH --job-name=vcc-scdfm-v7-portable
#SBATCH --account=allaccess
#SBATCH --partition=full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --gpus=h100:1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail
umask 077
unset PYTHONOPTIMIZE

readonly PROJECT_DIR="${VCC_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
readonly PYTHON_BIN="$PROJECT_DIR/.venv-state/bin/python"
readonly EXPECTED_LOCK_SHA256="f9c5b8675a1a46c561c056574009722d1faa4bfaf8b4e12e7d7017d3c5a9628d"
readonly EXPECTED_TRAINER_SHA256="eae081058bec960f012fd2116e9410ff62c9016264d9f82e9f7f6dd80c6bc5dc"
readonly EXPECTED_GATE_CONSUMER_SHA256="9e884bb50eec691dc973c811341f1e636358124514b8a696915c3fe147b9fa13"
readonly EXPECTED_LAUNCHER_SHA256="${VCC_EXPECTED_LAUNCHER_SHA256:?VCC_EXPECTED_LAUNCHER_SHA256 is required}"
readonly EXPECTED_GIT_COMMIT="${VCC_EXPECTED_GIT_COMMIT:?VCC_EXPECTED_GIT_COMMIT is required}"
readonly OUTPUT_JSON="${VCC_SCDFM_PORTABLE_RECEIPT:-$PROJECT_DIR/artifacts/scdfm/v7/training/portable_trainer_smoke_${SLURM_JOB_ID:?SLURM_JOB_ID is required}.json}"

cd "$PROJECT_DIR"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for required_path in \
  "$PYTHON_BIN" \
  artifacts/scdfm/v7/splits/jurkat_non_harm.json \
  artifacts/scdfm/v7/training/jurkat_training_preflight.json \
  artifacts/scdfm/v7/training/portable_training_gate_lock.json \
  configs/scdfm/vcc2026_v7_gamma1.toml \
  dataset/raw/replogle_nadig/GSE264667_jurkat_raw_singlecell_01.h5ad \
  dataset/state_support/extracted/ESM2_pert_features.pt \
  scripts/scdfm_portable_training_gate.py \
  scripts/train_scdfm_v7_portable.py \
  slurm/h100_train_scdfm_v7_portable_smoke.sbatch; do
  if [[ ! -r "$required_path" ]]; then
    echo "Missing portable trainer input: $required_path" >&2
    exit 2
  fi
done

if [[ "$(git rev-parse HEAD)" != "$EXPECTED_GIT_COMMIT" ]]; then
  echo "Git commit differs from VCC_EXPECTED_GIT_COMMIT" >&2
  exit 4
fi
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "H100 evidence requires a clean Git worktree" >&2
  exit 5
fi

if [[ -e "$OUTPUT_JSON" || -L "$OUTPUT_JSON" ]]; then
  echo "Refusing to overwrite portable trainer receipt: $OUTPUT_JSON" >&2
  exit 3
fi

date --iso-8601=seconds
hostname
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
nvidia-smi \
  --query-gpu=index,name,uuid,memory.total,driver_version,compute_cap \
  --format=csv,noheader

srun --nodes=1 --ntasks=1 \
  "$PYTHON_BIN" scripts/train_scdfm_v7_portable.py \
  --repository-root "$PROJECT_DIR" \
  --expected-lock-sha256 "$EXPECTED_LOCK_SHA256" \
  --expected-trainer-sha256 "$EXPECTED_TRAINER_SHA256" \
  --expected-gate-consumer-sha256 "$EXPECTED_GATE_CONSUMER_SHA256" \
  --launcher slurm/h100_train_scdfm_v7_portable_smoke.sbatch \
  --expected-launcher-sha256 "$EXPECTED_LAUNCHER_SHA256" \
  --expected-git-commit "$EXPECTED_GIT_COMMIT" \
  --output-json "$OUTPUT_JSON" \
  --device cuda \
  --require-cuda \
  --require-h100 \
  --batch-size 8 \
  --gene-count 64 \
  --optimizer-steps 1

test -s "$OUTPUT_JSON"
echo "Portable trainer smoke receipt: $OUTPUT_JSON"
date --iso-8601=seconds
