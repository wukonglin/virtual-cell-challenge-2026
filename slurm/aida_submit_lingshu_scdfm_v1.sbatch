#!/usr/bin/env bash
#SBATCH --job-name=vcc-lingshu-scdfm-submit-v1
#SBATCH --account=allaccess
#SBATCH --partition=regular
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/aida_lingshu_scdfm_submit_%j.out
#SBATCH --error=logs/aida_lingshu_scdfm_submit_%j.err
#SBATCH --no-requeue
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit through Slurm}"
: "${VCC_TOKEN:?Supply the authorized VCC token through the job environment}"
trap 'unset VCC_TOKEN' EXIT
readonly PROJECT_DIR=/home/fs01/yl4259/yl/virtual-cell-challenge-2026
readonly CLI="$PROJECT_DIR/.venv-vcc-lingshu/bin/vcc"
readonly RUN_DIR="$PROJECT_DIR/artifacts/lingshu_scdfm/v1"
cd "$PROJECT_DIR"
unset PYTHONHOME PYTHONPATH PYTHONSTARTUP PYTHONINSPECT LD_PRELOAD
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export VCC_CONFIG_DIR="$RUN_DIR/private_vcc_state"
mkdir -p "$VCC_CONFIG_DIR"
chmod 700 "$VCC_CONFIG_DIR"
hostname
date --iso-8601=seconds
test ! -e "$RUN_DIR/submission.json"
# Empirical public-data safeguards, byte identities, and format receipts must
# all pass before any submission API call. Never edit predictions to pass.
srun env -u VCC_TOKEN .venv-lingshu-scdfm/bin/python \
  scripts/audit_lingshu_scdfm_submission.py --run-dir "$RUN_DIR"
# VCC_TOKEN is read from the Slurm job environment, never a script argument or
# plaintext login store. Default daily-limit and one-in-flight checks remain on.
# One invocation only: no new-entry retry loop if a transfer/launch is ambiguous.
srun "$CLI" submit "$RUN_DIR/prediction.vcc" \
  --endpoint https://virtualcellchallenge.org \
  --model-name 'Lingshu-scDFM public-only 512g v1' \
  --description 'Frozen Lingshu-7B text features with a public K562-trained compact official scDFM PAD and learned projection. No challenge-treated cells used in fit or selection. 512 modeled genes; other genes retain sampled released-control counts. Zero-shot challenge contexts, not unseen identities for every target. Public target/context holdouts; this is an initial pilot.' \
  --json > "$RUN_DIR/submission.json"
SUBMISSION_ID=$(.venv-vcc-lingshu/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["entry_id"])' "$RUN_DIR/submission.json")
# Save the entry before waiting, so a scoring timeout never loses its identity.
srun "$CLI" status "$SUBMISSION_ID" --endpoint https://virtualcellchallenge.org \
  --wait --poll-interval 60 --wait-timeout 3600 --json > "$RUN_DIR/submission_status.json"
date --iso-8601=seconds
