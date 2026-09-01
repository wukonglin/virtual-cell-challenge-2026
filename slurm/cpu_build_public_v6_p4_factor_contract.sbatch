#!/usr/bin/env bash
#SBATCH --job-name=vcc-public-v6-p4-contract
#SBATCH --account=allaccess
#SBATCH --partition=full
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/cpu_public_v6_p4_contract_%j.out
#SBATCH --error=logs/cpu_public_v6_p4_contract_%j.err

set -euo pipefail
unset PYTHONOPTIMIZE

readonly PROJECT_DIR="${VCC_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
readonly PYTHON_BIN="$PROJECT_DIR/.venv-state/bin/python"
readonly PUBLIC_ROOT="${VCC_PUBLIC_V6_ROOT:-$PROJECT_DIR/artifacts/public_v6}"
readonly CANDIDATE_ROOT="$PUBLIC_ROOT/candidates/hepg2"
readonly OUTPUT_JSON="$PUBLIC_ROOT/contracts/hepg2_p4_state_gamma.json"

cd "$PROJECT_DIR"
mkdir -p logs
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

for required_path in \
  "$PYTHON_BIN" \
  scripts/build_public_v6_p4_factor_contract.py \
  "$CANDIDATE_ROOT/p4_g100_p0_a010/spec.json" \
  "$CANDIDATE_ROOT/p4_g100_p0_a010/generation_verified.json" \
  "$CANDIDATE_ROOT/p4_g100_p0_a010/prediction.h5ad" \
  "$CANDIDATE_ROOT/p4_g075_p0_a010/spec.json" \
  "$CANDIDATE_ROOT/p4_g075_p0_a010/generation_verified.json" \
  "$CANDIDATE_ROOT/p4_g075_p0_a010/prediction.h5ad" \
  "$CANDIDATE_ROOT/p4_g050_p0_a010/spec.json" \
  "$CANDIDATE_ROOT/p4_g050_p0_a010/generation_verified.json" \
  "$CANDIDATE_ROOT/p4_g050_p0_a010/prediction.h5ad"; do
  if [[ ! -r "$required_path" ]]; then
    echo "Missing P4 factor-contract input: $required_path" >&2
    exit 3
  fi
done

if [[ -e "$OUTPUT_JSON" ]]; then
  echo "Refusing to overwrite P4 factor contract: $OUTPUT_JSON" >&2
  exit 4
fi

"$PYTHON_BIN" scripts/build_public_v6_p4_factor_contract.py build \
  --g100-spec "$CANDIDATE_ROOT/p4_g100_p0_a010/spec.json" \
  --g100-generation-verification "$CANDIDATE_ROOT/p4_g100_p0_a010/generation_verified.json" \
  --g100-prediction "$CANDIDATE_ROOT/p4_g100_p0_a010/prediction.h5ad" \
  --g075-spec "$CANDIDATE_ROOT/p4_g075_p0_a010/spec.json" \
  --g075-generation-verification "$CANDIDATE_ROOT/p4_g075_p0_a010/generation_verified.json" \
  --g075-prediction "$CANDIDATE_ROOT/p4_g075_p0_a010/prediction.h5ad" \
  --g050-spec "$CANDIDATE_ROOT/p4_g050_p0_a010/spec.json" \
  --g050-generation-verification "$CANDIDATE_ROOT/p4_g050_p0_a010/generation_verified.json" \
  --g050-prediction "$CANDIDATE_ROOT/p4_g050_p0_a010/prediction.h5ad" \
  --output-json "$OUTPUT_JSON"

"$PYTHON_BIN" scripts/build_public_v6_p4_factor_contract.py validate \
  --receipt "$OUTPUT_JSON"

date --iso-8601=seconds
