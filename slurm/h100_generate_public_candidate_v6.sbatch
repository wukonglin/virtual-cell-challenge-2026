#!/usr/bin/env bash
#SBATCH --job-name=vcc-public-v6-generate
#SBATCH --account=allaccess
#SBATCH --partition=full
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=01:15:00
#SBATCH --output=logs/h100_public_v6_generate_%j.out
#SBATCH --error=logs/h100_public_v6_generate_%j.err

set -euo pipefail

# Required submission variables deliberately define one isolated candidate.
: "${VCC_CONTEXT:?Set VCC_CONTEXT to HepG2 or Jurkat}"
: "${VCC_OUTPUT_TAG:?Set a unique lowercase VCC_OUTPUT_TAG}"
: "${VCC_RESIDUAL_ALPHA:?Set VCC_RESIDUAL_ALPHA to a nonnegative number}"
: "${VCC_RESIDUAL_NPZ:?Set VCC_RESIDUAL_NPZ to the residual artifact}"
: "${VCC_RESIDUAL_JSON:?Set VCC_RESIDUAL_JSON to its authenticated sidecar}"
readonly REQUESTED_STATE_EFFECT_WEIGHT="${VCC_STATE_EFFECT_WEIGHT:-1.00}"
readonly REQUESTED_TARGET_REMAINING_FRACTION="${VCC_TARGET_REMAINING_FRACTION:-0.20}"

case "$REQUESTED_STATE_EFFECT_WEIGHT" in
  0.50|0.5) readonly STATE_EFFECT_WEIGHT="0.50" ;;
  0.75) readonly STATE_EFFECT_WEIGHT="0.75" ;;
  1.00|1.0|1) readonly STATE_EFFECT_WEIGHT="1.00" ;;
  *)
    echo "VCC_STATE_EFFECT_WEIGHT must be one of 0.50, 0.75, or 1.00" >&2
    exit 2
    ;;
esac

case "$REQUESTED_TARGET_REMAINING_FRACTION" in
  0.20|0.2) readonly TARGET_REMAINING_FRACTION="0.20" ;;
  0.40|0.4) readonly TARGET_REMAINING_FRACTION="0.40" ;;
  *)
    echo "VCC_TARGET_REMAINING_FRACTION must be the locked default 0.20 or pre-registered P3 value 0.40" >&2
    exit 2
    ;;
esac

readonly PROJECT_DIR="${VCC_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
readonly PYTHON_BIN="$PROJECT_DIR/.venv-state/bin/python"
readonly MODEL_DIR="$PROJECT_DIR/artifacts/state_runs/state_sm_20k_v0"
readonly CHECKPOINT_EXPECTED_SHA256="e407e51855792d0dc85483a977f63497d9ac9a6893db360d840c7d9ce556780d"
readonly PERTURBATION_MAP_EXPECTED_SHA256="6ab08c62dd117012b386051a9f375505abcd17b88ed76f0ad232199624291170"
readonly PUBLIC_ROOT="${VCC_PUBLIC_V6_ROOT:-$PROJECT_DIR/artifacts/public_v6}"
readonly GENERATOR_SCRIPT="$PROJECT_DIR/scripts/generate_native_state_anchor_counts.py"
readonly DIRECT_COUNTS_HELPER="$PROJECT_DIR/scripts/generate_state_direct_counts.py"
readonly INFERENCE_HELPER="$PROJECT_DIR/scripts/infer_state_effect_prior.py"
readonly PERTURBATION_MAP="$MODEL_DIR/pert_onehot_map.pt"
readonly VAR_DIMS="$MODEL_DIR/var_dims.pkl"
readonly STATE_CONFIG="$MODEL_DIR/config.yaml"
readonly STATE_SOURCE_DIR="${VCC_STATE_SOURCE_DIR:-$PROJECT_DIR/external/state_runtime_9bbfe78a}"

if [[ ! "$VCC_OUTPUT_TAG" =~ ^[a-z0-9][a-z0-9._-]{0,63}$ ]]; then
  echo "VCC_OUTPUT_TAG is unsafe" >&2
  exit 2
fi
readonly RESOLVED_PUBLIC_ROOT="$(realpath -m "$PUBLIC_ROOT")"
readonly V51_ROOT="$(realpath -m "$PROJECT_DIR/artifacts/public_v51")"
case "$RESOLVED_PUBLIC_ROOT" in
  "$V51_ROOT"|"$V51_ROOT"/*)
    echo "The V6 output root must not overlap artifacts/public_v51" >&2
    exit 2
    ;;
esac

case "$VCC_CONTEXT" in
  HepG2) readonly CONTEXT_PREFIX="hepg2" ;;
  Jurkat) readonly CONTEXT_PREFIX="jurkat" ;;
  *) echo "VCC_CONTEXT must be HepG2 or Jurkat" >&2; exit 2 ;;
esac

readonly RUN_DIR="$RESOLVED_PUBLIC_ROOT/candidates/$CONTEXT_PREFIX/$VCC_OUTPUT_TAG"
readonly SPEC_JSON="$RUN_DIR/spec.json"
readonly OUTPUT_H5AD="$RUN_DIR/prediction.h5ad"
readonly OUTPUT_JSON="$RUN_DIR/generation.json"
readonly VERIFICATION_JSON="$RUN_DIR/generation_verified.json"
readonly CONTROLS="$PROJECT_DIR/dataset/public_v51/${CONTEXT_PREFIX}_controls_only.h5ad"

cd "$PROJECT_DIR"
mkdir -p logs
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1
export CPATH="$PROJECT_DIR/.venv-state-headers/usr/include/python3.11"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for required_path in \
  "$PYTHON_BIN" \
  scripts/public_candidate_v6_contract.py \
  "$GENERATOR_SCRIPT" \
  "$DIRECT_COUNTS_HELPER" \
  "$INFERENCE_HELPER" \
  "$STATE_SOURCE_DIR" \
  "$CONTROLS" \
  dataset/public_v51/split_manifest.csv \
  dataset/public_v51/split_manifest.json \
  dataset/state_support/extracted/gene_names.csv \
  "$MODEL_DIR/checkpoints/step16000.ckpt" \
  "$MODEL_DIR/checkpoints/selected_checkpoint.json" \
  "$PERTURBATION_MAP" \
  "$VAR_DIMS" \
  "$VCC_RESIDUAL_NPZ" \
  "$VCC_RESIDUAL_JSON"; do
  if [[ ! -r "$required_path" ]]; then
    echo "Missing candidate input: $required_path" >&2
    exit 3
  fi
done

for output_path in "$SPEC_JSON" "$OUTPUT_H5AD" "$OUTPUT_JSON" "$VERIFICATION_JSON"; do
  if [[ -e "$output_path" ]]; then
    echo "Refusing to overwrite isolated candidate output: $output_path" >&2
    exit 4
  fi
done
mkdir -p "$RUN_DIR"

# The generator records config.yaml only when it exists. Bind the same optional
# presence decision in the immutable spec so a later appearance or omission
# cannot be mistaken for a matched P3 arm.
declare -a STATE_CONFIG_PLAN_ARGS=()
if [[ -e "$STATE_CONFIG" ]]; then
  if [[ ! -r "$STATE_CONFIG" ]]; then
    echo "STATE model config exists but is unreadable: $STATE_CONFIG" >&2
    exit 3
  fi
  STATE_CONFIG_PLAN_ARGS=(--state-config "$STATE_CONFIG")
fi

task_scratch="$(mktemp -d /tmp/vcc_public_v6_candidate_XXXXXX)"
readonly task_scratch
trap 'rm -rf -- "$task_scratch"' EXIT
export PYTHONPATH="$STATE_SOURCE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1

"$PYTHON_BIN" scripts/public_candidate_v6_contract.py plan \
  --context "$VCC_CONTEXT" \
  --output-tag "$VCC_OUTPUT_TAG" \
  --state-effect-weight "$STATE_EFFECT_WEIGHT" \
  --residual-alpha "$VCC_RESIDUAL_ALPHA" \
  --target-remaining-fraction "$TARGET_REMAINING_FRACTION" \
  --controls-h5ad "$CONTROLS" \
  --panel-manifest dataset/public_v51/split_manifest.json \
  --panel-csv dataset/public_v51/split_manifest.csv \
  --residual-npz "$VCC_RESIDUAL_NPZ" \
  --residual-json "$VCC_RESIDUAL_JSON" \
  --checkpoint "$MODEL_DIR/checkpoints/step16000.ckpt" \
  --selection-json "$MODEL_DIR/checkpoints/selected_checkpoint.json" \
  --support-genes dataset/state_support/extracted/gene_names.csv \
  --generator-script "$GENERATOR_SCRIPT" \
  --generate-state-direct-counts-helper "$DIRECT_COUNTS_HELPER" \
  --infer-state-effect-prior-helper "$INFERENCE_HELPER" \
  --state-source-dir "$STATE_SOURCE_DIR" \
  --perturbation-map "$PERTURBATION_MAP" \
  --var-dims "$VAR_DIMS" \
  "${STATE_CONFIG_PLAN_ARGS[@]}" \
  --output-json "$SPEC_JSON"

date --iso-8601=seconds
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unknown}"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader

"$PYTHON_BIN" "$GENERATOR_SCRIPT" \
  --controls-h5ad "$CONTROLS" \
  --panel-manifest dataset/public_v51/split_manifest.json \
  --panel-csv dataset/public_v51/split_manifest.csv \
  --context "$VCC_CONTEXT" \
  --model-dir "$MODEL_DIR" \
  --checkpoint step16000.ckpt \
  --checkpoint-expected-sha256 "$CHECKPOINT_EXPECTED_SHA256" \
  --perturbation-map-expected-sha256 "$PERTURBATION_MAP_EXPECTED_SHA256" \
  --selection-json "$MODEL_DIR/checkpoints/selected_checkpoint.json" \
  --support-genes dataset/state_support/extracted/gene_names.csv \
  --output-h5ad "$OUTPUT_H5AD" \
  --output-json "$OUTPUT_JSON" \
  --scratch-dir "$task_scratch" \
  --device cuda \
  --require-cuda \
  --require-h100 \
  --model-chunk-size 128 \
  --cells-per-group 400 \
  --state-effect-weight "$STATE_EFFECT_WEIGHT" \
  --state-effect-clip 0.60 \
  --state-effect-bounding tanh \
  --residual-npz "$VCC_RESIDUAL_NPZ" \
  --residual-alpha "$VCC_RESIDUAL_ALPHA" \
  --combined-effect-clip 0.65 \
  --target-policy force \
  --target-remaining-fraction "$TARGET_REMAINING_FRACTION" \
  --max-genes-per-cell 5900 \
  --integerization largest-remainder \
  --groups-per-shard 20 \
  --concat-max-loaded-elements 20000000 \
  --compression lzf \
  --seed 20260901

"$PYTHON_BIN" scripts/public_candidate_v6_contract.py verify \
  --spec "$SPEC_JSON" \
  --prediction-h5ad "$OUTPUT_H5AD" \
  --generation-json "$OUTPUT_JSON" \
  --output-json "$VERIFICATION_JSON"

date --iso-8601=seconds
