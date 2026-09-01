#!/usr/bin/env bash
#SBATCH --job-name=vcc-cross-prior-v4
#SBATCH --account=allaccess
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/cpu_cross_prior_v4_%j.out
#SBATCH --error=logs/cpu_cross_prior_v4_%j.err

set -euo pipefail

readonly PROJECT_DIR="${VCC_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
readonly PYTHON_BIN="$PROJECT_DIR/.venv-state/bin/python"
readonly SIGNATURE_NPZ="$PROJECT_DIR/artifacts/v2/k562_esm2_signature_v2.npz"
readonly OVERLAY_NPZ="$PROJECT_DIR/artifacts/v2/k562_esm2_bayesian_overlay_v1.npz"
readonly OVERLAY_JSON="$PROJECT_DIR/artifacts/v2/k562_esm2_bayesian_overlay_v1.json"
readonly EFFECT_NPZ="$PROJECT_DIR/artifacts/v2/cross_context_effect_prior_v4.npz"
readonly ROUTE_NPZ="$PROJECT_DIR/artifacts/v2/cross_context_effect_route_v4.npz"
readonly EFFECT_JSON="$PROJECT_DIR/artifacts/v2/cross_context_effect_prior_v4.json"
readonly PSEUDOBULK_NPZ="$PROJECT_DIR/artifacts/v2/cross_context_pseudobulk_v4.npz"
readonly PSEUDOBULK_JSON="$PROJECT_DIR/artifacts/v2/cross_context_pseudobulk_v4.json"

cd "$PROJECT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1

for required_path in \
  "$PYTHON_BIN" \
  "$SIGNATURE_NPZ" \
  artifacts/bayesian_prior_h100_v0.npz \
  artifacts/v2/k562_gwps_effect_atlas.npz \
  artifacts/v2/k562_essential_effect_atlas.npz \
  artifacts/v2/rpe1_effect_atlas.npz \
  dataset/controls/pert_counts.csv \
  dataset/controls/gene_names.csv \
  dataset/state_support/extracted/ESM2_pert_features.pt \
  scripts/overlay_effect_signature.py \
  scripts/calibrate_cross_context_effects.py \
  scripts/build_context_pseudobulk_prior.py; do
  if [[ ! -r "$required_path" ]]; then
    echo "Missing required input: $required_path" >&2
    exit 2
  fi
done

for output_path in \
  "$OVERLAY_NPZ" \
  "$OVERLAY_JSON" \
  "$EFFECT_NPZ" \
  "$ROUTE_NPZ" \
  "$EFFECT_JSON" \
  "$PSEUDOBULK_NPZ" \
  "$PSEUDOBULK_JSON"; do
  if [[ -e "$output_path" ]]; then
    echo "Refusing to overwrite output: $output_path" >&2
    exit 3
  fi
done

mkdir -p "$PROJECT_DIR/artifacts/v2"
date --iso-8601=seconds
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unknown}"

"$PYTHON_BIN" scripts/overlay_effect_signature.py \
  artifacts/bayesian_prior_h100_v0.npz \
  "$SIGNATURE_NPZ" \
  dataset/controls/pert_counts.csv \
  dataset/controls/gene_names.csv \
  "$OVERLAY_NPZ" \
  "$OVERLAY_JSON"

"$PYTHON_BIN" scripts/calibrate_cross_context_effects.py \
  artifacts/v2/k562_gwps_effect_atlas.npz \
  artifacts/v2/k562_essential_effect_atlas.npz \
  artifacts/v2/rpe1_effect_atlas.npz \
  "$OVERLAY_NPZ" \
  dataset/controls/pert_counts.csv \
  "$EFFECT_NPZ" \
  "$ROUTE_NPZ" \
  "$EFFECT_JSON" \
  --challenge-genes dataset/controls/gene_names.csv \
  --esm2-embeddings dataset/state_support/extracted/ESM2_pert_features.pt

"$PYTHON_BIN" scripts/build_context_pseudobulk_prior.py \
  "$EFFECT_NPZ" \
  dataset/controls \
  dataset/controls/pert_counts.csv \
  "$PSEUDOBULK_NPZ" \
  "$PSEUDOBULK_JSON" \
  --effect-clip 1.0

"$PYTHON_BIN" - \
  "$OVERLAY_JSON" \
  "$EFFECT_JSON" \
  "$PSEUDOBULK_JSON" \
  "$PSEUDOBULK_NPZ" <<'PY'
import hashlib
import json
import pathlib
import sys

overlay = json.load(open(sys.argv[1], encoding="utf-8"))
effect = json.load(open(sys.argv[2], encoding="utf-8"))
pseudobulk = json.load(open(sys.argv[3], encoding="utf-8"))
artifact = pathlib.Path(sys.argv[4])

assert overlay["schema"] == "vcc-effect-signature-overlay-v1"
assert overlay["axes"]["effect_shape"] == [3, 300, 18533]
assert overlay["contract"]["fallback_preserved_outside_overlay"] is True
assert overlay["overlay"]["written_coordinates"] > 0
assert effect["schema"] == "vcc-cross-context-effect-calibration-v1"
assert effect["dimensions"]["challenge_targets"] == 300
assert effect["dimensions"]["output_genes"] == 18533
assert pseudobulk["schema"] == "vcc-context-pseudobulk-prior-v1"
assert pseudobulk["output"]["shape"] == [3, 300, 18533]
assert pseudobulk["contract"]["model_derived"] is True
assert pseudobulk["contract"]["measured_perturbed_profiles_inserted"] is False

digest = hashlib.sha256()
with artifact.open("rb") as handle:
    while block := handle.read(8 << 20):
        digest.update(block)
assert pseudobulk["output"]["sha256"] == digest.hexdigest()
print("Cross-context v4 prior chain passed production gates")
PY

date --iso-8601=seconds
