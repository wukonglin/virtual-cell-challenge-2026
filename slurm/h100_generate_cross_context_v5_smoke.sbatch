#!/usr/bin/env bash
#SBATCH --job-name=vcc-cross-v5-smoke
#SBATCH --account=allaccess
#SBATCH --partition=full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=logs/h100_cross_v5_smoke_%j.out
#SBATCH --error=logs/h100_cross_v5_smoke_%j.err

set -euo pipefail

readonly PROJECT_DIR="${VCC_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
readonly PYTHON_BIN="$PROJECT_DIR/.venv-state/bin/python"
readonly MODEL_DIR="$PROJECT_DIR/artifacts/state_runs/state_sm_20k_v0"
readonly RESPONSE_NPZ="$PROJECT_DIR/artifacts/v2/cross_context_pseudobulk_v4.npz"
readonly OUTPUT_DIR="$PROJECT_DIR/artifacts/v2/smoke"
readonly OUTPUT_H5AD="$OUTPUT_DIR/cross_context_state_v5_smoke.h5ad"
readonly OUTPUT_JSON="$OUTPUT_DIR/cross_context_state_v5_smoke.json"

cd "$PROJECT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export CPATH="$PROJECT_DIR/.venv-state-headers/usr/include/python3.11"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

task_scratch="$(mktemp -d "/tmp/vcc_cross_v5_smoke_XXXXXX")"
readonly task_scratch
trap 'rm -rf -- "$task_scratch"' EXIT

for required_path in \
  "$PYTHON_BIN" \
  "$MODEL_DIR/checkpoints/selected.ckpt" \
  "$MODEL_DIR/checkpoints/selected_checkpoint.json" \
  "$MODEL_DIR/pert_onehot_map.pt" \
  "$MODEL_DIR/var_dims.pkl" \
  "$RESPONSE_NPZ" \
  scripts/generate_state_scorer_aware_counts.py \
  dataset/state_support/extracted/gene_names.csv \
  dataset/controls/gene_names.csv \
  dataset/controls/pert_counts.csv \
  dataset/controls/context_A.h5ad \
  dataset/controls/context_B.h5ad \
  dataset/controls/context_C.h5ad; do
  if [[ ! -r "$required_path" ]]; then
    echo "Missing required input: $required_path" >&2
    exit 2
  fi
done

mkdir -p "$OUTPUT_DIR"
for output_path in "$OUTPUT_H5AD" "$OUTPUT_JSON"; do
  if [[ -e "$output_path" ]]; then
    echo "Refusing to overwrite smoke artifact: $output_path" >&2
    exit 3
  fi
done

date --iso-8601=seconds
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unknown}"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader

"$PYTHON_BIN" scripts/generate_state_scorer_aware_counts.py \
  --model-dir "$MODEL_DIR" \
  --checkpoint checkpoints/selected.ckpt \
  --selection-json "$MODEL_DIR/checkpoints/selected_checkpoint.json" \
  --controls-dir dataset/controls \
  --support-genes dataset/state_support/extracted/gene_names.csv \
  --output-h5ad "$OUTPUT_H5AD" \
  --output-json "$OUTPUT_JSON" \
  --scratch-dir "$task_scratch" \
  --device cuda \
  --require-cuda \
  --require-h100 \
  --model-chunk-size 32 \
  --cells-per-group 32 \
  --target-limit 8 \
  --state-effect-weight 0.01 \
  --state-effect-clip 0.15 \
  --state-effect-bounding tanh \
  --response-npz "$RESPONSE_NPZ" \
  --common-response-weight 0.10 \
  --learned-response-weight 0.22 \
  --combined-effect-clip 0.35 \
  --pseudobulk-npz "$RESPONSE_NPZ" \
  --pseudobulk-blend-weight 0.0 \
  --zero-induction-scale 0.0 \
  --pseudobulk-depth-policy source-library \
  --target-policy off \
  --count-emission stochastic-round \
  --max-genes-per-cell 5900 \
  --max-dropped-count-fraction 0.03 \
  --max-absolute-library-drift-fraction 0.05 \
  --groups-per-shard 4 \
  --concat-max-loaded-elements 20000000 \
  --compression lzf \
  --seed 20260831

"$PYTHON_BIN" - "$OUTPUT_JSON" "$OUTPUT_H5AD" "$RESPONSE_NPZ" <<'PY'
import hashlib
import json
import pathlib
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
prediction = pathlib.Path(sys.argv[2]).resolve()
response = pathlib.Path(sys.argv[3]).resolve()

assert report["schema"] == "vcc-state-scorer-aware-counts-v2"
assert report["full_official_contract"] is False
assert report["configuration"]["device"].startswith("cuda")
assert "H100" in report["configuration"]["device_name"]
assert report["configuration"]["cells_per_group"] == 32
assert report["configuration"]["targets"] == 8
assert report["configuration"]["state_effect_weight"] == 0.01
assert report["configuration"]["common_response_weight"] == 0.10
assert report["configuration"]["learned_response_weight"] == 0.22
assert report["configuration"]["pseudobulk_blend_weight"] == 0.0
assert report["configuration"]["zero_induction_scale"] == 0.0
assert report["validation"]["shape"] == [768, 18533]
assert report["validation"]["groups"] == 24
assert report["validation"]["failed_checks"] == []
assert report["scientific_qc"]["induced_zero_emitted_entries"] == 0
assert report["scientific_qc"]["count_mass"]["passed"] is True
mass = report["scientific_qc"]["count_mass"]
assert mass["thresholds"]["max_dropped_count_fraction"] == 0.03
assert mass["thresholds"]["max_absolute_library_drift_fraction"] == 0.05
for scope in [mass["overall"], *mass["by_context"].values()]:
    assert scope["dropped_count_fraction"] <= 0.03
    assert scope["absolute_library_drift_fraction"] <= 0.05
assert mass["overall"]["dropped_count_fraction"] <= 0.025
groups = report["qc"]["groups"]
assert all(group["induced_zero_expected_entries"] == 0 for group in groups)
assert all(group["induced_zero_emitted_entries"] == 0 for group in groups)
assert all(group["combined_log_fold"]["std"] > 0 for group in groups)
outside = sum(group["combined_outside_bound"] for group in groups)
values = sum(group["effect_values"] for group in groups)
assert values > 0 and outside / values < 1e-3
assert pathlib.Path(report["provenance"]["response_artifact"]["path"]).resolve() == response
assert pathlib.Path(report["provenance"]["pseudobulk_artifact"]["path"]).resolve() == response
assert pathlib.Path(report["provenance"]["output_h5ad"]["path"]).resolve() == prediction
digest = hashlib.sha256(prediction.read_bytes()).hexdigest()
assert report["provenance"]["output_h5ad"]["sha256"] == digest
print("V5 single-H100 smoke gates passed")
PY

date --iso-8601=seconds
