#!/usr/bin/env bash
#SBATCH --job-name=vcc-cross-context
#SBATCH --account=allaccess
#SBATCH --partition=full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/h100_cross_context_%j.out
#SBATCH --error=logs/h100_cross_context_%j.err

set -euo pipefail

readonly PROJECT_DIR="${VCC_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
readonly PYTHON_BIN="$PROJECT_DIR/.venv-state/bin/python"
readonly MODEL_DIR="$PROJECT_DIR/artifacts/state_runs/state_sm_20k_v0"
readonly CHECKPOINT_EXPECTED_SHA256="e407e51855792d0dc85483a977f63497d9ac9a6893db360d840c7d9ce556780d"
readonly PERTURBATION_MAP_EXPECTED_SHA256="6ab08c62dd117012b386051a9f375505abcd17b88ed76f0ad232199624291170"
readonly CANDIDATE_VERSION="${VCC_CANDIDATE_VERSION:-v3}"
case "$CANDIDATE_VERSION" in
  v3|v4)
    readonly RESPONSE_VERSION="$CANDIDATE_VERSION"
    readonly STATE_EFFECT_WEIGHT="0.05"
    readonly STATE_EFFECT_CLIP="0.30"
    readonly COMMON_RESPONSE_WEIGHT="1.0"
    readonly LEARNED_RESPONSE_WEIGHT="1.0"
    readonly COMBINED_EFFECT_CLIP="1.0"
    readonly PSEUDOBULK_BLEND_WEIGHT="0.60"
    readonly ZERO_INDUCTION_SCALE="1.0"
    readonly MAX_DROPPED_COUNT_FRACTION="0.10"
    readonly MAX_LIBRARY_DRIFT_FRACTION="0.15"
    ;;
  v5)
    # V5 keeps the directional public-data response while removing the dense
    # absolute-pseudobulk branch that dominated V4 expression calibration.
    readonly RESPONSE_VERSION="v4"
    readonly STATE_EFFECT_WEIGHT="0.01"
    readonly STATE_EFFECT_CLIP="0.15"
    readonly COMMON_RESPONSE_WEIGHT="0.10"
    readonly LEARNED_RESPONSE_WEIGHT="0.22"
    readonly COMBINED_EFFECT_CLIP="0.35"
    readonly PSEUDOBULK_BLEND_WEIGHT="0.0"
    readonly ZERO_INDUCTION_SCALE="0.0"
    readonly MAX_DROPPED_COUNT_FRACTION="0.03"
    readonly MAX_LIBRARY_DRIFT_FRACTION="0.05"
    ;;
  *)
    echo "VCC_CANDIDATE_VERSION must be v3, v4, or v5" >&2
    exit 2
    ;;
esac
readonly RESPONSE_NPZ="$PROJECT_DIR/artifacts/v2/cross_context_pseudobulk_${RESPONSE_VERSION}.npz"
readonly OUTPUT_H5AD="$PROJECT_DIR/artifacts/v2/cross_context_state_${CANDIDATE_VERSION}.h5ad"
readonly OUTPUT_JSON="$PROJECT_DIR/artifacts/v2/cross_context_state_${CANDIDATE_VERSION}.json"

cd "$PROJECT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1
export CPATH="$PROJECT_DIR/.venv-state-headers/usr/include/python3.11"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

task_scratch="$(mktemp -d "/tmp/vcc_cross_context_${CANDIDATE_VERSION}_XXXXXX")"
readonly task_scratch
trap 'rm -rf -- "$task_scratch"' EXIT

for required_path in \
  "$PYTHON_BIN" \
  "$MODEL_DIR/checkpoints/step16000.ckpt" \
  "$MODEL_DIR/checkpoints/selected_checkpoint.json" \
  "$MODEL_DIR/pert_onehot_map.pt" \
  "$MODEL_DIR/var_dims.pkl" \
  "$RESPONSE_NPZ" \
  scripts/generate_state_scorer_aware_counts.py \
  scripts/generate_state_direct_counts.py \
  scripts/infer_state_effect_prior.py \
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

for output_path in "$OUTPUT_H5AD" "$OUTPUT_JSON"; do
  if [[ -e "$output_path" ]]; then
    echo "Refusing to overwrite production artifact: $output_path" >&2
    exit 3
  fi
done

scratch_available_bytes="$(df -PB1 /tmp | awk 'NR==2 {print $4}')"
readonly scratch_available_bytes
if [[ "$scratch_available_bytes" -lt 85899345920 ]]; then
  echo "Need at least 80 GiB free in /tmp for bounded H5AD generation" >&2
  exit 4
fi

date --iso-8601=seconds
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unknown}"
df -h "$PROJECT_DIR" /tmp
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader

"$PYTHON_BIN" scripts/generate_state_scorer_aware_counts.py \
  --model-dir "$MODEL_DIR" \
  --checkpoint checkpoints/step16000.ckpt \
  --checkpoint-expected-sha256 "$CHECKPOINT_EXPECTED_SHA256" \
  --perturbation-map-expected-sha256 "$PERTURBATION_MAP_EXPECTED_SHA256" \
  --selection-json "$MODEL_DIR/checkpoints/selected_checkpoint.json" \
  --controls-dir dataset/controls \
  --support-genes dataset/state_support/extracted/gene_names.csv \
  --output-h5ad "$OUTPUT_H5AD" \
  --output-json "$OUTPUT_JSON" \
  --scratch-dir "$task_scratch" \
  --device cuda \
  --require-cuda \
  --require-h100 \
  --model-chunk-size 128 \
  --cells-per-group 400 \
  --target-limit 0 \
  --state-effect-weight "$STATE_EFFECT_WEIGHT" \
  --state-effect-clip "$STATE_EFFECT_CLIP" \
  --state-effect-bounding tanh \
  --response-npz "$RESPONSE_NPZ" \
  --common-response-weight "$COMMON_RESPONSE_WEIGHT" \
  --learned-response-weight "$LEARNED_RESPONSE_WEIGHT" \
  --combined-effect-clip "$COMBINED_EFFECT_CLIP" \
  --pseudobulk-npz "$RESPONSE_NPZ" \
  --pseudobulk-blend-weight "$PSEUDOBULK_BLEND_WEIGHT" \
  --zero-induction-scale "$ZERO_INDUCTION_SCALE" \
  --pseudobulk-depth-policy source-library \
  --target-policy off \
  --count-emission stochastic-round \
  --max-genes-per-cell 5900 \
  --max-dropped-count-fraction "$MAX_DROPPED_COUNT_FRACTION" \
  --max-absolute-library-drift-fraction "$MAX_LIBRARY_DRIFT_FRACTION" \
  --groups-per-shard 20 \
  --concat-max-loaded-elements 20000000 \
  --compression lzf \
  --seed 20260831

test -s "$OUTPUT_H5AD"
test -s "$OUTPUT_JSON"

"$PYTHON_BIN" - \
  "$OUTPUT_JSON" \
  "$OUTPUT_H5AD" \
  "$RESPONSE_NPZ" \
  "$CANDIDATE_VERSION" <<'PY'
import hashlib
import json
import pathlib
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
prediction = pathlib.Path(sys.argv[2]).resolve()
response = pathlib.Path(sys.argv[3]).resolve()
candidate = sys.argv[4]

assert report["schema"] == "vcc-state-scorer-aware-counts-v2"
assert report["full_official_contract"] is True
config = report["configuration"]
assert config["device"].startswith("cuda")
assert "H100" in config["device_name"]
assert config["cells_per_group"] == 400
assert config["targets"] == 300
assert config["state_effect_bounding"] == "tanh"
expected = {
    "v3": (0.05, 0.30, 1.0, 1.0, 1.0, 0.60, 1.0),
    "v4": (0.05, 0.30, 1.0, 1.0, 1.0, 0.60, 1.0),
    "v5": (0.01, 0.15, 0.10, 0.22, 0.35, 0.0, 0.0),
}[candidate]
actual = (
    config["state_effect_weight"],
    config["state_effect_clip"],
    config["common_response_weight"],
    config["learned_response_weight"],
    config["combined_effect_clip"],
    config["pseudobulk_blend_weight"],
    config["zero_induction_scale"],
)
assert actual == expected, (actual, expected)
assert config["pseudobulk_depth_policy"] == "source-library"
assert config["target_policy"] == "off"
assert config["count_emission"] == "stochastic-round"
assert config["max_genes_per_cell"] == 5900
expected_mass_thresholds = {
    "v3": (0.10, 0.15),
    "v4": (0.10, 0.15),
    "v5": (0.03, 0.05),
}[candidate]
assert config["max_dropped_count_fraction"] == expected_mass_thresholds[0]
assert config["max_absolute_library_drift_fraction"] == expected_mass_thresholds[1]
assert config["library_policy"] == (
    "independent-gene-expectations; no total renormalization"
)
assert report["scorer_contract"]["perturbed_gene_excluded_from_all_six_metrics"] is True
assert report["scorer_contract"]["target_clamp_changes_other_coordinates"] is False
assert report["axes"]["contexts"] == ["A", "B", "C"]
assert report["axes"]["targets"] == 300
assert report["axes"]["current_genes"] == 18533
assert report["axes"]["support_genes"] == 18080
assert report["axes"]["shared_genes"] == 18077
validation = report["validation"]
assert validation["failed_checks"] == []
assert validation["shape"] == [360000, 18533]
assert validation["groups"] == 900
assert validation["nnz"] < 2140000000
assert all(validation["checks"].values())
assert report["scientific_qc"]["target_clamp_failures"] == []
if candidate == "v5":
    assert report["scientific_qc"]["induced_zero_emitted_entries"] == 0
else:
    assert report["scientific_qc"]["induced_zero_emitted_entries"] > 0
mass = report["scientific_qc"]["count_mass"]
assert mass["passed"] is True
assert mass["violations"] == []
assert mass["thresholds"] == {
    "max_dropped_count_fraction": expected_mass_thresholds[0],
    "max_absolute_library_drift_fraction": expected_mass_thresholds[1],
    "evaluation_scopes": ["overall", "A", "B", "C"],
}
assert set(mass["by_context"]) == {"A", "B", "C"}
for scope in [mass["overall"], *mass["by_context"].values()]:
    assert scope["emitted_count_total_before_cap"] == (
        scope["emitted_count_total_after_cap"] + scope["dropped_count_total"]
    )
    assert scope["dropped_count_fraction"] <= expected_mass_thresholds[0]
    assert scope["absolute_library_drift_fraction"] <= expected_mass_thresholds[1]
if candidate == "v5":
    assert mass["overall"]["dropped_count_fraction"] <= 0.025
assert mass["overall"]["dropped_count_total"] == report["scientific_qc"][
    "dropped_counts_without_redistribution"
]

for key in ("response_artifact", "pseudobulk_artifact"):
    assert pathlib.Path(report["provenance"][key]["path"]).resolve() == response
output_provenance = report["provenance"]["output_h5ad"]
assert pathlib.Path(output_provenance["path"]).resolve() == prediction
assert output_provenance["size_bytes"] == prediction.stat().st_size
digest = hashlib.sha256()
with prediction.open("rb") as handle:
    while block := handle.read(8 << 20):
        digest.update(block)
assert output_provenance["sha256"] == digest.hexdigest()
print("Full cross-context STATE generation gates passed")
PY

date --iso-8601=seconds
