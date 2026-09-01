#!/usr/bin/env bash
#SBATCH --job-name=vcc-state-anchor-v51-smoke
#SBATCH --account=allaccess
#SBATCH --partition=full
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=logs/h100_state_anchor_v51_smoke_%j.out
#SBATCH --error=logs/h100_state_anchor_v51_smoke_%j.err

set -euo pipefail
unset PYTHONOPTIMIZE

readonly PROJECT_DIR="${VCC_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
readonly PYTHON_BIN="$PROJECT_DIR/.venv-state/bin/python"
readonly MODEL_DIR="$PROJECT_DIR/artifacts/state_runs/state_sm_20k_v0"
readonly RESPONSE_NPZ="$PROJECT_DIR/artifacts/v2/state_anchored_residual_v51_top100.npz"
readonly OUTPUT_DIR="$PROJECT_DIR/artifacts/v2/smoke"
readonly DIRECT_H5AD="$OUTPUT_DIR/state_anchor_v51_direct_reference.h5ad"
readonly DIRECT_JSON="$OUTPUT_DIR/state_anchor_v51_direct_reference.json"
readonly ZERO_H5AD="$OUTPUT_DIR/state_anchor_v51_alpha000.h5ad"
readonly ZERO_JSON="$OUTPUT_DIR/state_anchor_v51_alpha000.json"
readonly V51_H5AD="$OUTPUT_DIR/state_anchor_v51_alpha005.h5ad"
readonly V51_JSON="$OUTPUT_DIR/state_anchor_v51_alpha005.json"

cd "$PROJECT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export CPATH="$PROJECT_DIR/.venv-state-headers/usr/include/python3.11"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

task_scratch="$(mktemp -d /tmp/vcc_state_anchor_v51_smoke_XXXXXX)"
readonly task_scratch
trap 'rm -rf -- "$task_scratch"' EXIT

for required_path in \
  "$PYTHON_BIN" \
  "$MODEL_DIR/checkpoints/selected.ckpt" \
  "$MODEL_DIR/checkpoints/selected_checkpoint.json" \
  "$MODEL_DIR/pert_onehot_map.pt" \
  "$MODEL_DIR/var_dims.pkl" \
  "$RESPONSE_NPZ" \
  scripts/generate_state_direct_counts.py \
  scripts/generate_state_scorer_aware_counts.py \
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

mkdir -p "$OUTPUT_DIR"
for output_path in \
  "$DIRECT_H5AD" "$DIRECT_JSON" \
  "$ZERO_H5AD" "$ZERO_JSON" \
  "$V51_H5AD" "$V51_JSON"; do
  if [[ -e "$output_path" ]]; then
    echo "Refusing to overwrite smoke artifact: $output_path" >&2
    exit 3
  fi
done

date --iso-8601=seconds
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unknown}"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader

"$PYTHON_BIN" scripts/generate_state_direct_counts.py \
  --model-dir "$MODEL_DIR" \
  --checkpoint selected.ckpt \
  --selection-json "$MODEL_DIR/checkpoints/selected_checkpoint.json" \
  --controls-dir dataset/controls \
  --support-genes dataset/state_support/extracted/gene_names.csv \
  --output-h5ad "$DIRECT_H5AD" \
  --output-json "$DIRECT_JSON" \
  --scratch-dir "$task_scratch/direct" \
  --device cuda \
  --require-cuda \
  --require-h100 \
  --model-chunk-size 32 \
  --cells-per-group 32 \
  --target-limit 8 \
  --effect-scale 1.0 \
  --effect-clip 0.60 \
  --effect-bounding tanh \
  --target-strategy force \
  --target-remaining-fraction 0.20 \
  --max-genes-per-cell 5900 \
  --integerization largest-remainder \
  --groups-per-shard 4 \
  --concat-max-loaded-elements 20000000 \
  --compression lzf \
  --seed 20260831

run_anchored_candidate() {
  local residual_weight="$1"
  local output_h5ad="$2"
  local output_json="$3"
  local scratch_name="$4"

  "$PYTHON_BIN" scripts/generate_state_scorer_aware_counts.py \
    --model-dir "$MODEL_DIR" \
    --checkpoint selected.ckpt \
    --selection-json "$MODEL_DIR/checkpoints/selected_checkpoint.json" \
    --controls-dir dataset/controls \
    --support-genes dataset/state_support/extracted/gene_names.csv \
    --output-h5ad "$output_h5ad" \
    --output-json "$output_json" \
    --scratch-dir "$task_scratch/$scratch_name" \
    --device cuda \
    --require-cuda \
    --require-h100 \
    --model-chunk-size 32 \
    --cells-per-group 32 \
    --target-limit 8 \
    --state-effect-weight 1.0 \
    --state-effect-clip 0.60 \
    --state-effect-bounding tanh \
    --response-npz "$RESPONSE_NPZ" \
    --common-response-weight 0.0 \
    --learned-response-weight "$residual_weight" \
    --combined-effect-clip 0.65 \
    --pseudobulk-blend-weight 0.0 \
    --zero-induction-scale 0.0 \
    --target-policy force \
    --target-max-remaining-fraction 0.20 \
    --count-emission exact-largest-remainder \
    --max-genes-per-cell 5900 \
    --max-dropped-count-fraction 0.0 \
    --max-absolute-library-drift-fraction 0.0 \
    --groups-per-shard 4 \
    --concat-max-loaded-elements 20000000 \
    --compression lzf \
    --seed 20260831
}

run_anchored_candidate 0.0 "$ZERO_H5AD" "$ZERO_JSON" alpha000
run_anchored_candidate 0.05 "$V51_H5AD" "$V51_JSON" alpha005

"$PYTHON_BIN" - \
  "$DIRECT_H5AD" "$DIRECT_JSON" \
  "$ZERO_H5AD" "$ZERO_JSON" \
  "$V51_H5AD" "$V51_JSON" \
  "$RESPONSE_NPZ" <<'PY'
import hashlib
import json
import pathlib
import sys

import anndata as ad
import numpy as np
import scipy.sparse as sp

direct_path, direct_report_path = map(pathlib.Path, sys.argv[1:3])
zero_path, zero_report_path = map(pathlib.Path, sys.argv[3:5])
v51_path, v51_report_path = map(pathlib.Path, sys.argv[5:7])
response_path = pathlib.Path(sys.argv[7]).resolve()

direct_report = json.loads(direct_report_path.read_text(encoding="utf-8"))
zero_report = json.loads(zero_report_path.read_text(encoding="utf-8"))
v51_report = json.loads(v51_report_path.read_text(encoding="utf-8"))
assert direct_report["schema"] == "vcc-state-direct-counts-v1"
for report, weight in ((zero_report, 0.0), (v51_report, 0.05)):
    assert report["schema"] == "vcc-state-scorer-aware-counts-v2"
    assert report["full_official_contract"] is False
    config = report["configuration"]
    assert config["device"].startswith("cuda")
    assert "H100" in config["device_name"]
    assert config["cells_per_group"] == 32
    assert config["targets"] == 8
    assert config["state_effect_weight"] == 1.0
    assert config["common_response_weight"] == 0.0
    assert config["learned_response_weight"] == weight
    assert config["combined_effect_clip"] == 0.65
    assert config["target_policy"] == "force"
    assert config["count_emission"] == "exact-largest-remainder"
    assert config["library_policy"] == "exact-source-library-composition"
    assert report["validation"]["shape"] == [768, 18533]
    assert report["validation"]["groups"] == 24
    assert report["validation"]["failed_checks"] == []
    assert report["scientific_qc"]["target_policy_failures"] == []
    assert report["scientific_qc"]["all_libraries_exact"] is True
    assert report["scientific_qc"]["all_current_only_counts_exact"] is True
    assert report["scientific_qc"]["induced_zero_emitted_entries"] == 0
    assert report["scientific_qc"]["dropped_counts_without_redistribution"] == 0
    assert report["scientific_qc"]["count_mass"]["passed"] is True
    assert pathlib.Path(report["provenance"]["response_artifact"]["path"]).resolve() == response_path

direct = ad.read_h5ad(direct_path)
zero = ad.read_h5ad(zero_path)
v51 = ad.read_h5ad(v51_path)
assert direct.shape == zero.shape == v51.shape == (768, 18533)
assert direct.obs_names.equals(zero.obs_names) and zero.obs_names.equals(v51.obs_names)
assert direct.var_names.equals(zero.var_names) and zero.var_names.equals(v51.var_names)
for column in ("context", "target_gene", "source_control_row"):
    assert direct.obs[column].equals(zero.obs[column])
    assert zero.obs[column].equals(v51.obs[column])

direct_matrix = direct.X.tocsr() if sp.issparse(direct.X) else sp.csr_matrix(direct.X)
zero_matrix = zero.X.tocsr() if sp.issparse(zero.X) else sp.csr_matrix(zero.X)
v51_matrix = v51.X.tocsr() if sp.issparse(v51.X) else sp.csr_matrix(v51.X)
identity_difference = direct_matrix != zero_matrix
assert identity_difference.nnz == 0, identity_difference.nnz

with np.load(response_path, allow_pickle=False) as response:
    direct_target = np.tile(np.asarray(response["direct_mask"][:8], dtype=bool), 3)
changed_per_row = np.asarray((v51_matrix != zero_matrix).sum(axis=1)).ravel()
group_changed = changed_per_row.reshape(24, 32).sum(axis=1)
assert np.all(group_changed[~direct_target] == 0)
assert np.any(group_changed[direct_target] > 0)

outside = sum(group["combined_outside_bound"] for group in v51_report["qc"]["groups"])
values = sum(group["effect_values"] for group in v51_report["qc"]["groups"])
assert values > 0 and outside / values < 1e-4
assert v51_report["scientific_qc"]["redistributed_shared_counts"] > 0

for path, report in ((zero_path, zero_report), (v51_path, v51_report)):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert report["provenance"]["output_h5ad"]["sha256"] == digest

print("V5.1 H100 smoke gates passed")
print(f"alpha=0 identity differences: {identity_difference.nnz}")
print(f"alpha=0.05 changed entries: {(v51_matrix != zero_matrix).nnz}")
print(f"combined clip fraction: {outside / values:.9g}")
PY

date --iso-8601=seconds
