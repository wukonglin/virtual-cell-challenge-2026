#!/usr/bin/env bash
#SBATCH --job-name=vcc-state-direct-smoke
#SBATCH --account=allaccess
#SBATCH --partition=full
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --constraint=gpu-h100
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/h100_state_direct_counts_smoke_v0_%j.out
#SBATCH --error=logs/h100_state_direct_counts_smoke_v0_%j.err

set -euo pipefail

readonly PROJECT_DIR="${VCC_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
readonly PYTHON_BIN="$PROJECT_DIR/.venv-state/bin/python"
readonly MODEL_DIR="$PROJECT_DIR/artifacts/state_runs/state_sm_20k_v0"
readonly CHECKPOINT_EXPECTED_SHA256="e407e51855792d0dc85483a977f63497d9ac9a6893db360d840c7d9ce556780d"
readonly PERTURBATION_MAP_EXPECTED_SHA256="6ab08c62dd117012b386051a9f375505abcd17b88ed76f0ad232199624291170"
readonly OUTPUT_H5AD="$PROJECT_DIR/artifacts/state_direct_counts_smoke_v0.h5ad"
readonly OUTPUT_JSON="$PROJECT_DIR/artifacts/state_direct_counts_smoke_v0.json"

cd "$PROJECT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export CPATH="$PROJECT_DIR/.venv-state-headers/usr/include/python3.11"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

task_scratch="$(mktemp -d /tmp/vcc_state_direct_smoke_XXXXXX)"
readonly task_scratch
trap 'rm -rf -- "$task_scratch"' EXIT

for required_path in \
  "$PYTHON_BIN" \
  "$MODEL_DIR/checkpoints/step16000.ckpt" \
  "$MODEL_DIR/checkpoints/selected_checkpoint.json" \
  "$MODEL_DIR/pert_onehot_map.pt" \
  "$MODEL_DIR/var_dims.pkl" \
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
    echo "Refusing to overwrite smoke artifact: $output_path" >&2
    exit 3
  fi
done

date --iso-8601=seconds
hostname
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader

"$PYTHON_BIN" scripts/generate_state_direct_counts.py \
  --model-dir "$MODEL_DIR" \
  --checkpoint step16000.ckpt \
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
  --model-chunk-size 16 \
  --cells-per-group 16 \
  --target-limit 1 \
  --effect-scale 1.0 \
  --effect-clip 0.60 \
  --effect-bounding tanh \
  --target-strategy force \
  --target-remaining-fraction 0.20 \
  --max-genes-per-cell 5900 \
  --integerization largest-remainder \
  --groups-per-shard 1 \
  --compression lzf \
  --seed 20260831

"$PYTHON_BIN" - "$OUTPUT_JSON" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
assert report["schema"] == "vcc-state-direct-counts-v1"
assert report["full_official_contract"] is False
assert report["configuration"]["effect_bounding"] == "tanh"
assert report["configuration"]["integerization"] == "largest-remainder"
assert report["configuration"]["target_strategy"] == "force"
assert report["validation"]["failed_checks"] == []
assert report["validation"]["shape"] == [48, 18533]
assert report["validation"]["groups"] == 3
assert report["scientific_qc"]["all_libraries_exact"] is True
assert report["scientific_qc"]["all_current_only_counts_exact"] is True
assert report["scientific_qc"]["target_knockdown_failures"] == []
for group in report["qc"]["groups"]:
    assert group["chunk_sizes"] == [16]
    assert group["library_size_mismatches"] == 0
    assert group["current_only_count_mismatches"] == 0
    assert "target_sum_before" in group and "target_sum_after" in group
print("Direct paired-residual STATE target-limit smoke gates passed")
PY

date --iso-8601=seconds
