#!/usr/bin/env bash
#SBATCH --job-name=vcc-state-direct
#SBATCH --account=allaccess
#SBATCH --partition=full
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --constraint=gpu-h100
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --output=logs/h100_state_direct_counts_v0_%j.out
#SBATCH --error=logs/h100_state_direct_counts_v0_%j.err

set -euo pipefail

readonly PROJECT_DIR="${VCC_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
readonly PYTHON_BIN="$PROJECT_DIR/.venv-state/bin/python"
readonly MODEL_DIR="$PROJECT_DIR/artifacts/state_runs/state_sm_20k_v0"
readonly OUTPUT_H5AD="$PROJECT_DIR/artifacts/state_direct_counts_20k_selected_v0.h5ad"
readonly OUTPUT_JSON="$PROJECT_DIR/artifacts/state_direct_counts_20k_selected_v0.json"

cd "$PROJECT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1
export CPATH="$PROJECT_DIR/.venv-state-headers/usr/include/python3.11"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

task_scratch="$(mktemp -d /tmp/vcc_state_direct_XXXXXX)"
readonly task_scratch
trap 'rm -rf -- "$task_scratch"' EXIT

for required_path in \
  "$PYTHON_BIN" \
  "$MODEL_DIR/checkpoints/selected.ckpt" \
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
    echo "Refusing to overwrite existing production artifact: $output_path" >&2
    exit 3
  fi
done

scratch_available_bytes="$(df -PB1 /tmp | awk 'NR==2 {print $4}')"
readonly scratch_available_bytes
if [[ "$scratch_available_bytes" -lt 85899345920 ]]; then
  echo "Need at least 80 GiB free in /tmp for bounded on-disk concatenation" >&2
  exit 4
fi

date --iso-8601=seconds
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unknown}"
df -h "$PROJECT_DIR" /tmp
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader

"$PYTHON_BIN" scripts/generate_state_direct_counts.py \
  --model-dir "$MODEL_DIR" \
  --checkpoint selected.ckpt \
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
  --effect-scale 1.0 \
  --effect-clip 0.60 \
  --effect-bounding tanh \
  --target-strategy force \
  --target-remaining-fraction 0.20 \
  --max-genes-per-cell 5900 \
  --integerization largest-remainder \
  --groups-per-shard 20 \
  --concat-max-loaded-elements 20000000 \
  --compression lzf \
  --seed 20260831

test -s "$OUTPUT_H5AD"
test -s "$OUTPUT_JSON"

"$PYTHON_BIN" - "$OUTPUT_JSON" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
assert report["schema"] == "vcc-state-direct-counts-v1"
assert report["full_official_contract"] is True
assert report["configuration"]["precision"] == "float32"
assert report["configuration"]["input_normalization"] == (
    "elementwise-log1p-raw-on-support-axis"
)
assert report["configuration"]["effect_reference"] == (
    "paired-model-control-on-identical-cell-chunks"
)
assert report["configuration"]["effect_bounding"] == "tanh"
assert report["configuration"]["target_strategy"] == "force"
assert report["configuration"]["target_remaining_fraction"] == 0.20
assert report["configuration"]["integerization"] == "largest-remainder"
assert report["axes"]["shared_genes"] == 18077
assert report["axes"]["current_only_genes_preserved"] == 456
assert report["validation"]["failed_checks"] == []
assert report["validation"]["shape"] == [360000, 18533]
assert report["validation"]["groups"] == 900
assert report["validation"]["nnz"] < 2140000000
assert report["scientific_qc"]["all_libraries_exact"] is True
assert report["scientific_qc"]["all_current_only_counts_exact"] is True
assert report["scientific_qc"]["target_knockdown_failures"] == []
print("Direct paired-residual STATE generation gates passed")
PY

date --iso-8601=seconds
