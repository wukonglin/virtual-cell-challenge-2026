#!/usr/bin/env bash
#SBATCH --job-name=vcc-k562-signature-v2
#SBATCH --account=allaccess
#SBATCH --partition=full
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --constraint=gpu-h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/h100_k562_signature_v2_%j.out
#SBATCH --error=logs/h100_k562_signature_v2_%j.err

set -euo pipefail

readonly PROJECT_DIR="${VCC_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
readonly PYTHON_BIN="$PROJECT_DIR/.venv-state/bin/python"
readonly EFFECT_ATLAS="$PROJECT_DIR/artifacts/v2/k562_gwps_effect_atlas.npz"
readonly ESM2="$PROJECT_DIR/dataset/state_support/extracted/ESM2_pert_features.pt"
readonly TARGETS="$PROJECT_DIR/dataset/controls/pert_counts.csv"
readonly GENES="$PROJECT_DIR/dataset/controls/gene_names.csv"
readonly OUTPUT_NPZ="$PROJECT_DIR/artifacts/v2/k562_esm2_signature_v2.npz"
readonly OUTPUT_CKPT="$PROJECT_DIR/artifacts/v2/k562_esm2_signature_v2.pt"
readonly OUTPUT_JSON="$PROJECT_DIR/artifacts/v2/k562_esm2_signature_v2.json"

cd "$PROJECT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for required_path in \
  "$PYTHON_BIN" \
  "$EFFECT_ATLAS" \
  "$ESM2" \
  "$TARGETS" \
  "$GENES" \
  scripts/train_target_signature_model.py; do
  if [[ ! -r "$required_path" ]]; then
    echo "Missing required input: $required_path" >&2
    exit 2
  fi
done

for output_path in "$OUTPUT_NPZ" "$OUTPUT_CKPT" "$OUTPUT_JSON"; do
  if [[ -e "$output_path" ]]; then
    echo "Refusing to overwrite production artifact: $output_path" >&2
    exit 3
  fi
done

mkdir -p "$(dirname "$OUTPUT_NPZ")"
date --iso-8601=seconds
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unknown}"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader

"$PYTHON_BIN" scripts/train_target_signature_model.py \
  "$EFFECT_ATLAS" \
  "$ESM2" \
  "$TARGETS" \
  "$OUTPUT_NPZ" \
  "$OUTPUT_CKPT" \
  "$OUTPUT_JSON" \
  --target-column target_gene \
  --challenge-genes "$GENES" \
  --input-rank 256 \
  --output-rank 128 \
  --hidden-dim 512 \
  --clusters 20 \
  --validation-fraction 0.20 \
  --epochs 120 \
  --batch-size 128 \
  --learning-rate 0.001 \
  --weight-decay 0.0001 \
  --device cuda \
  --require-cuda \
  --seed 20260831

test -s "$OUTPUT_NPZ"
test -s "$OUTPUT_CKPT"
test -s "$OUTPUT_JSON"

"$PYTHON_BIN" - "$OUTPUT_JSON" "$OUTPUT_NPZ" "$OUTPUT_CKPT" <<'PY'
import json
import pathlib
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
for artifact in sys.argv[2:]:
    path = pathlib.Path(artifact)
    assert path.is_file() and path.stat().st_size > 0, path
assert report["schema"] == "vcc-target-signature-model-v1"
assert report["runtime"]["device"].startswith("cuda")
assert "H100" in report["runtime"]["cuda_device"]
assert report["configuration"]["input_rank"] == 256
assert report["configuration"]["output_rank"] == 128
assert report["configuration"]["hidden_dim"] == 512
assert report["configuration"]["clusters"] == 20
assert report["configuration"]["validation_fraction"] == 0.20
data = report["data"]
assert data["training_targets_before_esm2_filter"] >= data["training_targets_after_esm2_filter"] > 1
assert data["excluded_training_targets_missing_esm2"] == (
    data["training_targets_before_esm2_filter"]
    - data["training_targets_after_esm2_filter"]
)
assert len(data["excluded_training_target_names"]) == data[
    "excluded_training_targets_missing_esm2"
]
assert data["prediction_targets_requested"] == 300
assert data["prediction_targets_missing_esm2"] == 0
print("K562 ESM2 signature artifacts passed production gates")
PY

date --iso-8601=seconds
