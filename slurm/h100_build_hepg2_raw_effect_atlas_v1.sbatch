#!/usr/bin/env bash
#SBATCH --job-name=vcc-hepg2-atlas
#SBATCH --account=allaccess
#SBATCH --partition=full
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --constraint=gpu-h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --output=logs/h100_hepg2_atlas_v1_%j.out
#SBATCH --error=logs/h100_hepg2_atlas_v1_%j.err

set -euo pipefail

readonly PROJECT_DIR="${VCC_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
readonly PYTHON_BIN="$PROJECT_DIR/.venv-state/bin/python"
readonly INPUT_H5AD="$PROJECT_DIR/dataset/raw/replogle_nadig/GSE264667_hepg2_raw_singlecell_01.h5ad"
readonly OUTPUT_NPZ="$PROJECT_DIR/artifacts/effect_atlas/hepg2_raw_min30_v1.npz"
readonly OUTPUT_JSON="$PROJECT_DIR/artifacts/effect_atlas/hepg2_raw_min30_v1.json"

cd "$PROJECT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

for required_path in \
  "$PYTHON_BIN" \
  "$INPUT_H5AD" \
  scripts/build_raw_effect_atlas.py \
  scripts/build_bulk_effect_atlas.py; do
  if [[ ! -r "$required_path" ]]; then
    echo "Missing required input: $required_path" >&2
    exit 2
  fi
done
for output_path in "$OUTPUT_NPZ" "$OUTPUT_JSON"; do
  if [[ -e "$output_path" ]]; then
    echo "Refusing to overwrite output: $output_path" >&2
    exit 3
  fi
done

mkdir -p "$(dirname "$OUTPUT_NPZ")"
date --iso-8601=seconds
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader

"$PYTHON_BIN" scripts/build_raw_effect_atlas.py \
  "$INPUT_H5AD" \
  "$OUTPUT_NPZ" \
  "$OUTPUT_JSON" \
  --pert-col gene \
  --batch-col gem_group \
  --gene-col gene_name \
  --control non-targeting \
  --min-cells 30 \
  --target-sum 50000 \
  --chunk-rows 2048 \
  --device cuda \
  --require-cuda

test -s "$OUTPUT_NPZ"
test -s "$OUTPUT_JSON"

"$PYTHON_BIN" - "$OUTPUT_NPZ" "$OUTPUT_JSON" <<'PY'
import hashlib
import json
import pathlib
import sys

import numpy as np


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


npz_path = pathlib.Path(sys.argv[1])
report = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
require(report["schema"] == "vcc-raw-effect-atlas-v1", "Unexpected schema")
require(report["contract"]["targets"] == 1818, "Unexpected target count")
require(report["contract"]["genes"] == 9623, "Unexpected gene count")
require(
    report["contract"]["input_gene_columns"] == 9624,
    "Unexpected input gene-column count",
)
require(report["contract"]["batches"] == 56, "Unexpected batch count")
require(
    report["contract"]["collapsed_duplicate_columns"] == 1,
    "Unexpected duplicate-column count",
)
require(
    report["contract"]["duplicate_symbols_collapsed_by_sum"] == {"HSPA14": 2},
    "Unexpected duplicate-symbol contract",
)
sources = report["contract"]["duplicate_gene_sources"]["HSPA14"]
require(
    {item["source_var_name"] for item in sources}
    == {"ENSG00000187522", "ENSG00000284024"},
    "Unexpected HSPA14 source identifiers",
)
require(report["runtime"]["device"].startswith("cuda"), "CUDA was not used")
require("H100" in report["runtime"]["cuda_device"], "H100 was not used")
require(
    report["output"]["sha256"]
    == hashlib.sha256(npz_path.read_bytes()).hexdigest(),
    "Output hash mismatch",
)
with np.load(npz_path, allow_pickle=False) as archive:
    require(archive["effects"].shape == (1818, 9623), "Unexpected effect shape")
    require(
        archive["matched_control_profiles"].shape == (1818, 9623),
        "Unexpected matched-control shape",
    )
    require(len(np.unique(archive["gene_names"])) == 9623, "Duplicate genes")
    require(len(np.unique(archive["target_names"])) == 1818, "Duplicate targets")
    require(int(archive["target_cell_counts"].min()) >= 30, "Cell-count gate failed")
    require(np.all(archive["control_batch_counts"] > 0), "Empty control batch")
    require(
        archive["gene_duplicate_positions"].tolist() == [4626],
        "Unexpected duplicate-gene position",
    )
print("HepG2 raw effect-atlas gates passed")
PY
date --iso-8601=seconds
