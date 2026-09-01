#!/usr/bin/env bash
#SBATCH --job-name=vcc-public-v51-data
#SBATCH --account=allaccess
#SBATCH --partition=full
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --array=0-1%1
#SBATCH --output=logs/cpu_public_v51_data_%A_%a.out
#SBATCH --error=logs/cpu_public_v51_data_%A_%a.err

set -euo pipefail
unset PYTHONOPTIMIZE

readonly PROJECT_DIR="${VCC_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
readonly PYTHON_BIN="$PROJECT_DIR/.venv-state/bin/python"
readonly OUTPUT_DIR="$PROJECT_DIR/dataset/public_v51"

case "${SLURM_ARRAY_TASK_ID:-}" in
  0)
    readonly CONTEXT="HepG2"
    readonly INPUT_H5AD="$PROJECT_DIR/dataset/raw/replogle_nadig/GSE264667_hepg2_raw_singlecell_01.h5ad"
    readonly PREFIX="hepg2"
    readonly EXPECTED_CONTROLS=4976
    readonly EXPECTED_TRUTH=40049
    readonly EXPECTED_GENES=9623
    readonly EXPECTED_STATE_OVERLAP=8909
    ;;
  1)
    readonly CONTEXT="Jurkat"
    readonly INPUT_H5AD="$PROJECT_DIR/dataset/raw/replogle_nadig/GSE264667_jurkat_raw_singlecell_01.h5ad"
    readonly PREFIX="jurkat"
    readonly EXPECTED_CONTROLS=12013
    readonly EXPECTED_TRUTH=73506
    readonly EXPECTED_GENES=8881
    readonly EXPECTED_STATE_OVERLAP=8181
    ;;
  *)
    echo "SLURM_ARRAY_TASK_ID must be 0 or 1" >&2
    exit 2
    ;;
esac

readonly OUTPUT_CONTROLS="$OUTPUT_DIR/${PREFIX}_controls_only.h5ad"
readonly OUTPUT_TRUTH="$OUTPUT_DIR/${PREFIX}_sealed_truth.h5ad"
readonly OUTPUT_JSON="$OUTPUT_DIR/${PREFIX}_data.json"

cd "$PROJECT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

for required_path in \
  "$PYTHON_BIN" \
  "$INPUT_H5AD" \
  scripts/prepare_public_validation_data.py \
  scripts/build_bulk_effect_atlas.py \
  dataset/public_v51/split_manifest.csv \
  dataset/public_v51/split_manifest.json \
  dataset/public_v51/common_genes.csv \
  dataset/public_v51/common_genes.json \
  dataset/state_support/extracted/gene_names.csv; do
  if [[ ! -r "$required_path" ]]; then
    echo "Missing required input: $required_path" >&2
    exit 3
  fi
done

for output_path in "$OUTPUT_CONTROLS" "$OUTPUT_TRUTH" "$OUTPUT_JSON"; do
  if [[ -e "$output_path" ]]; then
    echo "Refusing to overwrite validation data: $output_path" >&2
    exit 4
  fi
done

date --iso-8601=seconds
hostname
"$PYTHON_BIN" scripts/prepare_public_validation_data.py \
  "$INPUT_H5AD" \
  "$CONTEXT" \
  "$OUTPUT_CONTROLS" \
  "$OUTPUT_TRUTH" \
  "$OUTPUT_JSON" \
  --panel-csv dataset/public_v51/split_manifest.csv \
  --panel-json dataset/public_v51/split_manifest.json \
  --common-genes dataset/public_v51/common_genes.csv \
  --common-genes-json dataset/public_v51/common_genes.json \
  --state-genes dataset/state_support/extracted/gene_names.csv \
  --chunk-rows 2048 \
  --compression lzf

"$PYTHON_BIN" - \
  "$OUTPUT_CONTROLS" "$OUTPUT_TRUTH" "$OUTPUT_JSON" \
  "$EXPECTED_CONTROLS" "$EXPECTED_TRUTH" "$EXPECTED_GENES" \
  "$EXPECTED_STATE_OVERLAP" <<'PY'
import json
import pathlib
import sys

import anndata as ad
import numpy as np
import scipy.sparse as sp

controls_path = pathlib.Path(sys.argv[1])
truth_path = pathlib.Path(sys.argv[2])
report = json.loads(pathlib.Path(sys.argv[3]).read_text(encoding="utf-8"))
expected_controls, expected_truth, expected_genes, expected_state = map(int, sys.argv[4:8])
assert report["schema"] == "vcc-public-validation-data-v1"
assert report["axes"]["controls"] == expected_controls
assert report["axes"]["truth_cells_including_controls"] == expected_truth
assert report["axes"]["native_genes_after_duplicate_sum"] == expected_genes
assert report["axes"]["state_support_overlap"] == expected_state
assert report["axes"]["frozen_common_genes"] == 7107
assert report["axes"]["panel_targets"] == 300
assert report["duplicate_symbols_collapsed_by_sum"] == {"HSPA14": 2}

controls = ad.read_h5ad(controls_path, backed="r")
truth = ad.read_h5ad(truth_path, backed="r")
try:
    assert controls.shape == (expected_controls, expected_genes)
    assert truth.shape == (expected_truth, expected_genes)
    assert set(controls.obs["target_gene"].astype(str)) == {"non-targeting"}
    assert set(controls.obs["validation_route"].astype(str)) == {"control"}
    assert set(truth.obs["validation_route"].astype(str)) == {
        "control",
        "direct",
        "held_target",
    }
    assert truth.obs["target_gene"].nunique() == 301
    assert controls.var_names.equals(truth.var_names)
    assert int(controls.var["is_common_public_gene"].sum()) == 7107
    assert int(truth.var["is_state_support"].sum()) == expected_state
    assert controls.uns["public_validation"]["role"] == "controls-only-generator-input"
    assert truth.uns["public_validation"]["role"] == "sealed-scorer-input"
    assert controls.uns["public_validation"]["sealed_treated_profiles_present"] is False
    assert truth.uns["public_validation"]["sealed_treated_profiles_present"] is True
    control_positions = controls.obs["source_cell_index"].to_numpy(dtype=np.int64)
    truth_control_rows = np.flatnonzero(
        truth.obs["target_gene"].astype(str).to_numpy() == "non-targeting"
    )
    assert np.array_equal(
        control_positions,
        truth.obs["source_cell_index"].to_numpy(dtype=np.int64)[truth_control_rows],
    )
    for begin in range(0, expected_controls, 512):
        left = controls.X[begin : begin + 512]
        right = truth.X[truth_control_rows[begin : begin + 512]]
        if not sp.issparse(left):
            left = sp.csr_matrix(left)
        if not sp.issparse(right):
            right = sp.csr_matrix(right)
        assert (left != right).nnz == 0
finally:
    controls.file.close()
    truth.file.close()
print("Public validation data-firewall gates passed")
PY

date --iso-8601=seconds
