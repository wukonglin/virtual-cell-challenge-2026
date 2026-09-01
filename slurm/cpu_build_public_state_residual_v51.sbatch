#!/usr/bin/env bash
#SBATCH --job-name=vcc-public-v51-residual
#SBATCH --account=allaccess
#SBATCH --partition=full
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --array=0-1%1
#SBATCH --output=logs/cpu_public_v51_residual_%A_%a.out
#SBATCH --error=logs/cpu_public_v51_residual_%A_%a.err

set -euo pipefail
unset PYTHONOPTIMIZE

readonly PROJECT_DIR="${VCC_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
readonly PYTHON_BIN="$PROJECT_DIR/.venv-state/bin/python"

case "${SLURM_ARRAY_TASK_ID:-}" in
  0)
    readonly CONTEXT="HepG2"
    readonly PREFIX="hepg2"
    readonly EXPECTED_GENES=9623
    ;;
  1)
    readonly CONTEXT="Jurkat"
    readonly PREFIX="jurkat"
    readonly EXPECTED_GENES=8881
    ;;
  *)
    echo "SLURM_ARRAY_TASK_ID must be 0 or 1" >&2
    exit 2
    ;;
esac

readonly DATA_DIR="$PROJECT_DIR/dataset/public_v51"
readonly OUTPUT_DIR="$PROJECT_DIR/artifacts/public_v51"
readonly CONTROLS="$DATA_DIR/${PREFIX}_controls_only.h5ad"
readonly OUTPUT_NPZ="$OUTPUT_DIR/${PREFIX}_state_residual_top100_v1.npz"
readonly OUTPUT_JSON="$OUTPUT_DIR/${PREFIX}_state_residual_top100_v1.json"

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_DIR" logs
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

for required_path in \
  "$PYTHON_BIN" \
  scripts/build_public_state_residual.py \
  artifacts/v2/k562_essential_effect_atlas.npz \
  artifacts/v2/rpe1_effect_atlas.npz \
  dataset/public_v51/split_manifest.csv \
  dataset/public_v51/split_manifest.json \
  dataset/public_v51/common_genes.csv \
  dataset/public_v51/common_genes.json \
  "$CONTROLS"; do
  if [[ ! -r "$required_path" ]]; then
    echo "Missing required input: $required_path" >&2
    exit 3
  fi
done

for output_path in "$OUTPUT_NPZ" "$OUTPUT_JSON"; do
  if [[ -e "$output_path" ]]; then
    echo "Refusing to overwrite residual: $output_path" >&2
    exit 4
  fi
done

date --iso-8601=seconds
hostname
"$PYTHON_BIN" scripts/build_public_state_residual.py \
  --k562-atlas artifacts/v2/k562_essential_effect_atlas.npz \
  --rpe1-atlas artifacts/v2/rpe1_effect_atlas.npz \
  --panel-csv dataset/public_v51/split_manifest.csv \
  --panel-json dataset/public_v51/split_manifest.json \
  --common-genes dataset/public_v51/common_genes.csv \
  --common-genes-json dataset/public_v51/common_genes.json \
  --controls-only "$CONTROLS" \
  --context "$CONTEXT" \
  --output-npz "$OUTPUT_NPZ" \
  --output-json "$OUTPUT_JSON" \
  --top-k 100 \
  --direct-reliability-power 0.0

"$PYTHON_BIN" - "$OUTPUT_NPZ" "$OUTPUT_JSON" "$CONTEXT" "$EXPECTED_GENES" <<'PY'
import json
import pathlib
import sys

import numpy as np

npz_path = pathlib.Path(sys.argv[1])
report = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
context = sys.argv[3]
expected_genes = int(sys.argv[4])
assert report["schema"] == "vcc-public-state-residual-v1"
assert report["context"] == context
assert report["axes"]["targets"] == 300
assert report["axes"]["direct_targets"] == 267
assert report["axes"]["held_targets"] == 33
assert report["axes"]["native_genes"] == expected_genes
assert report["scientific_qc"]["truth_profiles_read"] is False
with np.load(npz_path, allow_pickle=False) as artifact:
    effects = artifact["effects"]
    direct = artifact["direct_mask"].astype(bool)
    fallback = artifact["fallback_mask"].astype(bool)
    panel = artifact["excluded_panel_target_mask"].astype(bool)
    frozen = artifact["frozen_scoring_gene_mask"].astype(bool)
    state = artifact["state_support_mask"].astype(bool)
    assert effects.shape == (1, 300, expected_genes)
    assert np.isfinite(effects).all()
    assert np.all(np.count_nonzero(effects, axis=2) <= 100)
    assert not np.any(effects[:, fallback])
    assert not np.any(effects[:, :, panel])
    assert not np.any(effects[:, :, ~frozen])
    assert not np.any(effects[:, :, ~state])
    assert np.count_nonzero(effects[:, direct]) > 0
print("Public V5.1 residual gates passed")
PY
date --iso-8601=seconds
