#!/usr/bin/env bash
#SBATCH --job-name=vcc-public-v51-smoke
#SBATCH --account=allaccess
#SBATCH --partition=full
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=logs/h100_public_v51_smoke_%j.out
#SBATCH --error=logs/h100_public_v51_smoke_%j.err

set -euo pipefail
unset PYTHONOPTIMIZE

readonly PROJECT_DIR="${VCC_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
readonly PYTHON_BIN="$PROJECT_DIR/.venv-state/bin/python"
readonly MODEL_DIR="$PROJECT_DIR/artifacts/state_runs/state_sm_20k_v0"
readonly OUTPUT_DIR="$PROJECT_DIR/artifacts/public_v51/smoke"
readonly SUMMARY_JSON="$OUTPUT_DIR/public_state_anchor_v51_smoke_summary.json"

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_DIR" logs
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export CPATH="$PROJECT_DIR/.venv-state-headers/usr/include/python3.11"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

task_scratch="$(mktemp -d /tmp/vcc_public_v51_smoke_XXXXXX)"
readonly task_scratch
trap 'rm -rf -- "$task_scratch"' EXIT

for required_path in \
  "$PYTHON_BIN" \
  "$MODEL_DIR/checkpoints/selected.ckpt" \
  "$MODEL_DIR/checkpoints/selected_checkpoint.json" \
  "$MODEL_DIR/pert_onehot_map.pt" \
  "$MODEL_DIR/var_dims.pkl" \
  scripts/generate_native_state_anchor_counts.py \
  dataset/state_support/extracted/gene_names.csv \
  dataset/public_v51/split_manifest.csv \
  dataset/public_v51/split_manifest.json \
  dataset/public_v51/hepg2_controls_only.h5ad \
  dataset/public_v51/jurkat_controls_only.h5ad \
  artifacts/public_v51/hepg2_state_residual_top100_v1.npz \
  artifacts/public_v51/jurkat_state_residual_top100_v1.npz; do
  if [[ ! -r "$required_path" ]]; then
    echo "Missing required input: $required_path" >&2
    exit 2
  fi
done

for context_prefix in hepg2 jurkat; do
  for candidate in anchor alpha010; do
    for suffix in h5ad json; do
      output_path="$OUTPUT_DIR/${context_prefix}_${candidate}.${suffix}"
      if [[ -e "$output_path" ]]; then
        echo "Refusing to overwrite smoke artifact: $output_path" >&2
        exit 3
      fi
    done
  done
done
if [[ -e "$SUMMARY_JSON" ]]; then
  echo "Refusing to overwrite smoke summary: $SUMMARY_JSON" >&2
  exit 3
fi

date --iso-8601=seconds
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unknown}"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader

run_candidate() {
  local context="$1"
  local prefix="$2"
  local alpha="$3"
  local label="$4"
  local residual="$5"
  local output_h5ad="$OUTPUT_DIR/${prefix}_${label}.h5ad"
  local output_json="$OUTPUT_DIR/${prefix}_${label}.json"
  local residual_args=()
  if [[ "$alpha" != "0.0" ]]; then
    residual_args=(--residual-npz "$residual")
  fi
  "$PYTHON_BIN" scripts/generate_native_state_anchor_counts.py \
    --controls-h5ad "dataset/public_v51/${prefix}_controls_only.h5ad" \
    --panel-manifest dataset/public_v51/split_manifest.json \
    --panel-csv dataset/public_v51/split_manifest.csv \
    --context "$context" \
    --model-dir "$MODEL_DIR" \
    --checkpoint selected.ckpt \
    --selection-json "$MODEL_DIR/checkpoints/selected_checkpoint.json" \
    --support-genes dataset/state_support/extracted/gene_names.csv \
    --output-h5ad "$output_h5ad" \
    --output-json "$output_json" \
    --scratch-dir "$task_scratch/${prefix}_${label}" \
    --device cuda \
    --require-cuda \
    --require-h100 \
    --model-chunk-size 32 \
    --cells-per-group 32 \
    --target-limit 8 \
    --state-effect-weight 1.0 \
    --state-effect-clip 0.60 \
    --state-effect-bounding tanh \
    "${residual_args[@]}" \
    --residual-alpha "$alpha" \
    --combined-effect-clip 0.65 \
    --target-policy force \
    --target-remaining-fraction 0.20 \
    --max-genes-per-cell 5900 \
    --integerization largest-remainder \
    --groups-per-shard 4 \
    --concat-max-loaded-elements 20000000 \
    --compression lzf \
    --seed 20260901
}

run_candidate HepG2 hepg2 0.0 anchor artifacts/public_v51/hepg2_state_residual_top100_v1.npz
run_candidate HepG2 hepg2 0.10 alpha010 artifacts/public_v51/hepg2_state_residual_top100_v1.npz
run_candidate Jurkat jurkat 0.0 anchor artifacts/public_v51/jurkat_state_residual_top100_v1.npz
run_candidate Jurkat jurkat 0.10 alpha010 artifacts/public_v51/jurkat_state_residual_top100_v1.npz

"$PYTHON_BIN" - "$OUTPUT_DIR" "$SUMMARY_JSON" <<'PY'
import hashlib
import json
import math
import os
import pathlib
import sys
import tempfile

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

output_dir = pathlib.Path(sys.argv[1])
summary_path = pathlib.Path(sys.argv[2])
manifest = pd.read_csv("dataset/public_v51/split_manifest.csv").iloc[:8]
direct = manifest["validation_route"].astype(str).to_numpy() == "direct"
summary = {"schema": "vcc-public-state-anchor-smoke-v1", "contexts": {}}
for context, prefix, genes in (("HepG2", "hepg2", 9623), ("Jurkat", "jurkat", 8881)):
    anchor_path = output_dir / f"{prefix}_anchor.h5ad"
    alpha_path = output_dir / f"{prefix}_alpha010.h5ad"
    anchor_report = json.loads((output_dir / f"{prefix}_anchor.json").read_text())
    alpha_report = json.loads((output_dir / f"{prefix}_alpha010.json").read_text())
    for report, alpha in ((anchor_report, 0.0), (alpha_report, 0.10)):
        assert report["schema"] == "vcc-native-state-anchor-counts-v1"
        assert report["configuration"]["device"].startswith("cuda")
        assert "H100" in report["configuration"]["device_name"].upper()
        assert report["configuration"]["residual_alpha"] == alpha
        assert report["validation"]["shape"] == [256, genes]
        assert report["validation"]["groups"] == 8
        assert report["validation"]["failed_checks"] == []
        assert report["scientific_qc"]["all_libraries_exact"] is True
        assert report["scientific_qc"]["all_native_non_support_counts_exact"] is True
        assert report["scientific_qc"]["target_knockdown_failures"] == []
        assert report["data_firewall"]["sealed_treated_profiles_read"] is False
        assert report["residual"]["artifact_read"] is (alpha > 0)
        for group in report["qc"]["groups"]:
            assert group["library_size_mismatches"] == 0
            assert group["native_non_support_count_mismatches"] == 0
            assert abs(group["final_target_delta"]["min"] - math.log(0.2)) < 1e-6
            assert abs(group["final_target_delta"]["max"] - math.log(0.2)) < 1e-6
    clip_fraction = sum(g["combined_outside_bound"] for g in alpha_report["qc"]["groups"]) / sum(
        g["effect_values"] for g in alpha_report["qc"]["groups"]
    )
    assert clip_fraction < 1e-4, clip_fraction

    anchor = ad.read_h5ad(anchor_path)
    candidate = ad.read_h5ad(alpha_path)
    assert anchor.shape == candidate.shape == (256, genes)
    assert anchor.obs_names.equals(candidate.obs_names)
    assert anchor.var_names.equals(candidate.var_names)
    for column in ("context", "target_gene", "source_control_row", "source_control_cell_id"):
        assert anchor.obs[column].equals(candidate.obs[column])
    anchor_x = anchor.X.tocsr() if sp.issparse(anchor.X) else sp.csr_matrix(anchor.X)
    candidate_x = candidate.X.tocsr() if sp.issparse(candidate.X) else sp.csr_matrix(candidate.X)
    changed_by_cell = np.asarray((anchor_x != candidate_x).sum(axis=1)).ravel()
    changed_by_group = changed_by_cell.reshape(8, 32).sum(axis=1)
    assert np.all(changed_by_group[~direct] == 0), changed_by_group.tolist()
    assert np.any(changed_by_group[direct] > 0), changed_by_group.tolist()
    summary["contexts"][context] = {
        "clip_fraction": clip_fraction,
        "changed_entries": int((anchor_x != candidate_x).nnz),
        "changed_cells": int(np.count_nonzero(changed_by_cell)),
        "direct_groups_with_changes": int(np.count_nonzero(changed_by_group[direct])),
        "direct_groups": int(np.count_nonzero(direct)),
        "fallback_changed_entries": int(changed_by_group[~direct].sum()),
        "anchor_sha256": hashlib.sha256(anchor_path.read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(alpha_path.read_bytes()).hexdigest(),
    }

summary_path.parent.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile(
    mode="w", encoding="utf-8", prefix=f".{summary_path.name}.",
    suffix=".tmp", dir=summary_path.parent, delete=False
) as handle:
    temporary = pathlib.Path(handle.name)
    json.dump(summary, handle, indent=2, sort_keys=True)
    handle.write("\n")
try:
    os.replace(temporary, summary_path)
finally:
    if temporary.exists():
        temporary.unlink()
print(json.dumps(summary, indent=2, sort_keys=True))
PY

date --iso-8601=seconds
