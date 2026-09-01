#!/usr/bin/env bash
#SBATCH --job-name=vcc-public-v51-full
#SBATCH --account=allaccess
#SBATCH --partition=full
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=02:00:00
#SBATCH --output=logs/h100_public_v51_full_%j.out
#SBATCH --error=logs/h100_public_v51_full_%j.err

set -euo pipefail
unset PYTHONOPTIMIZE

readonly PROJECT_DIR="${VCC_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
readonly PYTHON_BIN="$PROJECT_DIR/.venv-state/bin/python"
readonly MODEL_DIR="$PROJECT_DIR/artifacts/state_runs/state_sm_20k_v0"
readonly OUTPUT_DIR="$PROJECT_DIR/artifacts/public_v51/full"
readonly SMOKE_SUMMARY="$PROJECT_DIR/artifacts/public_v51/smoke/public_state_anchor_v51_smoke_summary.json"
readonly SUMMARY_JSON="$OUTPUT_DIR/public_state_anchor_v51_full_summary.json"

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_DIR" logs
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1
export CPATH="$PROJECT_DIR/.venv-state-headers/usr/include/python3.11"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

task_scratch="$(mktemp -d /tmp/vcc_public_v51_full_XXXXXX)"
readonly task_scratch
trap 'rm -rf -- "$task_scratch"' EXIT

for required_path in \
  "$PYTHON_BIN" \
  "$MODEL_DIR/checkpoints/selected.ckpt" \
  "$MODEL_DIR/checkpoints/selected_checkpoint.json" \
  "$MODEL_DIR/pert_onehot_map.pt" \
  "$MODEL_DIR/var_dims.pkl" \
  "$SMOKE_SUMMARY" \
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
  for candidate in anchor v51_alpha010; do
    for suffix in h5ad json; do
      output_path="$OUTPUT_DIR/${context_prefix}_${candidate}.${suffix}"
      if [[ -e "$output_path" ]]; then
        echo "Refusing to overwrite full validation artifact: $output_path" >&2
        exit 3
      fi
    done
  done
done
if [[ -e "$SUMMARY_JSON" ]]; then
  echo "Refusing to overwrite full summary: $SUMMARY_JSON" >&2
  exit 3
fi

"$PYTHON_BIN" - "$SMOKE_SUMMARY" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
assert summary["schema"] == "vcc-public-state-anchor-smoke-v1"
assert set(summary["contexts"]) == {"HepG2", "Jurkat"}
for context in summary["contexts"].values():
    assert context["clip_fraction"] < 1e-4
    assert context["fallback_changed_entries"] == 0
    assert context["direct_groups_with_changes"] > 0
PY

scratch_available_bytes="$(df -PB1 /tmp | awk 'NR==2 {print $4}')"
readonly scratch_available_bytes
if [[ "$scratch_available_bytes" -lt 64424509440 ]]; then
  echo "Need at least 60 GiB free in /tmp for full public generation" >&2
  exit 4
fi

date --iso-8601=seconds
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unknown}"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader
df -h "$PROJECT_DIR" /tmp

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
    --model-chunk-size 128 \
    --cells-per-group 400 \
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
    --groups-per-shard 20 \
    --concat-max-loaded-elements 20000000 \
    --compression lzf \
    --seed 20260901
}

run_candidate HepG2 hepg2 0.0 anchor artifacts/public_v51/hepg2_state_residual_top100_v1.npz
run_candidate HepG2 hepg2 0.10 v51_alpha010 artifacts/public_v51/hepg2_state_residual_top100_v1.npz
run_candidate Jurkat jurkat 0.0 anchor artifacts/public_v51/jurkat_state_residual_top100_v1.npz
run_candidate Jurkat jurkat 0.10 v51_alpha010 artifacts/public_v51/jurkat_state_residual_top100_v1.npz

"$PYTHON_BIN" - "$OUTPUT_DIR" "$SUMMARY_JSON" <<'PY'
import hashlib
import json
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
manifest = pd.read_csv("dataset/public_v51/split_manifest.csv")
direct = manifest["validation_route"].astype(str).to_numpy() == "direct"
summary = {"schema": "vcc-public-state-anchor-full-v1", "contexts": {}}

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

for context, prefix, genes in (("HepG2", "hepg2", 9623), ("Jurkat", "jurkat", 8881)):
    anchor_path = output_dir / f"{prefix}_anchor.h5ad"
    candidate_path = output_dir / f"{prefix}_v51_alpha010.h5ad"
    anchor_report = json.loads((output_dir / f"{prefix}_anchor.json").read_text())
    candidate_report = json.loads((output_dir / f"{prefix}_v51_alpha010.json").read_text())
    for report, alpha in ((anchor_report, 0.0), (candidate_report, 0.10)):
        assert report["full_frozen_panel_contract"] is True
        assert report["validation"]["shape"] == [120000, genes]
        assert report["validation"]["groups"] == 300
        assert report["validation"]["failed_checks"] == []
        assert report["configuration"]["residual_alpha"] == alpha
        assert report["scientific_qc"]["all_libraries_exact"] is True
        assert report["scientific_qc"]["all_native_non_support_counts_exact"] is True
        assert report["scientific_qc"]["target_knockdown_failures"] == []
        assert report["data_firewall"]["sealed_treated_profiles_read"] is False
    clip_fraction = sum(g["combined_outside_bound"] for g in candidate_report["qc"]["groups"]) / sum(
        g["effect_values"] for g in candidate_report["qc"]["groups"]
    )
    assert clip_fraction < 1e-4, clip_fraction

    anchor = ad.read_h5ad(anchor_path, backed="r")
    candidate = ad.read_h5ad(candidate_path, backed="r")
    try:
        assert anchor.shape == candidate.shape == (120000, genes)
        assert anchor.obs_names.equals(candidate.obs_names)
        assert anchor.var_names.equals(candidate.var_names)
        for column in ("context", "target_gene", "source_control_row", "source_control_cell_id"):
            assert anchor.obs[column].equals(candidate.obs[column])
        fallback_targets = manifest.loc[~direct, "target_gene"].astype(str).tolist()
        fallback_rows = anchor.obs["target_gene"].astype(str).isin(fallback_targets).to_numpy()
        fallback_changed = 0
        direct_changed = 0
        for start in range(0, anchor.n_obs, 2048):
            stop = min(start + 2048, anchor.n_obs)
            difference = (anchor.X[start:stop] != candidate.X[start:stop]).tocsr()
            row_changes = np.asarray(difference.getnnz(axis=1)).ravel()
            local_fallback = fallback_rows[start:stop]
            fallback_changed += int(row_changes[local_fallback].sum())
            direct_changed += int(row_changes[~local_fallback].sum())
        assert fallback_changed == 0
        assert direct_changed > 0
        summary["contexts"][context] = {
            "clip_fraction": clip_fraction,
            "direct_changed_entries": direct_changed,
            "fallback_changed_entries": fallback_changed,
            "anchor_sha256": sha256_file(anchor_path),
            "candidate_sha256": sha256_file(candidate_path),
        }
    finally:
        anchor.file.close()
        candidate.file.close()

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
