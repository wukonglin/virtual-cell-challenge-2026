#!/usr/bin/env bash
#SBATCH --job-name=vcc-state-direct-pack
#SBATCH --account=allaccess
#SBATCH --partition=full
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --output=logs/cpu_state_direct_pack_v0_%j.out
#SBATCH --error=logs/cpu_state_direct_pack_v0_%j.err

set -euo pipefail

readonly PROJECT_DIR="${VCC_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
readonly VCC_BIN="$PROJECT_DIR/.venv-vcc/bin/vcc"
readonly VCC_PY="$PROJECT_DIR/.venv-vcc/bin/python"
readonly PRED_H5AD="$PROJECT_DIR/artifacts/state_direct_counts_20k_selected_v0.h5ad"
readonly PRED_JSON="$PROJECT_DIR/artifacts/state_direct_counts_20k_selected_v0.json"
readonly DRYRUN_JSON="$PROJECT_DIR/artifacts/state_direct_counts_20k_selected_v0_dryrun.json"
readonly VCC_FILE="$PROJECT_DIR/artifacts/state_direct_counts_20k_selected_v0.vcc"
readonly PREP_JSON="$PROJECT_DIR/artifacts/state_direct_counts_20k_selected_v0_prep.json"
readonly SUMS_FILE="$PROJECT_DIR/artifacts/state_direct_counts_20k_selected_v0_SHA256SUMS"

cd "$PROJECT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1

task_tmp="$(mktemp -d /tmp/vcc_state_direct_pack_XXXXXX)"
readonly task_tmp
export TMPDIR="$task_tmp"
trap 'rm -rf -- "$task_tmp"' EXIT

for required_path in \
  "$VCC_BIN" "$VCC_PY" "$PRED_H5AD" "$PRED_JSON" \
  dataset/controls/gene_names.csv \
  dataset/controls/pert_counts.csv; do
  if [[ ! -r "$required_path" ]]; then
    echo "Missing required input: $required_path" >&2
    exit 2
  fi
done

for output_path in "$DRYRUN_JSON" "$VCC_FILE" "$PREP_JSON" "$SUMS_FILE"; do
  if [[ -e "$output_path" ]]; then
    echo "Refusing to overwrite existing production artifact: $output_path" >&2
    exit 3
  fi
done

tmp_available_bytes="$(df -PB1 /tmp | awk 'NR==2 {print $4}')"
readonly tmp_available_bytes
if [[ "$tmp_available_bytes" -lt 53687091200 ]]; then
  echo "Need at least 50 GiB free in /tmp for VCC packaging" >&2
  exit 4
fi

date --iso-8601=seconds
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unknown}"
df -h "$PROJECT_DIR" /tmp

"$VCC_PY" - "$PRED_JSON" "$PRED_H5AD" <<'PY'
import hashlib
import json
import pathlib
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
prediction = pathlib.Path(sys.argv[2]).resolve()
assert report["schema"] == "vcc-state-direct-counts-v1"
assert report["full_official_contract"] is True
assert report["configuration"]["effect_bounding"] == "tanh"
assert report["configuration"]["effect_reference"] == (
    "paired-model-control-on-identical-cell-chunks"
)
assert report["configuration"]["target_strategy"] == "force"
assert report["configuration"]["integerization"] == "largest-remainder"
assert report["configuration"]["current_only_policy"] == "copy-raw-counts-exactly"
assert report["configuration"]["library_policy"] == "exact-source-library-per-cell"
assert report["validation"]["failed_checks"] == []
assert report["validation"]["shape"] == [360000, 18533]
assert report["validation"]["groups"] == 900
assert report["validation"]["nnz"] < 2140000000
assert report["scientific_qc"]["all_libraries_exact"] is True
assert report["scientific_qc"]["all_current_only_counts_exact"] is True
assert report["scientific_qc"]["target_knockdown_failures"] == []
provenance = report["provenance"]["output_h5ad"]
assert pathlib.Path(provenance["path"]).resolve() == prediction
assert provenance["size_bytes"] == prediction.stat().st_size
digest = hashlib.sha256()
with prediction.open("rb") as handle:
    while block := handle.read(8 << 20):
        digest.update(block)
assert provenance["sha256"] == digest.hexdigest()
print("Direct STATE provenance and generation assertions passed")
PY

"$VCC_BIN" prep "$PRED_H5AD" \
  -g dataset/controls/gene_names.csv \
  --perts dataset/controls/pert_counts.csv \
  --pert-col target_gene \
  --context-col context \
  --contexts A,B,C \
  --verify-targets \
  --check-cell-counts \
  --require-counts \
  --reject-controls \
  --encoding 32 \
  --dry-run \
  --json | tee "$DRYRUN_JSON"

"$VCC_PY" - "$DRYRUN_JSON" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
assert report["dry_run"] is True
assert report["n_cells"] == 360000
assert report["n_genes"] == 18533
assert report["normalization"] == "counts-preserved"
assert report["verified_targets"] is True
assert report["cells_per_context"] == {"A": 120000, "B": 120000, "C": 120000}
assert report["nnz"] is not None and report["nnz"] < 2140000000
print("Official direct STATE dry-run gates passed")
PY

"$VCC_BIN" prep "$PRED_H5AD" \
  -g dataset/controls/gene_names.csv \
  --perts dataset/controls/pert_counts.csv \
  --pert-col target_gene \
  --context-col context \
  --contexts A,B,C \
  --verify-targets \
  --check-cell-counts \
  --require-counts \
  --reject-controls \
  --encoding 32 \
  --output "$VCC_FILE" \
  --json | tee "$PREP_JSON"

"$VCC_PY" - "$PREP_JSON" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
assert report["dry_run"] is False
assert report["n_cells"] == 360000
assert report["n_genes"] == 18533
assert report["normalization"] == "counts-preserved"
assert report["verified_targets"] is True
assert report["cells_per_context"] == {"A": 120000, "B": 120000, "C": 120000}
assert report["nnz"] is not None and report["nnz"] < 2140000000
print("Official direct STATE package gates passed")
PY

"$VCC_PY" -c 'import sys; from vcc.vccfile import validate_vcc; validate_vcc(sys.argv[1]); print("container_valid")' "$VCC_FILE"
tar -tf "$VCC_FILE"

sha256sum \
  "$PRED_H5AD" "$PRED_JSON" "$DRYRUN_JSON" "$VCC_FILE" "$PREP_JSON" \
  dataset/controls/gene_names.csv \
  dataset/controls/pert_counts.csv \
  scripts/generate_state_direct_counts.py \
  scripts/infer_state_effect_prior.py \
  | tee "$SUMS_FILE"

date --iso-8601=seconds
