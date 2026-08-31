#!/usr/bin/env bash
#SBATCH --job-name=vcc-state-pack-20k
#SBATCH --account=allaccess
#SBATCH --partition=full
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --output=logs/cpu_state_pack_20k_best_v0_%j.out
#SBATCH --error=logs/cpu_state_pack_20k_best_v0_%j.err

set -euo pipefail

readonly PROJECT_DIR="${VCC_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
readonly PYTHON_BIN="/mnt/beegfs/bulk/mirror/yl4259/miniconda3/envs/scanpy/bin/python"
readonly VCC_BIN="$PROJECT_DIR/.venv-vcc/bin/vcc"
readonly VCC_PY="$PROJECT_DIR/.venv-vcc/bin/python"
readonly PRIOR_NPZ="$PROJECT_DIR/artifacts/state_effect_prior_20k_best_v0.npz"
readonly PRIOR_JSON="$PROJECT_DIR/artifacts/state_effect_prior_20k_best_v0.json"
readonly BAYES_NPZ="$PROJECT_DIR/artifacts/bayesian_prior_h100_v0.npz"
readonly QC_JSON="$PROJECT_DIR/artifacts/state_vs_bayes_20k_best_v0.json"
readonly PRED_H5AD="$PROJECT_DIR/artifacts/state_20k_best_v0.h5ad"
readonly PRED_JSON="$PROJECT_DIR/artifacts/state_20k_best_v0.json"
readonly DRYRUN_JSON="$PROJECT_DIR/artifacts/state_20k_best_v0_dryrun.json"
readonly VCC_FILE="$PROJECT_DIR/artifacts/state_20k_best_v0.vcc"
readonly PREP_JSON="$PROJECT_DIR/artifacts/state_20k_best_v0_prep.json"

cd "$PROJECT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1

task_tmp="$(mktemp -d /tmp/vcc_state_pack_20k_XXXXXX)"
readonly task_tmp
export TMPDIR="$task_tmp"
trap 'rm -rf -- "$task_tmp"' EXIT

for required_path in \
  "$PYTHON_BIN" "$VCC_BIN" "$VCC_PY" "$PRIOR_NPZ" "$PRIOR_JSON" "$BAYES_NPZ" \
  scripts/qc_effect_priors.py \
  scripts/generate_bayesian_submission.py \
  dataset/controls/context_A.h5ad \
  dataset/controls/context_B.h5ad \
  dataset/controls/context_C.h5ad \
  dataset/controls/gene_names.csv \
  dataset/controls/pert_counts.csv; do
  if [[ ! -r "$required_path" ]]; then
    echo "Missing required input: $required_path" >&2
    exit 2
  fi
done

for protected_output in \
  "$QC_JSON" "$PRED_H5AD" "$PRED_JSON" "$DRYRUN_JSON" "$VCC_FILE" "$PREP_JSON"; do
  if [[ -e "$protected_output" ]]; then
    echo "Refusing to overwrite existing production artifact: $protected_output" >&2
    exit 3
  fi
done

date --iso-8601=seconds
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unknown}"
df -h "$PROJECT_DIR" /tmp
tmp_available_bytes="$(df -PB1 /tmp | awk 'NR==2 {print $4}')"
readonly tmp_available_bytes
if [[ "$tmp_available_bytes" -lt 53687091200 ]]; then
  echo "Need at least 50 GiB free in /tmp for VCC packaging" >&2
  exit 4
fi

"$VCC_PY" - "$PRIOR_JSON" "$PROJECT_DIR" <<'PY'
import json
import pathlib
import sys

report = json.load(open(sys.argv[1]))
project_dir = pathlib.Path(sys.argv[2]).resolve()
configuration = report["configuration"]
model = report["model"]
provenance = report["provenance"]

assert configuration["effect_reference"] == "model-control"
assert configuration["precision"] == "float32"
assert configuration["input_normalization"] == "log1p-raw"
assert configuration["sets_per_context"] == 4
assert configuration["cells_per_context"] == 512
assert configuration["top_k_including_target"] == 161
assert model["cell_sentence_len"] == 128
assert model["input_dim"] == 18080
assert model["output_dim"] == 18080
assert model["pert_dim"] == 5120
assert model["parameters"] > 150_000_000
assert pathlib.Path(provenance["checkpoint"]["path"]).name == "best.ckpt"
assert provenance["repository_commit"] is not None
assert provenance["state_source_commit"] is not None
assert pathlib.Path(provenance["output_npz"]["path"]).resolve() == pathlib.Path(
    project_dir / "artifacts/state_effect_prior_20k_best_v0.npz"
)
print("STATE inference provenance assertions passed")
PY

"$VCC_PY" scripts/qc_effect_priors.py \
  "$PRIOR_NPZ" "$BAYES_NPZ" \
  --controls-dir dataset/controls \
  --report "$QC_JSON"

"$VCC_PY" - "$QC_JSON" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
state = report["priors"][0]
assert state["failed_hard_checks"] == [], state["failed_hard_checks"]
for context, structure in state["context_structure"].items():
    assert structure["spectral_effective_rank"] >= 5.0, context
    assert structure["nearest_other_cosine"]["0.5"] < 0.98, context
    assert structure["panel_excluded_profile_norm"]["0.1"] > 1e-8, context
print("STATE structural QC gates passed")
PY

"$PYTHON_BIN" scripts/generate_bayesian_submission.py \
  --prior "$PRIOR_NPZ" \
  --controls-dir dataset/controls \
  --output-h5ad "$PRED_H5AD" \
  --output-json "$PRED_JSON" \
  --cells-per-group 400 \
  --max-base-genes 5780 \
  --compression lzf \
  --seed 20260831

"$PYTHON_BIN" - "$PRED_JSON" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
assert report["full_official_contract"] is True
assert report["validation"]["failed_checks"] == []
assert report["validation"]["shape"] == [360000, 18533]
assert report["validation"]["groups"] == 900
assert report["validation"]["nnz"] < 2_140_000_000
assert report["scientific_qc"]["all_groups_have_realized_delta"] is True
assert report["scientific_qc"]["target_knockdown_failures"] == []
for context in "ABC":
    cap = report["context_statistics"][context]["sparsity_cap_qc"]
    assert cap["pseudobulk_cosine"] >= 0.995
    assert cap["pseudobulk_l1_fraction"] <= 0.06
print("STATE generator assertions passed")
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
import json, sys
report = json.load(open(sys.argv[1]))
assert report["dry_run"] is True
assert report["n_cells"] == 360000
assert report["n_genes"] == 18533
assert report["normalization"] == "counts-preserved"
assert report["verified_targets"] is True
assert report["cells_per_context"] == {"A": 120000, "B": 120000, "C": 120000}
assert report["nnz"] is not None and report["nnz"] < 2_140_000_000
print("Official dry-run assertions passed")
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
import json, sys
report = json.load(open(sys.argv[1]))
assert report["dry_run"] is False
assert report["n_cells"] == 360000
assert report["n_genes"] == 18533
assert report["normalization"] == "counts-preserved"
assert report["verified_targets"] is True
assert report["cells_per_context"] == {"A": 120000, "B": 120000, "C": 120000}
assert report["nnz"] is not None and report["nnz"] < 2_140_000_000
print("Official package assertions passed")
PY

"$VCC_PY" -c 'import sys; from vcc.vccfile import validate_vcc; validate_vcc(sys.argv[1]); print("container_valid")' "$VCC_FILE"
tar -tf "$VCC_FILE"

sha256sum \
  "$PRIOR_NPZ" "$PRIOR_JSON" "$QC_JSON" "$PRED_H5AD" "$PRED_JSON" \
  "$DRYRUN_JSON" "$VCC_FILE" "$PREP_JSON" \
  dataset/controls/context_A.h5ad \
  dataset/controls/context_B.h5ad \
  dataset/controls/context_C.h5ad \
  dataset/controls/gene_names.csv \
  dataset/controls/pert_counts.csv \
  scripts/infer_state_effect_prior.py \
  scripts/qc_effect_priors.py \
  scripts/generate_bayesian_submission.py \
  | tee artifacts/state_20k_best_v0_SHA256SUMS

date --iso-8601=seconds
