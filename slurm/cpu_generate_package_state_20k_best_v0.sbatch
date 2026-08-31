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
readonly SELECTION_JSON="$PROJECT_DIR/artifacts/state_runs/state_sm_20k_v0/checkpoints/selected_checkpoint.json"
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
  "$PYTHON_BIN" "$VCC_BIN" "$VCC_PY" "$PRIOR_NPZ" "$PRIOR_JSON" \
  "$SELECTION_JSON" "$BAYES_NPZ" \
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

"$VCC_PY" - "$PRIOR_JSON" "$PROJECT_DIR" "$SELECTION_JSON" <<'PY'
import hashlib
import json
import pathlib
import sys

report = json.load(open(sys.argv[1]))
project_dir = pathlib.Path(sys.argv[2]).resolve()
selection = json.load(open(sys.argv[3]))
configuration = report["configuration"]
model = report["model"]
provenance = report["provenance"]

assert report["schema"] == "vcc-state-effect-prior-v2"
assert configuration["effect_reference"] == "model-control"
assert configuration["precision"] == "float32"
assert configuration["input_normalization"] == "log1p-raw"
assert configuration["effect_space"] == (
    "log1p-target-sum-normalized-arithmetic-pseudobulk-delta"
)
assert configuration["effect_bulk_target_sum"] == 50_000.0
assert configuration["effect_normalization_genes"] == 18_077
assert configuration["sets_per_context"] == 4
assert configuration["cells_per_context"] == 512
assert configuration["top_k_including_target"] == 161
assert configuration["min_abs_effect"] == 0.01
assert configuration["effect_clip"] == 0.60
for context in "ABC":
    assert report["qc"]["controls"][context]["state_reference"]["bulk_axis_genes"] == 18_077
assert model["cell_sentence_len"] == 128
assert model["input_dim"] == 18080
assert model["output_dim"] == 18080
assert model["pert_dim"] == 5120
assert model["parameters"] > 150_000_000
assert selection["schema"] == "vcc-state-checkpoint-selection-v1"
assert pathlib.Path(provenance["checkpoint"]["path"]).resolve() == pathlib.Path(
    selection["selected"]["checkpoint"]
).resolve()
assert provenance["checkpoint"]["size_bytes"] == selection["selected"][
    "checkpoint_size_bytes"
]
assert provenance["checkpoint"]["sha256"] == selection["selected"][
    "checkpoint_sha256"
]
assert provenance["repository_commit"] is not None
assert provenance["state_source_commit"] is not None
prior_path = pathlib.Path(
    project_dir / "artifacts/state_effect_prior_20k_best_v0.npz"
)
assert pathlib.Path(provenance["output_npz"]["path"]).resolve() == prior_path
assert provenance["output_npz"]["size_bytes"] == prior_path.stat().st_size
digest = hashlib.sha256()
with prior_path.open("rb") as handle:
    while block := handle.read(8 << 20):
        digest.update(block)
assert provenance["output_npz"]["sha256"] == digest.hexdigest()
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
  --effect-baseline-target-sum 50000 \
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
assert report["generation"]["effect_baseline_target_sum"] == 50_000.0
assert report["input_prior_effect_space"] == (
    "log1p-target-sum-normalized-arithmetic-pseudobulk-delta"
)
assert report["input_prior_effect_target_sum"] == 50_000.0
assert report["input_prior_effect_normalization_genes"] == 18_077
assert report["generation"]["effect_normalization_genes"] == 18_077
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
