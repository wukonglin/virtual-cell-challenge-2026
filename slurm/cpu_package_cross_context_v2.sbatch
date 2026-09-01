#!/usr/bin/env bash
#SBATCH --job-name=vcc-cross-context-pack
#SBATCH --account=allaccess
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/cpu_cross_context_pack_%j.out
#SBATCH --error=logs/cpu_cross_context_pack_%j.err

set -euo pipefail

readonly PROJECT_DIR="${VCC_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
readonly VCC_BIN="$PROJECT_DIR/.venv-vcc/bin/vcc"
readonly VCC_PY="$PROJECT_DIR/.venv-vcc/bin/python"
readonly CANDIDATE_VERSION="${VCC_CANDIDATE_VERSION:-v3}"
case "$CANDIDATE_VERSION" in
  v3|v4) readonly RESPONSE_VERSION="$CANDIDATE_VERSION" ;;
  v5) readonly RESPONSE_VERSION="v4" ;;
  *)
    echo "VCC_CANDIDATE_VERSION must be v3, v4, or v5" >&2
    exit 2
    ;;
esac
readonly PRED_H5AD="$PROJECT_DIR/artifacts/v2/cross_context_state_${CANDIDATE_VERSION}.h5ad"
readonly PRED_JSON="$PROJECT_DIR/artifacts/v2/cross_context_state_${CANDIDATE_VERSION}.json"
readonly RESPONSE_NPZ="$PROJECT_DIR/artifacts/v2/cross_context_pseudobulk_${RESPONSE_VERSION}.npz"
readonly DRYRUN_JSON="$PROJECT_DIR/artifacts/v2/cross_context_state_${CANDIDATE_VERSION}_dryrun.json"
readonly VCC_FILE="$PROJECT_DIR/artifacts/v2/cross_context_state_${CANDIDATE_VERSION}.vcc"
readonly PREP_JSON="$PROJECT_DIR/artifacts/v2/cross_context_state_${CANDIDATE_VERSION}_prep.json"
readonly SUMS_FILE="$PROJECT_DIR/artifacts/v2/cross_context_state_${CANDIDATE_VERSION}_SHA256SUMS"

cd "$PROJECT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1

task_tmp="$(mktemp -d "/tmp/vcc_cross_context_pack_${CANDIDATE_VERSION}_XXXXXX")"
readonly task_tmp
export TMPDIR="$task_tmp"
trap 'rm -rf -- "$task_tmp"' EXIT

for required_path in \
  "$VCC_BIN" \
  "$VCC_PY" \
  "$PRED_H5AD" \
  "$PRED_JSON" \
  "$RESPONSE_NPZ" \
  scripts/generate_state_scorer_aware_counts.py \
  dataset/controls/gene_names.csv \
  dataset/state_support/extracted/gene_names.csv \
  dataset/controls/pert_counts.csv \
  dataset/controls/context_A.h5ad \
  dataset/controls/context_B.h5ad \
  dataset/controls/context_C.h5ad; do
  if [[ ! -r "$required_path" ]]; then
    echo "Missing required input: $required_path" >&2
    exit 2
  fi
done

for output_path in "$DRYRUN_JSON" "$VCC_FILE" "$PREP_JSON" "$SUMS_FILE"; do
  if [[ -e "$output_path" ]]; then
    echo "Refusing to overwrite production artifact: $output_path" >&2
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

"$VCC_PY" - \
  "$PRED_JSON" \
  "$PRED_H5AD" \
  "$RESPONSE_NPZ" \
  scripts/generate_state_scorer_aware_counts.py \
  dataset/controls/gene_names.csv \
  dataset/state_support/extracted/gene_names.csv \
  dataset/controls/pert_counts.csv \
  dataset/controls/context_A.h5ad \
  dataset/controls/context_B.h5ad \
  dataset/controls/context_C.h5ad \
  "$CANDIDATE_VERSION" <<'PY'
import hashlib
import json
import pathlib
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
prediction = pathlib.Path(sys.argv[2]).resolve()
response = pathlib.Path(sys.argv[3]).resolve()
implementation = pathlib.Path(sys.argv[4]).resolve()
current_genes = pathlib.Path(sys.argv[5]).resolve()
support_genes = pathlib.Path(sys.argv[6]).resolve()
targets = pathlib.Path(sys.argv[7]).resolve()
controls = {
    "A": pathlib.Path(sys.argv[8]).resolve(),
    "B": pathlib.Path(sys.argv[9]).resolve(),
    "C": pathlib.Path(sys.argv[10]).resolve(),
}
candidate = sys.argv[11]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def assert_manifest_file(record, path, label):
    assert isinstance(record, dict), f"{label}: missing manifest record"
    assert pathlib.Path(record["path"]).resolve() == path, f"{label}: path mismatch"
    assert record["size_bytes"] == path.stat().st_size, f"{label}: size mismatch"
    assert record["sha256"] == sha256_file(path), f"{label}: SHA256 mismatch"

assert report["schema"] == "vcc-state-scorer-aware-counts-v2"
assert report["full_official_contract"] is True
config = report["configuration"]
assert config["cells_per_group"] == 400
assert config["targets"] == 300
assert config["state_effect_bounding"] == "tanh"
expected = {
    "v3": (0.05, 0.30, 1.0, 1.0, 1.0, 0.60, 1.0),
    "v4": (0.05, 0.30, 1.0, 1.0, 1.0, 0.60, 1.0),
    "v5": (0.01, 0.15, 0.10, 0.22, 0.35, 0.0, 0.0),
}[candidate]
actual = (
    config["state_effect_weight"],
    config["state_effect_clip"],
    config["common_response_weight"],
    config["learned_response_weight"],
    config["combined_effect_clip"],
    config["pseudobulk_blend_weight"],
    config["zero_induction_scale"],
)
assert actual == expected, (actual, expected)
assert config["pseudobulk_depth_policy"] == "source-library"
assert config["target_policy"] == "off"
assert config["count_emission"] == "stochastic-round"
assert config["max_genes_per_cell"] == 5900
expected_mass_thresholds = {
    "v3": (0.10, 0.15),
    "v4": (0.10, 0.15),
    "v5": (0.03, 0.05),
}[candidate]
assert config["max_dropped_count_fraction"] == expected_mass_thresholds[0]
assert config["max_absolute_library_drift_fraction"] == expected_mass_thresholds[1]
assert report["scorer_contract"]["perturbed_gene_excluded_from_all_six_metrics"] is True
assert report["axes"]["contexts"] == ["A", "B", "C"]
assert report["axes"]["targets"] == 300
assert report["axes"]["current_genes"] == 18533
assert report["validation"]["failed_checks"] == []
assert report["validation"]["shape"] == [360000, 18533]
assert report["validation"]["groups"] == 900
assert report["validation"]["nnz"] < 2140000000
assert all(report["validation"]["checks"].values())
assert report["scientific_qc"]["target_clamp_failures"] == []
mass = report["scientific_qc"]["count_mass"]
assert mass["passed"] is True
assert mass["violations"] == []
assert mass["thresholds"] == {
    "max_dropped_count_fraction": expected_mass_thresholds[0],
    "max_absolute_library_drift_fraction": expected_mass_thresholds[1],
    "evaluation_scopes": ["overall", "A", "B", "C"],
}
assert set(mass["by_context"]) == {"A", "B", "C"}
for scope in [mass["overall"], *mass["by_context"].values()]:
    assert scope["emitted_count_total_before_cap"] == (
        scope["emitted_count_total_after_cap"] + scope["dropped_count_total"]
    )
    assert scope["dropped_count_fraction"] <= expected_mass_thresholds[0]
    assert scope["absolute_library_drift_fraction"] <= expected_mass_thresholds[1]
if candidate == "v5":
    assert mass["overall"]["dropped_count_fraction"] <= 0.025
assert mass["overall"]["dropped_count_total"] == report["scientific_qc"][
    "dropped_counts_without_redistribution"
]

for key in ("response_artifact", "pseudobulk_artifact"):
    assert_manifest_file(report["provenance"][key], response, key)
assert_manifest_file(report["provenance"]["script"], implementation, "implementation")
assert_manifest_file(
    report["provenance"]["current_gene_axis"], current_genes, "current_gene_axis"
)
assert_manifest_file(
    report["provenance"]["support_gene_axis"], support_genes, "support_gene_axis"
)
assert_manifest_file(report["provenance"]["target_axis"], targets, "target_axis")
for context, path in controls.items():
    assert_manifest_file(
        report["provenance"]["controls"][context], path, f"control_{context}"
    )
output_provenance = report["provenance"]["output_h5ad"]
assert pathlib.Path(output_provenance["path"]).resolve() == prediction
assert output_provenance["size_bytes"] == prediction.stat().st_size
assert output_provenance["sha256"] == sha256_file(prediction)
print("Cross-context provenance and generation assertions passed")
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
print("Official VCC dry-run gates passed")
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
print("Official VCC package gates passed")
PY

"$VCC_PY" -c 'import sys; from vcc.vccfile import validate_vcc; validate_vcc(sys.argv[1]); print("container_valid")' "$VCC_FILE"
tar -tf "$VCC_FILE"

sha256sum \
  "$PRED_H5AD" \
  "$PRED_JSON" \
  "$RESPONSE_NPZ" \
  "$DRYRUN_JSON" \
  "$VCC_FILE" \
  "$PREP_JSON" \
  dataset/controls/gene_names.csv \
  dataset/controls/pert_counts.csv \
  scripts/generate_state_scorer_aware_counts.py \
  | tee "$SUMS_FILE"

date --iso-8601=seconds
