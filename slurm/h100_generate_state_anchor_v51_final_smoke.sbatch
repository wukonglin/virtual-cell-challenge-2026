#!/usr/bin/env bash
#SBATCH --job-name=vcc-state-anchor-v51-final
#SBATCH --account=allaccess
#SBATCH --partition=full
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --output=logs/h100_state_anchor_v51_final_%j.out
#SBATCH --error=logs/h100_state_anchor_v51_final_%j.err

set -euo pipefail
unset PYTHONOPTIMIZE

readonly PROJECT_DIR="${VCC_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
readonly PYTHON_BIN="$PROJECT_DIR/.venv-state/bin/python"
readonly MODEL_DIR="$PROJECT_DIR/artifacts/state_runs/state_sm_20k_v0"
readonly RESPONSE_NPZ="$PROJECT_DIR/artifacts/v2/state_anchored_residual_v51_top100.npz"
readonly RESPONSE_JSON="$PROJECT_DIR/artifacts/v2/state_anchored_residual_v51_top100.json"
readonly OUTPUT_DIR="$PROJECT_DIR/artifacts/v2/smoke_final"
readonly DIRECT_H5AD="$OUTPUT_DIR/state_anchor_v51_final_direct.h5ad"
readonly DIRECT_JSON="$OUTPUT_DIR/state_anchor_v51_final_direct.json"
readonly ZERO_H5AD="$OUTPUT_DIR/state_anchor_v51_final_alpha000.h5ad"
readonly ZERO_JSON="$OUTPUT_DIR/state_anchor_v51_final_alpha000.json"
readonly ALPHA010_H5AD="$OUTPUT_DIR/state_anchor_v51_final_alpha010.h5ad"
readonly ALPHA010_JSON="$OUTPUT_DIR/state_anchor_v51_final_alpha010.json"
readonly SUMMARY_JSON="$OUTPUT_DIR/state_anchor_v51_final_summary.json"
readonly EXPECTED_RESPONSE_SHA256="f5875a01fcdbf9f0de27a5e8493900cbaed0b3971a5a19f760b929eb996a1bba"

cd "$PROJECT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export CPATH="$PROJECT_DIR/.venv-state-headers/usr/include/python3.11"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

required_paths=(
  "$PYTHON_BIN"
  "$MODEL_DIR/checkpoints/selected.ckpt"
  "$MODEL_DIR/checkpoints/selected_checkpoint.json"
  "$MODEL_DIR/pert_onehot_map.pt"
  "$MODEL_DIR/var_dims.pkl"
  "$RESPONSE_NPZ"
  "$RESPONSE_JSON"
  scripts/generate_state_direct_counts.py
  scripts/generate_state_scorer_aware_counts.py
  dataset/state_support/extracted/gene_names.csv
  dataset/controls/gene_names.csv
  dataset/controls/pert_counts.csv
  dataset/controls/context_A.h5ad
  dataset/controls/context_B.h5ad
  dataset/controls/context_C.h5ad
)
for required_path in "${required_paths[@]}"; do
  if [[ ! -r "$required_path" ]]; then
    echo "Missing required input: $required_path" >&2
    exit 2
  fi
done

mkdir -p "$OUTPUT_DIR"
for output_path in \
  "$DIRECT_H5AD" "$DIRECT_JSON" \
  "$ZERO_H5AD" "$ZERO_JSON" \
  "$ALPHA010_H5AD" "$ALPHA010_JSON" "$SUMMARY_JSON"; do
  if [[ -e "$output_path" ]]; then
    echo "Refusing to overwrite final smoke artifact: $output_path" >&2
    exit 3
  fi
done

task_scratch="$(mktemp -d /tmp/vcc_state_anchor_v51_final_XXXXXX)"
readonly task_scratch
trap 'rm -rf -- "$task_scratch"' EXIT

# Gate the complete residual contract before spending H100 time on generation.
"$PYTHON_BIN" - \
  "$RESPONSE_NPZ" "$RESPONSE_JSON" "$EXPECTED_RESPONSE_SHA256" \
  dataset/controls/gene_names.csv dataset/controls/pert_counts.csv \
  dataset/state_support/extracted/gene_names.csv \
  dataset/controls/context_A.h5ad dataset/controls/context_B.h5ad \
  dataset/controls/context_C.h5ad <<'PY'
import csv
import hashlib
import json
import pathlib
import sys

import numpy as np


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_axis(path: pathlib.Path, column: str) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != [column]:
            raise RuntimeError(f"{path}: expected one column named {column}")
        values = [row[column] for row in reader]
    if not values or len(values) != len(set(values)):
        raise RuntimeError(f"{path}: axis is empty or duplicated")
    return values


def read_headerless_axis(path: pathlib.Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    values = [value for value in values if value]
    if not values or len(values) != len(set(values)):
        raise RuntimeError(f"{path}: axis is empty or duplicated")
    return values


response_path = pathlib.Path(sys.argv[1]).resolve()
report_path = pathlib.Path(sys.argv[2]).resolve()
expected_sha = sys.argv[3]
gene_axis_path = pathlib.Path(sys.argv[4]).resolve()
target_axis_path = pathlib.Path(sys.argv[5]).resolve()
support_axis_path = pathlib.Path(sys.argv[6]).resolve()
control_paths = [pathlib.Path(value).resolve() for value in sys.argv[7:10]]
genes = read_axis(gene_axis_path, "gene_name")
targets = read_axis(target_axis_path, "target_gene")
support_genes = read_headerless_axis(support_axis_path)
report = json.loads(report_path.read_text(encoding="utf-8"))
actual_sha = sha256_file(response_path)
assert actual_sha == expected_sha, (actual_sha, expected_sha)
assert report["schema"] == "vcc-state-anchored-residual-v1"
config = report["configuration"]
assert config == {
    "top_k": 100,
    "min_context_detection_rate": 0.01,
    "min_context_mean_count": 0.01,
    "direct_reliability_power": 0.0,
    "fallback_reliability": 0.0,
    "centering": "mean",
    "shared_response_weight": 0.0,
    "preserve_current_only_genes": True,
    "exclude_all_panel_target_genes": True,
}
axes = report["axes"]
assert axes["contexts"] == ["A", "B", "C"]
assert axes["targets"] == len(targets) == 300
assert axes["genes"] == len(genes) == 18533
assert axes["support_overlap_genes"] == 18077
assert axes["protected_current_only_genes"] == 456
assert axes["public_query_genes"] == 7681
assert axes["excluded_panel_target_genes"] == 300
assert axes["direct_targets"] == 267
assert axes["fallback_targets"] == 33
assert axes["control_cells"] == {"A": 18400, "B": 18400, "C": 18400}
output_descriptor = report["provenance"]["output_npz"]
assert pathlib.Path(output_descriptor["path"]).resolve() == response_path
assert output_descriptor["sha256"] == actual_sha
assert output_descriptor["size_bytes"] == response_path.stat().st_size
assert report["provenance"]["support_genes"]["sha256"] == sha256_file(support_axis_path)
for context, path in zip(("A", "B", "C"), control_paths, strict=True):
    descriptor = report["provenance"]["controls"][context]
    assert pathlib.Path(descriptor["path"]).resolve() == path
    assert descriptor["sha256"] == sha256_file(path)

required_keys = {
    "genes", "targets", "contexts", "learned_log_fold", "direct_mask",
    "fallback_mask", "source_reliability", "applied_target_reliability",
    "context_detection_rate", "context_mean_count", "protected_current_only_mask",
    "query_gene_mask", "excluded_panel_target_mask",
}
with np.load(response_path, allow_pickle=False) as archive:
    assert set(archive.files) == required_keys, sorted(archive.files)
    archive_genes = np.asarray(archive["genes"]).astype(str)
    archive_targets = np.asarray(archive["targets"]).astype(str)
    contexts = np.asarray(archive["contexts"]).astype(str)
    residual = np.asarray(archive["learned_log_fold"])
    direct = np.asarray(archive["direct_mask"])
    fallback = np.asarray(archive["fallback_mask"])
    source_reliability = np.asarray(archive["source_reliability"])
    applied_reliability = np.asarray(archive["applied_target_reliability"])
    detection = np.asarray(archive["context_detection_rate"])
    mean_count = np.asarray(archive["context_mean_count"])
    protected = np.asarray(archive["protected_current_only_mask"])
    query = np.asarray(archive["query_gene_mask"])
    excluded = np.asarray(archive["excluded_panel_target_mask"])

assert np.array_equal(archive_genes, np.asarray(genes))
assert np.array_equal(archive_targets, np.asarray(targets))
assert np.array_equal(contexts, np.asarray(["A", "B", "C"]))
assert residual.shape == (3, 300, 18533) and residual.dtype == np.float32
assert np.isfinite(residual).all()
for name, mask, shape in (
    ("direct", direct, (300,)), ("fallback", fallback, (300,)),
    ("protected", protected, (18533,)), ("query", query, (18533,)),
    ("excluded", excluded, (18533,)),
):
    assert mask.shape == shape and mask.dtype == np.bool_, name
assert np.all(np.logical_xor(direct, fallback))
assert int(direct.sum()) == 267 and int(fallback.sum()) == 33
assert source_reliability.shape == (300,) and applied_reliability.shape == (300,)
assert np.isfinite(source_reliability).all() and np.isfinite(applied_reliability).all()
assert np.all((source_reliability >= 0) & (source_reliability <= 1))
assert np.all(source_reliability[direct] > 0) and np.all(source_reliability[fallback] == 0)
assert np.all(applied_reliability[direct] == 1) and np.all(applied_reliability[fallback] == 0)
assert detection.shape == mean_count.shape == (3, 18533)
assert np.isfinite(detection).all() and np.isfinite(mean_count).all()
assert np.all((detection >= 0) & (detection <= 1)) and np.all(mean_count >= 0)
expected_protected = ~np.isin(archive_genes, np.asarray(support_genes))
expected_excluded = np.isin(archive_genes, archive_targets)
assert np.array_equal(protected, expected_protected)
assert np.array_equal(excluded, expected_excluded)
assert int(protected.sum()) == 456 and int(query.sum()) == 7681 and int(excluded.sum()) == 300
assert not np.any(residual[:, :, protected])
assert not np.any(residual[:, :, ~query])
assert not np.any(residual[:, :, excluded])
assert not np.any(residual[:, fallback])
nonzero_per_group = np.count_nonzero(residual, axis=2)
assert np.all(nonzero_per_group[:, direct] == 100)
assert np.all(nonzero_per_group[:, fallback] == 0)
scientific = report["scientific_qc"]
assert scientific["nonzero_residuals"] == int(np.count_nonzero(residual)) == 80100
assert scientific["fallback_residuals_are_zero"] is True
assert scientific["common_response_emitted"] is False
assert scientific["nonzero_per_group"]["max"] == 100.0
assert scientific["applied_target_reliability"]["min"] == 0.0
assert scientific["applied_target_reliability"]["max"] == 1.0
print(json.dumps({
    "residual_contract": "passed",
    "sha256": actual_sha,
    "direct_targets": int(direct.sum()),
    "fallback_targets": int(fallback.sum()),
    "nonzero_residuals": int(np.count_nonzero(residual)),
}, indent=2))
PY

date --iso-8601=seconds
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unknown}"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader

"$PYTHON_BIN" scripts/generate_state_direct_counts.py \
  --model-dir "$MODEL_DIR" \
  --checkpoint selected.ckpt \
  --selection-json "$MODEL_DIR/checkpoints/selected_checkpoint.json" \
  --controls-dir dataset/controls \
  --support-genes dataset/state_support/extracted/gene_names.csv \
  --output-h5ad "$DIRECT_H5AD" \
  --output-json "$DIRECT_JSON" \
  --scratch-dir "$task_scratch/direct" \
  --device cuda \
  --require-cuda \
  --require-h100 \
  --model-chunk-size 32 \
  --cells-per-group 32 \
  --target-limit 8 \
  --effect-scale 1.0 \
  --effect-clip 0.60 \
  --effect-bounding tanh \
  --target-strategy force \
  --target-remaining-fraction 0.20 \
  --max-genes-per-cell 5900 \
  --integerization largest-remainder \
  --groups-per-shard 4 \
  --concat-max-loaded-elements 20000000 \
  --compression lzf \
  --seed 20260831

run_anchored_candidate() {
  local residual_weight="$1"
  local output_h5ad="$2"
  local output_json="$3"
  local scratch_name="$4"

  "$PYTHON_BIN" scripts/generate_state_scorer_aware_counts.py \
    --model-dir "$MODEL_DIR" \
    --checkpoint selected.ckpt \
    --selection-json "$MODEL_DIR/checkpoints/selected_checkpoint.json" \
    --controls-dir dataset/controls \
    --support-genes dataset/state_support/extracted/gene_names.csv \
    --output-h5ad "$output_h5ad" \
    --output-json "$output_json" \
    --scratch-dir "$task_scratch/$scratch_name" \
    --device cuda \
    --require-cuda \
    --require-h100 \
    --model-chunk-size 32 \
    --cells-per-group 32 \
    --target-limit 8 \
    --state-effect-weight 1.0 \
    --state-effect-clip 0.60 \
    --state-effect-bounding tanh \
    --response-npz "$RESPONSE_NPZ" \
    --common-response-weight 0.0 \
    --learned-response-weight "$residual_weight" \
    --combined-effect-clip 0.65 \
    --pseudobulk-blend-weight 0.0 \
    --zero-induction-scale 0.0 \
    --target-policy force \
    --target-max-remaining-fraction 0.20 \
    --count-emission exact-largest-remainder \
    --max-genes-per-cell 5900 \
    --max-dropped-count-fraction 0.0 \
    --max-absolute-library-drift-fraction 0.0 \
    --groups-per-shard 4 \
    --concat-max-loaded-elements 20000000 \
    --compression lzf \
    --seed 20260831
}

run_anchored_candidate 0.0 "$ZERO_H5AD" "$ZERO_JSON" alpha000
run_anchored_candidate 0.10 "$ALPHA010_H5AD" "$ALPHA010_JSON" alpha010

"$PYTHON_BIN" - \
  "$DIRECT_H5AD" "$DIRECT_JSON" \
  "$ZERO_H5AD" "$ZERO_JSON" \
  "$ALPHA010_H5AD" "$ALPHA010_JSON" \
  "$RESPONSE_NPZ" "$EXPECTED_RESPONSE_SHA256" "$SUMMARY_JSON" \
  scripts/generate_state_direct_counts.py scripts/generate_state_scorer_aware_counts.py \
  dataset/controls/context_A.h5ad dataset/controls/context_B.h5ad \
  dataset/controls/context_C.h5ad <<'PY'
import hashlib
import json
import os
import pathlib
import sys
import tempfile

import anndata as ad
import numpy as np
import scipy.sparse as sp


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: pathlib.Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
        suffix=".tmp", delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = pathlib.Path(handle.name)
    os.replace(temporary, path)


def sparse_matrix(data: ad.AnnData) -> sp.csr_matrix:
    matrix = data.X.tocsr() if sp.issparse(data.X) else sp.csr_matrix(data.X)
    matrix.sort_indices()
    return matrix


def validate_scorer_group_contract(groups: list[dict]) -> None:
    assert len(groups) == 24
    forced_delta = float(np.log(0.20))
    for group in groups:
        assert group["library_size_mismatches"] == 0
        assert group["current_only_count_mismatches"] == 0
        assert group["count_mass"]["dropped_count_fraction"] == 0.0
        assert group["count_mass"]["absolute_library_drift_fraction"] == 0.0
        normalization = group["composition_normalization_factor"]
        assert normalization["count"] == 32
        assert np.isfinite(normalization["min"]) and normalization["min"] > 0
        assert np.isfinite(normalization["max"]) and normalization["max"] > 0
        target_delta = group["final_target_delta"]
        assert target_delta["count"] == 32
        assert np.isclose(target_delta["min"], forced_delta, rtol=0, atol=1e-7)
        assert np.isclose(target_delta["max"], forced_delta, rtol=0, atol=1e-7)


def validate_scorer_report(
    report: dict, weight: float, output_path: pathlib.Path,
    response_path: pathlib.Path, response_sha: str, script_sha: str,
) -> None:
    assert report["schema"] == "vcc-state-scorer-aware-counts-v2"
    assert report["full_official_contract"] is False
    config = report["configuration"]
    assert config["device"].startswith("cuda") and "H100" in config["device_name"]
    assert config["cells_per_group"] == 32 and config["targets"] == 8
    assert config["state_effect_weight"] == 1.0
    assert config["state_effect_clip"] == 0.60
    assert config["state_effect_bounding"] == "tanh"
    assert config["common_response_weight"] == 0.0
    assert config["learned_response_weight"] == weight
    assert config["combined_effect_clip"] == 0.65
    assert config["pseudobulk_blend_weight"] == 0.0
    assert config["zero_induction_scale"] == 0.0
    assert config["target_policy"] == "force"
    assert config["target_max_remaining_fraction"] == 0.20
    assert config["count_emission"] == "exact-largest-remainder"
    assert config["emission_contract_version"] == "state-composition-v1"
    assert config["library_policy"] == "exact-source-library-composition"
    assert config["max_genes_per_cell"] == 5900
    scorer = report["scorer_contract"]
    assert scorer["target_force_order"] == "after-combined-effect-clip-before-composition-normalization"
    assert scorer["combined_effect_qc_includes_target_force"] is False
    assert scorer["target_policy_changes_other_coordinates"] is True
    assert report["validation"]["shape"] == [768, 18533]
    assert report["validation"]["groups"] == 24
    assert report["validation"]["failed_checks"] == []
    scientific = report["scientific_qc"]
    assert scientific["all_libraries_exact"] is True
    assert scientific["all_current_only_counts_exact"] is True
    assert scientific["target_policy_failures"] == []
    assert scientific["final_target_delta_failures"] == []
    assert np.isclose(scientific["forced_target_delta"], np.log(0.20), rtol=0, atol=1e-7)
    assert scientific["induced_zero_emitted_entries"] == 0
    assert scientific["dropped_counts_without_redistribution"] == 0
    assert scientific["count_mass"]["passed"] is True
    validate_scorer_group_contract(report["qc"]["groups"])
    response = report["provenance"]["response_artifact"]
    assert pathlib.Path(response["path"]).resolve() == response_path
    assert response["sha256"] == response_sha
    assert report["provenance"]["script"]["sha256"] == script_sha
    assert report["provenance"]["output_h5ad"]["sha256"] == sha256_file(output_path)


def validate_direct_report(
    report: dict, output_path: pathlib.Path, script_sha: str,
) -> None:
    assert report["schema"] == "vcc-state-direct-counts-v1"
    assert report["full_official_contract"] is False
    config = report["configuration"]
    assert config["device"].startswith("cuda") and "H100" in config["device_name"]
    assert config["cells_per_group"] == 32 and config["targets"] == 8
    assert config["effect_scale"] == 1.0 and config["effect_clip"] == 0.60
    assert config["effect_bounding"] == "tanh"
    assert config["target_strategy"] == "force"
    assert config["target_remaining_fraction"] == 0.20
    assert config["integerization"] == "largest-remainder"
    assert config["max_genes_per_cell"] == 5900
    assert config["library_policy"] == "exact-source-library-per-cell"
    assert config["current_only_policy"] == "copy-raw-counts-exactly"
    assert report["validation"]["shape"] == [768, 18533]
    assert report["validation"]["groups"] == 24
    assert report["validation"]["failed_checks"] == []
    scientific = report["scientific_qc"]
    assert scientific["all_libraries_exact"] is True
    assert scientific["all_current_only_counts_exact"] is True
    assert scientific["library_failures"] == []
    assert scientific["current_only_failures"] == []
    assert scientific["target_knockdown_failures"] == []
    groups = report["qc"]["groups"]
    assert len(groups) == 24
    forced_delta = float(np.log(0.20))
    for group in groups:
        assert group["library_size_mismatches"] == 0
        assert group["current_only_count_mismatches"] == 0
        assert group["integerization"] == "largest-remainder"
        assert np.isclose(group["final_target_delta"]["min"], forced_delta, rtol=0, atol=1e-7)
        assert np.isclose(group["final_target_delta"]["max"], forced_delta, rtol=0, atol=1e-7)
    assert report["provenance"]["script"]["sha256"] == script_sha
    assert report["provenance"]["output_h5ad"]["sha256"] == sha256_file(output_path)


direct_path, direct_report_path = map(pathlib.Path, sys.argv[1:3])
zero_path, zero_report_path = map(pathlib.Path, sys.argv[3:5])
alpha_path, alpha_report_path = map(pathlib.Path, sys.argv[5:7])
response_path = pathlib.Path(sys.argv[7]).resolve()
expected_response_sha = sys.argv[8]
summary_path = pathlib.Path(sys.argv[9]).resolve()
direct_script = pathlib.Path(sys.argv[10]).resolve()
scorer_script = pathlib.Path(sys.argv[11]).resolve()
control_paths = {
    context: pathlib.Path(path).resolve()
    for context, path in zip(("A", "B", "C"), sys.argv[12:15], strict=True)
}
response_sha = sha256_file(response_path)
assert response_sha == expected_response_sha

direct_report = json.loads(direct_report_path.read_text(encoding="utf-8"))
zero_report = json.loads(zero_report_path.read_text(encoding="utf-8"))
alpha_report = json.loads(alpha_report_path.read_text(encoding="utf-8"))
validate_direct_report(direct_report, direct_path, sha256_file(direct_script))
validate_scorer_report(
    zero_report, 0.0, zero_path, response_path, response_sha, sha256_file(scorer_script)
)
validate_scorer_report(
    alpha_report, 0.10, alpha_path, response_path, response_sha, sha256_file(scorer_script)
)

direct = ad.read_h5ad(direct_path)
zero = ad.read_h5ad(zero_path)
alpha = ad.read_h5ad(alpha_path)
assert direct.shape == zero.shape == alpha.shape == (768, 18533)
assert direct.obs_names.equals(zero.obs_names) and zero.obs_names.equals(alpha.obs_names)
assert direct.var_names.equals(zero.var_names) and zero.var_names.equals(alpha.var_names)
for column in ("context", "target_gene", "source_control_row"):
    assert direct.obs[column].equals(zero.obs[column])
    assert zero.obs[column].equals(alpha.obs[column])

direct_matrix = sparse_matrix(direct)
zero_matrix = sparse_matrix(zero)
alpha_matrix = sparse_matrix(alpha)
for matrix in (direct_matrix, zero_matrix, alpha_matrix):
    assert matrix.dtype.kind in "iu"
    assert np.all(matrix.data > 0)
    assert np.all(np.diff(matrix.indptr) <= 5900)

identity_difference = direct_matrix != zero_matrix
assert identity_difference.nnz == 0, identity_difference.nnz
assert np.array_equal(
    np.asarray(direct_matrix.sum(axis=1)).ravel(),
    np.asarray(alpha_matrix.sum(axis=1)).ravel(),
)

with np.load(response_path, allow_pickle=False) as response:
    direct_mask = np.asarray(response["direct_mask"], dtype=bool)
    fallback_mask = np.asarray(response["fallback_mask"], dtype=bool)
    protected_mask = np.asarray(response["protected_current_only_mask"], dtype=bool)
    targets = np.asarray(response["targets"]).astype(str)
expected_contexts = np.repeat(np.asarray(["A", "B", "C"]), 8 * 32)
expected_targets = np.tile(np.repeat(targets[:8], 32), 3)
assert np.array_equal(np.asarray(direct.obs["context"].astype(str)), expected_contexts)
assert np.array_equal(np.asarray(direct.obs["target_gene"].astype(str)), expected_targets)
group_is_direct = np.tile(direct_mask[:8], 3)
assert np.array_equal(~group_is_direct, np.tile(fallback_mask[:8], 3))

# Independently reconstruct source libraries and protected coordinates from controls.
source_libraries = np.empty(direct.n_obs, dtype=np.int64)
source_protected_parts: list[sp.csr_matrix] = []
for context in ("A", "B", "C"):
    output_positions = np.flatnonzero(np.asarray(direct.obs["context"].astype(str)) == context)
    source_rows = np.asarray(
        direct.obs.iloc[output_positions]["source_control_row"], dtype=np.int64
    )
    unique_rows, inverse = np.unique(source_rows, return_inverse=True)
    control = ad.read_h5ad(control_paths[context], backed="r")
    try:
        source = control[unique_rows].X
        source = source.tocsr() if sp.issparse(source) else sp.csr_matrix(source)
        source = source[inverse]
        source_libraries[output_positions] = np.asarray(source.sum(axis=1)).ravel()
        source_protected_parts.append(source[:, protected_mask])
    finally:
        control.file.close()
source_protected = sp.vstack(source_protected_parts, format="csr")
for matrix in (direct_matrix, zero_matrix, alpha_matrix):
    assert np.array_equal(np.asarray(matrix.sum(axis=1)).ravel(), source_libraries)
    assert (matrix[:, protected_mask] != source_protected).nnz == 0

changed = alpha_matrix != zero_matrix
changed_per_row = np.asarray(changed.sum(axis=1)).ravel()
changed_by_group = changed_per_row.reshape(24, 32).sum(axis=1)
fallback_changes = int(changed_by_group[~group_is_direct].sum())
changed_direct_groups = int(np.count_nonzero(changed_by_group[group_is_direct]))
total_direct_groups = int(np.count_nonzero(group_is_direct))
materialization = changed_direct_groups / total_direct_groups
assert changed.nnz > 0
assert fallback_changes == 0
assert (alpha_matrix[:, protected_mask] != zero_matrix[:, protected_mask]).nnz == 0
assert materialization >= 0.90
groups = alpha_report["qc"]["groups"]
outside = sum(group["combined_outside_bound"] for group in groups)
values = sum(group["effect_values"] for group in groups)
clip_fraction = outside / values
assert clip_fraction < 1e-4

summary = {
    "schema": "vcc-state-anchor-v51-final-smoke-v1",
    "status": "passed",
    "residual_contract": {
        "path": str(response_path),
        "sha256": response_sha,
        "direct_targets": int(direct_mask.sum()),
        "fallback_targets": int(fallback_mask.sum()),
    },
    "outputs": {
        "direct": {"h5ad": str(direct_path.resolve()), "report": str(direct_report_path.resolve())},
        "alpha000": {"h5ad": str(zero_path.resolve()), "report": str(zero_report_path.resolve())},
        "alpha010": {"h5ad": str(alpha_path.resolve()), "report": str(alpha_report_path.resolve())},
    },
    "exact_output_qc": {
        "cells": int(direct.n_obs),
        "genes": int(direct.n_vars),
        "max_genes_per_cell": 5900,
        "source_library_mismatches": 0,
        "current_only_count_mismatches": 0,
    },
    "alpha000_identity": {
        "different_entries_vs_direct": int(identity_difference.nnz),
        "byte_identical_sparse_matrix": True,
    },
    "alpha010": {
        "changed_entries_vs_alpha000": int(changed.nnz),
        "changed_cells_vs_alpha000": int(np.count_nonzero(changed_per_row)),
        "direct_groups_with_changes": changed_direct_groups,
        "direct_groups_total": total_direct_groups,
        "direct_group_materialization_fraction": float(materialization),
        "fallback_changed_entries": fallback_changes,
        "combined_clip_fraction": float(clip_fraction),
    },
}
atomic_write_json(summary_path, summary)
print(json.dumps(summary, indent=2, sort_keys=True))
PY

date --iso-8601=seconds
echo "Final smoke summary: $SUMMARY_JSON"
