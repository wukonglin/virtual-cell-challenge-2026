#!/usr/bin/env bash
#SBATCH --job-name=vcc-state-anchor-v51-grid
#SBATCH --account=allaccess
#SBATCH --partition=full
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=logs/h100_state_anchor_v51_grid_%j.out
#SBATCH --error=logs/h100_state_anchor_v51_grid_%j.err

set -euo pipefail

readonly PROJECT_DIR="${VCC_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
readonly PYTHON_BIN="$PROJECT_DIR/.venv-state/bin/python"
readonly MODEL_DIR="$PROJECT_DIR/artifacts/state_runs/state_sm_20k_v0"
readonly RESPONSE_NPZ="$PROJECT_DIR/artifacts/v2/state_anchored_residual_v51_top100.npz"
readonly FINAL_SMOKE_DIR="$PROJECT_DIR/artifacts/v2/smoke_final"
readonly ANCHOR_H5AD="$FINAL_SMOKE_DIR/state_anchor_v51_final_alpha000.h5ad"
readonly ANCHOR_JSON="$FINAL_SMOKE_DIR/state_anchor_v51_final_alpha000.json"
readonly GRID_ROOT="$PROJECT_DIR/artifacts/v2/grid_runs"

cd "$PROJECT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export CPATH="$PROJECT_DIR/.venv-state-headers/usr/include/python3.11"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mkdir -p "$GRID_ROOT"
run_tag="${SLURM_JOB_ID:-manual}"
RUN_DIR="$(mktemp -d "$GRID_ROOT/state_anchor_v51_grid_${run_tag}_XXXXXX")"
readonly RUN_DIR
readonly ALPHA010_H5AD="$RUN_DIR/state_anchor_v51_grid_alpha010.h5ad"
readonly ALPHA010_JSON="$RUN_DIR/state_anchor_v51_grid_alpha010.json"
readonly ALPHA020_H5AD="$RUN_DIR/state_anchor_v51_grid_alpha020.h5ad"
readonly ALPHA020_JSON="$RUN_DIR/state_anchor_v51_grid_alpha020.json"
readonly SUMMARY_JSON="$RUN_DIR/state_anchor_v51_grid_summary.json"

task_scratch="$(mktemp -d /tmp/vcc_state_anchor_v51_grid_XXXXXX)"
readonly task_scratch
trap 'rm -rf -- "$task_scratch"' EXIT

# Create the summary before preflight so every allocated grid run has a durable record.
"$PYTHON_BIN" - "$SUMMARY_JSON" "$RUN_DIR" "${SLURM_JOB_ID:-unknown}" <<'PY'
import json
import os
import pathlib
import sys
import tempfile

summary_path = pathlib.Path(sys.argv[1])
payload = {
    "schema": "vcc-state-anchor-v51-grid-smoke-v2",
    "status": "initialized",
    "evaluation_completed": False,
    "slurm_job_id": sys.argv[3],
    "run_directory": str(pathlib.Path(sys.argv[2]).resolve()),
    "candidates": [],
}
with tempfile.NamedTemporaryFile(
    mode="w", encoding="utf-8", dir=summary_path.parent, prefix=f".{summary_path.name}.",
    suffix=".tmp", delete=False,
) as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    temporary = pathlib.Path(handle.name)
os.replace(temporary, summary_path)
PY

required_paths=(
  "$PYTHON_BIN"
  "$MODEL_DIR/checkpoints/selected.ckpt"
  "$MODEL_DIR/checkpoints/selected_checkpoint.json"
  "$MODEL_DIR/pert_onehot_map.pt"
  "$MODEL_DIR/var_dims.pkl"
  "$RESPONSE_NPZ"
  "$ANCHOR_H5AD"
  "$ANCHOR_JSON"
  scripts/generate_state_scorer_aware_counts.py
  dataset/state_support/extracted/gene_names.csv
  dataset/controls/gene_names.csv
  dataset/controls/pert_counts.csv
  dataset/controls/context_A.h5ad
  dataset/controls/context_B.h5ad
  dataset/controls/context_C.h5ad
)
missing_paths=()
for required_path in "${required_paths[@]}"; do
  if [[ ! -r "$required_path" ]]; then
    missing_paths+=("$required_path")
  fi
done
if (( ${#missing_paths[@]} > 0 )); then
  "$PYTHON_BIN" - "$SUMMARY_JSON" "${missing_paths[@]}" <<'PY'
import json
import os
import pathlib
import sys
import tempfile

summary_path = pathlib.Path(sys.argv[1])
payload = json.loads(summary_path.read_text(encoding="utf-8"))
payload.update({"status": "preflight_failed", "missing_inputs": sys.argv[2:]})
with tempfile.NamedTemporaryFile(
    mode="w", encoding="utf-8", dir=summary_path.parent, prefix=f".{summary_path.name}.",
    suffix=".tmp", delete=False,
) as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    temporary = pathlib.Path(handle.name)
os.replace(temporary, summary_path)
PY
  printf 'Grid preflight failed; see %s\n' "$SUMMARY_JSON" >&2
  exit 2
fi

date --iso-8601=seconds
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unknown}"
echo "RUN_DIR=$RUN_DIR"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader

run_candidate() {
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

# A failed candidate must not prevent the other candidate or the summary from running.
set +e
run_candidate 0.10 "$ALPHA010_H5AD" "$ALPHA010_JSON" alpha010
status010=$?
run_candidate 0.20 "$ALPHA020_H5AD" "$ALPHA020_JSON" alpha020
status020=$?
set -e

set +e
"$PYTHON_BIN" - \
  "$ANCHOR_H5AD" "$ANCHOR_JSON" \
  "$ALPHA010_H5AD" "$ALPHA010_JSON" "$status010" \
  "$ALPHA020_H5AD" "$ALPHA020_JSON" "$status020" \
  "$RESPONSE_NPZ" "$SUMMARY_JSON" \
  scripts/generate_state_scorer_aware_counts.py <<'PY'
import hashlib
import json
import os
import pathlib
import sys
import tempfile
import traceback

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
    return data.X.tocsr() if sp.issparse(data.X) else sp.csr_matrix(data.X)


def exact_report_violations(
    report: dict, expected_weight: float, response_path: pathlib.Path,
    response_sha: str, script_sha: str,
) -> list[str]:
    violations: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            violations.append(message)

    config = report.get("configuration", {})
    scorer = report.get("scorer_contract", {})
    scientific = report.get("scientific_qc", {})
    validation = report.get("validation", {})
    provenance = report.get("provenance", {})
    response = provenance.get("response_artifact") or {}
    script = provenance.get("script") or {}
    check(report.get("schema") == "vcc-state-scorer-aware-counts-v2", "schema mismatch")
    check(report.get("full_official_contract") is False, "expected smoke contract")
    check(config.get("learned_response_weight") == expected_weight, "alpha mismatch")
    check(config.get("emission_contract_version") == "state-composition-v1", "emission contract mismatch")
    check(config.get("count_emission") == "exact-largest-remainder", "count emitter mismatch")
    check(config.get("library_policy") == "exact-source-library-composition", "library policy mismatch")
    check(config.get("max_genes_per_cell") == 5900, "5900-gene cap mismatch")
    check(config.get("target_policy") == "force", "target policy mismatch")
    check(config.get("target_max_remaining_fraction") == 0.20, "target force mismatch")
    check(config.get("common_response_weight") == 0.0, "common response must be disabled")
    check(config.get("pseudobulk_blend_weight") == 0.0, "pseudobulk blend must be disabled")
    check(config.get("zero_induction_scale") == 0.0, "zero induction must be disabled")
    check(scorer.get("target_force_order") == "after-combined-effect-clip-before-composition-normalization", "target-force order mismatch")
    check(scorer.get("combined_effect_qc_includes_target_force") is False, "combined-effect QC stage mismatch")
    check(validation.get("shape") == [768, 18533], "output shape mismatch")
    check(validation.get("groups") == 24, "group count mismatch")
    check(validation.get("failed_checks") == [], "official structural checks failed")
    check(scientific.get("all_libraries_exact") is True, "library exactness failed")
    check(scientific.get("all_current_only_counts_exact") is True, "current-only preservation failed")
    check(scientific.get("target_policy_failures") == [], "target-policy QC failed")
    check(scientific.get("final_target_delta_failures") == [], "final target-delta QC failed")
    check(scientific.get("induced_zero_emitted_entries") == 0, "source-zero induction occurred")
    check(scientific.get("dropped_counts_without_redistribution") == 0, "counts were dropped")
    check((scientific.get("count_mass") or {}).get("passed") is True, "count-mass QC failed")
    check(pathlib.Path(response.get("path", "")).resolve() == response_path, "response path mismatch")
    check(response.get("sha256") == response_sha, "response hash mismatch")
    check(script.get("sha256") == script_sha, "generator script is stale")
    groups = (report.get("qc") or {}).get("groups", [])
    check(len(groups) == 24, "group QC count mismatch")
    forced_delta = float(np.log(0.20))
    for index, group in enumerate(groups):
        prefix = f"group {index}"
        check(group.get("library_size_mismatches") == 0, f"{prefix}: library mismatch")
        check(group.get("current_only_count_mismatches") == 0, f"{prefix}: current-only mismatch")
        normalization = group.get("composition_normalization_factor") or {}
        check(normalization.get("count") == 32, f"{prefix}: normalization count mismatch")
        check(np.isfinite(normalization.get("min", np.nan)) and normalization.get("min", 0) > 0, f"{prefix}: invalid normalization")
        target_delta = group.get("final_target_delta") or {}
        check(target_delta.get("count") == 32, f"{prefix}: target-delta count mismatch")
        check(np.isclose(target_delta.get("min", np.nan), forced_delta, rtol=0, atol=1e-7), f"{prefix}: target-delta minimum mismatch")
        check(np.isclose(target_delta.get("max", np.nan), forced_delta, rtol=0, atol=1e-7), f"{prefix}: target-delta maximum mismatch")
    return violations


anchor_path = pathlib.Path(sys.argv[1]).resolve()
anchor_report_path = pathlib.Path(sys.argv[2]).resolve()
candidate_inputs = (
    (0.10, pathlib.Path(sys.argv[3]).resolve(), pathlib.Path(sys.argv[4]).resolve(), int(sys.argv[5])),
    (0.20, pathlib.Path(sys.argv[6]).resolve(), pathlib.Path(sys.argv[7]).resolve(), int(sys.argv[8])),
)
response_path = pathlib.Path(sys.argv[9]).resolve()
summary_path = pathlib.Path(sys.argv[10]).resolve()
script_path = pathlib.Path(sys.argv[11]).resolve()
summary = json.loads(summary_path.read_text(encoding="utf-8"))
summary.update(
    {
        "anchor_h5ad": str(anchor_path),
        "anchor_report": str(anchor_report_path),
        "response_artifact": str(response_path),
        "clip_fraction_limit": 1e-4,
        "minimum_direct_group_materialization_fraction": 0.90,
    }
)
exit_code = 0
try:
    response_sha = sha256_file(response_path)
    script_sha = sha256_file(script_path)
    with np.load(response_path, allow_pickle=False) as response:
        direct_mask = np.asarray(response["direct_mask"], dtype=bool)
        fallback_mask = np.asarray(response["fallback_mask"], dtype=bool)
        protected_mask = np.asarray(response["protected_current_only_mask"], dtype=bool)
        targets = np.asarray(response["targets"]).astype(str)
    if direct_mask.shape != (300,) or fallback_mask.shape != (300,):
        raise RuntimeError("Response target masks do not have the official shape")
    if not np.all(np.logical_xor(direct_mask, fallback_mask)):
        raise RuntimeError("Response direct/fallback masks do not partition targets")

    anchor_report = json.loads(anchor_report_path.read_text(encoding="utf-8"))
    anchor_violations = exact_report_violations(
        anchor_report, 0.0, response_path, response_sha, script_sha
    )
    if anchor_report.get("provenance", {}).get("output_h5ad", {}).get("sha256") != sha256_file(anchor_path):
        anchor_violations.append("anchor output hash mismatch")
    summary["anchor_contract_violations"] = anchor_violations
    if anchor_violations:
        summary["status"] = "invalid_anchor"
        summary["evaluation_completed"] = False
        exit_code = 5
    else:
        anchor = ad.read_h5ad(anchor_path)
        anchor_matrix = sparse_matrix(anchor)
        expected_contexts = np.repeat(np.asarray(["A", "B", "C"]), 8 * 32)
        expected_targets = np.tile(np.repeat(targets[:8], 32), 3)
        if not np.array_equal(np.asarray(anchor.obs["context"].astype(str)), expected_contexts):
            raise RuntimeError("Anchor context/group order is not canonical")
        if not np.array_equal(np.asarray(anchor.obs["target_gene"].astype(str)), expected_targets):
            raise RuntimeError("Anchor target/group order is not canonical")
        group_is_direct = np.tile(direct_mask[:8], 3)
        summary["candidates"] = []
        hard_error = False
        for weight, h5ad_path, report_path, generation_status in candidate_inputs:
            record = {
                "residual_weight": weight,
                "generation_exit_code": generation_status,
                "output_h5ad": str(h5ad_path),
                "output_report": str(report_path),
                "status": "pending",
                "structural_violations": [],
                "rejection_reasons": [],
            }
            if generation_status != 0:
                record["status"] = "generation_error"
                record["structural_violations"].append("generator returned a non-zero exit code")
                hard_error = True
                summary["candidates"].append(record)
                continue
            missing = [str(path) for path in (h5ad_path, report_path) if not path.is_file()]
            if missing:
                record["status"] = "artifact_missing"
                record["structural_violations"].append("missing outputs: " + ", ".join(missing))
                hard_error = True
                summary["candidates"].append(record)
                continue
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                record["structural_violations"].extend(
                    exact_report_violations(
                        report, weight, response_path, response_sha, script_sha
                    )
                )
                output_hash = sha256_file(h5ad_path)
                record["output_sha256"] = output_hash
                if report.get("provenance", {}).get("output_h5ad", {}).get("sha256") != output_hash:
                    record["structural_violations"].append("output hash mismatch")

                candidate = ad.read_h5ad(h5ad_path)
                matrix = sparse_matrix(candidate)
                if not candidate.obs_names.equals(anchor.obs_names):
                    record["structural_violations"].append("observation axis mismatch")
                if not candidate.var_names.equals(anchor.var_names):
                    record["structural_violations"].append("gene axis mismatch")
                for column in ("context", "target_gene", "source_control_row"):
                    if not candidate.obs[column].equals(anchor.obs[column]):
                        record["structural_violations"].append(f"{column} mismatch")
                if matrix.shape != anchor_matrix.shape:
                    record["structural_violations"].append("matrix shape mismatch")
                else:
                    changed = matrix != anchor_matrix
                    changed_per_row = np.asarray(changed.sum(axis=1)).ravel()
                    changed_by_group = changed_per_row.reshape(24, 32).sum(axis=1)
                    fallback_changes = int(changed_by_group[~group_is_direct].sum())
                    changed_direct_groups = int(np.count_nonzero(changed_by_group[group_is_direct]))
                    total_direct_groups = int(np.count_nonzero(group_is_direct))
                    materialization = changed_direct_groups / total_direct_groups
                    record.update(
                        {
                            "changed_entries_vs_anchor": int(changed.nnz),
                            "changed_cells_vs_anchor": int(np.count_nonzero(changed_per_row)),
                            "direct_groups_with_changes": changed_direct_groups,
                            "direct_groups_total": total_direct_groups,
                            "direct_group_materialization_fraction": float(materialization),
                            "fallback_changed_entries": fallback_changes,
                            "current_only_changed_entries": int(
                                (matrix[:, protected_mask] != anchor_matrix[:, protected_mask]).nnz
                            ),
                        }
                    )
                    if int(changed.nnz) == 0:
                        record["rejection_reasons"].append("candidate did not differ from alpha=0")
                    if fallback_changes != 0:
                        record["rejection_reasons"].append("fallback groups changed")
                    if record["current_only_changed_entries"] != 0:
                        record["rejection_reasons"].append("current-only genes changed")
                    if materialization < 0.90:
                        record["rejection_reasons"].append("insufficient direct-group materialization")
                    if not np.array_equal(
                        np.asarray(matrix.sum(axis=1)).ravel(),
                        np.asarray(anchor_matrix.sum(axis=1)).ravel(),
                    ):
                        record["structural_violations"].append("per-cell library mismatch vs anchor")

                groups = report.get("qc", {}).get("groups", [])
                outside = sum(int(group.get("combined_outside_bound", 0)) for group in groups)
                values = sum(int(group.get("effect_values", 0)) for group in groups)
                clip_fraction = float(outside / values) if values > 0 else float("inf")
                record["combined_outside_bound"] = outside
                record["combined_effect_values"] = values
                record["combined_clip_fraction"] = clip_fraction
                if not np.isfinite(clip_fraction) or clip_fraction >= 1e-4:
                    record["rejection_reasons"].append("combined clip fraction is not below 1e-4")
            except Exception as error:
                record["structural_violations"].append(
                    f"candidate validation raised {type(error).__name__}: {error}"
                )
                record["validation_traceback"] = traceback.format_exc()

            if record["structural_violations"]:
                record["status"] = "invalid"
                hard_error = True
            elif record["rejection_reasons"]:
                record["status"] = "rejected"
            else:
                record["status"] = "accepted"
            summary["candidates"].append(record)

        summary["evaluation_completed"] = not hard_error
        summary["status"] = "completed" if not hard_error else "invalid_candidate_artifact"
        summary["accepted_weights"] = [
            item["residual_weight"] for item in summary["candidates"]
            if item["status"] == "accepted"
        ]
        summary["rejected_weights"] = [
            item["residual_weight"] for item in summary["candidates"]
            if item["status"] == "rejected"
        ]
        if hard_error:
            exit_code = 6
except Exception as error:
    summary["status"] = "validation_error"
    summary["evaluation_completed"] = False
    summary["error"] = f"{type(error).__name__}: {error}"
    summary["traceback"] = traceback.format_exc()
    exit_code = 7
finally:
    atomic_write_json(summary_path, summary)

print(json.dumps(summary, indent=2, sort_keys=True))
sys.exit(exit_code)
PY
validation_status=$?
set -e

date --iso-8601=seconds
echo "Grid summary: $SUMMARY_JSON"
exit "$validation_status"
