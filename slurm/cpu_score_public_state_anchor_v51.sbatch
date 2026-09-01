#!/usr/bin/env bash
#SBATCH --job-name=vcc-public-v51-score
#SBATCH --account=allaccess
#SBATCH --partition=full
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/cpu_public_v51_score_%A_%a.out
#SBATCH --error=logs/cpu_public_v51_score_%A_%a.err

set -euo pipefail
unset PYTHONOPTIMIZE

readonly PROJECT_DIR="${VCC_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
readonly PYTHON_BIN="$PROJECT_DIR/.venv-state/bin/python"
readonly FULL_DIR="$PROJECT_DIR/artifacts/public_v51/full"
readonly SCORE_ROOT="$PROJECT_DIR/artifacts/public_v51/scoring"
readonly FULL_SUMMARY="$FULL_DIR/public_state_anchor_v51_full_summary.json"
readonly CELL_EVAL_DIR="$PROJECT_DIR/external/cell-eval2"
readonly EXPECTED_CELL_EVAL_COMMIT="5e64833518a6603a0301cbe28185d49c30f4a986"
readonly EXPECTED_CELL_EVAL_VERSION="0.16.0"
readonly EXPECTED_PDEX_VERSION="0.3.0"
case "${VCC_RESUME_AUTHENTICATED_ANCHOR:-0}" in
  0|1) readonly RESUME_AUTHENTICATED_ANCHOR="${VCC_RESUME_AUTHENTICATED_ANCHOR:-0}" ;;
  *)
    echo "VCC_RESUME_AUTHENTICATED_ANCHOR must be 0 or 1" >&2
    exit 2
    ;;
esac

case "${SLURM_ARRAY_TASK_ID:-}" in
  0)
    readonly CONTEXT="HepG2"
    readonly PREFIX="hepg2"
    readonly EXPECTED_TRUTH_CELLS=40049
    ;;
  1)
    readonly CONTEXT="Jurkat"
    readonly PREFIX="jurkat"
    readonly EXPECTED_TRUTH_CELLS=73506
    ;;
  *)
    echo "SLURM_ARRAY_TASK_ID must be 0 or 1" >&2
    exit 2
    ;;
esac

readonly CONTEXT_DIR="$SCORE_ROOT/$PREFIX"
readonly CACHE_REAL="$CONTEXT_DIR/cache_real"
readonly SCORING_RECEIPT="$CONTEXT_DIR/scoring_receipt.json"
readonly CONTROLS="$PROJECT_DIR/dataset/public_v51/${PREFIX}_controls_only.h5ad"
readonly SEALED_TRUTH="$PROJECT_DIR/dataset/public_v51/${PREFIX}_sealed_truth.h5ad"
readonly ANCHOR_PREDICTION="$FULL_DIR/${PREFIX}_anchor.h5ad"
readonly V51_PREDICTION="$FULL_DIR/${PREFIX}_v51_alpha010.h5ad"

cd "$PROJECT_DIR"
mkdir -p "$CONTEXT_DIR" logs
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR/external/cell-eval2/src${PYTHONPATH:+:$PYTHONPATH}"

for required_path in \
  "$PYTHON_BIN" \
  "$FULL_SUMMARY" \
  scripts/prepare_public_scoring_views.py \
  dataset/public_v51/common_genes.csv \
  "$CELL_EVAL_DIR/pyproject.toml" \
  "$CELL_EVAL_DIR/src/cell_eval2/cli.py" \
  "$ANCHOR_PREDICTION" \
  "$V51_PREDICTION" \
  "$CONTROLS" \
  "$SEALED_TRUTH"; do
  if [[ ! -r "$required_path" ]]; then
    echo "Missing required input: $required_path" >&2
    exit 3
  fi
done
if [[ ! -d "$CELL_EVAL_DIR/.git" ]]; then
  echo "cell-eval2 checkout has no Git metadata" >&2
  exit 3
fi

if [[ "$RESUME_AUTHENTICATED_ANCHOR" == 1 ]]; then
  for required_resume_path in \
    "$SCORING_RECEIPT" \
    "$CACHE_REAL/manifest.json" \
    "$CONTEXT_DIR/anchor_prediction_view.h5ad" \
    "$CONTEXT_DIR/anchor_truth_view.h5ad" \
    "$CONTEXT_DIR/anchor_views.json" \
    "$CONTEXT_DIR/anchor_cell_eval2/results.csv" \
    "$CONTEXT_DIR/anchor_cell_eval2/agg_results.csv" \
    "$CONTEXT_DIR/anchor_cell_eval2/metric_aggregation.csv" \
    "$CONTEXT_DIR/anchor_cell_eval2/run_meta.json"; do
    if [[ ! -r "$required_resume_path" ]]; then
      echo "Authenticated resume input is missing: $required_resume_path" >&2
      exit 4
    fi
  done
  for forbidden_candidate_path in \
    "$CONTEXT_DIR/v51_alpha010_prediction_view.h5ad" \
    "$CONTEXT_DIR/v51_alpha010_truth_view.h5ad" \
    "$CONTEXT_DIR/v51_alpha010_views.json" \
    "$CONTEXT_DIR/v51_alpha010_cell_eval2"; do
    if [[ -e "$forbidden_candidate_path" ]]; then
      echo "Refusing candidate-only resume over an existing candidate artifact: $forbidden_candidate_path" >&2
      exit 4
    fi
  done
else
  for forbidden_path in "$CACHE_REAL" "$SCORING_RECEIPT"; do
    if [[ -e "$forbidden_path" ]]; then
      echo "Refusing to reuse an unstamped scoring artifact: $forbidden_path" >&2
      exit 4
    fi
  done
  for candidate in anchor v51_alpha010; do
    for output_path in \
      "$CONTEXT_DIR/${candidate}_prediction_view.h5ad" \
      "$CONTEXT_DIR/${candidate}_truth_view.h5ad" \
      "$CONTEXT_DIR/${candidate}_views.json" \
      "$CONTEXT_DIR/${candidate}_cell_eval2"; do
      if [[ -e "$output_path" ]]; then
        echo "Refusing to reuse a scoring artifact: $output_path" >&2
        exit 4
      fi
    done
  done
fi

actual_cell_eval_commit="$(git -C "$CELL_EVAL_DIR" rev-parse HEAD)"
readonly actual_cell_eval_commit
if [[ "$actual_cell_eval_commit" != "$EXPECTED_CELL_EVAL_COMMIT" ]]; then
  echo "cell-eval2 HEAD is not the pinned scoring revision" >&2
  exit 4
fi
if [[ -n "$(git -C "$CELL_EVAL_DIR" status --porcelain --untracked-files=normal)" ]]; then
  echo "cell-eval2 checkout is dirty" >&2
  exit 4
fi

"$PYTHON_BIN" - \
  "$CELL_EVAL_DIR" "$EXPECTED_CELL_EVAL_VERSION" "$EXPECTED_PDEX_VERSION" <<'PY'
import importlib.metadata
import json
import pathlib
import sys
import tomllib

import cell_eval2

checkout = pathlib.Path(sys.argv[1]).resolve()
declared_expected = sys.argv[2]
pdex_expected = sys.argv[3]
module_path = pathlib.Path(cell_eval2.__file__).resolve()
if not module_path.is_relative_to(checkout / "src"):
    raise RuntimeError(f"cell_eval2 did not import from the pinned checkout: {module_path}")
with (checkout / "pyproject.toml").open("rb") as handle:
    declared = tomllib.load(handle)["project"]["version"]
if declared != declared_expected:
    raise RuntimeError(f"Unexpected cell-eval2 declared version: {declared}")
pdex_version = importlib.metadata.version("pdex")
if pdex_version != pdex_expected:
    raise RuntimeError(f"Unexpected pdex version: {pdex_version}")
print(json.dumps({
    "cell_eval2_declared_version": declared,
    "cell_eval2_runtime_version": cell_eval2.__version__,
    "cell_eval2_module": str(module_path),
    "pdex_version": pdex_version,
}, indent=2, sort_keys=True))
PY

# Authenticate the large predictions once, before preparing any truth-side views.
"$PYTHON_BIN" - \
  "$FULL_SUMMARY" "$CONTEXT" "$ANCHOR_PREDICTION" "$V51_PREDICTION" \
  "$SCORING_RECEIPT" "$EXPECTED_CELL_EVAL_COMMIT" \
  "$EXPECTED_CELL_EVAL_VERSION" "$EXPECTED_PDEX_VERSION" \
  "$RESUME_AUTHENTICATED_ANCHOR" <<'PY'
import hashlib
import importlib.metadata
import json
import os
import pathlib
import re
import sys
import tempfile
import tomllib

import cell_eval2
import numpy as np


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path, payload):
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as handle:
        temporary = pathlib.Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


summary_path = pathlib.Path(sys.argv[1]).resolve()
context = sys.argv[2]
anchor_path = pathlib.Path(sys.argv[3]).resolve()
candidate_path = pathlib.Path(sys.argv[4]).resolve()
receipt_path = pathlib.Path(sys.argv[5]).resolve()
commit = sys.argv[6]
declared_expected = sys.argv[7]
pdex_expected = sys.argv[8]
resume_authenticated_anchor = sys.argv[9] == "1"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
if set(summary) != {"schema", "contexts"}:
    raise RuntimeError("Unexpected full-generation summary top-level fields")
if summary.get("schema") != "vcc-public-state-anchor-full-v1":
    raise RuntimeError("Bad full-generation summary schema")
contexts = summary.get("contexts")
if not isinstance(contexts, dict) or set(contexts) != {"HepG2", "Jurkat"}:
    raise RuntimeError("Full-generation summary context set is incomplete")
expected_context_fields = {
    "clip_fraction",
    "direct_changed_entries",
    "fallback_changed_entries",
    "anchor_sha256",
    "candidate_sha256",
}
for summary_context, record in contexts.items():
    if not isinstance(record, dict) or set(record) != expected_context_fields:
        raise RuntimeError(f"{summary_context}: malformed full-generation context record")
    clip_fraction = record["clip_fraction"]
    direct_changed = record["direct_changed_entries"]
    fallback_changed = record["fallback_changed_entries"]
    if (
        isinstance(clip_fraction, bool)
        or not isinstance(clip_fraction, (int, float))
        or not np.isfinite(float(clip_fraction))
        or not 0.0 <= float(clip_fraction) < 1e-4
    ):
        raise RuntimeError(f"{summary_context}: invalid clip-fraction gate")
    if (
        isinstance(direct_changed, bool)
        or not isinstance(direct_changed, int)
        or direct_changed <= 0
    ):
        raise RuntimeError(f"{summary_context}: invalid direct-change gate")
    if (
        isinstance(fallback_changed, bool)
        or not isinstance(fallback_changed, int)
        or fallback_changed != 0
    ):
        raise RuntimeError(f"{summary_context}: held-target output is not identical")
    for hash_field in ("anchor_sha256", "candidate_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(record[hash_field])):
            raise RuntimeError(f"{summary_context}: malformed {hash_field}")
context_summary = contexts[context]
anchor_sha = sha256_file(anchor_path)
candidate_sha = sha256_file(candidate_path)
if anchor_sha != context_summary.get("anchor_sha256"):
    raise RuntimeError("Anchor H5AD hash differs from the full-generation summary")
if candidate_sha != context_summary.get("candidate_sha256"):
    raise RuntimeError("Candidate H5AD hash differs from the full-generation summary")
checkout = pathlib.Path("external/cell-eval2").resolve()
with (checkout / "pyproject.toml").open("rb") as handle:
    declared = tomllib.load(handle)["project"]["version"]
pdex_version = importlib.metadata.version("pdex")
if declared != declared_expected or pdex_version != pdex_expected:
    raise RuntimeError("Scoring package version drifted after preflight")
payload = {
    "schema": "vcc-public-validation-scoring-receipt-v1",
    "status": "inputs_authenticated",
    "context": context,
    "full_generation_summary": {
        "path": str(summary_path),
        "sha256": sha256_file(summary_path),
    },
    "predictions": {
        "anchor": {"path": str(anchor_path), "sha256": anchor_sha},
        "v51_alpha010": {"path": str(candidate_path), "sha256": candidate_sha},
    },
    "generation_qc": context_summary,
    "scoring_software": {
        "cell_eval2_git_commit": commit,
        "cell_eval2_checkout_clean": True,
        "cell_eval2_declared_version": declared,
        "cell_eval2_runtime_version": cell_eval2.__version__,
        "cell_eval2_module": str(pathlib.Path(cell_eval2.__file__).resolve()),
        "pdex_version": pdex_version,
    },
}
if resume_authenticated_anchor:
    existing = json.loads(receipt_path.read_text(encoding="utf-8"))
    if existing != payload:
        raise RuntimeError(
            "Existing input-authentication receipt does not exactly match current inputs"
        )
else:
    atomic_write(receipt_path, payload)
print(json.dumps(payload, indent=2, sort_keys=True))
PY

date --iso-8601=seconds
hostname

for candidate in anchor v51_alpha010; do
  prediction="$FULL_DIR/${PREFIX}_${candidate}.h5ad"
  pred_view="$CONTEXT_DIR/${candidate}_prediction_view.h5ad"
  truth_view="$CONTEXT_DIR/${candidate}_truth_view.h5ad"
  views_json="$CONTEXT_DIR/${candidate}_views.json"
  score_dir="$CONTEXT_DIR/${candidate}_cell_eval2"
  if [[ ! -r "$prediction" ]]; then
    echo "Missing generated prediction: $prediction" >&2
    exit 5
  fi
  if [[ "$RESUME_AUTHENTICATED_ANCHOR" == 1 && "$candidate" == anchor ]]; then
    echo "Authenticating and reusing completed anchor score for $CONTEXT"
  else
    for output_path in "$pred_view" "$truth_view" "$views_json" "$score_dir"; do
      if [[ -e "$output_path" ]]; then
        echo "Refusing to overwrite scoring artifact: $output_path" >&2
        exit 6
      fi
    done

    "$PYTHON_BIN" scripts/prepare_public_scoring_views.py \
      "$prediction" \
      "$CONTROLS" \
      "$SEALED_TRUTH" \
      dataset/public_v51/common_genes.csv \
      "$pred_view" \
      "$truth_view" \
      "$views_json" \
      --chunk-rows 2048 \
      --compression lzf
  fi

  "$PYTHON_BIN" - \
    "$views_json" "$CONTEXT" "$EXPECTED_TRUTH_CELLS" "$candidate" \
    "$SCORING_RECEIPT" "$prediction" "$pred_view" "$truth_view" <<'PY'
import hashlib
import json
import pathlib
import sys


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


report = json.load(open(sys.argv[1], encoding="utf-8"))
candidate = sys.argv[4]
receipt = json.load(open(sys.argv[5], encoding="utf-8"))
prediction = pathlib.Path(sys.argv[6]).resolve()
prediction_view = pathlib.Path(sys.argv[7]).resolve()
truth_view = pathlib.Path(sys.argv[8]).resolve()
provenance = report.get("provenance") or {}
prediction_stamp = provenance.get("prediction") or {}
prediction_receipt = (receipt.get("predictions") or {}).get(candidate) or {}
output_prediction = provenance.get("output_prediction") or {}
output_truth = provenance.get("output_truth") or {}
checks = {
    "schema": report.get("schema") == "vcc-public-scoring-views-v1",
    "context": report.get("context") == sys.argv[2],
    "genes": (report.get("axes") or {}).get("genes") == 7107,
    "targets": (report.get("axes") or {}).get("targets") == 300,
    "prediction_treated_cells": (
        (report.get("axes") or {}).get("prediction_treated_cells") == 120000
    ),
    "truth_view_cells": (
        (report.get("axes") or {}).get("truth_view_cells") == int(sys.argv[3])
    ),
    "truth_firewall": (
        (report.get("contract") or {}).get("model_read_treated_truth") is False
    ),
    "input_type": (
        (report.get("contract") or {}).get("input_type")
        == "raw-nonnegative-integer-counts"
    ),
    "prediction_path": pathlib.Path(str(prediction_stamp.get("path", ""))).resolve()
    == prediction,
    "prediction_hash_matches_receipt": prediction_stamp.get("sha256")
    == prediction_receipt.get("sha256"),
    "prediction_view_path": pathlib.Path(
        str(output_prediction.get("path", ""))
    ).resolve()
    == prediction_view,
    "truth_view_path": pathlib.Path(str(output_truth.get("path", ""))).resolve()
    == truth_view,
    "prediction_view_hash": prediction_view.is_file()
    and sha256_file(prediction_view) == output_prediction.get("sha256"),
    "truth_view_hash": truth_view.is_file()
    and sha256_file(truth_view) == output_truth.get("sha256"),
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise RuntimeError(f"Scoring-view QC failed: {failed}")
PY

  if [[ ! ( "$RESUME_AUTHENTICATED_ANCHOR" == 1 && "$candidate" == anchor ) ]]; then
    "$PYTHON_BIN" -c 'from cell_eval2.cli import main; main()' run \
      --adata-pred "$pred_view" \
      --adata-real "$truth_view" \
      --preset vcc2026 \
      --input-type counts \
      --pert-col target_gene \
      --control non-targeting \
      --cache-real "$CACHE_REAL" \
      --set de.backend=pdex \
      --set device=cpu \
      --set num_threads=16 \
      --set gather_threads=16 \
      --outdir "$score_dir"
  fi

  "$PYTHON_BIN" - \
    "$score_dir/results.csv" dataset/public_v51/split_manifest.csv \
    "$score_dir/run_meta.json" "$CACHE_REAL/manifest.json" "$truth_view" <<'PY'
import json
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path("scripts").resolve()))
from summarize_public_validation import (  # noqa: E402
    MSE_METRIC,
    _mean_summary,
    _read_results,
    _validate_cell_eval_sidecars,
)

path = sys.argv[1]
manifest_path = sys.argv[2]
run_meta_path = sys.argv[3]
cache_manifest_path = pathlib.Path(sys.argv[4]).resolve()
truth_view_path = pathlib.Path(sys.argv[5]).resolve()
run_meta = json.load(open(run_meta_path, encoding="utf-8"))
if run_meta.get("resolved_device") != "cpu":
    raise RuntimeError("cell-eval2 did not resolve to CPU")
if run_meta.get("resolved_de_backend") != "pdex":
    raise RuntimeError("cell-eval2 did not resolve to pdex")
if run_meta.get("source_fingerprint_strict") is not True:
    raise RuntimeError("cell-eval2 strict source fingerprinting is disabled")
if run_meta.get("input_type_real_effective") != "counts":
    raise RuntimeError("cell-eval2 real input type is not counts")
if run_meta.get("input_type_pred_effective") != "counts":
    raise RuntimeError("cell-eval2 prediction input type is not counts")
if run_meta.get("comparator") != "bulk_lognorm":
    raise RuntimeError("cell-eval2 did not resolve the vcc2026 count comparator")
pdex = (run_meta.get("environment") or {}).get("pdex") or {}
if pdex.get("version") != "0.3.0":
    raise RuntimeError("run_meta does not record pdex 0.3.0")
config_digest = run_meta.get("config_digest")
if not isinstance(config_digest, str) or not config_digest:
    raise RuntimeError("run_meta has no scorer config digest")
if pathlib.Path(str(run_meta.get("source", ""))).resolve() != truth_view_path:
    raise RuntimeError("run_meta real-side source differs from the authenticated truth view")
manifest = pd.read_csv(manifest_path)
if "target_gene" not in manifest.columns:
    raise RuntimeError("Validation manifest lacks target_gene")
manifest_targets = manifest["target_gene"].astype(str)
if len(manifest_targets) != 300 or not manifest_targets.is_unique:
    raise RuntimeError("Validation manifest does not define 300 unique targets")
targets = set(manifest_targets)
cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
if cache_manifest.get("cache_format_version") != 2:
    raise RuntimeError("Real-side cache manifest version mismatch")
cache_artifacts = cache_manifest.get("artifacts") or {}
source_fingerprint = run_meta.get("source_fingerprint")
for cache_key in ("de_wilcoxon_table", "pseudobulk_moments_bulk_lognorm"):
    cache_record = cache_artifacts.get(cache_key) or {}
    if cache_record.get("fingerprint") != source_fingerprint:
        raise RuntimeError(f"Real-side cache fingerprint mismatch: {cache_key}")
    cache_file = cache_manifest_path.parent / str(cache_record.get("filename", ""))
    if not cache_file.is_file():
        raise RuntimeError(f"Real-side cache artifact is missing: {cache_key}")
frame = _read_results(pathlib.Path(path), targets)
summary = _mean_summary(frame, len(targets))
validation = _validate_cell_eval_sidecars(
    pathlib.Path(path), frame, summary, len(targets)
)
print(json.dumps({"mse_ratio_of_sums": summary[MSE_METRIC], **validation}, indent=2))
PY
done

"$PYTHON_BIN" - \
  "$CONTEXT_DIR/anchor_cell_eval2/results.csv" \
  "$CONTEXT_DIR/v51_alpha010_cell_eval2/results.csv" <<'PY'
import sys

import pandas as pd

metric = "de_wilcoxon_lfc_nmae"
target_sets = []
for path in sys.argv[1:]:
    results = pd.read_csv(path)
    subset = results.loc[results["metric"].astype(str) == metric, "perturbation"]
    targets = frozenset(subset.astype(str))
    if not targets:
        raise RuntimeError(f"NMAE has no scored targets: {path}")
    target_sets.append(targets)
if target_sets[0] != target_sets[1]:
    raise RuntimeError("Anchor and candidate NMAE target subsets differ")
print(f"Matched NMAE target subset: {len(target_sets[0])} targets")
PY

"$PYTHON_BIN" - \
  "$SCORING_RECEIPT" "$CONTEXT" \
  "$CONTEXT_DIR/anchor_views.json" "$CONTEXT_DIR/v51_alpha010_views.json" \
  "$CONTEXT_DIR/anchor_cell_eval2" "$CONTEXT_DIR/v51_alpha010_cell_eval2" <<'PY'
import hashlib
import json
import os
import pathlib
import sys
import tempfile


def descriptor(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def atomic_write(path, payload):
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as handle:
        temporary = pathlib.Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


receipt_path = pathlib.Path(sys.argv[1]).resolve()
context = sys.argv[2]
view_paths = [pathlib.Path(value).resolve() for value in sys.argv[3:5]]
score_dirs = [pathlib.Path(value).resolve() for value in sys.argv[5:7]]
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
if receipt.get("schema") != "vcc-public-validation-scoring-receipt-v1":
    raise RuntimeError("Bad scoring receipt schema")
if receipt.get("status") != "inputs_authenticated" or receipt.get("context") != context:
    raise RuntimeError("Scoring receipt input authentication is incomplete")
labels = ("anchor", "v51_alpha010")
run_meta = {}
outputs = {}
for label, view_path, score_dir in zip(labels, view_paths, score_dirs, strict=True):
    views = json.loads(view_path.read_text(encoding="utf-8"))
    if views.get("context") != context:
        raise RuntimeError(f"{label}: scoring-view context mismatch")
    meta_path = score_dir / "run_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("resolved_device") != "cpu" or meta.get("resolved_de_backend") != "pdex":
        raise RuntimeError(f"{label}: resolved scorer identity mismatch")
    if meta.get("source_fingerprint_strict") is not True:
        raise RuntimeError(f"{label}: strict scorer fingerprint is disabled")
    pdex_version = ((meta.get("environment") or {}).get("pdex") or {}).get("version")
    if pdex_version != "0.3.0":
        raise RuntimeError(f"{label}: pdex version differs from 0.3.0")
    config_digest = meta.get("config_digest")
    if not isinstance(config_digest, str) or not config_digest:
        raise RuntimeError(f"{label}: scorer config digest is absent")
    run_meta[label] = {
        "config_digest": config_digest,
        "resolved_device": meta.get("resolved_device"),
        "resolved_de_backend": meta.get("resolved_de_backend"),
        "cell_eval2_version": meta.get("cell_eval2_version"),
        "pdex_version": pdex_version,
    }
    outputs[label] = {
        "views_report": descriptor(view_path),
        "results": descriptor(score_dir / "results.csv"),
        "aggregate_results": descriptor(score_dir / "agg_results.csv"),
        "metric_aggregation": descriptor(score_dir / "metric_aggregation.csv"),
        "run_meta": descriptor(meta_path),
    }
if run_meta["anchor"]["config_digest"] != run_meta["v51_alpha010"]["config_digest"]:
    raise RuntimeError("Anchor and candidate cell-eval2 config digests differ")
receipt.update({
    "status": "scoring_complete",
    "resolved_scoring": run_meta,
    "outputs": outputs,
})
atomic_write(receipt_path, receipt)
print(json.dumps(receipt, indent=2, sort_keys=True))
PY

date --iso-8601=seconds
