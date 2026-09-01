#!/usr/bin/env bash
#SBATCH --job-name=vcc-public-v6-score
#SBATCH --account=allaccess
#SBATCH --partition=full
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=03:00:00
#SBATCH --output=logs/cpu_public_v6_score_%j.out
#SBATCH --error=logs/cpu_public_v6_score_%j.err

set -euo pipefail
unset PYTHONOPTIMIZE

: "${VCC_CONTEXT:?Set VCC_CONTEXT to HepG2 or Jurkat}"
: "${VCC_OUTPUT_TAG:?Set VCC_OUTPUT_TAG to the generated candidate tag}"

readonly PROJECT_DIR="${VCC_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
readonly PYTHON_BIN="$PROJECT_DIR/.venv-state/bin/python"
readonly PUBLIC_ROOT="${VCC_PUBLIC_V6_ROOT:-$PROJECT_DIR/artifacts/public_v6}"
readonly CELL_EVAL_CHECKOUT="$PROJECT_DIR/external/cell-eval2"
readonly EXPECTED_CELL_EVAL_COMMIT="5e64833518a6603a0301cbe28185d49c30f4a986"
readonly EXPECTED_CELL_EVAL_VERSION="0.16.0"
readonly EXPECTED_PDEX_VERSION="0.3.0"

if [[ ! "$VCC_OUTPUT_TAG" =~ ^[a-z0-9][a-z0-9._-]{0,63}$ ]]; then
  echo "VCC_OUTPUT_TAG is unsafe" >&2
  exit 2
fi
readonly RESOLVED_PUBLIC_ROOT="$(realpath -m "$PUBLIC_ROOT")"
readonly V51_ROOT="$(realpath -m "$PROJECT_DIR/artifacts/public_v51")"
case "$RESOLVED_PUBLIC_ROOT" in
  "$V51_ROOT"|"$V51_ROOT"/*)
    echo "The V6 output root must not overlap artifacts/public_v51" >&2
    exit 2
    ;;
esac

case "$VCC_CONTEXT" in
  HepG2) readonly CONTEXT_PREFIX="hepg2"; readonly EXPECTED_TRUTH_CELLS=40049 ;;
  Jurkat) readonly CONTEXT_PREFIX="jurkat"; readonly EXPECTED_TRUTH_CELLS=73506 ;;
  *) echo "VCC_CONTEXT must be HepG2 or Jurkat" >&2; exit 2 ;;
esac

readonly RUN_DIR="$RESOLVED_PUBLIC_ROOT/candidates/$CONTEXT_PREFIX/$VCC_OUTPUT_TAG"
readonly SCORE_DIR="$RESOLVED_PUBLIC_ROOT/scoring/$CONTEXT_PREFIX/$VCC_OUTPUT_TAG"
readonly SPEC_JSON="$RUN_DIR/spec.json"
readonly PREDICTION="$RUN_DIR/prediction.h5ad"
readonly GENERATION_JSON="$RUN_DIR/generation.json"
readonly VERIFICATION_JSON="$RUN_DIR/generation_verified.json"
readonly CONTROLS="$PROJECT_DIR/dataset/public_v51/${CONTEXT_PREFIX}_controls_only.h5ad"
readonly SEALED_TRUTH="$PROJECT_DIR/dataset/public_v51/${CONTEXT_PREFIX}_sealed_truth.h5ad"
readonly PRED_VIEW="$SCORE_DIR/prediction_view.h5ad"
readonly TRUTH_VIEW="$SCORE_DIR/truth_view.h5ad"
readonly VIEWS_JSON="$SCORE_DIR/views.json"
readonly CELL_EVAL_DIR="$SCORE_DIR/cell_eval2"
readonly CACHE_REAL="$SCORE_DIR/cache_real"
readonly SCORE_VERIFICATION="$SCORE_DIR/scoring_verified.json"

cd "$PROJECT_DIR"
mkdir -p logs
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR/external/cell-eval2/src${PYTHONPATH:+:$PYTHONPATH}"

for required_path in \
  "$PYTHON_BIN" \
  scripts/prepare_public_scoring_views.py \
  scripts/validate_public_cell_eval_run.py \
  "$CELL_EVAL_CHECKOUT/pyproject.toml" \
  "$CELL_EVAL_CHECKOUT/src/cell_eval2/cli.py" \
  "$SPEC_JSON" \
  "$PREDICTION" \
  "$GENERATION_JSON" \
  "$VERIFICATION_JSON" \
  "$CONTROLS" \
  "$SEALED_TRUTH" \
  dataset/public_v51/common_genes.csv \
  dataset/public_v51/common_genes.json; do
  if [[ ! -r "$required_path" ]]; then
    echo "Missing verified scoring input: $required_path" >&2
    exit 3
  fi
done
if [[ ! -d "$CELL_EVAL_CHECKOUT/.git" ]]; then
  echo "cell-eval2 checkout lacks Git metadata" >&2
  exit 3
fi
if [[ -e "$SCORE_DIR" ]]; then
  echo "Refusing to overwrite isolated score directory: $SCORE_DIR" >&2
  exit 4
fi

actual_cell_eval_commit="$(git -C "$CELL_EVAL_CHECKOUT" rev-parse HEAD)"
readonly actual_cell_eval_commit
if [[ "$actual_cell_eval_commit" != "$EXPECTED_CELL_EVAL_COMMIT" ]]; then
  echo "cell-eval2 HEAD differs from the pinned scoring revision" >&2
  exit 4
fi
if [[ -n "$(git -C "$CELL_EVAL_CHECKOUT" status --porcelain --untracked-files=normal)" ]]; then
  echo "cell-eval2 checkout is dirty" >&2
  exit 4
fi

"$PYTHON_BIN" - \
  "$CELL_EVAL_CHECKOUT" "$EXPECTED_CELL_EVAL_VERSION" "$EXPECTED_PDEX_VERSION" <<'PY'
import importlib.metadata
import pathlib
import sys
import tomllib

import cell_eval2

checkout = pathlib.Path(sys.argv[1]).resolve()
expected_version = sys.argv[2]
expected_pdex = sys.argv[3]
module_path = pathlib.Path(cell_eval2.__file__).resolve()
if not module_path.is_relative_to(checkout / "src"):
    raise RuntimeError(f"cell_eval2 imported outside the pinned checkout: {module_path}")
with (checkout / "pyproject.toml").open("rb") as handle:
    declared_version = tomllib.load(handle)["project"]["version"]
if declared_version != expected_version:
    raise RuntimeError(f"Unexpected cell-eval2 version: {declared_version}")
pdex_version = importlib.metadata.version("pdex")
if pdex_version != expected_pdex:
    raise RuntimeError(f"Unexpected pdex version: {pdex_version}")
PY

# This check happens before the scorer-only truth path is passed to any tool.
"$PYTHON_BIN" - "$SPEC_JSON" "$GENERATION_JSON" "$VERIFICATION_JSON" "$PREDICTION" "$VCC_CONTEXT" "$VCC_OUTPUT_TAG" <<'PY'
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path.cwd() / "scripts"))
from public_candidate_v6_contract import (  # noqa: E402
    SPEC_SCHEMA,
    _validate_strict_spec_provenance,
)

spec_path, generation_path, verification_path, prediction_path = map(pathlib.Path, sys.argv[1:5])
context, output_tag = sys.argv[5:7]
spec = json.loads(spec_path.read_text(encoding="utf-8"))
generation = json.loads(generation_path.read_text(encoding="utf-8"))
verification = json.loads(verification_path.read_text(encoding="utf-8"))
_validate_strict_spec_provenance(spec)

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

checks = {
    # New score jobs accept only the strict v2 provenance contract. Existing
    # immutable v1 P2 artifacts remain available to read-only comparators via
    # an explicit legacy flag; they must never enter this production scorer.
    "spec_schema": spec.get("schema") == SPEC_SCHEMA,
    "strict_spec_provenance": (
        (spec.get("provenance_contract") or {}).get("schema")
        == "vcc-public-generator-provenance-v2"
        and (spec.get("provenance_contract") or {}).get("mode") == "strict"
    ),
    "verification_schema": (
        verification.get("schema") == "vcc-public-candidate-verification-v1"
    ),
    "verification_status": verification.get("status") == "passed",
    "verification_mode": verification.get("validation_mode") == "strict-v2",
    "verification_configuration": (
        verification.get("configuration") == spec.get("configuration")
    ),
    "context": spec.get("context") == verification.get("context") == context,
    "tag": spec.get("output_tag") == verification.get("output_tag") == output_tag,
    "truth_firewall": (
        (generation.get("data_firewall") or {}).get("sealed_treated_profiles_read") is False
    ),
    "prediction_report_hash": (
        ((generation.get("provenance") or {}).get("output_h5ad") or {}).get("sha256")
        == sha256_file(prediction_path)
    ),
    "spec_receipt_hash": (
        ((verification.get("provenance") or {}).get("spec") or {}).get("sha256")
        == sha256_file(spec_path)
    ),
    "generation_receipt_hash": (
        ((verification.get("provenance") or {}).get("generation_json") or {}).get("sha256")
        == sha256_file(generation_path)
    ),
    "prediction_receipt_hash": (
        ((verification.get("provenance") or {}).get("prediction_h5ad") or {}).get("sha256")
        == sha256_file(prediction_path)
    ),
    "verification_checks": bool(verification.get("checks"))
    and all(value is True for value in verification["checks"].values())
    and verification["checks"].get("strict_generator_provenance_authenticated")
    is True,
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise RuntimeError(f"Candidate pre-score authentication failed: {failed}")
PY

mkdir -p "$SCORE_DIR"
"$PYTHON_BIN" scripts/prepare_public_scoring_views.py \
  "$PREDICTION" \
  "$CONTROLS" \
  "$SEALED_TRUTH" \
  dataset/public_v51/common_genes.csv \
  "$PRED_VIEW" \
  "$TRUTH_VIEW" \
  "$VIEWS_JSON" \
  --chunk-rows 2048 \
  --compression lzf

"$PYTHON_BIN" - "$VIEWS_JSON" "$VCC_CONTEXT" "$EXPECTED_TRUTH_CELLS" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
checks = {
    "schema": report.get("schema") == "vcc-public-scoring-views-v1",
    "context": report.get("context") == sys.argv[2],
    "genes": (report.get("axes") or {}).get("genes") == 7107,
    "targets": (report.get("axes") or {}).get("targets") == 300,
    "prediction_cells": (
        (report.get("axes") or {}).get("prediction_treated_cells") == 120000
    ),
    "truth_cells": (
        (report.get("axes") or {}).get("truth_view_cells") == int(sys.argv[3])
    ),
    "truth_firewall": (
        (report.get("contract") or {}).get("model_read_treated_truth") is False
    ),
    "input_type": (
        (report.get("contract") or {}).get("input_type")
        == "raw-nonnegative-integer-counts"
    ),
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise RuntimeError(f"Scoring-view validation failed: {failed}")
PY

"$PYTHON_BIN" -c 'from cell_eval2.cli import main; main()' run \
  --adata-pred "$PRED_VIEW" \
  --adata-real "$TRUTH_VIEW" \
  --preset vcc2026 \
  --input-type counts \
  --pert-col target_gene \
  --control non-targeting \
  --cache-real "$CACHE_REAL" \
  --set de.backend=pdex \
  --set device=cpu \
  --set num_threads=16 \
  --set gather_threads=16 \
  --outdir "$CELL_EVAL_DIR"

"$PYTHON_BIN" scripts/validate_public_cell_eval_run.py \
  --context "$VCC_CONTEXT" \
  --output-tag "$VCC_OUTPUT_TAG" \
  --manifest dataset/public_v51/split_manifest.csv \
  --candidate-spec "$SPEC_JSON" \
  --generation-verification "$VERIFICATION_JSON" \
  --views-json "$VIEWS_JSON" \
  --results-csv "$CELL_EVAL_DIR/results.csv" \
  --run-meta "$CELL_EVAL_DIR/run_meta.json" \
  --cell-eval-checkout "$CELL_EVAL_CHECKOUT" \
  --expected-commit "$EXPECTED_CELL_EVAL_COMMIT" \
  --expected-version "$EXPECTED_CELL_EVAL_VERSION" \
  --expected-pdex "$EXPECTED_PDEX_VERSION" \
  --output-json "$SCORE_VERIFICATION"

date --iso-8601=seconds
