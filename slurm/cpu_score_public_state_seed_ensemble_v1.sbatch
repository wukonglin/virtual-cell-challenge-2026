#!/usr/bin/env bash
# Score the one pre-registered, truth-blind four-seed STATE ensemble locally.
# This launcher does not package or submit a challenge artifact.
#SBATCH --job-name=vcc-state-seed-ensemble-score-v1
#SBATCH --account=allaccess
#SBATCH --partition=full
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=03:00:00
#SBATCH --output=logs/cpu_state_seed_ensemble_score_v1_%j.out
#SBATCH --error=logs/cpu_state_seed_ensemble_score_v1_%j.err

set -euo pipefail
unset PYTHONOPTIMIZE
umask 027

readonly PROJECT_INPUT="${VCC_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PROJECT_DIR="$(realpath -- "$PROJECT_INPUT")"
readonly PROJECT_DIR
readonly PUBLIC_ROOT="$PROJECT_DIR/artifacts/anvil_state_reconstruction"
readonly CONTEXT="HepG2"
readonly CONTEXT_PREFIX="hepg2"
readonly OUTPUT_TAG="state_sm_anvil_reconstructed_hepg2_uniform4_mean_v1"
readonly PLAN="$PROJECT_DIR/configs/state/anvil_seed_ensemble_v1.json"
readonly ENSEMBLE_RUN="$PUBLIC_ROOT/ensembles/$CONTEXT_PREFIX/$OUTPUT_TAG"
readonly ENSEMBLE_VERIFICATION="$ENSEMBLE_RUN/ensemble_verified.json"
readonly PREDICTION="$ENSEMBLE_RUN/prediction.h5ad"
readonly SCORE_DIR="$PUBLIC_ROOT/ensemble_scoring/$CONTEXT_PREFIX/$OUTPUT_TAG"
readonly SCORE_PARENT="$PUBLIC_ROOT/ensemble_scoring/$CONTEXT_PREFIX"
readonly PRESCORE_AUTH="$SCORE_DIR/ensemble_prescore_authenticated.json"
readonly PRED_VIEW="$SCORE_DIR/prediction_view.h5ad"
readonly TRUTH_VIEW="$SCORE_DIR/truth_view.h5ad"
readonly VIEWS_JSON="$SCORE_DIR/views.json"
readonly CELL_EVAL_DIR="$SCORE_DIR/cell_eval2"
readonly CACHE_REAL="$SCORE_DIR/cache_real"
readonly SCORE_VERIFICATION="$SCORE_DIR/scoring_verified.json"
readonly CONTROLS="$PROJECT_DIR/dataset/public_v51/hepg2_controls_only.h5ad"
readonly MANIFEST="$PROJECT_DIR/dataset/public_v51/split_manifest.csv"
readonly COMMON_GENES="$PROJECT_DIR/dataset/public_v51/common_genes.csv"
readonly COMMON_GENES_JSON="$PROJECT_DIR/dataset/public_v51/common_genes.json"
readonly DATA_RECEIPT="$PROJECT_DIR/dataset/public_v51/hepg2_data.json"
readonly CELL_EVAL_CHECKOUT="$PROJECT_DIR/external/cell-eval2"
readonly EXPECTED_CELL_EVAL_COMMIT="5e64833518a6603a0301cbe28185d49c30f4a986"
readonly EXPECTED_CELL_EVAL_VERSION="0.16.0"
readonly EXPECTED_PDEX_VERSION="0.3.0"
readonly PRESCORE_SCRIPT="$PROJECT_DIR/scripts/authenticate_state_seed_ensemble_for_scoring.py"
readonly SCORE_VALIDATOR="$PROJECT_DIR/scripts/validate_public_state_seed_ensemble_cell_eval_run.py"
readonly LAUNCHER_PATH="$(realpath -- "$0")"

readonly SCORE_PYTHON="$PROJECT_DIR/.venv-score/bin/python"
readonly CONTRACT_PYTHON="$PROJECT_DIR/.venv-state/bin/python"

cd "$PROJECT_DIR"
mkdir -p logs
unset PYTHONHOME PYTHONINSPECT PYTHONSTARTUP PYTHONUSERBASE LD_PRELOAD
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR/external/cell-eval2/src"

# This required-input gate intentionally excludes the scorer-only treated file.
for required_path in \
  "$SCORE_PYTHON" \
  "$CONTRACT_PYTHON" \
  "$PLAN" \
  "$ENSEMBLE_VERIFICATION" \
  "$PREDICTION" \
  "$PRESCORE_SCRIPT" \
  "$SCORE_VALIDATOR" \
  scripts/ensemble_state_seed_candidates.py \
  scripts/materialize_public_candidate_transport.py \
  scripts/prepare_public_scoring_views.py \
  "$CONTROLS" \
  "$MANIFEST" \
  "$COMMON_GENES" \
  "$COMMON_GENES_JSON" \
  "$DATA_RECEIPT" \
  "$CELL_EVAL_CHECKOUT/pyproject.toml" \
  "$CELL_EVAL_CHECKOUT/src/cell_eval2/cli.py" \
  "$LAUNCHER_PATH"; do
  if [[ ! -r "$required_path" ]]; then
    echo "Missing ensemble scoring input: $required_path" >&2
    exit 3
  fi
done
if [[ ! -x "$SCORE_PYTHON" || ! -x "$CONTRACT_PYTHON" ]]; then
  echo "Scoring and contract Python executables are required" >&2
  exit 3
fi
if [[ -e "$SCORE_DIR" || -L "$SCORE_DIR" ]]; then
  echo "Refusing to overwrite isolated ensemble score directory: $SCORE_DIR" >&2
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

"$SCORE_PYTHON" - \
  "$CELL_EVAL_CHECKOUT" "$EXPECTED_CELL_EVAL_VERSION" "$EXPECTED_PDEX_VERSION" <<'PY'
import importlib.metadata
import pathlib
import sys
import tomllib

import cell_eval2

checkout = pathlib.Path(sys.argv[1]).resolve()
module_path = pathlib.Path(cell_eval2.__file__).resolve()
if not module_path.is_relative_to(checkout / "src"):
    raise RuntimeError(f"cell_eval2 imported outside the pinned checkout: {module_path}")
with (checkout / "pyproject.toml").open("rb") as handle:
    declared_version = tomllib.load(handle)["project"]["version"]
if declared_version != sys.argv[2]:
    raise RuntimeError(f"Unexpected cell-eval2 version: {declared_version}")
if importlib.metadata.version("pdex") != sys.argv[3]:
    raise RuntimeError("Unexpected pdex version")
PY

# Reserve the isolated directory, then authenticate every truth-blind source
# and the derived output.  The command has no scorer-only data argument.
if [[ -L "$PUBLIC_ROOT/ensemble_scoring" || -L "$SCORE_PARENT" ]]; then
  echo "Refusing symbolic-link ensemble scoring parent" >&2
  exit 4
fi
mkdir -p "$SCORE_PARENT"
if [[ -L "$PUBLIC_ROOT/ensemble_scoring" || -L "$SCORE_PARENT" ]] \
  || [[ "$(realpath -e -- "$SCORE_PARENT")" != "$SCORE_PARENT" ]]; then
  echo "Ensemble scoring parent is not canonical" >&2
  exit 4
fi
mkdir -m 700 "$SCORE_DIR"
"$CONTRACT_PYTHON" "$PRESCORE_SCRIPT" build \
  --project-dir "$PROJECT_DIR" \
  --public-root "$PUBLIC_ROOT" \
  --plan "$PLAN" \
  --ensemble-run-dir "$ENSEMBLE_RUN" \
  --scoring-launcher "$LAUNCHER_PATH" \
  --output-json "$PRESCORE_AUTH"
test -s "$PRESCORE_AUTH"
chmod 0440 "$PRESCORE_AUTH"
echo "ensemble_prescore_sha256=$(sha256sum "$PRESCORE_AUTH" | cut -d' ' -f1)"

# The scorer-only path is introduced only after pre-score authentication has
# completed successfully.
readonly SEALED_TRUTH="$PROJECT_DIR/dataset/public_v51/hepg2_sealed_truth.h5ad"
if [[ ! -r "$SEALED_TRUTH" ]]; then
  echo "Missing sealed scorer input" >&2
  exit 3
fi

# Authenticate the scorer-only file against the canonical V5.1 preparation
# receipt.  This is deliberately after the ensemble pre-score receipt exists.
"$SCORE_PYTHON" - "$DATA_RECEIPT" "$SEALED_TRUTH" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

sidecar_path = pathlib.Path(sys.argv[1]).resolve(strict=True)
truth_arg = pathlib.Path(sys.argv[2])
if truth_arg.is_symlink():
    raise RuntimeError("Sealed truth must not be a symbolic link")
truth_path = truth_arg.resolve(strict=True)
expected_path = (sidecar_path.parent / "hepg2_sealed_truth.h5ad").resolve(strict=True)
if truth_path != expected_path:
    raise RuntimeError("Sealed truth is not at the canonical registered path")
payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
if payload.get("schema") != "vcc-public-validation-data-v1":
    raise RuntimeError("Bad V5.1 data receipt schema")
if payload.get("context") != "HepG2":
    raise RuntimeError("V5.1 data receipt context differs")
recorded = (payload.get("provenance") or {}).get("truth_output")
if not isinstance(recorded, dict):
    raise RuntimeError("V5.1 data receipt lacks the truth descriptor")
if pathlib.Path(str(recorded.get("path", ""))).resolve(strict=True) != truth_path:
    raise RuntimeError("V5.1 truth descriptor path differs")
before = os.stat(truth_path, follow_symlinks=False)
digest = hashlib.sha256()
with truth_path.open("rb") as handle:
    for block in iter(lambda: handle.read(8 << 20), b""):
        digest.update(block)
after = os.stat(truth_path, follow_symlinks=False)
identity = (after.st_size, after.st_mtime_ns)
if identity != (before.st_size, before.st_mtime_ns):
    raise RuntimeError("Sealed truth changed while being authenticated")
if type(recorded.get("size_bytes")) is not int or recorded["size_bytes"] != after.st_size:
    raise RuntimeError("Sealed truth size differs from V5.1 receipt")
if recorded.get("sha256") != digest.hexdigest():
    raise RuntimeError("Sealed truth SHA-256 differs from V5.1 receipt")
print(f"sealed_truth_sha256={digest.hexdigest()}")
PY

"$SCORE_PYTHON" scripts/prepare_public_scoring_views.py \
  "$PREDICTION" \
  "$CONTROLS" \
  "$SEALED_TRUTH" \
  "$COMMON_GENES" \
  "$PRED_VIEW" \
  "$TRUTH_VIEW" \
  "$VIEWS_JSON" \
  --chunk-rows 2048 \
  --compression lzf

"$SCORE_PYTHON" - "$VIEWS_JSON" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
axes = report.get("axes") or {}
checks = {
    "schema": report.get("schema") == "vcc-public-scoring-views-v1",
    "context": report.get("context") == "HepG2",
    "genes": axes.get("genes") == 7107,
    "targets": axes.get("targets") == 300,
    "prediction_cells": axes.get("prediction_treated_cells") == 120000,
    "truth_cells": axes.get("truth_view_cells") == 40049,
    "firewall": (report.get("contract") or {}).get("model_read_treated_truth") is False,
    "counts": (report.get("contract") or {}).get("input_type")
    == "raw-nonnegative-integer-counts",
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise RuntimeError(f"Ensemble scoring-view validation failed: {failed}")
PY

"$SCORE_PYTHON" -c 'from cell_eval2.cli import main; main()' run \
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

"$SCORE_PYTHON" "$SCORE_VALIDATOR" \
  --context "$CONTEXT" \
  --output-tag "$OUTPUT_TAG" \
  --manifest "$MANIFEST" \
  --ensemble-plan "$PLAN" \
  --ensemble-verification "$ENSEMBLE_VERIFICATION" \
  --prescore-authentication "$PRESCORE_AUTH" \
  --ensemble-run-dir "$ENSEMBLE_RUN" \
  --views-json "$VIEWS_JSON" \
  --results-csv "$CELL_EVAL_DIR/results.csv" \
  --run-meta "$CELL_EVAL_DIR/run_meta.json" \
  --cell-eval-checkout "$CELL_EVAL_CHECKOUT" \
  --expected-commit "$EXPECTED_CELL_EVAL_COMMIT" \
  --expected-version "$EXPECTED_CELL_EVAL_VERSION" \
  --expected-pdex "$EXPECTED_PDEX_VERSION" \
  --output-json "$SCORE_VERIFICATION"

test -s "$SCORE_VERIFICATION"
date --iso-8601=seconds
