#!/usr/bin/env bash
# Submit only as a task-correlated dependency of the matching v1 training
# array. The explicit environment binding must name the same array job:
#   VCC_TRAINING_ARRAY_JOB_ID=<TRAIN_ARRAY_JOB_ID> \
#     sbatch --export=ALL --dependency=aftercorr:<TRAIN_ARRAY_JOB_ID> \
#       slurm/anvil_h100_generate_state_hepg2_reconstruction_array_v1.sbatch
# This launcher creates reconstruction evidence only. It never scores, packages,
# submits, writes under artifacts/public_v6, or claims historical P4 lineage.
#SBATCH --job-name=vcc-state-anvil-hepg2-recon-v1
#SBATCH --account=bio250247-ai
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=02:00:00
#SBATCH --array=0-3%4
#SBATCH --output=logs/anvil_h100_state_hepg2_recon_v1_%A_%a.out
#SBATCH --error=logs/anvil_h100_state_hepg2_recon_v1_%A_%a.err
#SBATCH --open-mode=append
#SBATCH --no-requeue

set -euo pipefail
umask 027

# Array indices must remain correlated with the registered training array.
#   task 0 -> training seed 42 (historical default-seed reconstruction)
#   task 1 -> training seed 20260901
#   task 2 -> training seed 20260902
#   task 3 -> training seed 20260903
readonly -a REGISTERED_TRAIN_SEEDS=(42 20260901 20260902 20260903)
# Keep candidate-cell randomness matched across model seeds. This value is also
# locked by the strict public-candidate generator contract.
readonly GENERATOR_SEED=20260901
readonly STATE_EFFECT_WEIGHT=1.00
readonly RESIDUAL_ALPHA=0.10
readonly TARGET_REMAINING_FRACTION=0.20

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This launcher must run inside a Slurm allocation" >&2
  exit 2
fi
case "${SLURM_ARRAY_TASK_ID:-}" in
  0|1|2|3) ;;
  *)
    echo "SLURM_ARRAY_TASK_ID must be one of 0, 1, 2, or 3" >&2
    exit 2
    ;;
esac
readonly ARRAY_INDEX="$SLURM_ARRAY_TASK_ID"
readonly TRAIN_SEED="${REGISTERED_TRAIN_SEEDS[$ARRAY_INDEX]}"
if [[ ! "${VCC_TRAINING_ARRAY_JOB_ID:-}" =~ ^[1-9][0-9]*$ ]]; then
  echo "VCC_TRAINING_ARRAY_JOB_ID must be the upstream array's numeric job ID" >&2
  exit 2
fi
readonly TRAINING_ARRAY_JOB_ID="$VCC_TRAINING_ARRAY_JOB_ID"
if [[ "${SLURM_JOB_DEPENDENCY:-}" != "aftercorr:$TRAINING_ARRAY_JOB_ID" ]]; then
  echo "This launcher requires exactly aftercorr:$TRAINING_ARRAY_JOB_ID" >&2
  exit 2
fi
if [[ "${SLURM_ARRAY_TASK_COUNT:-}" != "4" ]]; then
  echo "This launcher requires exactly four correlated array tasks" >&2
  exit 2
fi

readonly PROJECT_INPUT="${VCC_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$PROJECT_INPUT" ]]; then
  echo "Set VCC_PROJECT_DIR or submit from the repository root" >&2
  exit 2
fi
PROJECT_DIR="$(realpath -- "$PROJECT_INPUT")"
readonly PROJECT_DIR
readonly PYTHON_BIN="$PROJECT_DIR/.venv-state/bin/python"
readonly PYTHON_INCLUDE="$PROJECT_DIR/.venv-state/include/python3.11"
readonly STATE_SOURCE_DIR="$PROJECT_DIR/external/state_runtime_9bbfe78a"
readonly STATE_SOURCE_COMMIT="9bbfe78a434a55205e4de834e1ea99f85f7a3add"

readonly MODEL_RUN_NAME="state_sm_anvil_20k_seed${TRAIN_SEED}_v1"
readonly MODEL_DIR="$PROJECT_DIR/artifacts/state_runs/$MODEL_RUN_NAME"
readonly TRAINING_RUNTIME="$MODEL_DIR/anvil_launcher_runtime.json"
readonly SELECTION_JSON="$MODEL_DIR/checkpoints/selected_checkpoint.json"
readonly SELECTED_LINK="$MODEL_DIR/checkpoints/selected.ckpt"
readonly PERTURBATION_MAP="$MODEL_DIR/pert_onehot_map.pt"
readonly VAR_DIMS="$MODEL_DIR/var_dims.pkl"
readonly STATE_CONFIG="$MODEL_DIR/config.yaml"

readonly CONTROLS="$PROJECT_DIR/dataset/public_v51/hepg2_controls_only.h5ad"
readonly CONTROLS_EXPECTED_SHA256="b9fdeea0e64735ced2c19642a7c9ac1f61bf5bac6047ed854756b0e880d05d2b"
readonly CONTROLS_EXPECTED_SIZE=116744594
readonly PANEL_CSV="$PROJECT_DIR/dataset/public_v51/split_manifest.csv"
readonly PANEL_JSON_SOURCE="$PROJECT_DIR/dataset/public_v51/split_manifest.json"
readonly PANEL_JSON_SOURCE_EXPECTED_SHA256="ad69eaa3438a050f1b995e816d4ee43e25d1f59bd88b6ba79c447d2103af2569"
readonly RESIDUAL_NPZ="$PROJECT_DIR/artifacts/public_v51/hepg2_state_residual_top100_v1.npz"
readonly RESIDUAL_JSON_SOURCE="$PROJECT_DIR/artifacts/public_v51/hepg2_state_residual_top100_v1.json"
readonly RESIDUAL_JSON_SOURCE_EXPECTED_SHA256="3ff865285d4454d0764aae118807bc6a0dca65f52520b40bbae8d50263ca4113"
readonly SUPPORT_GENES="$PROJECT_DIR/dataset/state_support/extracted/gene_names.csv"
readonly SUPPORT_GENES_EXPECTED_SHA256="e29ff3512b4c2440759bfb2a34577049b2f49df786b3e19c2a91b565de47b38e"
readonly SUPPORT_GENES_EXPECTED_SIZE=116023

readonly MATERIALIZER="$PROJECT_DIR/scripts/materialize_public_v51_transport.py"
readonly LINEAGE_VALIDATOR="$PROJECT_DIR/scripts/anvil_inference_lineage.py"
readonly COMPLETION_WRITER="$PROJECT_DIR/scripts/write_anvil_candidate_completion.py"
readonly CONTRACT_SCRIPT="$PROJECT_DIR/scripts/public_candidate_v6_contract.py"
readonly GENERATOR_SCRIPT="$PROJECT_DIR/scripts/generate_native_state_anchor_counts.py"
readonly DIRECT_COUNTS_HELPER="$PROJECT_DIR/scripts/generate_state_direct_counts.py"
readonly INFERENCE_HELPER="$PROJECT_DIR/scripts/infer_state_effect_prior.py"
readonly LAUNCHER_PATH="$(realpath -- "$0")"

readonly PUBLIC_ROOT="$PROJECT_DIR/artifacts/anvil_state_reconstruction"
readonly OUTPUT_ROOT="$PUBLIC_ROOT/candidates/hepg2"
readonly OUTPUT_TAG="state_sm_anvil_reconstructed_hepg2_seed${TRAIN_SEED}_v1"
readonly RUN_DIR="$OUTPUT_ROOT/$OUTPUT_TAG"
readonly TRANSPORT_DIR="$RUN_DIR/transport"
readonly PANEL_JSON="$TRANSPORT_DIR/split_manifest.json"
readonly RESIDUAL_JSON="$TRANSPORT_DIR/hepg2_state_residual_top100_v1.json"
readonly SPEC_JSON="$RUN_DIR/spec.json"
readonly OUTPUT_H5AD="$RUN_DIR/prediction.h5ad"
readonly OUTPUT_JSON="$RUN_DIR/generation.json"
readonly VERIFICATION_JSON="$RUN_DIR/generation_verified.json"
readonly RUNTIME_RECORD="$RUN_DIR/anvil_inference_runtime.json"
readonly COMPLETION_JSON="$RUN_DIR/anvil_candidate_completion.json"
readonly SCRATCH_DIR="$RUN_DIR/scratch"

cd "$PROJECT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CPATH="$PYTHON_INCLUDE${CPATH:+:$CPATH}"
export PYTHONPATH="$STATE_SOURCE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=disabled
export WANDB_DISABLED=true

readonly -a REQUIRED_PATHS=(
  "$PYTHON_BIN"
  "$PYTHON_INCLUDE/Python.h"
  "$STATE_SOURCE_DIR"
  "$TRAINING_RUNTIME"
  "$SELECTION_JSON"
  "$SELECTED_LINK"
  "$PERTURBATION_MAP"
  "$VAR_DIMS"
  "$STATE_CONFIG"
  "$CONTROLS"
  "$PANEL_CSV"
  "$PANEL_JSON_SOURCE"
  "$RESIDUAL_NPZ"
  "$RESIDUAL_JSON_SOURCE"
  "$SUPPORT_GENES"
  "$MATERIALIZER"
  "$LINEAGE_VALIDATOR"
  "$COMPLETION_WRITER"
  "$CONTRACT_SCRIPT"
  "$GENERATOR_SCRIPT"
  "$DIRECT_COUNTS_HELPER"
  "$INFERENCE_HELPER"
  "$LAUNCHER_PATH"
)
for required_path in "${REQUIRED_PATHS[@]}"; do
  if [[ ! -r "$required_path" ]]; then
    echo "Missing reconstructed inference input: $required_path" >&2
    exit 3
  fi
done
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "STATE Python must be executable: $PYTHON_BIN" >&2
  exit 3
fi

verify_file_identity() {
  local expected_sha256="$1"
  local expected_size="$2"
  local path="$3"
  local actual_sha256
  local actual_size
  actual_sha256="$(sha256sum -- "$path")"
  actual_sha256="${actual_sha256%% *}"
  actual_size="$(stat -c '%s' -- "$path")"
  if [[ "$actual_sha256" != "$expected_sha256" || "$actual_size" != "$expected_size" ]]; then
    echo "Authenticated input identity mismatch: $path" >&2
    exit 3
  fi
}

verify_file_identity "$CONTROLS_EXPECTED_SHA256" "$CONTROLS_EXPECTED_SIZE" "$CONTROLS"
verify_file_identity "$SUPPORT_GENES_EXPECTED_SHA256" "$SUPPORT_GENES_EXPECTED_SIZE" "$SUPPORT_GENES"
verify_file_identity "$PANEL_JSON_SOURCE_EXPECTED_SHA256" 4072 "$PANEL_JSON_SOURCE"
verify_file_identity "$RESIDUAL_JSON_SOURCE_EXPECTED_SHA256" 4396 "$RESIDUAL_JSON_SOURCE"

observed_state_commit="$(git -C "$STATE_SOURCE_DIR" rev-parse HEAD)"
readonly observed_state_commit
if [[ "$observed_state_commit" != "$STATE_SOURCE_COMMIT" ]]; then
  echo "STATE source commit mismatch: $observed_state_commit" >&2
  exit 3
fi

# Resolve the selected regular checkpoint from the receipt, never from a
# historical fixed step or the selected.ckpt symlink.
mapfile -t CHECKPOINT_FIELDS < <(
  "$PYTHON_BIN" - "$MODEL_DIR" "$TRAIN_SEED" "$MODEL_RUN_NAME" <<'PY'
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

from scripts.anvil_inference_lineage import (
    validate_correlated_training_runtime,
    validate_selected_earliest_argmin,
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


model_dir = Path(sys.argv[1]).resolve(strict=True)
expected_seed = int(sys.argv[2])
expected_run_name = sys.argv[3]
runtime_path = model_dir / "anvil_launcher_runtime.json"
if runtime_path.is_symlink() or not runtime_path.is_file():
    fail("Upstream training runtime must be a regular file")
runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
correlated_lineage = validate_correlated_training_runtime(
    runtime,
    os.environ,
    expected_seed=expected_seed,
    expected_run_name=expected_run_name,
)

checkpoint_dir = (model_dir / "checkpoints").resolve(strict=True)
selection_path = checkpoint_dir / "selected_checkpoint.json"
if selection_path.is_symlink() or not selection_path.is_file():
    fail("Checkpoint-selection receipt must be a regular file")
selection = json.loads(selection_path.read_text(encoding="utf-8"))
if selection.get("schema") != "vcc-state-checkpoint-selection-v1":
    fail("Bad checkpoint-selection schema")
if selection.get("checkpoint_layout") != "validation_step_archives":
    fail("Corrected v1 inference requires validation-step checkpoint archives")
if selection.get("unarchived_validation_steps") != []:
    fail("Checkpoint-selection receipt reports unarchived validations")
expected_steps = list(range(2_000, 20_001, 2_000))
validation_results = selection.get("validation_results")
archived_candidates = selection.get("archived_candidates")
if not isinstance(validation_results, list) or not all(
    isinstance(item, dict) for item in validation_results
):
    fail("Checkpoint-selection validation results are invalid")
if not isinstance(archived_candidates, list) or not all(
    isinstance(item, dict) for item in archived_candidates
):
    fail("Checkpoint-selection archived candidates are invalid")
layout_authentication = selection.get("layout_authentication")
if not isinstance(layout_authentication, dict):
    fail("Checkpoint-selection layout authentication is invalid")
if layout_authentication.get("best_last_fallback_used") is not False:
    fail("Corrected v1 inference forbids best/last checkpoint fallback")
if layout_authentication.get("all_finite_validation_steps_archived") is not True:
    fail("Checkpoint-selection receipt does not authenticate every validation")
if (
    layout_authentication.get("archive_callback_schema")
    != "vcc-state-validation-step-archive-v1"
):
    fail("Checkpoint-selection archive callback schema mismatch")
selected = selection.get("selected")
validated_selection = validate_selected_earliest_argmin(
    validation_results,
    archived_candidates,
    selected,
    expected_steps=expected_steps,
)
global_step = int(validated_selection["global_step"])
val_loss = float(validated_selection["val_loss"])
if global_step <= 0 or global_step > 20_000 or global_step % 2_000:
    fail("Selected checkpoint step is outside the registered 20k archive")
if not math.isfinite(val_loss):
    fail("Selected checkpoint validation loss is invalid")

recorded_checkpoint = Path(str(validated_selection["checkpoint"]))
if not recorded_checkpoint.is_absolute():
    fail("Selected checkpoint path is not absolute")
checkpoint = recorded_checkpoint.resolve(strict=True)
if checkpoint.parent != checkpoint_dir:
    fail("Selected checkpoint is outside the seed run checkpoint directory")
if checkpoint.name != f"step{global_step:05d}.ckpt":
    fail("Selected checkpoint filename and global step differ")
if checkpoint.is_symlink() or not checkpoint.is_file():
    fail("Selected checkpoint must be a regular file")

expected_size = int(selected.get("checkpoint_size_bytes", -1))
expected_sha256 = str(selected.get("checkpoint_sha256", "")).lower()
if checkpoint.stat().st_size != expected_size:
    fail("Selected checkpoint size differs from its receipt")
observed_sha256 = sha256_file(checkpoint)
if observed_sha256 != expected_sha256:
    fail("Selected checkpoint SHA-256 differs from its receipt")

selected_link = checkpoint_dir / "selected.ckpt"
recorded_link = Path(str(selected.get("link", "")))
if Path(os.path.abspath(recorded_link)) != selected_link:
    fail("Selected checkpoint link path differs from its receipt")
if not selected_link.is_symlink():
    fail("Selected checkpoint link is absent")
if selected_link.resolve(strict=True) != checkpoint:
    fail("Selected checkpoint link targets a different archive")

perturbation_map = model_dir / "pert_onehot_map.pt"
if not perturbation_map.is_file():
    fail("Missing perturbation map")
fields = (
    str(checkpoint),
    observed_sha256,
    sha256_file(perturbation_map),
    str(global_step),
    format(val_loss, ".17g"),
    sha256_file(runtime_path),
    correlated_lineage["training_task_job_id"],
)
if any("\n" in value or "\r" in value for value in fields):
    fail("Unsafe newline in resolved checkpoint fields")
print("\n".join(fields))
PY
)
if [[ "${#CHECKPOINT_FIELDS[@]}" -ne 7 ]]; then
  echo "Unable to resolve the seed-specific selected checkpoint" >&2
  exit 3
fi
readonly CHECKPOINT="${CHECKPOINT_FIELDS[0]}"
readonly CHECKPOINT_EXPECTED_SHA256="${CHECKPOINT_FIELDS[1]}"
readonly PERTURBATION_MAP_EXPECTED_SHA256="${CHECKPOINT_FIELDS[2]}"
readonly SELECTED_STEP="${CHECKPOINT_FIELDS[3]}"
readonly SELECTED_VAL_LOSS="${CHECKPOINT_FIELDS[4]}"
readonly TRAINING_RUNTIME_EXPECTED_SHA256="${CHECKPOINT_FIELDS[5]}"
readonly UPSTREAM_TRAINING_TASK_JOB_ID="${CHECKPOINT_FIELDS[6]}"
if [[ ! "$CHECKPOINT_EXPECTED_SHA256" =~ ^[0-9a-f]{64}$ || ! "$PERTURBATION_MAP_EXPECTED_SHA256" =~ ^[0-9a-f]{64}$ || ! "$TRAINING_RUNTIME_EXPECTED_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "Resolved model hashes are malformed" >&2
  exit 3
fi
if [[ ! "$UPSTREAM_TRAINING_TASK_JOB_ID" =~ ^[1-9][0-9]*$ ]]; then
  echo "Resolved upstream training task job ID is malformed" >&2
  exit 3
fi

mkdir -p -- "$OUTPUT_ROOT"
if ! mkdir -- "$RUN_DIR"; then
  echo "Refusing to overwrite or resume reconstructed inference: $RUN_DIR" >&2
  exit 4
fi
# A failed task remains reserved. Use a new launcher version for any rerun.
mkdir -- "$SCRATCH_DIR"

readonly -a MATERIALIZER_ARGS=(
  "$MATERIALIZER"
  --panel-json-source
  "$PANEL_JSON_SOURCE"
  --panel-json-expected-sha256
  "$PANEL_JSON_SOURCE_EXPECTED_SHA256"
  --panel-csv
  "$PANEL_CSV"
  --residual-json-source
  "$RESIDUAL_JSON_SOURCE"
  --residual-json-expected-sha256
  "$RESIDUAL_JSON_SOURCE_EXPECTED_SHA256"
  --residual-npz
  "$RESIDUAL_NPZ"
  --output-panel-json
  "$PANEL_JSON"
  --output-residual-json
  "$RESIDUAL_JSON"
)
"$PYTHON_BIN" "${MATERIALIZER_ARGS[@]}"

# Record the task-correlated training seed, dynamically selected checkpoint,
# transported inputs, dependency metadata, and the single allocated H100.
readonly -a RUNTIME_ARGS=(
  "$RUNTIME_RECORD"
  "$TRAIN_SEED"
  "$GENERATOR_SEED"
  "$OUTPUT_TAG"
  "$CHECKPOINT"
  "$CHECKPOINT_EXPECTED_SHA256"
  "$SELECTED_STEP"
  "$SELECTED_VAL_LOSS"
  "$SELECTION_JSON"
  "$TRAINING_RUNTIME"
  "$TRAINING_RUNTIME_EXPECTED_SHA256"
  "$UPSTREAM_TRAINING_TASK_JOB_ID"
  "$PANEL_JSON"
  "$RESIDUAL_JSON"
  "$LAUNCHER_PATH"
  "$LINEAGE_VALIDATOR"
)
"$PYTHON_BIN" - "${RUNTIME_ARGS[@]}" <<'PY'
from __future__ import annotations

import datetime
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path

import torch


def describe(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return {
        "path": str(path.resolve(strict=True)),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


(
    record_path,
    train_seed,
    generator_seed,
    output_tag,
    checkpoint,
    checkpoint_sha256,
    selected_step,
    selected_val_loss,
    selection_json,
    training_runtime,
    training_runtime_sha256,
    upstream_training_task_job_id,
    panel_json,
    residual_json,
    launcher_path,
    lineage_validator,
) = sys.argv[1:]
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")
if torch.cuda.device_count() != 1:
    raise RuntimeError(
        f"Expected exactly one visible GPU, found {torch.cuda.device_count()}"
    )
device_name = torch.cuda.get_device_name(0)
if "H100" not in device_name:
    raise RuntimeError(f"Anvil ai allocation is not an H100: {device_name}")
nvidia_smi = subprocess.run(
    [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,driver_version",
        "--format=csv,noheader",
    ],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip().splitlines()
training_runtime_description = describe(Path(training_runtime))
if training_runtime_description["sha256"] != training_runtime_sha256:
    raise RuntimeError("Upstream training runtime changed after lineage validation")
payload = {
    "schema": "vcc-anvil-state-reconstructed-inference-runtime-v2",
    "recorded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "lineage": "anvil-portable-reconstruction-v2-correlated-not-historical-p4",
    "output_tag": output_tag,
    "training_seed": int(train_seed),
    "candidate_generator_seed": int(generator_seed),
    "parallelism": "independent_slurm_array_task_single_gpu_no_ddp",
    "leaderboard_submission_authorized": False,
    "slurm": {
        "job_id": os.environ["SLURM_JOB_ID"],
        "array_job_id": os.environ["SLURM_ARRAY_JOB_ID"],
        "array_task_id": os.environ["SLURM_ARRAY_TASK_ID"],
        "job_dependency": os.environ.get("SLURM_JOB_DEPENDENCY"),
        "declared_training_array_job_id": os.environ["VCC_TRAINING_ARRAY_JOB_ID"],
        "correlated_upstream": {
            "array_job_id": os.environ["VCC_TRAINING_ARRAY_JOB_ID"],
            "array_task_id": os.environ["SLURM_ARRAY_TASK_ID"],
            "task_job_id": upstream_training_task_job_id,
            "dependency_type": "aftercorr",
        },
        "job_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    },
    "runtime": {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "arc_state": importlib.metadata.version("arc-state"),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "visible_gpu_count": torch.cuda.device_count(),
        "visible_gpu_name": device_name,
        "nvidia_smi": nvidia_smi,
    },
    "checkpoint_selection": {
        "selected_step": int(selected_step),
        "selected_val_loss": float(selected_val_loss),
        "checkpoint": describe(Path(checkpoint)),
        "expected_checkpoint_sha256": checkpoint_sha256,
        "selection_json": describe(Path(selection_json)),
    },
    "upstream_training_runtime": {
        **training_runtime_description,
        "expected_sha256": training_runtime_sha256,
        "required_schema": "vcc-anvil-state-training-runtime-v2",
    },
    "transported_inputs": {
        "panel_json": describe(Path(panel_json)),
        "residual_json": describe(Path(residual_json)),
    },
    "source": {
        "launcher": describe(Path(launcher_path)),
        "lineage_validator": describe(Path(lineage_validator)),
    },
}
temporary = Path(record_path).with_name(f".{Path(record_path).name}.tmp")
with temporary.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
try:
    os.link(temporary, record_path)
finally:
    temporary.unlink(missing_ok=True)
print(json.dumps(payload, sort_keys=True))
PY

date --iso-8601=seconds
hostname
echo "training_seed=$TRAIN_SEED generator_seed=$GENERATOR_SEED selected_step=$SELECTED_STEP"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader

readonly -a PLAN_ARGS=(
  "$CONTRACT_SCRIPT"
  plan
  --context
  HepG2
  --output-tag
  "$OUTPUT_TAG"
  --state-effect-weight
  "$STATE_EFFECT_WEIGHT"
  --residual-alpha
  "$RESIDUAL_ALPHA"
  --target-remaining-fraction
  "$TARGET_REMAINING_FRACTION"
  --controls-h5ad
  "$CONTROLS"
  --panel-manifest
  "$PANEL_JSON"
  --panel-csv
  "$PANEL_CSV"
  --residual-npz
  "$RESIDUAL_NPZ"
  --residual-json
  "$RESIDUAL_JSON"
  --checkpoint
  "$CHECKPOINT"
  --selection-json
  "$SELECTION_JSON"
  --support-genes
  "$SUPPORT_GENES"
  --generator-script
  "$GENERATOR_SCRIPT"
  --generate-state-direct-counts-helper
  "$DIRECT_COUNTS_HELPER"
  --infer-state-effect-prior-helper
  "$INFERENCE_HELPER"
  --state-source-dir
  "$STATE_SOURCE_DIR"
  --perturbation-map
  "$PERTURBATION_MAP"
  --var-dims
  "$VAR_DIMS"
  --state-config
  "$STATE_CONFIG"
  --output-json
  "$SPEC_JSON"
)
"$PYTHON_BIN" "${PLAN_ARGS[@]}"

readonly -a GENERATOR_ARGS=(
  "$GENERATOR_SCRIPT"
  --controls-h5ad
  "$CONTROLS"
  --panel-manifest
  "$PANEL_JSON"
  --panel-csv
  "$PANEL_CSV"
  --context
  HepG2
  --model-dir
  "$MODEL_DIR"
  --checkpoint
  "$CHECKPOINT"
  --checkpoint-expected-sha256
  "$CHECKPOINT_EXPECTED_SHA256"
  --perturbation-map-expected-sha256
  "$PERTURBATION_MAP_EXPECTED_SHA256"
  --selection-json
  "$SELECTION_JSON"
  --support-genes
  "$SUPPORT_GENES"
  --output-h5ad
  "$OUTPUT_H5AD"
  --output-json
  "$OUTPUT_JSON"
  --scratch-dir
  "$SCRATCH_DIR"
  --device
  cuda
  --require-cuda
  --require-h100
  --model-chunk-size
  128
  --cells-per-group
  400
  --state-effect-weight
  "$STATE_EFFECT_WEIGHT"
  --state-effect-clip
  0.60
  --state-effect-bounding
  tanh
  --residual-npz
  "$RESIDUAL_NPZ"
  --residual-alpha
  "$RESIDUAL_ALPHA"
  --combined-effect-clip
  0.65
  --target-policy
  force
  --target-remaining-fraction
  "$TARGET_REMAINING_FRACTION"
  --max-genes-per-cell
  5900
  --integerization
  largest-remainder
  --groups-per-shard
  20
  --concat-max-loaded-elements
  20000000
  --compression
  lzf
  --seed
  "$GENERATOR_SEED"
)
"$PYTHON_BIN" "${GENERATOR_ARGS[@]}"

readonly -a VERIFY_ARGS=(
  "$CONTRACT_SCRIPT"
  verify
  --spec
  "$SPEC_JSON"
  --prediction-h5ad
  "$OUTPUT_H5AD"
  --generation-json
  "$OUTPUT_JSON"
  --output-json
  "$VERIFICATION_JSON"
)
"$PYTHON_BIN" "${VERIFY_ARGS[@]}"

# This is deliberately the final candidate write. The completion receipt
# independently authenticates every transport-required artifact and will not
# replace an existing path.
readonly -a COMPLETION_ARGS=(
  "$COMPLETION_WRITER"
  --runtime
  "$RUNTIME_RECORD"
  --selection
  "$SELECTION_JSON"
  --spec
  "$SPEC_JSON"
  --generation
  "$OUTPUT_JSON"
  --prediction
  "$OUTPUT_H5AD"
  --verification
  "$VERIFICATION_JSON"
  --output
  "$COMPLETION_JSON"
  --output-tag
  "$OUTPUT_TAG"
  --context
  HepG2
)

test -s "$RUNTIME_RECORD"
test -s "$PANEL_JSON"
test -s "$RESIDUAL_JSON"
test -s "$SPEC_JSON"
test -s "$OUTPUT_H5AD"
test -s "$OUTPUT_JSON"
test -s "$VERIFICATION_JSON"
"$PYTHON_BIN" "${COMPLETION_ARGS[@]}"
test -s "$COMPLETION_JSON"
date --iso-8601=seconds
