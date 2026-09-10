#!/usr/bin/env bash
# Submit from the repository root; Slurm opens the tracked logs/ directory
# before this script begins.
#SBATCH --job-name=vcc-state-anvil-20k-v1
#SBATCH --account=bio250247-ai
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --array=0-3%4
#SBATCH --output=logs/anvil_h100_state_sm_20k_v1_%A_%a.out
#SBATCH --error=logs/anvil_h100_state_sm_20k_v1_%A_%a.err
#SBATCH --open-mode=append
#SBATCH --no-requeue

set -euo pipefail
umask 027

# Registered v1 array map. These are four independent single-GPU fits, not DDP:
#   task 0 -> seed 42 (historical default-seed incumbent)
#   task 1 -> seed 20260901
#   task 2 -> seed 20260902
#   task 3 -> seed 20260903
# Changing this map requires a new launcher version and new immutable run names.
readonly -a REGISTERED_SEEDS=(42 20260901 20260902 20260903)

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
readonly TRAIN_SEED="${REGISTERED_SEEDS[$ARRAY_INDEX]}"

readonly PROJECT_INPUT="${VCC_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$PROJECT_INPUT" ]]; then
  echo "Set VCC_PROJECT_DIR or submit from the repository root" >&2
  exit 2
fi

readonly PROJECT_DIR="$(realpath -- "$PROJECT_INPUT")"
readonly PYTHON_BIN="$PROJECT_DIR/.venv-state/bin/python"
readonly STATE_BIN="$PROJECT_DIR/.venv-state/bin/state"
readonly PYTHON_INCLUDE="$PROJECT_DIR/.venv-state/include/python3.11"
readonly STATE_SOURCE_DIR="$PROJECT_DIR/external/state_runtime_9bbfe78a"
readonly STATE_SOURCE_COMMIT="9bbfe78a434a55205e4de834e1ea99f85f7a3add"
readonly STATE_SOURCE_AUTHENTICATOR="$PROJECT_DIR/scripts/public_candidate_v6_contract.py"
readonly CONFIG="$PROJECT_DIR/configs/state/vcc2026_support.toml"
readonly CHECKPOINT_SHIM="$PROJECT_DIR/scripts/run_state_train_with_archives.py"
readonly CHECKPOINT_SELECTOR="$PROJECT_DIR/scripts/select_state_checkpoint.py"
readonly SUPPORT_DIR="$PROJECT_DIR/dataset/state_support/state_compatible"
readonly FEATURES="$PROJECT_DIR/dataset/state_support/extracted/ESM2_pert_features.pt"
readonly RUN_ROOT="$PROJECT_DIR/artifacts/state_runs"
readonly RUN_NAME="state_sm_anvil_20k_seed${TRAIN_SEED}_v1"
readonly RUN_DIR="$RUN_ROOT/$RUN_NAME"
readonly LAUNCHER_PATH="$(realpath -- "$0")"
readonly -a SUPPORT_FILES=(
  competition_train.h5
  k562_gwps.h5
  rpe1.h5
  jurkat.h5
  k562.h5
  hepg2.h5
)

cd "$PROJECT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export HYDRA_FULL_ERROR=1
export PYTHONHASHSEED="$TRAIN_SEED"
export CPATH="$PYTHON_INCLUDE${CPATH:+:$CPATH}"
export PYTHONPATH="$STATE_SOURCE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=disabled
export WANDB_DISABLED=true

readonly -a REQUIRED_PATHS=(
  "$PYTHON_BIN"
  "$STATE_BIN"
  "$PYTHON_INCLUDE/Python.h"
  "$STATE_SOURCE_DIR"
  "$STATE_SOURCE_AUTHENTICATOR"
  "$CONFIG"
  "$CHECKPOINT_SHIM"
  "$CHECKPOINT_SELECTOR"
  "$FEATURES"
  "$LAUNCHER_PATH"
)
for required_path in "${REQUIRED_PATHS[@]}"; do
  if [[ ! -r "$required_path" ]]; then
    echo "Missing required input: $required_path" >&2
    exit 3
  fi
done
if [[ ! -x "$PYTHON_BIN" || ! -x "$STATE_BIN" ]]; then
  echo "STATE Python and CLI must be executable" >&2
  exit 3
fi
for filename in "${SUPPORT_FILES[@]}"; do
  if [[ ! -r "$SUPPORT_DIR/$filename" ]]; then
    echo "Missing repository STATE wrapper: $SUPPORT_DIR/$filename" >&2
    exit 3
  fi
done

# Authenticate before support-data access, explicit CUDA/H100 use, or run reservation. The
# environment above is therefore active for the first Python process.
STATE_SOURCE_AUTHENTICATION="$(
  "$PYTHON_BIN" - "$PROJECT_DIR" "$STATE_SOURCE_DIR" "$STATE_SOURCE_COMMIT" <<'PY'
import json
import sys
from pathlib import Path


project_dir = Path(sys.argv[1]).resolve(strict=True)
source_dir = Path(sys.argv[2])
expected_commit = sys.argv[3]
sys.path.insert(0, str(project_dir / "scripts"))
from public_candidate_v6_contract import authenticate_state_source

authentication = authenticate_state_source(
    source_dir,
    require_active_isolation=True,
)
if authentication.get("repository_commit") != expected_commit:
    raise RuntimeError(
        "STATE source commit mismatch: "
        f"{authentication.get('repository_commit')} != {expected_commit}"
    )
print(json.dumps(authentication, sort_keys=True))
PY
)"
readonly STATE_SOURCE_AUTHENTICATION

# Opening every wrapper validates its external links before reserving a run.
"$PYTHON_BIN" - "$SUPPORT_DIR" <<'PY'
import sys
from pathlib import Path



import h5py

support_dir = Path(sys.argv[1])
for name in (
    "competition_train.h5",
    "k562_gwps.h5",
    "rpe1.h5",
    "jurkat.h5",
    "k562.h5",
    "hepg2.h5",
):
    path = support_dir / name
    with h5py.File(path, "r") as handle:
        missing = {"X", "obs", "var", "uns"}.difference(handle.keys())
        if missing:
            raise RuntimeError(f"{path} is missing groups: {sorted(missing)}")
        if "log1p" not in handle["uns"]:
            raise RuntimeError(f"{path} is missing uns/log1p")
        matrix = handle["X"]
        shape = matrix.shape if isinstance(matrix, h5py.Dataset) else tuple(matrix.attrs["shape"])
        if int(shape[1]) != 18_080:
            raise RuntimeError(f"Unexpected gene dimension in {path}: {shape}")
        print({"wrapper": str(path), "shape": tuple(int(value) for value in shape)})
PY

mkdir -p -- "$RUN_ROOT"
if ! mkdir -- "$RUN_DIR"; then
  echo "Refusing to overwrite or resume existing seed run: $RUN_DIR" >&2
  exit 4
fi
readonly RUNTIME_RECORD="$RUN_DIR/anvil_launcher_runtime.json"
export TORCH_EXTENSIONS_DIR="$RUN_DIR/torch_extensions"
mkdir -- "$TORCH_EXTENSIONS_DIR"

# Record the allocated GPU, software runtime, immutable launcher, and registered seed.
"$PYTHON_BIN" - \
  "$RUNTIME_RECORD" \
  "$TRAIN_SEED" \
  "$RUN_NAME" \
  "$PROJECT_DIR" \
  "$LAUNCHER_PATH" \
  "$CHECKPOINT_SHIM" \
  "$STATE_SOURCE_AUTHENTICATION" \
  "$STATE_SOURCE_COMMIT" \
  "$CONFIG" \
  "$FEATURES" \
  "$SUPPORT_DIR" <<'PY'
from __future__ import annotations

import datetime
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import stat
import subprocess
import sys
from pathlib import Path


import torch

import h5py

record_path = Path(sys.argv[1])
seed = int(sys.argv[2])
run_name = sys.argv[3]
project_dir = Path(sys.argv[4])
launcher_path = Path(sys.argv[5])
checkpoint_shim = Path(sys.argv[6])
state_source_authentication = json.loads(sys.argv[7])
expected_state_source_commit = sys.argv[8]
config_path = Path(sys.argv[9])
features_path = Path(sys.argv[10])
support_dir = Path(sys.argv[11])
if state_source_authentication.get("repository_commit") != expected_state_source_commit:
    raise RuntimeError("Preflight STATE source authentication commit changed")
if state_source_authentication.get("tracked_worktree_clean") is not True:
    raise RuntimeError("Preflight STATE source was not tracked-clean")
if state_source_authentication.get("untracked_and_ignored_runtime_files_absent") is not True:
    raise RuntimeError("Preflight STATE source had import-affecting files")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")
if torch.cuda.device_count() != 1:
    raise RuntimeError(f"Expected exactly one visible GPU, found {torch.cuda.device_count()}")
device_name = torch.cuda.get_device_name(0)
if "H100" not in device_name:
    raise RuntimeError(f"Anvil ai allocation is not an H100: {device_name}")

def authenticated_regular_file(path: Path, label: str) -> dict[str, object]:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or path.is_symlink() or before.st_size == 0:
        raise RuntimeError(f"{label} must be a nonempty regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    after = path.lstat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError(f"{label} changed while it was hashed: {path}")
    return {
        "path": str(path),
        "size_bytes": int(after.st_size),
        "sha256": digest.hexdigest(),
    }


launcher_source = authenticated_regular_file(launcher_path, "launcher")
shim_source = authenticated_regular_file(checkpoint_shim, "checkpoint archive shim")
EXPECTED_FEATURES_SHA256 = "a210e1cc7901513999b2bca3836ba9e2f203cd008be4e9a9d6412a2267de9748"
EXPECTED_FEATURES_SIZE = 410_886_729


def regular_stat_identity(path: Path, label: str) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    before = resolved.lstat()
    if not stat.S_ISREG(before.st_mode) or resolved.is_symlink() or before.st_size == 0:
        raise RuntimeError(f"{label} must be a nonempty regular file: {resolved}")
    after = resolved.lstat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError(f"{label} changed while it was described: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": int(after.st_size),
        "mtime_ns": int(after.st_mtime_ns),
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
        "identity_scope": "runtime_filesystem_stat",
    }


config_source = authenticated_regular_file(config_path, "STATE support config")
features_source = authenticated_regular_file(features_path, "ESM2 target features")
if (
    features_source["sha256"] != EXPECTED_FEATURES_SHA256
    or features_source["size_bytes"] != EXPECTED_FEATURES_SIZE
):
    raise RuntimeError("ESM2 target-feature identity differs from the registered artifact")

support_inputs = []
for filename in (
    "competition_train.h5",
    "k562_gwps.h5",
    "rpe1.h5",
    "jurkat.h5",
    "k562.h5",
    "hepg2.h5",
):
    wrapper = support_dir / filename
    wrapper_source = authenticated_regular_file(wrapper, f"STATE wrapper {filename}")
    external_targets: set[Path] = set()
    with h5py.File(wrapper, "r") as handle:
        groups = [handle]
        if "uns" in handle and isinstance(handle["uns"], h5py.Group):
            groups.append(handle["uns"])
        for group in groups:
            for key in group.keys():
                link = group.get(key, getlink=True)
                if isinstance(link, h5py.ExternalLink):
                    target = Path(link.filename)
                    if not target.is_absolute():
                        target = wrapper.parent / target
                    external_targets.add(target.resolve(strict=True))
    if len(external_targets) != 1:
        raise RuntimeError(
            f"STATE wrapper {filename} must reference exactly one payload: {external_targets}"
        )
    linked_payload = next(iter(external_targets))
    expected_payload = (support_dir.parent / "extracted" / filename).resolve(strict=True)
    if linked_payload != expected_payload:
        raise RuntimeError(
            f"STATE wrapper {filename} links outside the registered extracted payload"
        )
    support_inputs.append(
        {
            "filename": filename,
            "wrapper": wrapper_source,
            "linked_payload": regular_stat_identity(
                linked_payload, f"STATE linked payload {filename}"
            ),
            "linked_payload_content_opened_by_preflight": True,
            "linked_payload_cryptographic_hash_computed_per_task": False,
        }
    )

training_inputs = {
    "state_config": config_source,
    "esm2_target_features": {
        **features_source,
        "registered_sha256_match": True,
        "registered_size_match": True,
    },
    "state_support_wrappers": support_inputs,
}

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
git_commit = subprocess.run(
    ["git", "-C", str(project_dir), "rev-parse", "HEAD"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
git_status = subprocess.run(
    ["git", "-C", str(project_dir), "status", "--short"],
    check=True,
    capture_output=True,
    text=True,
).stdout.splitlines()
payload = {
    "schema": "vcc-anvil-state-training-runtime-v2",
    "recorded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "run_name": run_name,
    "registered_seed": seed,
    "lineage": "anvil-state-training-production-clean-source-archives-v1",
    "training_devices": 1,
    "parallelism": "independent_slurm_array_task_single_gpu_no_ddp",
    "wandb_enabled": False,
    "slurm": {
        "job_id": os.environ["SLURM_JOB_ID"],
        "array_job_id": os.environ["SLURM_ARRAY_JOB_ID"],
        "array_task_id": os.environ["SLURM_ARRAY_TASK_ID"],
        "array_task_count": os.environ.get("SLURM_ARRAY_TASK_COUNT"),
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
    "source": {
        "git_commit": git_commit,
        "git_status_short": git_status,
        "state_source_authentication": state_source_authentication,
        "training_inputs": training_inputs,
        "launcher_path": launcher_source["path"],
        "launcher_size_bytes": launcher_source["size_bytes"],
        "launcher_sha256": launcher_source["sha256"],
        "checkpoint_archive_shim": {
            **shim_source,
            "callback_state_schema": "vcc-state-validation-step-archive-v1",
        },
    },
}
temporary = record_path.with_name(f".{record_path.name}.tmp")
with temporary.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, record_path)
print(json.dumps(payload, sort_keys=True))
PY

date --iso-8601=seconds
hostname
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader

readonly -a STATE_ARGS=(
  tx
  train
  "data.kwargs.toml_config_path=$CONFIG"
  data.kwargs.num_workers=16
  data.kwargs.batch_col=batch_var
  data.kwargs.pert_col=target_gene
  data.kwargs.cell_type_key=cell_type
  data.kwargs.control_pert=non-targeting
  "data.kwargs.perturbation_features_file=$FEATURES"
  model=state_sm
  '~training.cumulative_flops_use_backward'
  training.batch_size=16
  training.max_steps=20000
  training.val_freq=2000
  training.ckpt_every_n_steps=2000
  "training.train_seed=$TRAIN_SEED"
  training.devices=1
  training.strategy=auto
  training.wandb_track=false
  use_wandb=false
  overwrite=false
  "output_dir=$RUN_ROOT"
  "name=$RUN_NAME"
)
"$PYTHON_BIN" "$CHECKPOINT_SHIM" "${STATE_ARGS[@]}"

test -s "$RUN_DIR/config.yaml"
test -s "$RUN_DIR/pert_onehot_map.pt"
test -s "$RUN_DIR/var_dims.pkl"
test -s "$RUN_DIR/checkpoints/best.ckpt"
test -s "$RUN_DIR/checkpoints/last.ckpt"
test -s "$RUN_DIR/checkpoints/final.ckpt"

# Authenticate the complete validation/archive bijection before selection.  In
# particular, each archive must carry current step/loss metadata rather than
# the one-validation-late score observed in STATE's original best callback.
"$PYTHON_BIN" - "$RUN_DIR" <<'PY'
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import secrets
import stat
import sys
from pathlib import Path


import torch


run_dir = Path(sys.argv[1])
checkpoint_dir = run_dir / "checkpoints"
metrics_path = run_dir / "version_0" / "metrics.csv"
manifest_path = checkpoint_dir / "validation_step_archives.json"
expected_steps = list(range(2_000, 20_001, 2_000))
archive_schema = "vcc-state-validation-step-archive-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


if not metrics_path.is_file():
    raise RuntimeError(f"Missing metrics CSV: {metrics_path}")
metrics_hash_before = sha256_file(metrics_path)
validation: dict[int, float] = {}
with metrics_path.open("r", encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle):
        raw_loss = row.get("val_loss")
        if raw_loss in (None, ""):
            continue
        loss = float(raw_loss)
        step = int(row["step"]) + 1
        if not math.isfinite(loss):
            raise RuntimeError(f"Nonfinite val_loss at global step {step}: {loss}")
        if step in validation:
            raise RuntimeError(f"Duplicate val_loss at global step {step}")
        validation[step] = loss
metrics_hash_after = sha256_file(metrics_path)
if metrics_hash_before != metrics_hash_after:
    raise RuntimeError("Metrics CSV changed while it was read")
if sorted(validation) != expected_steps:
    raise RuntimeError(
        f"Validation steps differ from registered cadence: {sorted(validation)}"
    )

actual_archive_names = sorted(path.name for path in checkpoint_dir.glob("step*.ckpt"))
expected_archive_names = [f"step{step:05d}.ckpt" for step in expected_steps]
if actual_archive_names != expected_archive_names:
    raise RuntimeError(
        f"Archive filenames differ from registered cadence: {actual_archive_names}"
    )

archives = []
for archive_count, step in enumerate(expected_steps, start=1):
    path = checkpoint_dir / f"step{step:05d}.ckpt"
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or path.is_symlink() or before.st_size == 0:
        raise RuntimeError(f"Archive is not a nonempty regular file: {path}")
    with path.open("rb") as handle:
        checkpoint = torch.load(handle, map_location="cpu", weights_only=True)
    after = path.lstat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError(f"Archive changed while it was authenticated: {path}")
    if int(checkpoint.get("global_step", -1)) != step:
        raise RuntimeError(f"Checkpoint root global_step mismatch: {path}")
    callback_states = checkpoint.get("callbacks")
    if not isinstance(callback_states, dict):
        raise RuntimeError(f"Archive has no callback state mapping: {path}")
    matching = [
        state
        for state in callback_states.values()
        if isinstance(state, dict) and state.get("schema") == archive_schema
    ]
    if len(matching) != 1:
        raise RuntimeError(f"Archive must contain exactly one shim state: {path}")
    state = matching[0]
    expected_filename = path.name
    checks = {
        "global_step": state.get("global_step") == step,
        "val_loss": math.isclose(
            float(state.get("val_loss", math.nan)),
            validation[step],
            rel_tol=1e-6,
            abs_tol=1e-8,
        ),
        "filename": state.get("filename") == expected_filename,
        "archive_count": state.get("archive_count") == archive_count,
        "hook": state.get("hook") == "on_validation_end",
        "publish_method": state.get("publish_method")
        == "same_directory_temporary_then_atomic_hard_link_no_overwrite",
    }
    if not all(checks.values()):
        raise RuntimeError(f"Archive shim metadata mismatch for {path}: {checks}")
    archives.append(
        {
            "global_step": step,
            "val_loss": validation[step],
            "filename": expected_filename,
            "size_bytes": int(after.st_size),
            "sha256": sha256_file(path),
            "callback_metadata_checks": checks,
        }
    )

payload = {
    "schema": "vcc-state-validation-step-archive-manifest-v1",
    "run_dir": str(run_dir),
    "metrics": str(metrics_path),
    "metrics_sha256": metrics_hash_before,
    "registered_validation_steps": expected_steps,
    "archive_count": len(archives),
    "archives": archives,
}
if os.path.lexists(manifest_path):
    raise FileExistsError(f"Refusing to overwrite archive manifest: {manifest_path}")
temporary = manifest_path.with_name(
    f".{manifest_path.name}.{os.getpid()}.{secrets.token_hex(12)}.tmp"
)
try:
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, manifest_path, follow_symlinks=False)
finally:
    try:
        temporary.unlink()
    except FileNotFoundError:
        pass
print(json.dumps(payload, sort_keys=True))
PY

"$PYTHON_BIN" "$CHECKPOINT_SELECTOR" --run-dir "$RUN_DIR"
test -L "$RUN_DIR/checkpoints/selected.ckpt"
test -s "$RUN_DIR/checkpoints/selected_checkpoint.json"
"$PYTHON_BIN" - "$RUN_DIR/checkpoints/selected_checkpoint.json" <<'PY'
import json
import sys
from pathlib import Path


receipt = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if receipt.get("checkpoint_layout") != "validation_step_archives":
    raise RuntimeError("Selector did not use validation-step archives")
if receipt.get("unarchived_validation_steps") != []:
    raise RuntimeError(
        f"Selector reports unarchived validations: {receipt.get('unarchived_validation_steps')}"
    )
if len(receipt.get("archived_candidates", [])) != 10:
    raise RuntimeError("Selector did not register all ten validation-step archives")
PY
date --iso-8601=seconds
