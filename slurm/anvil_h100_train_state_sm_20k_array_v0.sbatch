#!/usr/bin/env bash
# Submit from the repository root; Slurm opens the tracked logs/ directory
# before this script begins.
#SBATCH --job-name=vcc-state-anvil-20k
#SBATCH --account=bio250247-ai
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --array=0-3%4
#SBATCH --output=logs/anvil_h100_state_sm_20k_v0_%A_%a.out
#SBATCH --error=logs/anvil_h100_state_sm_20k_v0_%A_%a.err
#SBATCH --open-mode=append

set -euo pipefail
umask 027

# Registered v0 array map. These are four independent single-GPU fits, not DDP:
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
readonly CONFIG="$PROJECT_DIR/configs/state/vcc2026_support.toml"
readonly CHECKPOINT_SELECTOR="$PROJECT_DIR/scripts/select_state_checkpoint.py"
readonly SUPPORT_DIR="$PROJECT_DIR/dataset/state_support/state_compatible"
readonly FEATURES="$PROJECT_DIR/dataset/state_support/extracted/ESM2_pert_features.pt"
readonly RUN_ROOT="$PROJECT_DIR/artifacts/state_runs"
readonly RUN_NAME="state_sm_anvil_20k_seed${TRAIN_SEED}_v0"
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
export HYDRA_FULL_ERROR=1
export PYTHONHASHSEED="$TRAIN_SEED"
export CPATH="$PYTHON_INCLUDE${CPATH:+:$CPATH}"
export WANDB_MODE=disabled
export WANDB_DISABLED=true

readonly -a REQUIRED_PATHS=(
  "$PYTHON_BIN"
  "$STATE_BIN"
  "$PYTHON_INCLUDE/Python.h"
  "$CONFIG"
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
"$PYTHON_BIN" - "$RUNTIME_RECORD" "$TRAIN_SEED" "$RUN_NAME" "$PROJECT_DIR" "$LAUNCHER_PATH" <<'PY'
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

record_path = Path(sys.argv[1])
seed = int(sys.argv[2])
run_name = sys.argv[3]
project_dir = Path(sys.argv[4])
launcher_path = Path(sys.argv[5])
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")
if torch.cuda.device_count() != 1:
    raise RuntimeError(f"Expected exactly one visible GPU, found {torch.cuda.device_count()}")
device_name = torch.cuda.get_device_name(0)
if "H100" not in device_name:
    raise RuntimeError(f"Anvil ai allocation is not an H100: {device_name}")

digest = hashlib.sha256()
with launcher_path.open("rb") as handle:
    while block := handle.read(1 << 20):
        digest.update(block)
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
    "schema": "vcc-anvil-state-training-runtime-v1",
    "recorded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "run_name": run_name,
    "registered_seed": seed,
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
        "launcher_path": str(launcher_path),
        "launcher_sha256": digest.hexdigest(),
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
"$STATE_BIN" "${STATE_ARGS[@]}"

test -s "$RUN_DIR/config.yaml"
test -s "$RUN_DIR/pert_onehot_map.pt"
test -s "$RUN_DIR/var_dims.pkl"
test -s "$RUN_DIR/checkpoints/final.ckpt"
"$PYTHON_BIN" "$CHECKPOINT_SELECTOR" --run-dir "$RUN_DIR"
test -L "$RUN_DIR/checkpoints/selected.ckpt"
test -s "$RUN_DIR/checkpoints/selected_checkpoint.json"
date --iso-8601=seconds
