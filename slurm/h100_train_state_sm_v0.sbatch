#!/usr/bin/env bash
#SBATCH --job-name=vcc-state-20k
#SBATCH --account=allaccess
#SBATCH --partition=full
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --constraint=gpu-h100
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=logs/h100_state_sm_20k_v0_%j.out
#SBATCH --error=logs/h100_state_sm_20k_v0_%j.err

set -euo pipefail

readonly PROJECT_DIR="${VCC_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
readonly STATE_BIN="$PROJECT_DIR/.venv-state/bin/state"
readonly RUN_DIR="$PROJECT_DIR/artifacts/state_runs/state_sm_20k_v0"
readonly SOURCE_STAGE="/tmp/vcc_state_support_source_v1"
readonly WRAPPED_STAGE="/tmp/vcc_state_support_local_v1"

cd "$PROJECT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export CPATH="$PROJECT_DIR/.venv-state-headers/usr/include/python3.11"

for required_path in \
  "$STATE_BIN" \
  .venv-state-headers/usr/include/python3.11/Python.h \
  configs/state/vcc2026_support_local_v1.toml \
  scripts/prepare_state_support_links.py \
  dataset/state_support/extracted/ESM2_pert_features.pt \
  dataset/state_support/extracted/competition_train.h5 \
  dataset/state_support/extracted/hepg2.h5; do
  if [[ ! -r "$required_path" ]]; then
    echo "Missing required input: $required_path" >&2
    exit 2
  fi
done

if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing to overwrite existing production run: $RUN_DIR" >&2
  exit 3
fi

if [[ -e "$SOURCE_STAGE" || -e "$WRAPPED_STAGE" ]]; then
  echo "Refusing to reuse an existing node-local staging directory" >&2
  exit 4
fi

mkdir -p "$SOURCE_STAGE"
trap 'rm -rf -- "$SOURCE_STAGE" "$WRAPPED_STAGE"' EXIT

available_bytes="$(df -PB1 /tmp | awk 'NR==2 {print $4}')"
readonly available_bytes
if [[ "$available_bytes" -lt 53687091200 ]]; then
  echo "Need at least 50 GiB free in node-local /tmp" >&2
  exit 5
fi

date --iso-8601=seconds
hostname
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader
"$PROJECT_DIR/.venv-state/bin/python" -c 'import torch; assert torch.cuda.is_available(); print({"torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0)})'

for filename in \
  competition_train.h5 \
  k562_gwps.h5 \
  rpe1.h5 \
  jurkat.h5 \
  k562.h5 \
  hepg2.h5; do
  cp --reflink=auto \
    "$PROJECT_DIR/dataset/state_support/extracted/$filename" \
    "$SOURCE_STAGE/$filename"
done

"$PROJECT_DIR/.venv-state/bin/python" scripts/prepare_state_support_links.py \
  --source-dir "$SOURCE_STAGE" \
  --output-dir "$WRAPPED_STAGE"

"$STATE_BIN" tx train \
  data.kwargs.toml_config_path="$PROJECT_DIR/configs/state/vcc2026_support_local_v1.toml" \
  data.kwargs.num_workers=16 \
  data.kwargs.batch_col=batch_var \
  data.kwargs.pert_col=target_gene \
  data.kwargs.cell_type_key=cell_type \
  data.kwargs.control_pert=non-targeting \
  data.kwargs.perturbation_features_file="$PROJECT_DIR/dataset/state_support/extracted/ESM2_pert_features.pt" \
  model=state_sm \
  '~training.cumulative_flops_use_backward' \
  training.batch_size=16 \
  training.max_steps=20000 \
  training.val_freq=2000 \
  training.ckpt_every_n_steps=2000 \
  use_wandb=false \
  output_dir="$PROJECT_DIR/artifacts/state_runs" \
  name=state_sm_20k_v0

test -s "$RUN_DIR/config.yaml"
test -s "$RUN_DIR/pert_onehot_map.pt"
test -s "$RUN_DIR/var_dims.pkl"
test -s "$RUN_DIR/checkpoints/final.ckpt"
date --iso-8601=seconds
