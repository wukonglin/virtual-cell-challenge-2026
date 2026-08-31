#!/usr/bin/env bash
#SBATCH --job-name=vcc-state-smoke
#SBATCH --account=allaccess
#SBATCH --partition=full
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --constraint=gpu-h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/h100_state_smoke_v0_%j.out
#SBATCH --error=logs/h100_state_smoke_v0_%j.err

set -euo pipefail

readonly PROJECT_DIR="${VCC_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
readonly STATE_BIN="$PROJECT_DIR/.venv-state/bin/state"
readonly RUN_DIR="$PROJECT_DIR/artifacts/state_runs/state_sm_smoke_v0"

cd "$PROJECT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export CPATH="$PROJECT_DIR/.venv-state-headers/usr/include/python3.11"

for required_path in \
  "$STATE_BIN" \
  .venv-state-headers/usr/include/python3.11/Python.h \
  configs/state/vcc2026_support.toml \
  dataset/state_support/extracted/ESM2_pert_features.pt \
  dataset/state_support/extracted/competition_train.h5 \
  dataset/state_support/extracted/hepg2.h5; do
  if [[ ! -r "$required_path" ]]; then
    echo "Missing required input: $required_path" >&2
    exit 2
  fi
done

if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing to overwrite existing smoke run: $RUN_DIR" >&2
  exit 3
fi

date --iso-8601=seconds
hostname
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader
"$PROJECT_DIR/.venv-state/bin/python" -c 'import torch; assert torch.cuda.is_available(); print({"torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0)})'

"$STATE_BIN" tx train \
  data.kwargs.toml_config_path="$PROJECT_DIR/configs/state/vcc2026_support.toml" \
  data.kwargs.num_workers=8 \
  data.kwargs.batch_col=batch_var \
  data.kwargs.pert_col=target_gene \
  data.kwargs.cell_type_key=cell_type \
  data.kwargs.control_pert=non-targeting \
  data.kwargs.perturbation_features_file="$PROJECT_DIR/dataset/state_support/extracted/ESM2_pert_features.pt" \
  model=state_sm \
  '~training.cumulative_flops_use_backward' \
  training.batch_size=16 \
  training.max_steps=20 \
  training.val_freq=20 \
  training.ckpt_every_n_steps=20 \
  use_wandb=false \
  output_dir="$PROJECT_DIR/artifacts/state_runs" \
  name=state_sm_smoke_v0

test -s "$RUN_DIR/config.yaml"
test -s "$RUN_DIR/pert_onehot_map.pt"
test -s "$RUN_DIR/var_dims.pkl"
test -s "$RUN_DIR/checkpoints/final.ckpt"
date --iso-8601=seconds
