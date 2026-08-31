#!/usr/bin/env bash
#SBATCH --job-name=vcc-jurkat-atlas
#SBATCH --account=allaccess
#SBATCH --partition=full
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --constraint=gpu-h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --output=logs/h100_jurkat_atlas_v1_%j.out
#SBATCH --error=logs/h100_jurkat_atlas_v1_%j.err

set -euo pipefail

readonly PROJECT_DIR="${VCC_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
readonly PYTHON_BIN="$PROJECT_DIR/.venv-state/bin/python"
readonly INPUT_H5AD="$PROJECT_DIR/dataset/raw/replogle_nadig/GSE264667_jurkat_raw_singlecell_01.h5ad"
readonly OUTPUT_NPZ="$PROJECT_DIR/artifacts/effect_atlas/jurkat_raw_min30_v1.npz"
readonly OUTPUT_JSON="$PROJECT_DIR/artifacts/effect_atlas/jurkat_raw_min30_v1.json"

cd "$PROJECT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

for required_path in "$PYTHON_BIN" "$INPUT_H5AD" scripts/build_raw_effect_atlas.py; do
  if [[ ! -r "$required_path" ]]; then
    echo "Missing required input: $required_path" >&2
    exit 2
  fi
done

for output_path in "$OUTPUT_NPZ" "$OUTPUT_JSON"; do
  if [[ -e "$output_path" ]]; then
    echo "Refusing to overwrite output: $output_path" >&2
    exit 3
  fi
done

mkdir -p "$(dirname "$OUTPUT_NPZ")"
date --iso-8601=seconds
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unknown}"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader

"$PYTHON_BIN" scripts/build_raw_effect_atlas.py \
  "$INPUT_H5AD" \
  "$OUTPUT_NPZ" \
  "$OUTPUT_JSON" \
  --pert-col gene \
  --batch-col gem_group \
  --gene-col gene_name \
  --control non-targeting \
  --min-cells 30 \
  --target-sum 50000 \
  --chunk-rows 2048 \
  --device cuda \
  --require-cuda

test -s "$OUTPUT_NPZ"
test -s "$OUTPUT_JSON"
date --iso-8601=seconds
