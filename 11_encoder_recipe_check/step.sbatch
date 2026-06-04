#!/bin/bash -l
#SBATCH -A brics.b5ac
#SBATCH -p workq
#SBATCH --job-name=r11_deberta_grid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=04:00:00
# Submit via submit_grid.sh

STEP=11_encoder_recipe_check
set -euo pipefail
: "${REPO:?export REPO=/path/to/project_1 before sbatch}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO}/../projects/project_1}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export TMPDIR="${SLURM_TMPDIR:-${OUTPUT_ROOT}/tmp/${SLURM_JOB_ID:-local}}"
mkdir -p "${TMPDIR}"
export TEMP="${TMPDIR}" TMP="${TMPDIR}"

source "${HOME}/miniforge3/etc/profile.d/conda.sh" && conda activate hf-hpc
cd "${REPO}/${STEP}"
echo "=== Encoder recipe check train job ${SLURM_JOB_ID:-local} ==="
export STEP_ARGS="${STEP_ARGS:---train-only}"
python run.py ${STEP_ARGS}
echo "=== Encoder recipe check train complete ==="
