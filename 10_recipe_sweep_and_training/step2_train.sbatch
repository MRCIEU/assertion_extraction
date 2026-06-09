#!/bin/bash -l
#SBATCH --job-name=step2_train
#SBATCH -A brics.b5ac
#SBATCH -p workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

STEP="10_recipe_sweep_and_training"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  SWEEP_DIR="${SLURM_SUBMIT_DIR}"
else
  SWEEP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
REPO="$(cd "${SWEEP_DIR}/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO}/../projects/project_1}"
RUNS="${OUTPUT_ROOT}/runs/${STEP}"
mkdir -p "${RUNS}"

JOB_TAG="${SLURM_JOB_ID:-local}"
JOB_TMP="${RUNS}/tmp_${JOB_TAG}"
mkdir -p "${JOB_TMP}"
exec > >(tee "${RUNS}/train_${JOB_TAG}.out") 2> >(tee "${RUNS}/train_${JOB_TAG}.err" >&2)

echo "=== Step 2: full-matrix training ==="
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "SLURM_JOB_NAME=${SLURM_JOB_NAME:-step2_train}"
echo "Start: $(date -u)"
echo "REPO=${REPO}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "STEP_ARGS=${STEP_ARGS:-<all 9 encoders>}"
echo "Log: ${RUNS}/train_${JOB_TAG}.out"
echo "Ensure CHOSEN_RECIPE is set in config.py (not None) before submitting."

source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate hf-hpc

export OUTPUT_ROOT
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export TMPDIR="${JOB_TMP}"
export TEMP="${JOB_TMP}"
export TMP="${JOB_TMP}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

echo "TMPDIR=${TMPDIR}"

cd "${SWEEP_DIR}"
python run.py --train-only ${STEP_ARGS:-}
echo "End: $(date -u)"
