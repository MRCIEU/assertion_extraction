#!/bin/bash -l
#SBATCH -A brics.b5ac
#SBATCH -p workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=04:00:00

set -euo pipefail

: "${REPO:?export REPO before sbatch}"
: "${SWEEP_MODEL:?export SWEEP_MODEL=pubmedbert_base|roberta_base|distilbert_base}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO}/../projects/project_1}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

# Compute nodes may not have /local/user/<uid>; use SLURM scratch or project tmp.
export TMPDIR="${SLURM_TMPDIR:-${OUTPUT_ROOT}/tmp/sweep/${SLURM_JOB_ID:-local}}"
mkdir -p "${TMPDIR}"
export TEMP="${TMPDIR}" TMP="${TMPDIR}"

source "${HOME}/miniforge3/etc/profile.d/conda.sh" && conda activate hf-hpc
cd "${REPO}/10_round1_benchmark_kb"

echo "=== Sweep job ${SLURM_JOB_ID:-local} model=${SWEEP_MODEL} ==="
echo "TMPDIR=${TMPDIR}"
python -m sweep.run_sweep --model "${SWEEP_MODEL}"
echo "=== Sweep job complete ==="
