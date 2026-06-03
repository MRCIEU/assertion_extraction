#!/bin/bash -l
#SBATCH --job-name=10_round1_benchmark_kb
#SBATCH -A brics.b5ac
#SBATCH -p workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=12:00:00
# Prefer submit_round1_parallel.sh for the full matrix. This script is for ad-hoc subsets.
#SBATCH --output=10_round1_benchmark_kb_%j.slurm.out
#SBATCH --error=10_round1_benchmark_kb_%j.slurm.err

STEP=10_round1_benchmark_kb
: "${REPO:?export REPO=/path/to/project_1 before sbatch}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO}/../projects/project_1}"
RUNS="${OUTPUT_ROOT}/runs/${STEP}"
mkdir -p "${RUNS}"
exec > >(tee "${RUNS}/${SLURM_JOB_ID:-local}.out") 2> >(tee "${RUNS}/${SLURM_JOB_ID:-local}.err" >&2)
source "${HOME}/miniforge3/etc/profile.d/conda.sh" && conda activate hf-hpc
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
cd "${REPO}/${STEP}"
# Default to train+eval only; use --analyze-only explicitly for finalisation.
export STEP_ARGS="${STEP_ARGS:---train-eval-only}"
python run.py ${STEP_ARGS}
