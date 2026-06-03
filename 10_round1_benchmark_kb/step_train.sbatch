#!/bin/bash -l
#SBATCH -A brics.b5ac
#SBATCH -p workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=12:00:00
# Submit with explicit --job-name, --output, --error from submit_round1_parallel.sh

STEP=10_round1_benchmark_kb
: "${REPO:?export REPO=/path/to/project_1 before sbatch}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO}/../projects/project_1}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

RUNS="${OUTPUT_ROOT}/runs/${STEP}"
mkdir -p "${RUNS}"

# Default: train all assigned models/seeds and evaluate (no cross-model analysis).
export STEP_ARGS="${STEP_ARGS:---train-eval-only}"

source "${HOME}/miniforge3/etc/profile.d/conda.sh" && conda activate hf-hpc
cd "${REPO}/${STEP}"
echo "=== Round 1 train/eval job ${SLURM_JOB_ID:-local} ${JOB_LABEL:-} ==="
echo "STEP_ARGS=${STEP_ARGS}"
python run.py ${STEP_ARGS}
echo "=== Round 1 train/eval complete ${SLURM_JOB_ID:-local} ==="
