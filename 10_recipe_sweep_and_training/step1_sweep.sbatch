#!/bin/bash -l
#SBATCH --job-name=step1_sweep
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
SWEEP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SWEEP_DIR}/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO}/../projects/project_1}"
RUNS="${OUTPUT_ROOT}/runs/${STEP}"
mkdir -p "${RUNS}"

JOB_TAG="${SLURM_JOB_ID:-local}"
exec > >(tee "${RUNS}/sweep_${JOB_TAG}.out") 2> >(tee "${RUNS}/sweep_${JOB_TAG}.err" >&2)

echo "=== Step 1: recipe sweep ==="
echo "Host: $(hostname)"
echo "Start: $(date -u)"

source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate hf-hpc

export OUTPUT_ROOT
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

cd "${SWEEP_DIR}"
python run.py --sweep-only ${STEP_ARGS:-}
python run.py --sweep-advisory-only
echo "End: $(date -u)"
