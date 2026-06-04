#!/bin/bash -l
#SBATCH -A brics.b5ac
#SBATCH -p workq
#SBATCH --job-name=r2_diag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

STEP="20_round2_diagnostic"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO}/../projects/project_1}"
RUNS="${OUTPUT_ROOT}/runs/${STEP}"
mkdir -p "${RUNS}"

JOB_TAG="${SLURM_JOB_ID:-local}"
exec > >(tee "${RUNS}/diag_${JOB_TAG}.out") 2> >(tee "${RUNS}/diag_${JOB_TAG}.err" >&2)

echo "=== Round 2 diagnostic ==="
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate hf-hpc

export OUTPUT_ROOT
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

cd "${REPO}/${STEP}"
# Fast path (no epoch KB): omit --rescore-epochs
python run.py ${STEP_ARGS:---rescore-epochs}
echo "End: $(date -u)"
