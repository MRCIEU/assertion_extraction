#!/bin/bash -l
#SBATCH --job-name=r1_analyze
#SBATCH -A brics.b5ac
#SBATCH -p workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

STEP="11_round1_analysis"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  STEP_DIR="${SLURM_SUBMIT_DIR}"
else
  STEP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
REPO="$(cd "${STEP_DIR}/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO}/../projects/project_1}"
RUNS="${OUTPUT_ROOT}/runs/${STEP}"
mkdir -p "${RUNS}"

JOB_TAG="${SLURM_JOB_ID:-local}"
exec > >(tee "${RUNS}/analyze_${JOB_TAG}.out") 2> >(tee "${RUNS}/analyze_${JOB_TAG}.err" >&2)

echo "=== Round 1 analysis (stage 2, CPU) ==="
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "Start: $(date -u)"
echo "REPO=${REPO}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"

source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate hf-hpc

export OUTPUT_ROOT
export PYTHONUNBUFFERED=1

cd "${STEP_DIR}"
python run.py --analyze-only ${STEP_ARGS:-}
echo "End: $(date -u)"
