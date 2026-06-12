#!/bin/bash -l
#SBATCH --job-name=r2_stratum
#SBATCH -A brics.b5ac
#SBATCH -p workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -euo pipefail

STEP="20_round2_diagnostic"
STEP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO="$(cd "${STEP_DIR}/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO}/../projects/project_1}"
RUNS="${OUTPUT_ROOT}/runs/${STEP}"
mkdir -p "${RUNS}"

JOB_TAG="${SLURM_JOB_ID:-local}"
exec > >(tee "${RUNS}/stratum_epoch1_${JOB_TAG}.out") 2> >(tee "${RUNS}/stratum_epoch1_${JOB_TAG}.err" >&2)

source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate hf-hpc
export OUTPUT_ROOT PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=8
export TMPDIR="${RUNS}/tmp/${JOB_TAG}"
mkdir -p "${TMPDIR}"

cd "${STEP_DIR}"
python run.py --stratum-epoch1-only ${STEP_ARGS:-}
echo "End: $(date -u)"
