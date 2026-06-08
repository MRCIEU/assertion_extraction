#!/bin/bash -l
#SBATCH --job-name=r2_analyze
#SBATCH -A brics.b5ac
#SBATCH -p workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

STEP="20_round2_diagnostic"
# SLURM_SUBMIT_DIR is the cwd at sbatch time, not necessarily this step folder.
if [[ -f "${SLURM_SUBMIT_DIR:-}/run.py" ]]; then
  STEP_DIR="${SLURM_SUBMIT_DIR}"
elif [[ -f "${SLURM_SUBMIT_DIR:-}/${STEP}/run.py" ]]; then
  STEP_DIR="${SLURM_SUBMIT_DIR}/${STEP}"
elif [[ -n "${REPO:-}" && -f "${REPO}/${STEP}/run.py" ]]; then
  STEP_DIR="${REPO}/${STEP}"
else
  STEP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
fi
REPO="$(cd "${STEP_DIR}/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO}/../projects/project_1}"
RUNS="${OUTPUT_ROOT}/runs/${STEP}"
mkdir -p "${RUNS}"

JOB_TAG="${SLURM_JOB_ID:-local}"
exec > >(tee "${RUNS}/analyze_${JOB_TAG}.out") 2> >(tee "${RUNS}/analyze_${JOB_TAG}.err" >&2)

echo "=== Round 2 diagnostic analysis (CPU) ==="
echo "STEP_DIR=${STEP_DIR}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate hf-hpc

export OUTPUT_ROOT
export PYTHONUNBUFFERED=1

cd "${STEP_DIR}"
python run.py --analyze-only ${STEP_ARGS:-} || echo "WARNING: analyze-only failed (continuing to three-point timing)"
python run.py --three-point-timing
echo "End: $(date -u)"
