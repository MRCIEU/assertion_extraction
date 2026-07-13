#!/bin/bash -l
#SBATCH --job-name=manuscript_regen
#SBATCH -A brics.b5ac
#SBATCH -p workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -euo pipefail

STEP="manuscript_regenerate"
if [[ -f "${SLURM_SUBMIT_DIR:-}/run_all.py" ]]; then
  STEP_DIR="${SLURM_SUBMIT_DIR}"
elif [[ -f "${SLURM_SUBMIT_DIR:-}/${STEP}/run_all.py" ]]; then
  STEP_DIR="${SLURM_SUBMIT_DIR}/${STEP}"
elif [[ -n "${REPO:-}" && -f "${REPO}/${STEP}/run_all.py" ]]; then
  STEP_DIR="${REPO}/${STEP}"
else
  STEP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
fi
REPO="$(cd "${STEP_DIR}/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO}/../projects/project_1}"
RUNS="${OUTPUT_ROOT}/runs/${STEP}"
mkdir -p "${RUNS}"
# shellcheck source=/dev/null
source "${REPO}/_slurm_env.sh"
setup_slurm_tmpdir "${RUNS}"

JOB_TAG="${SLURM_JOB_ID:-local}"
exec > >(tee "${RUNS}/manuscript_${JOB_TAG}.out") 2> >(tee "${RUNS}/manuscript_${JOB_TAG}.err" >&2)

echo "=== Manuscript regeneration (CPU) ==="
echo "REPO=${REPO}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate hf-hpc

export OUTPUT_ROOT
export PYTHONUNBUFFERED=1
export SKIP_GIT_COMMIT=1

cd "${REPO}"
python -m manuscript_regenerate.run_all
echo "End: $(date -u)"
