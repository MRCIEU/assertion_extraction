#!/bin/bash -l
#SBATCH --job-name=prep_step
#SBATCH -A brics.b5ac
#SBATCH -p workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail

STEP="${1:?step folder name required}"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO}/../projects/project_1}"
RUNS="${OUTPUT_ROOT}/runs/${STEP}"
mkdir -p "${RUNS}"

JOB_TAG="${SLURM_JOB_ID:-local}"
exec > >(tee "${RUNS}/${JOB_TAG}.out") 2> >(tee "${RUNS}/${JOB_TAG}.err" >&2)

echo "=== ${STEP} ==="
echo "Host: $(hostname)"
echo "Start: $(date -u)"
echo "REPO=${REPO}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"

if [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniforge3/etc/profile.d/conda.sh"
  conda activate hf-hpc
fi

export OUTPUT_ROOT
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

cd "${REPO}/${STEP}"
python run.py ${STEP_ARGS:-}
echo "End: $(date -u)"
