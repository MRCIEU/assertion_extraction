#!/bin/bash -l
#SBATCH --job-name=r1_untrained
#SBATCH -A brics.b5ac
#SBATCH -p workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00

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
exec > >(tee "${RUNS}/score_untrained_${JOB_TAG}.out") 2> >(tee "${RUNS}/score_untrained_${JOB_TAG}.err" >&2)

echo "=== Round 1 untrained-floor scoring (stage 1b, GPU) ==="
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "Start: $(date -u)"
echo "REPO=${REPO}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "STEP_ARGS=${STEP_ARGS:-<all 9 encoders>}"

source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate hf-hpc

export OUTPUT_ROOT
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

cd "${STEP_DIR}"
python run.py --score-untrained-only ${STEP_ARGS:-} ${FORCE_SCORE:+--force-score}
echo "End: $(date -u)"
echo "Untrained markers: $(find "${OUTPUT_ROOT}/data/11_round1_analysis/scores" -name untrained_scoring_complete.json 2>/dev/null | wc -l)/9"
