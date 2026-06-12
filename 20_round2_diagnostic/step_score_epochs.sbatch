#!/bin/bash -l
#SBATCH --job-name=r2_score_ep
#SBATCH -A brics.b5ac
#SBATCH -p workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

STEP="20_round2_diagnostic"
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
exec > >(tee "${RUNS}/score_epochs_${JOB_TAG}.out") 2> >(tee "${RUNS}/score_epochs_${JOB_TAG}.err" >&2)

echo "=== Round 2 epoch scoring (GPU) ==="
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate hf-hpc

export OUTPUT_ROOT
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
# SLURM_TMPDIR often points at /local/user/<uid> which may not exist on the node.
# Use a Lustre-backed job temp dir so torch import (distributed.nn jit) succeeds.
export TMPDIR="${RUNS}/tmp/${JOB_TAG}"
mkdir -p "${TMPDIR}"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
echo "TMPDIR=${TMPDIR}"

cd "${STEP_DIR}"
python run.py --score-epochs-only ${STEP_ARGS:-} ${FORCE_RESCORE:+--force-rescore}
echo "End: $(date -u)"
python -c "import json; from pathlib import Path; p=Path('${OUTPUT_ROOT}/data/20_round2_diagnostic/epoch_scoring_complete.json'); print(json.loads(p.read_text()) if p.exists() else 'no marker')"
echo "Epoch score files: $(find '${OUTPUT_ROOT}/data/20_round2_diagnostic/scores' -name 'epoch_*.json' 2>/dev/null | wc -l)"
