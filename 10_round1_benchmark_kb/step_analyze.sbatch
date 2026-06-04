#!/bin/bash -l
#SBATCH -A brics.b5ac
#SBATCH -p workq
#SBATCH --job-name=r1_analyze
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
# CPU-only re-analysis from stored Round-1 results (no GPU, no rescoring).
# Submit: ./submit_analyze.sh

STEP=10_round1_benchmark_kb
set -euo pipefail
: "${REPO:?export REPO=/path/to/project_1 before sbatch}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO}/../projects/project_1}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TMPDIR="${SLURM_TMPDIR:-${OUTPUT_ROOT}/tmp/analyze_${SLURM_JOB_ID:-local}}"
mkdir -p "${TMPDIR}"
export TEMP="${TMPDIR}" TMP="${TMPDIR}"

OUT="${OUTPUT_ROOT}/outputs/${STEP}"
DATA="${OUTPUT_ROOT}/data/${STEP}"

echo "=== Round 1 re-analysis job ${SLURM_JOB_ID:-local} $(date -Is) ==="
echo "REPO=${REPO}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"

N_MARKERS=$(find "${DATA}" -name 'round1_complete.json' 2>/dev/null | wc -l | tr -d ' ')
echo "Completion markers: ${N_MARKERS}/72"
if [[ "${N_MARKERS}" -lt 72 ]]; then
  echo "ERROR: need 72 round1_complete.json markers." >&2
  exit 1
fi

for f in 10_per_run_scores.csv 10_easy_hard_ranking.csv; do
  if [[ ! -f "${OUT}/${f}" ]]; then
    echo "ERROR: missing ${OUT}/${f} (required on disk; will not rescore)." >&2
    exit 1
  fi
done

source "${HOME}/miniforge3/etc/profile.d/conda.sh" && conda activate hf-hpc
cd "${REPO}/${STEP}"
python run.py --analyze-only
echo "=== Round 1 re-analysis complete ${SLURM_JOB_ID:-local} $(date -Is) ==="
