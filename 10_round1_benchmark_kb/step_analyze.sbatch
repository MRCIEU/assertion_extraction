#!/bin/bash -l
#SBATCH -A brics.b5ac
#SBATCH -p workq
#SBATCH --job-name=r1_analyze
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
# Submit via submit_analyze.sh (sets --output/--error under runs/)

STEP=10_round1_benchmark_kb
set -euo pipefail
: "${REPO:?export REPO=/path/to/project_1 before sbatch}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO}/../projects/project_1}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
export TOKENIZERS_PARALLELISM=false
export TMPDIR="${SLURM_TMPDIR:-${OUTPUT_ROOT}/tmp/analyze_${SLURM_JOB_ID:-local}}"
mkdir -p "${TMPDIR}"
export TEMP="${TMPDIR}" TMP="${TMPDIR}"

source "${HOME}/miniforge3/etc/profile.d/conda.sh" && conda activate hf-hpc
cd "${REPO}/${STEP}"
echo "=== Round 1 full analysis job ${SLURM_JOB_ID:-local} ==="
echo "REPO=${REPO} OUTPUT_ROOT=${OUTPUT_ROOT}"
python run.py --analyze-only
echo "=== Round 1 analysis complete ${SLURM_JOB_ID:-local} ==="
