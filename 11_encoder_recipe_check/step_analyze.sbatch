#!/bin/bash -l
#SBATCH -A brics.b5ac
#SBATCH -p workq
#SBATCH --job-name=r11_analyze
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
# CPU-only finalisation (no GPU)

STEP=11_encoder_recipe_check
set -euo pipefail
: "${REPO:?export REPO=/path/to/project_1 before sbatch}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO}/../projects/project_1}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg

source "${HOME}/miniforge3/etc/profile.d/conda.sh" && conda activate hf-hpc
cd "${REPO}/${STEP}"
echo "=== Encoder recipe check analysis ${SLURM_JOB_ID:-local} ==="
python run.py --analyze-only
echo "=== Analysis complete ==="
