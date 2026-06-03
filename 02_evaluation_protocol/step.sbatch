#!/bin/bash -l
#SBATCH --job-name=02_evaluation_protocol
#SBATCH -A brics.b5ac
#SBATCH -p workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=02_evaluation_protocol_%j.slurm.out
#SBATCH --error=02_evaluation_protocol_%j.slurm.err

STEP=02_evaluation_protocol
: "${REPO:?export REPO=/path/to/project_1 before sbatch}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO}/../projects/project_1}"
RUNS="${OUTPUT_ROOT}/runs/${STEP}"
mkdir -p "${RUNS}"
exec > >(tee "${RUNS}/${SLURM_JOB_ID:-local}.out") 2> >(tee "${RUNS}/${SLURM_JOB_ID:-local}.err" >&2)
source "${HOME}/miniforge3/etc/profile.d/conda.sh" && conda activate hf-hpc
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
cd "${REPO}/${STEP}"
python run.py ${STEP_ARGS:-}
