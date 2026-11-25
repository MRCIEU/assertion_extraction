#!/bin/bash
# Tokenise the remaining few-shot and supervised workloads. No GPU is requested.
#SBATCH --job-name=p1-qwen-rest
#SBATCH --partition=workq
#SBATCH --account=brics.b5ac
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --export=NIL

set -euo pipefail
set +x

if [[ -n "${SLURM_JOB_GPUS:-}" || -n "${SLURM_GPUS:-}" || -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
  echo "This script refuses a GPU allocation." >&2
  exit 2
fi

export PATH=/home/b5ac/freddieyu.b5ac/miniforge3/envs/hf-hpc/bin:/usr/bin:/bin
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/b5ac/freddieyu.b5ac/project_1/src
export SCHEMA_STUDY_STORAGE_ROOT=/home/b5ac/freddieyu.b5ac/projects/project_1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /home/b5ac/freddieyu.b5ac/project_1
exec /home/b5ac/freddieyu.b5ac/miniforge3/envs/hf-hpc/bin/python -m schema_study.qwen_workload remaining
