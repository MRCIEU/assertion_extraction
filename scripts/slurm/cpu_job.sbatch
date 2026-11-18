#!/bin/bash
# CPU-only batch script. The allocation starts from --export=NIL.
# NIL cannot carry submit-time variables, so every path below is absolute.
# This file does not source a login shell or a user startup file.
#SBATCH --job-name=p1-schema-cpu
#SBATCH --partition=workq
#SBATCH --account=brics.b5ac
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --export=NIL

set -euo pipefail
set +x

if [[ -n "${SLURM_JOB_GPUS:-}" || -n "${SLURM_GPUS:-}" || -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
  echo "This script refuses a GPU allocation." >&2
  exit 2
fi

# Assigned inside the job. These are not read from the login environment.
export PATH=/usr/bin:/bin
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/b5ac/freddieyu.b5ac/project_1/src
export SCHEMA_STUDY_STORAGE_ROOT=/home/b5ac/freddieyu.b5ac/projects/project_1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR=/home/b5ac/freddieyu.b5ac/projects/project_1/cache/matplotlib

# srun on this cluster does not offer --export=NIL. Do not call srun here:
# an srun step can select a different export mode and reload a login environment.
cd /home/b5ac/freddieyu.b5ac/project_1
exec /home/b5ac/freddieyu.b5ac/projects/project_1/environments/analysis/bin/python -m schema_study.cli "$@"
