#!/bin/bash
# Activate hf-hpc and dispatch one prepared task.
# CPU preparation runs without an approval file.
# GPU training and generation exit 3 until the approval file matches the manifest.
#SBATCH --job-name=p1-hf
#SBATCH --partition=workq
#SBATCH --account=brics.b5ac
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --export=ALL

set -euo pipefail
set +x

source /home/b5ac/freddieyu.b5ac/miniforge3/etc/profile.d/conda.sh
conda activate hf-hpc

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/b5ac/freddieyu.b5ac/project_1/src
export SCHEMA_STUDY_STORAGE_ROOT=/home/b5ac/freddieyu.b5ac/projects/project_1
export TOKENIZERS_PARALLELISM=false

cd /home/b5ac/freddieyu.b5ac/project_1
python -m schema_study.production_dispatch "$@"
