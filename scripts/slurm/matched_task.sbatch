#!/bin/bash
# One matched-programme task. The historical formal_20261115 launcher does not call this script.
#SBATCH --job-name=p1-matched
#SBATCH --partition=workq
#SBATCH --account=brics.b5ac
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --export=ALL

set -euo pipefail
set +x

source /home/b5ac/freddieyu.b5ac/miniforge3/etc/profile.d/conda.sh
conda activate hf-hpc
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/home/b5ac/freddieyu.b5ac/project_1/src
export SCHEMA_STUDY_STORAGE_ROOT=/home/b5ac/freddieyu.b5ac/projects/project_1
export SCHEMA_STUDY_PROGRAMME=formal_matched_e0_e7

cd /home/b5ac/freddieyu.b5ac/project_1
gpus="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS:-1}}"
if [[ "$gpus" -gt 1 ]]; then
  exec torchrun --nproc_per_node="$gpus" -m schema_study.production_dispatch run "$1" "$2"
fi
exec python -m schema_study.production_dispatch run "$1" "$2"
