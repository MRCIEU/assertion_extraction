#!/bin/bash
# One bounded controller step. It requeues itself only when the approval file
# matches and work remains. It does not poll from a login node.
#SBATCH --job-name=p1-controller
#SBATCH --partition=workq
#SBATCH --account=brics.b5ac
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --export=NIL

set -euo pipefail
set +x

export PATH=/home/b5ac/freddieyu.b5ac/miniforge3/envs/hf-hpc/bin:/usr/bin:/bin
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/b5ac/freddieyu.b5ac/project_1/src
export SCHEMA_STUDY_STORAGE_ROOT=/home/b5ac/freddieyu.b5ac/projects/project_1

cd /home/b5ac/freddieyu.b5ac/project_1
# --control submits only when the approval file matches this checkout.
# Until then it exits 3 and this script does not requeue and does not call sbatch.
/home/b5ac/freddieyu.b5ac/miniforge3/envs/hf-hpc/bin/python -m schema_study.formal_launch matched --control
status=$?
exit "$status"
