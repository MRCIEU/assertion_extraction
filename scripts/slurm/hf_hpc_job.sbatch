#!/bin/bash
# Non-interactive activation of the existing hf-hpc environment.
# This script does not submit itself. Formal GPU work stays blocked
# until formal_execution_approval.json is present.
#SBATCH --job-name=p1-hf
#SBATCH --partition=workq
#SBATCH --account=brics.b5ac
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --export=NIL

set -euo pipefail
set +x

source /home/b5ac/freddieyu.b5ac/miniforge3/etc/profile.d/conda.sh
conda activate hf-hpc

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/b5ac/freddieyu.b5ac/project_1/src
export SCHEMA_STUDY_STORAGE_ROOT=/home/b5ac/freddieyu.b5ac/projects/project_1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -c 'import sys; print(sys.executable)'
echo "formal_gpu_submission=blocked"
exit 3
