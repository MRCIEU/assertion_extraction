#!/bin/bash
#SBATCH --job-name=p1-inputs
#SBATCH --partition=workq
#SBATCH --account=brics.b5ac
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --export=ALL

set -euo pipefail
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
mkdir -p "$TMPDIR"
source /home/b5ac/freddieyu.b5ac/miniforge3/etc/profile.d/conda.sh
conda activate hf-hpc
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
CODE_ROOT="${SCHEMA_STUDY_CODE_ROOT:-/home/b5ac/freddieyu.b5ac/project_1}"
export PYTHONPATH="${CODE_ROOT}/src"
cd "$CODE_ROOT"
python scripts/build_mesh_ancestor_inputs.py
