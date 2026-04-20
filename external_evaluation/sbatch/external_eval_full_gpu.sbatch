#!/bin/bash -l
# Full external evaluation on GPU. Do not run the Python driver on the login node for this workload.
#SBATCH -J ext_eval_full
#SBATCH -A brics.b5ac
#SBATCH -p workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=/lus/lfs1aip2/projects/b5ac/project_1/external_evaluation/logs/ext_eval_%j.out
#SBATCH --error=/lus/lfs1aip2/projects/b5ac/project_1/external_evaluation/logs/ext_eval_%j.err

set -euo pipefail

_LUSTRE_P1="/lus/lfs1aip2/projects/b5ac/project_1"
export PROJECT_1_DATA_ROOT="${PROJECT_1_DATA_ROOT:-${_LUSTRE_P1}}"
export EXTERNAL_EVAL_ROOT="${EXTERNAL_EVAL_ROOT:-${_LUSTRE_P1}/external_evaluation}"
export PYTHONPATH="${PYTHONPATH:-/home/b5ac/freddieyu.b5ac/project_1}"
export EXT_EVAL_SKIP_RESET=1
export EXT_EVAL_SKIP_MIRROR="${EXT_EVAL_SKIP_MIRROR:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

mkdir -p "${EXTERNAL_EVAL_ROOT}/logs"
cd /home/b5ac/freddieyu.b5ac/project_1

/usr/bin/python3.11 -m external_evaluation.run_external_evaluation --run-eval

echo "external_eval_job_ok"
