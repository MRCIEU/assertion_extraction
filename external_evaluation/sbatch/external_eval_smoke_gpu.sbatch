#!/bin/bash -l
# GPU smoke test; outputs under smoke_runs/<jobid>/. Do not run eval on login node.
#SBATCH -J ext_eval_smoke
#SBATCH -A brics.b5ac
#SBATCH -p workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=00:45:00
#SBATCH --output=/lus/lfs1aip2/projects/b5ac/project_1/external_evaluation/logs/smoke_%j.out
#SBATCH --error=/lus/lfs1aip2/projects/b5ac/project_1/external_evaluation/logs/smoke_%j.err

set -euo pipefail

_LUSTRE_P1="/lus/lfs1aip2/projects/b5ac/project_1"
_EXT_BASE="${EXTERNAL_EVAL_BASE:-${_LUSTRE_P1}/external_evaluation}"
export EXTERNAL_EVAL_ROOT="${_EXT_BASE}/smoke_runs/${SLURM_JOB_ID}"
export PROJECT_1_DATA_ROOT="${PROJECT_1_DATA_ROOT:-${_LUSTRE_P1}}"
export PYTHONPATH="${PYTHONPATH:-/home/b5ac/freddieyu.b5ac/project_1}"

export EXT_EVAL_MAX_EXAMPLES="${EXT_EVAL_MAX_EXAMPLES:-64}"
export EXT_EVAL_SEEDS="${EXT_EVAL_SEEDS:-1}"
export EXT_EVAL_BATCH_SIZE="${EXT_EVAL_BATCH_SIZE:-8}"
export EXT_EVAL_SKIP_RESET=1
export EXT_EVAL_SKIP_MIRROR=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

mkdir -p "${EXTERNAL_EVAL_ROOT}/logs"
mkdir -p "${_EXT_BASE}/logs"
cd /home/b5ac/freddieyu.b5ac/project_1

/usr/bin/python3.11 -m external_evaluation.run_external_evaluation --run-eval

echo "external_eval_smoke_ok job=${SLURM_JOB_ID} root=${EXTERNAL_EVAL_ROOT}"
