#!/bin/bash -l
#SBATCH -J PA_smoke
#SBATCH -A brics.b5ac
#SBATCH -p workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --array=1-3
#SBATCH --output=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/logs/schema_exp/PA_smoke_%A_%a.out
#SBATCH --error=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/logs/schema_exp/PA_smoke_%A_%a.err

# Phase A smoke — 3 cells before the full 120-run submission.
#   task 1: PA_PB_Sflat_s01   (baseline encoder × baseline schema)
#   task 2: PA_PB_Spair_s01   (exercises Bug A fix: 8-label classifier head)
#   task 3: PA_BL_Sflat_s01   (non-PubMedBERT encoder path)

set -euo pipefail
export PROJECT_1_DATA_ROOT=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/..
export PYTHONPATH=/home/b5ac/freddieyu.b5ac/project_1
export HF_HOME=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/../hf_hub_cache
export TRANSFORMERS_CACHE=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/../hf_hub_cache
cd /home/b5ac/freddieyu.b5ac/project_1

declare -a IDS=( PA_PB_Sflat_s01 PA_PB_Spair_s01 PA_BL_Sflat_s01 )
EXP_ID="${IDS[$((SLURM_ARRAY_TASK_ID - 1))]}"
CFG=/home/b5ac/freddieyu.b5ac/project_1/fine_tuning_experiments/schema_exp/configs/${EXP_ID}.yaml
RUN_ROOT=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/runs/schema_exp

if [[ ! -f "$CFG" ]]; then
  echo "ERROR: config not found: $CFG" >&2; exit 1
fi

mkdir -p "$RUN_ROOT" "/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/logs/schema_exp"
echo "[$(date -u +%FT%TZ)] Starting ${EXP_ID} (task ${SLURM_ARRAY_TASK_ID})"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"

/usr/bin/python3.11 -m fine_tuning_experiments.train.run_experiment \
    --experiment-id "${EXP_ID}" \
    --config-path "${CFG}" \
    --run-root "${RUN_ROOT}"

echo "[$(date -u +%FT%TZ)] Completed ${EXP_ID}"
