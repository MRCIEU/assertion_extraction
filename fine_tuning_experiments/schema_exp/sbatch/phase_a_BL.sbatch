#!/bin/bash -l
#SBATCH -J PA_BL
#SBATCH -A brics.b5ac
#SBATCH -p workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --array=1-30%6
#SBATCH --output=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/logs/schema_exp/PA_BL_%A_%a.out
#SBATCH --error=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/logs/schema_exp/PA_BL_%A_%a.err

# Phase A — BioLinkBERT-base encoder
# 30 tasks: 3 schemas × 10 seeds

set -euo pipefail
export PROJECT_1_DATA_ROOT=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/..
export PYTHONPATH=/home/b5ac/freddieyu.b5ac/project_1
export HF_HOME=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/../hf_hub_cache
export TRANSFORMERS_CACHE=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/../hf_hub_cache
cd /home/b5ac/freddieyu.b5ac/project_1

IDS_FILE=/home/b5ac/freddieyu.b5ac/project_1/fine_tuning_experiments/schema_exp/phase_a_bl_ids.txt
EXP_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$IDS_FILE")
CFG=/home/b5ac/freddieyu.b5ac/project_1/fine_tuning_experiments/schema_exp/configs/${EXP_ID}.yaml
RUN_ROOT=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/runs/schema_exp

if [[ -z "${EXP_ID:-}" ]]; then
  echo "ERROR: empty experiment ID for task $SLURM_ARRAY_TASK_ID" >&2; exit 1
fi
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
