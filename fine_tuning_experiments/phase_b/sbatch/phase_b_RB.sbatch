#!/bin/bash -l
#SBATCH -J PB_RB
#SBATCH -A brics.b5ac
#SBATCH -p workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --array=1-10%6
#SBATCH --output=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/logs/phase_b/PB_RB_%A_%a.out
#SBATCH --error=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/logs/phase_b/PB_RB_%A_%a.err

# Phase B — RoBERTa-base reference (anchor for H1 encoder contrast).
# 10 tasks: FT × T2 × 10 seeds.  RB is the out-of-domain reference;
# it is pinned to the anchor cell (FT, T2) so it contrasts only against
# PB-FT-T2 / BL-FT-T2 / PL-FT-T2 (the H1 target cells).

set -euo pipefail
export PROJECT_1_DATA_ROOT=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/..
export PYTHONPATH=/home/b5ac/freddieyu.b5ac/project_1
export HF_HOME=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/../hf_hub_cache
export TRANSFORMERS_CACHE=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/../hf_hub_cache
cd /home/b5ac/freddieyu.b5ac/project_1

IDS_FILE=/home/b5ac/freddieyu.b5ac/project_1/fine_tuning_experiments/phase_b/phase_b_rb_ids.txt
EXP_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$IDS_FILE")
CFG=/home/b5ac/freddieyu.b5ac/project_1/fine_tuning_experiments/phase_b/configs/${EXP_ID}.yaml
RUN_ROOT=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/runs/phase_b

if [[ -z "${EXP_ID:-}" ]]; then
  echo "ERROR: empty experiment ID for task $SLURM_ARRAY_TASK_ID" >&2; exit 1
fi
if [[ ! -f "$CFG" ]]; then
  echo "ERROR: config not found: $CFG" >&2; exit 1
fi

mkdir -p "$RUN_ROOT" "/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/logs/phase_b"
echo "[$(date -u +%FT%TZ)] Starting ${EXP_ID} (task ${SLURM_ARRAY_TASK_ID})"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"

/usr/bin/python3.11 -m fine_tuning_experiments.train.run_experiment \
    --experiment-id "${EXP_ID}" \
    --config-path "${CFG}" \
    --run-root "${RUN_ROOT}"

echo "[$(date -u +%FT%TZ)] Completed ${EXP_ID}"
