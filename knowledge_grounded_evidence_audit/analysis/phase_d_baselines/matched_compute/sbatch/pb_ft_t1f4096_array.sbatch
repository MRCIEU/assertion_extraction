#!/bin/bash -l
#SBATCH -J PB_T1F4096
#SBATCH -A brics.b5ac
#SBATCH -p workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --array=1-20%6
#SBATCH --output=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/logs/phase_d_pb_t1f4096/%A_%a.out
#SBATCH --error=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/logs/phase_d_pb_t1f4096/%A_%a.err

# Phase 2C — PubMedBERT-base × FT × T1F with max_updates=4096 (matched compute).
# Requires COMMITMENT.md sign-off + configs generated via gen_t1f4096_configs.py.
# Concurrency 6 matches the Phase B PL driver.

set -euo pipefail
export PROJECT_1_DATA_ROOT=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/..
export PYTHONPATH=/home/b5ac/freddieyu.b5ac/project_1
export HF_HOME=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/../hf_hub_cache
export TRANSFORMERS_CACHE=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/../hf_hub_cache
cd /home/b5ac/freddieyu.b5ac/project_1

IDS_FILE=/home/b5ac/freddieyu.b5ac/project_1/knowledge_grounded_evidence_audit/analysis/phase_d_baselines/matched_compute/pb_ft_t1f4096_ids.txt
EXP_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$IDS_FILE")
CFG=/home/b5ac/freddieyu.b5ac/project_1/knowledge_grounded_evidence_audit/analysis/phase_d_baselines/matched_compute/configs/${EXP_ID}.yaml
RUN_ROOT=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/runs/phase_b

mkdir -p "$RUN_ROOT" "/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/logs/phase_d_pb_t1f4096"
if [[ -z "${EXP_ID:-}" ]]; then
  echo "ERROR: empty experiment ID for task $SLURM_ARRAY_TASK_ID" >&2; exit 1
fi
if [[ ! -f "$CFG" ]]; then
  echo "ERROR: config not found: $CFG (run gen_t1f4096_configs.py?)" >&2; exit 1
fi

echo "[$(date -u +%FT%TZ)] Starting ${EXP_ID} (task ${SLURM_ARRAY_TASK_ID})"
/usr/bin/python3.11 -m fine_tuning_experiments.train.run_experiment \
    --experiment-id "${EXP_ID}" \
    --config-path "${CFG}" \
    --run-root "${RUN_ROOT}"
echo "[$(date -u +%FT%TZ)] Completed ${EXP_ID}"
