#!/bin/bash -l
#SBATCH -J PA_EV_V1
#SBATCH -A brics.b5ac
#SBATCH -p workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=00:45:00
#SBATCH --array=1-120%12
#SBATCH --output=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/logs/schema_exp_eval/PA_EV_V1_%A_%a.out
#SBATCH --error=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/logs/schema_exp_eval/PA_EV_V1_%A_%a.err

# Step 0.5 / Step 8b: re-evaluate all 120 Phase A checkpoints under the frozen
# eval-pipeline-v1.0 tag (method A/B/C + full pre-softmax logits saved).
# Overwrites existing phase_a_eval.json + kb_surface_targets.jsonl per run.

set -euo pipefail
export PROJECT_1_DATA_ROOT=/lus/lfs1aip2/projects/b5ac/project_1
export PYTHONPATH=/home/b5ac/freddieyu.b5ac/project_1
export HF_HOME=/lus/lfs1aip2/projects/b5ac/project_1/hf_hub_cache
export TRANSFORMERS_CACHE=/lus/lfs1aip2/projects/b5ac/project_1/hf_hub_cache
cd /home/b5ac/freddieyu.b5ac/project_1

IDS_FILE=/home/b5ac/freddieyu.b5ac/project_1/fine_tuning_experiments/schema_exp/phase_a_experiment_ids.txt
EXP_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$IDS_FILE")
RUN_ROOT=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/runs/schema_exp
RUN_DIR="${RUN_ROOT}/${EXP_ID}"

mkdir -p /lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/logs/schema_exp_eval

if [[ -z "${EXP_ID:-}" ]]; then
  echo "ERROR: empty experiment ID for task $SLURM_ARRAY_TASK_ID" >&2; exit 1
fi
if [[ ! -d "$RUN_DIR" ]]; then
  echo "ERROR: run dir not found: $RUN_DIR" >&2; exit 1
fi

echo "[$(date -u +%FT%TZ)] v1.0 re-eval ${EXP_ID} (task ${SLURM_ARRAY_TASK_ID})"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"

/usr/bin/python3.11 -m fine_tuning_experiments.schema_exp.eval.eval_one_run \
    --run-dir "${RUN_DIR}" \
    --overwrite

echo "[$(date -u +%FT%TZ)] Completed v1.0 re-eval for ${EXP_ID}"
