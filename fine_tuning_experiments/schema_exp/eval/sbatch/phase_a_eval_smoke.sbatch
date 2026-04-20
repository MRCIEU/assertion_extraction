#!/bin/bash -l
#SBATCH -J PA_EV_SMK
#SBATCH -A brics.b5ac
#SBATCH -p workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/logs/schema_exp_eval/smoke_%j.out
#SBATCH --error=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/logs/schema_exp_eval/smoke_%j.err

# One-shot smoke test for Phase A-eval: run eval_one_run.py on a single PA_ run.
set -euo pipefail
export PROJECT_1_DATA_ROOT=/lus/lfs1aip2/projects/b5ac/project_1
export PYTHONPATH=/home/b5ac/freddieyu.b5ac/project_1
export HF_HOME=/lus/lfs1aip2/projects/b5ac/project_1/hf_hub_cache
export TRANSFORMERS_CACHE=/lus/lfs1aip2/projects/b5ac/project_1/hf_hub_cache
cd /home/b5ac/freddieyu.b5ac/project_1
mkdir -p /lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/logs/schema_exp_eval

RUN_DIR=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/runs/schema_exp/PA_PB_Sflat_s01
echo "[$(date -u +%FT%TZ)] SMOKE test on ${RUN_DIR}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

/usr/bin/python3.11 -m fine_tuning_experiments.schema_exp.eval.eval_one_run \
    --run-dir "${RUN_DIR}" \
    --overwrite

echo "[$(date -u +%FT%TZ)] SMOKE done"
ls -la "${RUN_DIR}/eval/"
