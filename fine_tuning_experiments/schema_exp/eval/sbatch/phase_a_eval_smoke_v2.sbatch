#!/bin/bash -l
#SBATCH -J EV_SMOKE_V2
#SBATCH -A brics.b5ac
#SBATCH -p workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/logs/schema_exp_eval/smoke_v2_%j.out
#SBATCH --error=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/logs/schema_exp_eval/smoke_v2_%j.err

# Step 9a smoke test: verify the v2 eval pipeline (Method A/B/C + full logits save)
# runs end-to-end on one Phase A checkpoint per schema.

set -euo pipefail
export PROJECT_1_DATA_ROOT=/lus/lfs1aip2/projects/b5ac/project_1
export PYTHONPATH=/home/b5ac/freddieyu.b5ac/project_1
export HF_HOME=/lus/lfs1aip2/projects/b5ac/project_1/hf_hub_cache
export TRANSFORMERS_CACHE=/lus/lfs1aip2/projects/b5ac/project_1/hf_hub_cache
cd /home/b5ac/freddieyu.b5ac/project_1
mkdir -p /lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/logs/schema_exp_eval

RUNS=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/runs/schema_exp

echo "[$(date -u +%FT%TZ)] SMOKE v2 — one cell per schema"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

for cell in PA_PB_Sflat_s01 PA_PB_Spair_s01 PA_PB_Smech_s01; do
  echo "---"
  echo "[$(date -u +%FT%TZ)] $cell"
  /usr/bin/python3.11 -m fine_tuning_experiments.schema_exp.eval.eval_one_run \
      --run-dir "${RUNS}/${cell}" \
      --overwrite
done

echo "---"
echo "[$(date -u +%FT%TZ)] Done. Inspect outputs:"
for cell in PA_PB_Sflat_s01 PA_PB_Spair_s01 PA_PB_Smech_s01; do
  f="${RUNS}/${cell}/eval/phase_a_eval.json"
  echo "  ${cell}: $(ls -la ${f} 2>/dev/null | awk '{print $5, $9}')"
  tf="${RUNS}/${cell}/eval/kb_surface_targets.jsonl"
  echo "    targets jsonl: $(ls -la ${tf} 2>/dev/null | awk '{print $5}')"
done
