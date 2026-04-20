#!/bin/bash -l
#SBATCH -J BRIDGE_EQV
#SBATCH -A brics.b5ac
#SBATCH -p workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gpus=1
#SBATCH --time=00:45:00
#SBATCH --output=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/logs/phase_b/bridge_eqv_%j.out
#SBATCH --error=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/logs/phase_b/bridge_eqv_%j.err

# Bridge equivalence smoke: re-run PA_PB_Sflat_s01 config with the new clean
# .py trainer.  Compare to the saved Phase A artifacts.
#
# Tier 3 acceptance (per §11.6 of paper design):
#   * final dev macro-F1 per stage within ±0.01 of Phase A's saved value
#   * label2id matches exactly
#   * checkpoint schema (5 expected keys) matches
#
# Tier 1 (step-level loss to 1e-6) and Tier 2 (1e-4) are NOT expected to pass
# — the new trainer has different RNG consumption order.

set -euo pipefail
export PROJECT_1_DATA_ROOT=/lus/lfs1aip2/projects/b5ac/project_1
export PYTHONPATH=/home/b5ac/freddieyu.b5ac/project_1
export HF_HOME=/lus/lfs1aip2/projects/b5ac/project_1/hf_hub_cache
export TRANSFORMERS_CACHE=/lus/lfs1aip2/projects/b5ac/project_1/hf_hub_cache
cd /home/b5ac/freddieyu.b5ac/project_1
mkdir -p /lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/logs/phase_b

RUN_ROOT=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/runs/bridge_equivalence
CFG=/home/b5ac/freddieyu.b5ac/project_1/fine_tuning_experiments/schema_exp/configs/PA_PB_Sflat_s01.yaml
# Use a distinct experiment-id so outputs don't collide with the Phase A run
EXP_ID=BRIDGE_PB_Sflat_s01

echo "[$(date -u +%FT%TZ)] bridge equivalence: ${EXP_ID}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"

/usr/bin/python3.11 -m fine_tuning_experiments.phase_b.trainer.run_experiment \
    --experiment-id "${EXP_ID}" \
    --config-path "${CFG}" \
    --run-root "${RUN_ROOT}"

echo "[$(date -u +%FT%TZ)] bridge run complete; invoking comparison"

/usr/bin/python3.11 -m fine_tuning_experiments.phase_b.trainer.bridge_equivalence_compare \
    --phase-a-run /lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/runs/schema_exp/PA_PB_Sflat_s01 \
    --phase-b-run "${RUN_ROOT}/${EXP_ID}"

echo "[$(date -u +%FT%TZ)] done"
