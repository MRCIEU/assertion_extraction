#!/bin/bash -l
#SBATCH -J PB_PLFT2bf
#SBATCH -A brics.b5ac
#SBATCH -p workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --array=1-2%2
#SBATCH --output=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/logs/phase_b/PB_PLFT2bf_%A_%a.out
#SBATCH --error=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/logs/phase_b/PB_PLFT2bf_%A_%a.err

# Backfill the two missing PL_FT_T2 seeds (s17, s19) so the cell reaches
# 20 seeds.  Independent of the LoRA decision (B.10 drops LoRA but FT data
# is unaffected).  Train + Phase B eval per task.
#
# Safety:
#   - set -euo pipefail
#   - idempotent skip if best.pt + phase_b_eval.json already exist
#   - explicit check that update_regime is full_finetune (defensive vs
#     stale config tree)

set -euo pipefail
export PYTHONPATH=/home/b5ac/freddieyu.b5ac/project_1
export HF_HOME=/lus/lfs1aip2/projects/b5ac/project_1/hf_hub_cache
export TRANSFORMERS_CACHE=/lus/lfs1aip2/projects/b5ac/project_1/hf_hub_cache
cd /home/b5ac/freddieyu.b5ac/project_1

case "$SLURM_ARRAY_TASK_ID" in
    1) EXP_ID=PB_PL_FT_T2_s17 ;;
    2) EXP_ID=PB_PL_FT_T2_s19 ;;
    *) echo "ERROR: unexpected task id $SLURM_ARRAY_TASK_ID" >&2; exit 1 ;;
esac

CFG=/home/b5ac/freddieyu.b5ac/project_1/fine_tuning_experiments/phase_b/configs/${EXP_ID}.yaml
RUN_ROOT=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/runs/phase_b
RUN_DIR=${RUN_ROOT}/${EXP_ID}

if [[ ! -f "$CFG" ]]; then
    echo "ERROR: config not found: $CFG" >&2
    exit 1
fi

cfg_regime=$(grep -E '^update_regime:' "$CFG" | head -1 | awk '{print $2}')
if [[ "$cfg_regime" != "full_finetune" ]]; then
    echo "ERROR: expected update_regime=full_finetune, got $cfg_regime" >&2
    exit 1
fi

if [[ -f "$RUN_DIR/checkpoints/best.pt" && -f "$RUN_DIR/eval/phase_b_eval.json" ]]; then
    echo "[$(date -u +%FT%TZ)] SKIP ${EXP_ID}: already complete"
    exit 0
fi

echo "[$(date -u +%FT%TZ)] START ${EXP_ID}"
echo "  host=$(hostname)"
echo "  trainer commit=$(cd /home/b5ac/freddieyu.b5ac/project_1 && git rev-parse --short HEAD 2>/dev/null || echo 'no-git')"

/usr/bin/python3.11 -m fine_tuning_experiments.phase_b.trainer.run_experiment \
    --experiment-id "${EXP_ID}" \
    --config-path "${CFG}" \
    --run-root "${RUN_ROOT}"

/usr/bin/python3.11 -m fine_tuning_experiments.phase_b.eval.eval_one_run \
    --run-dir "${RUN_DIR}" --overwrite

for f in "$RUN_DIR/checkpoints/best.pt" "$RUN_DIR/run_manifest.json" "$RUN_DIR/eval/phase_b_eval.json"; do
    if [[ ! -s "$f" ]]; then
        echo "ERROR: expected artefact missing or empty: $f" >&2
        exit 2
    fi
done

echo "[$(date -u +%FT%TZ)] DONE ${EXP_ID}"
