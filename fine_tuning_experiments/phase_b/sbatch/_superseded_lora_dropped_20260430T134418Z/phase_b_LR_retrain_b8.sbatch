#!/bin/bash -l
#SBATCH -J PB_LR_B8
#SBATCH -A brics.b5ac
#SBATCH -p workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --array=1-182%10
#SBATCH --output=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/logs/phase_b/PB_LR_B8_%A_%a.out
#SBATCH --error=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/logs/phase_b/PB_LR_B8_%A_%a.err

# Phase B — Amendment B.8 retrain: 180 LoRA runs at LR=3.0e-4 + 2 PL_FT_T2 seeds.
#
# Replaces the superseded phase_b_LR_retrain.sbatch (job 4260475), which was
# cancelled after the first 26 completed runs were discovered to be degenerate
# (bit-identical dev_macro_f1 = 0.12624... at LR=2e-5 across all seeds).
#
# Amendment B.8 changes EXACTLY ONE hyperparameter:
#   LoRA learning_rate : 2.0e-05 -> 3.0e-04  (Hu et al. 2021 / PEFT default).
#
# All other LoRA specs (r=16, alpha=32, dropout=0.05, target Q/V, classifier
# modules_to_save, bias=none, seeds, schedules, max_updates, early stopping,
# selection metric, batch size, max_length) are identical to the locked
# pre-registration.  Configs were bulk-edited in place by
# `scripts/update_lora_lr_to_3e4.py` and committed before this array was
# submitted.
#
# Precondition: smoke job PB_PB_LR_T1B_s99 (SLURM 4268527) must PASS all three
# non-degeneracy criteria (dev_f1 > 0.20 by step 512, eval-point variation > 1e-4,
# macro_f1_excluding_negative > 0 at some step).
#
# Safety properties (same as the superseded v1 retrain):
#   - Idempotent (skip if best.pt + eval exists)
#   - No `rm -rf` before Python
#   - set -euo pipefail
#   - Explicit fail-fast on missing config / missing module.
# Wall-clock estimate: 182 / 10 ≈ 19 slots × avg 15 min ≈ 5 h wall.

set -euo pipefail
export PYTHONPATH=/home/b5ac/freddieyu.b5ac/project_1
export HF_HOME=/lus/lfs1aip2/projects/b5ac/project_1/hf_hub_cache
export TRANSFORMERS_CACHE=/lus/lfs1aip2/projects/b5ac/project_1/hf_hub_cache
cd /home/b5ac/freddieyu.b5ac/project_1

IDS_FILE=/home/b5ac/freddieyu.b5ac/project_1/fine_tuning_experiments/phase_b/phase_b_retrain_v2_ids.txt
EXP_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$IDS_FILE")
CFG=/home/b5ac/freddieyu.b5ac/project_1/fine_tuning_experiments/phase_b/configs/${EXP_ID}.yaml
RUN_ROOT=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/runs/phase_b
RUN_DIR=${RUN_ROOT}/${EXP_ID}

if [[ -z "${EXP_ID:-}" ]]; then
    echo "ERROR: empty experiment ID for task $SLURM_ARRAY_TASK_ID" >&2
    exit 1
fi
if [[ ! -f "$CFG" ]]; then
    echo "ERROR: config not found: $CFG" >&2
    exit 1
fi

# Verify the config's LR is the amended value (defensive: catches stale checkouts)
cfg_lr=$(grep -E '^[[:space:]]+learning_rate:' "$CFG" | head -1 | awk '{print $2}')
if [[ "$cfg_lr" != "3.0e-04" ]]; then
    # LoRA configs must be 3.0e-04; FT (PL_FT_T2_s17, s19) retains 2.0e-05
    if [[ "$EXP_ID" != *"_FT_"* ]]; then
        echo "ERROR: LoRA config $CFG has learning_rate=$cfg_lr, expected 3.0e-04" >&2
        exit 1
    fi
fi

if [[ -f "$RUN_DIR/checkpoints/best.pt" && -f "$RUN_DIR/eval/phase_b_eval.json" ]]; then
    echo "[$(date -u +%FT%TZ)] SKIP ${EXP_ID}: already complete"
    exit 0
fi

echo "[$(date -u +%FT%TZ)] START task=${SLURM_ARRAY_TASK_ID} exp=${EXP_ID} lr=${cfg_lr}"
echo "  node=$(hostname)"
echo "  trainer commit=$(cd /home/b5ac/freddieyu.b5ac/project_1 && git rev-parse --short HEAD)"

/usr/bin/python3.11 -m fine_tuning_experiments.phase_b.trainer.run_experiment \
    --experiment-id "${EXP_ID}" \
    --config-path "${CFG}" \
    --run-root "${RUN_ROOT}"

/usr/bin/python3.11 -m fine_tuning_experiments.phase_b.eval.eval_one_run \
    --run-dir "${RUN_DIR}" \
    --overwrite

for f in "$RUN_DIR/checkpoints/best.pt" "$RUN_DIR/run_manifest.json" "$RUN_DIR/eval/phase_b_eval.json"; do
    if [[ ! -s "$f" ]]; then
        echo "ERROR: expected artefact missing or empty: $f" >&2
        exit 2
    fi
done

echo "[$(date -u +%FT%TZ)] DONE ${EXP_ID}"
