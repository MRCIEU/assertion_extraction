#!/bin/bash -l
#SBATCH -J PB_LR_v2
#SBATCH -A brics.b5ac
#SBATCH -p workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --array=1-182%10
#SBATCH --output=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/logs/phase_b/PB_LR_v2_%A_%a.out
#SBATCH --error=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/logs/phase_b/PB_LR_v2_%A_%a.err

# Phase B — Option A: full retrain of 180 LoRA runs under v2 trainer.
#
# Scope:
#   180 LR runs   = {PB, BL, PL} × {T1B, T1F, T2} × 20 seeds
#   +   2 PL_FT_T2 runs (seeds 17 and 19 — the two that failed in the
#       original PL encoder sbatch due to the source-tree deletion hitting
#       tasks after their scientific_trainer.py was wiped)
#   = 182 total
#
# The v2 trainer (commit 5f93544, tag phase_b_lora_v2) re-implements
# LoRA support after the 2026-04-24 source-tree deletion destroyed the
# uncommitted v1 implementation.  Functionally identical to v1
# (r=16, α=32, dropout=0.05, query/value targets, classifier modules_to_save);
# verified bug-clean via sbatch job 4259839 (7/7 checks PASS).
#
# Safety properties of this sbatch (vs the buggy phase_b_retrain.sbatch
# that was deleted):
#   - NO `rm -rf` before Python (previous bug: run dir wiped before launch
#     meant failed tasks left no partial state for triage).
#   - Idempotent: skips tasks whose run dir already has best.pt + eval.
#   - Explicit fail-fast on missing config / missing module.
#   - `set -euo pipefail`.
#
# Wall-clock estimate: 182 tasks / 10 concurrent ≈ 19 slots × avg 15 min
#   ≈ 5 h wall (PL runs are ~25 min, PB/BL ~10-15 min).

set -euo pipefail
export PROJECT_1_DATA_ROOT=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/..
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

# Idempotence: skip if the run already succeeded end-to-end (both best.pt
# AND phase_b_eval.json present).  This lets us safely re-submit this
# array to retry only failures.
if [[ -f "$RUN_DIR/checkpoints/best.pt" && -f "$RUN_DIR/eval/phase_b_eval.json" ]]; then
    echo "[$(date -u +%FT%TZ)] SKIP ${EXP_ID}: already complete (best.pt + phase_b_eval.json)"
    exit 0
fi

echo "[$(date -u +%FT%TZ)] START task=${SLURM_ARRAY_TASK_ID} exp=${EXP_ID}"
echo "  node=$(hostname)  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo ?)"
echo "  trainer=$(cd /home/b5ac/freddieyu.b5ac/project_1 && git rev-parse --short HEAD 2>/dev/null)  (expected: 5f93544 or newer)"

# Train — writes run_dir/checkpoints/best.pt + run_manifest.json +
# training.log + predictions + metrics.
/usr/bin/python3.11 -m fine_tuning_experiments.train.run_experiment \
    --experiment-id "${EXP_ID}" \
    --config-path "${CFG}" \
    --run-root "${RUN_ROOT}"

# Eval — writes run_dir/eval/phase_b_eval.json + kb_surface_targets.jsonl.
/usr/bin/python3.11 -m fine_tuning_experiments.phase_b.eval.eval_one_run \
    --run-dir "${RUN_DIR}" \
    --overwrite

# Sanity: confirm both output artefacts exist and are non-empty.
for f in "$RUN_DIR/checkpoints/best.pt" "$RUN_DIR/run_manifest.json" "$RUN_DIR/eval/phase_b_eval.json"; do
    if [[ ! -s "$f" ]]; then
        echo "ERROR: expected artefact missing or empty: $f" >&2
        exit 2
    fi
done

echo "[$(date -u +%FT%TZ)] DONE ${EXP_ID}"
