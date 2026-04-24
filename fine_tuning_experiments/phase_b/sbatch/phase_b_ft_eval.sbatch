#!/bin/bash -l
#SBATCH -J PB_FT_eval
#SBATCH -A brics.b5ac
#SBATCH -p workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --array=1-188%15
#SBATCH --output=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/logs/phase_b/PB_FT_eval_%A_%a.out
#SBATCH --error=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/logs/phase_b/PB_FT_eval_%A_%a.err

# Eval backlog: 188 pre-deletion FT runs (PB/BL/PL/RB) were trained but
# never evaluated with the Phase B eval pipeline.  This array invokes
# `phase_b.eval.eval_one_run` to produce phase_b_eval.json + the
# kb_surface_targets.jsonl sidecar for each FT run.  Independent of
# the LoRA LR=3e-4 retrain (amendment B.8).
#
# No training, no checkpoint writing, no dependency ordering — pure
# forward-only inference on BioRED + BC5CDR test + KB surface targets.
# Expected wall-clock: ~10 min per run × 188 / 15 concurrent ≈ 2 h total.

set -euo pipefail
export PYTHONPATH=/home/b5ac/freddieyu.b5ac/project_1
export HF_HOME=/lus/lfs1aip2/projects/b5ac/project_1/hf_hub_cache
export TRANSFORMERS_CACHE=/lus/lfs1aip2/projects/b5ac/project_1/hf_hub_cache
cd /home/b5ac/freddieyu.b5ac/project_1

IDS_FILE=/home/b5ac/freddieyu.b5ac/project_1/fine_tuning_experiments/phase_b/phase_b_ft_eval_ids.txt
EXP_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$IDS_FILE")
RUN_ROOT=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/runs/phase_b
RUN_DIR=${RUN_ROOT}/${EXP_ID}

if [[ -z "${EXP_ID:-}" ]]; then
    echo "ERROR: empty experiment ID for task $SLURM_ARRAY_TASK_ID" >&2
    exit 1
fi
if [[ ! -f "$RUN_DIR/checkpoints/best.pt" ]]; then
    echo "ERROR: best.pt missing for $EXP_ID" >&2
    exit 1
fi

# Idempotent skip
if [[ -f "$RUN_DIR/eval/phase_b_eval.json" ]]; then
    echo "[$(date -u +%FT%TZ)] SKIP ${EXP_ID}: eval already exists"
    exit 0
fi

echo "[$(date -u +%FT%TZ)] START ${EXP_ID}"
echo "  node=$(hostname)  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo ?)"

/usr/bin/python3.11 -m fine_tuning_experiments.phase_b.eval.eval_one_run \
    --run-dir "${RUN_DIR}" \
    --overwrite

if [[ ! -s "$RUN_DIR/eval/phase_b_eval.json" ]]; then
    echo "ERROR: eval file missing or empty after run" >&2
    exit 2
fi

echo "[$(date -u +%FT%TZ)] DONE ${EXP_ID}"
