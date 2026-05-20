#!/bin/bash -l
#SBATCH -J PB_T1F4096_eval
#SBATCH -A brics.b5ac
#SBATCH -p workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=01:30:00
#SBATCH --array=1-20%6
#SBATCH --output=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/logs/phase_d_pb_t1f4096_eval/%A_%a.out
#SBATCH --error=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/logs/phase_d_pb_t1f4096_eval/%A_%a.err

# Phase 2C follow-up — Phase B eval only for PB_PB_FT_T1F4096_s{01..20}.
# Training is produced by pb_ft_t1f4096_array.sbatch; this job writes
# eval/phase_b_eval.json + eval/kb_surface_targets.jsonl (same contract as
# other Phase B FT cells). Idempotent: skips if eval JSON already exists.

set -euo pipefail
export PYTHONPATH=/home/b5ac/freddieyu.b5ac/project_1
export HF_HOME=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/../hf_hub_cache
export TRANSFORMERS_CACHE=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/../hf_hub_cache
cd /home/b5ac/freddieyu.b5ac/project_1

IDS_FILE=/home/b5ac/freddieyu.b5ac/project_1/knowledge_grounded_evidence_audit/analysis/phase_d_baselines/matched_compute/pb_ft_t1f4096_ids.txt
EXP_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$IDS_FILE")
RUN_ROOT=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/runs/phase_b
RUN_DIR="${RUN_ROOT}/${EXP_ID}"

mkdir -p "/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/logs/phase_d_pb_t1f4096_eval"

if [[ -z "${EXP_ID:-}" ]]; then
  echo "ERROR: empty experiment ID for task $SLURM_ARRAY_TASK_ID" >&2
  exit 1
fi
if [[ ! -f "$RUN_DIR/checkpoints/best.pt" ]]; then
  echo "ERROR: best.pt missing for $EXP_ID" >&2
  exit 1
fi

if [[ -f "$RUN_DIR/eval/phase_b_eval.json" ]]; then
  echo "[$(date -u +%FT%TZ)] SKIP ${EXP_ID}: eval already exists"
  exit 0
fi

echo "[$(date -u +%FT%TZ)] START ${EXP_ID}"
echo "  node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo ?)"

/usr/bin/python3.11 -m fine_tuning_experiments.phase_b.eval.eval_one_run \
    --run-dir "${RUN_DIR}"

if [[ ! -s "$RUN_DIR/eval/phase_b_eval.json" ]]; then
  echo "ERROR: eval file missing or empty after run" >&2
  exit 2
fi

echo "[$(date -u +%FT%TZ)] DONE ${EXP_ID}"
