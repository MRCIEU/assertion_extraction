#!/bin/bash -l
#SBATCH -J PB_post_an
#SBATCH -A brics.b5ac
#SBATCH -p workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/logs/phase_b/PB_post_analysis_%j.out
#SBATCH --error=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/logs/phase_b/PB_post_analysis_%j.err

# Phase B post-lock-v3 analysis bundle.
#
# Runs after PL_FT_T2 backfill (SLURM 4427892) succeeds:
#   1. aggregate_phase_b.py -> 190-row CSV
#   2. analyze_phase_b.py -> H1/H2/H3/H4-methodological-null/H5/H7/R_B/ordinal
#   3. h6_coupling_slopes.py -> H6 slopes (Phase A eval dirs + Phase B CSV)
#   4. rq3_encoder_kb_interaction.py -> exploratory encoder × KB-metric audit
#
# No GPU is requested; this is CPU-only analysis over already-frozen eval JSONs.

set -euo pipefail
export PYTHONPATH=/home/b5ac/freddieyu.b5ac/project_1
cd /home/b5ac/freddieyu.b5ac/project_1

RUN_ROOT=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/runs/phase_b
PHASE_A_RUN_ROOT=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/runs/schema_exp
OUT_DIR=/home/b5ac/freddieyu.b5ac/project_1/fine_tuning_experiments/phase_b/analysis/output
TS=$(date -u +%Y%m%dT%H%M%SZ)

mkdir -p "$OUT_DIR"

echo "[$(date -u +%FT%TZ)] Phase B post-lock analysis"
echo "  host=$(hostname)"
echo "  commit=$(git rev-parse --short HEAD 2>/dev/null || echo no-git)"
echo "  timestamp=$TS"

AGG="$OUT_DIR/phase_b_eval_aggregate_${TS}.csv"
AN_JSON="$OUT_DIR/phase_b_analysis_${TS}.json"
AN_MD="$OUT_DIR/phase_b_analysis_${TS}.md"
H6_JSON="$OUT_DIR/h6_coupling_slopes_${TS}.json"
RQ3_JSON="$OUT_DIR/rq3_encoder_kb_interaction_${TS}.json"
RQ3_MD="$OUT_DIR/rq3_encoder_kb_interaction_${TS}.md"

echo ""
echo "=== [1/4] Aggregate Phase B eval JSONs ==="
/usr/bin/python3.11 fine_tuning_experiments/phase_b/aggregate_phase_b.py \
    --run-root "$RUN_ROOT" \
    --out "$AGG"

n_rows=$(($(wc -l < "$AGG") - 1))
echo "  aggregate rows: $n_rows"
if [[ "$n_rows" -ne 190 ]]; then
    echo "ERROR: expected 190 rows after PL_FT_T2 backfill, got $n_rows" >&2
    exit 2
fi
ln -sf "$(basename "$AGG")" "$OUT_DIR/phase_b_eval_aggregate_LATEST.csv"

echo ""
echo "=== [2/4] H1/H2/H3/H4/H5/H7 + R_B + ordinal instability ==="
/usr/bin/python3.11 fine_tuning_experiments/phase_b/analysis/analyze_phase_b.py \
    --input "$AGG" \
    --out-json "$AN_JSON" \
    --out-md "$AN_MD"
ln -sf "$(basename "$AN_JSON")" "$OUT_DIR/phase_b_analysis_LATEST.json"
ln -sf "$(basename "$AN_MD")" "$OUT_DIR/phase_b_analysis_LATEST.md"

echo ""
echo "=== [3/4] H6 mechanism-stratified coupling slopes ==="
/usr/bin/python3.11 fine_tuning_experiments/phase_b/analysis/h6_coupling_slopes.py \
    --input "$AGG" \
    --input-dirs "$PHASE_A_RUN_ROOT" \
    --out "$H6_JSON" \
    --n-boot 5000 \
    --seed 20260416
ln -sf "$(basename "$H6_JSON")" "$OUT_DIR/h6_coupling_slopes_LATEST.json"

echo ""
echo "=== [4/4] RQ3 exploratory encoder × KB-metric interaction ==="
/usr/bin/python3.11 fine_tuning_experiments/phase_b/analysis/rq3_encoder_kb_interaction.py \
    --input "$AGG" \
    --out-json "$RQ3_JSON" \
    --out-md "$RQ3_MD"
ln -sf "$(basename "$RQ3_JSON")" "$OUT_DIR/rq3_encoder_kb_interaction_LATEST.json"
ln -sf "$(basename "$RQ3_MD")" "$OUT_DIR/rq3_encoder_kb_interaction_LATEST.md"

echo ""
echo "=== Output manifest ==="
sha256sum "$AGG" "$AN_JSON" "$AN_MD" "$H6_JSON" "$RQ3_JSON" "$RQ3_MD" \
    | tee "$OUT_DIR/phase_b_postlock_analysis_${TS}.sha256"
ln -sf "phase_b_postlock_analysis_${TS}.sha256" "$OUT_DIR/phase_b_postlock_analysis_LATEST.sha256"

echo ""
echo "[$(date -u +%FT%TZ)] DONE"
