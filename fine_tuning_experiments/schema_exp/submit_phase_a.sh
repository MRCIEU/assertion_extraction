#!/bin/bash
# Submit all Phase A sbatch jobs
# Usage: bash submit_phase_a.sh [--dry-run]
#
# Phase A: 4 encoders × 3 schemas × 10 seeds = 120 runs
# Submitted as 4 sbatch array jobs (one per encoder, 30 tasks each)
# Concurrency: 6 simultaneous tasks per array (--array=1-30%6)
# Estimated runtime: ~2h per task → all done within ~12h on 6 GPUs

set -euo pipefail
DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

SBATCH_DIR="$(dirname "$0")/sbatch"
LOG_DIR="/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/logs/schema_exp"
mkdir -p "$LOG_DIR"
mkdir -p "/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/runs/schema_exp"

echo "=== Phase A Submission ==="
echo "Timestamp: $(date -u +%FT%TZ)"
echo "Dry run: $DRY_RUN"
echo ""

declare -A JOBIDS
for enc in RB PB BL PL; do
    sbatch_file="$SBATCH_DIR/phase_a_${enc}.sbatch"
    echo "Submitting: phase_a_${enc}.sbatch (30 tasks: 3 schemas × 10 seeds)"
    if [ "$DRY_RUN" = "true" ]; then
        echo "  [DRY RUN] would run: sbatch $sbatch_file"
        JOBIDS[$enc]="DRY_RUN"
    else
        JOB_OUT=$(sbatch "$sbatch_file")
        JOB_ID=$(echo "$JOB_OUT" | grep -oP '\d+')
        JOBIDS[$enc]="$JOB_ID"
        echo "  Submitted job: $JOB_ID"
    fi
done

echo ""
echo "=== Submission Summary ==="
for enc in RB PB BL PL; do
    echo "  Phase A ${enc}: job ${JOBIDS[$enc]}"
done

# Write submission record
RECORD_FILE="$(dirname "$0")/phase_a_submission_record.json"
cat > "$RECORD_FILE" << JSONEOF
{
  "submitted_at": "$(date -u +%FT%TZ)",
  "phase": "phase_a_schema_selection",
  "total_experiments": 120,
  "array_jobs": {
    "RB": "${JOBIDS[RB]}",
    "PB": "${JOBIDS[PB]}",
    "BL": "${JOBIDS[BL]}",
    "PL": "${JOBIDS[PL]}"
  },
  "n_tasks_per_job": 30,
  "schemas": ["S_flat", "S_pair", "S_mech"],
  "seeds_per_group": 10,
  "output_root": "/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/runs/schema_exp",
  "log_dir": "$LOG_DIR",
  "note": "Phase A: schema × encoder joint exploration. Requires RoBERTa-base download."
}
JSONEOF
echo ""
echo "Record written: $RECORD_FILE"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f $LOG_DIR/PA_PB_*.out"
