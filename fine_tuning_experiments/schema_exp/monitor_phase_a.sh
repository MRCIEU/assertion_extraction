#!/bin/bash
# Monitor Phase A job progress
# Usage: bash monitor_phase_a.sh
#        bash monitor_phase_a.sh --wait  (polls every 10 min until done)

WAIT_MODE=false
[[ "${1:-}" == "--wait" ]] && WAIT_MODE=true

JOBS=(3836854 3836855 3836856 3836857)
JOB_NAMES=(PA_RB PA_PB PA_BL PA_PL)
LOG=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/logs/schema_exp
RUNS=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/runs/schema_exp

check_status() {
    echo "=== Phase A Status at $(date -u +%FT%TZ) ==="
    echo ""
    echo "Slurm queue:"
    squeue -u $USER --format="%.10i %.8j %.8T %.10M %.6D %R" 2>/dev/null | grep -E "JOBID|PA_" || echo "  (no PA jobs in queue)"
    echo ""
    echo "Completed run directories:"
    n=$(ls "$RUNS" 2>/dev/null | grep "^PA_" | wc -l)
    echo "  $n / 120"
    echo ""
    echo "Recent log output (last 3 lines of newest .out file):"
    newest=$(ls -t $LOG/*.out 2>/dev/null | head -1)
    if [ -n "$newest" ]; then
        echo "  File: $(basename $newest)"
        tail -3 "$newest" | sed 's/^/  /'
    fi
    echo ""
    
    # Check for errors
    n_err=$(grep -l "Error\|Traceback\|FAILED" $LOG/*.err 2>/dev/null | wc -l)
    if [ "$n_err" -gt 0 ]; then
        echo "WARNING: $n_err error files detected"
        for f in $(grep -l "Error\|Traceback" $LOG/*.err 2>/dev/null | head -3); do
            echo "  $f:"
            tail -3 "$f" | sed 's/^/    /'
        done
    fi
}

check_status

if [ "$WAIT_MODE" = "true" ]; then
    while squeue -u $USER 2>/dev/null | grep -q "PA_"; do
        echo "Jobs still running. Waiting 10 minutes..."
        sleep 600
        check_status
    done
    echo "All PA jobs complete. Running evaluation..."
    cd /home/b5ac/freddieyu.b5ac/project_1
    PYTHONPATH=/home/b5ac/freddieyu.b5ac/project_1 \
        python3.11 fine_tuning_experiments/schema_exp/evaluate_phase_a.py
fi
