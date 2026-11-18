#!/bin/bash
# Submit one CPU job with a clean Slurm environment.
# --export=NIL does not accept extra variables. The batch script uses absolute paths.
set -euo pipefail
if [[ "${1:-}" == *gpu* || "${SLURM_GPUS_PER_TASK:-0}" != "0" ]]; then
  echo "GPU submission is not available from this script." >&2
  exit 2
fi
TIME=00:20:00
CPUS=8
MEM=16G
if [[ "${1:-}" == "--time" ]]; then
  TIME="${2:?}"
  shift 2
fi
CODE_ROOT=/home/b5ac/freddieyu.b5ac/project_1
LOG_DIR=/home/b5ac/freddieyu.b5ac/projects/project_1/job_logs
mkdir -p "$LOG_DIR"
if [[ "$#" -lt 1 ]]; then
  echo "CPU command arguments are required." >&2
  exit 2
fi
sbatch \
  --export=NIL \
  --time="${TIME}" \
  --cpus-per-task="${CPUS}" \
  --mem="${MEM}" \
  --output="${LOG_DIR}/cpu-%j.out" \
  --error="${LOG_DIR}/cpu-%j.err" \
  "${CODE_ROOT}/scripts/slurm/cpu_job.sbatch" \
  "$@"
