#!/bin/bash -l
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
export REPO
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO}/../projects/project_1}"
RUNS="${OUTPUT_ROOT}/runs/20_round2_diagnostic"
mkdir -p "${RUNS}"
jid=$(sbatch --parsable \
  --chdir="${REPO}/20_round2_diagnostic" \
  --output="${RUNS}/diag_%j.out" \
  --error="${RUNS}/diag_%j.err" \
  --export=ALL,REPO="${REPO}",OUTPUT_ROOT="${OUTPUT_ROOT}" \
  "${REPO}/20_round2_diagnostic/step_diagnostic.sbatch")
echo "Submitted ${jid}"
echo "Logs: ${RUNS}/diag_${jid}.{out,err}"
