#!/usr/bin/env bash
# Submit GPU epoch scoring, then CPU analysis when scoring succeeds.
set -euo pipefail

STEP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "${STEP_DIR}/.." && pwd)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO}/../projects/project_1}"

echo "=== Round 2 diagnostic submit ==="
echo "REPO=${REPO}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"

score_out=$(sbatch --job-name=r2_score_ep --export=ALL "${STEP_DIR}/step_score_epochs.sbatch")
score_jid="${score_out##* }"
echo "  epoch scoring -> ${score_jid}"

analyze_out=$(sbatch --job-name=r2_analyze --dependency=afterok:${score_jid} \
  --chdir="${STEP_DIR}" \
  --export=ALL,REPO="${REPO}",OUTPUT_ROOT="${OUTPUT_ROOT}" \
  "${STEP_DIR}/step_analyze.sbatch")
analyze_jid="${analyze_out##* }"
echo "  analysis -> ${analyze_jid} (after scoring succeeds)"
echo ""
echo "Check scoring: cat ${OUTPUT_ROOT}/data/20_round2_diagnostic/epoch_scoring_complete.json"
