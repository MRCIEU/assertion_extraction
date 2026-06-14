#!/usr/bin/env bash
# Build epoch-1 stratum cache (GPU), then run full CPU analysis + reports.
set -euo pipefail

STEP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Submit stratum cache + dependent analyze ==="
stratum_jid=$(sbatch --parsable --chdir="${STEP_DIR}" "${STEP_DIR}/step_stratum_epoch1.sbatch")
echo "  stratum (GPU): ${stratum_jid}"

analyze_jid=$(sbatch --parsable --chdir="${STEP_DIR}" --dependency="afterok:${stratum_jid}" \
  "${STEP_DIR}/step_analyze.sbatch")
echo "  analyze (CPU, after stratum): ${analyze_jid}"
echo ""
echo "Monitor:"
echo "  sacct -j ${stratum_jid},${analyze_jid}"
echo "  tail -f ${STEP_DIR}/../projects/project_1/runs/20_round2_diagnostic/stratum_epoch1_${stratum_jid}.out"
