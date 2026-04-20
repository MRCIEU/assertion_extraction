#!/usr/bin/env bash
# Submit Tier-2 downstream transfer GPU job (multi-seed × selected families).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SBATCH="${ROOT}/scripts/run_transfer_tier2_gpu.sbatch"
echo "Submitting ${SBATCH}"
jid="$(sbatch --parsable "${SBATCH}")"
echo "job_id=${jid}"
LOG="${HOME}/projects/project_1/knowledge_grounded_evidence_audit/manifests/tier2_job_submission_log.csv"
TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
mkdir -p "$(dirname "${LOG}")"
if ! grep -q '^job_id,' "${LOG}" 2>/dev/null; then
  echo 'job_id,submitted_utc,completed_utc,sbatch_script,partition,status,notes' >> "${LOG}"
fi
echo "${jid},${TS},,project_1/knowledge_grounded_evidence_audit/scripts/run_transfer_tier2_gpu.sbatch,workq,submitted,tier2 all_tier2_final" >> "${LOG}"
echo "Appended to ${LOG}"
