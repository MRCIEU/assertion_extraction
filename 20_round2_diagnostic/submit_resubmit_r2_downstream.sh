#!/usr/bin/env bash
# Resubmit failed R2 stratum jobs, then analyze + manuscript.
# Skips AUC-PR supplement jobs (epoch score JSON already has kb_auc_pr fields).
set -euo pipefail

STEP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "${STEP_DIR}/.." && pwd)}"
export REPO OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO}/../projects/project_1}"

# Encoders whose r2_st_* jobs failed with TMPDIR /local/user errors (2026-06-29 batch).
ST_ENCODERS=(
  pubmedbert_base
  bluebert_base
  biolinkbert_base
  biobert_base
  scibert_base
  bert_base
  deberta_base
)

echo "=== Resubmit R2 downstream (stratum -> analyze -> manuscript) ==="
echo "REPO=${REPO}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"

scancel 5420030 5420031 2>/dev/null || true
echo "Cancelled stuck jobs 5420030 (r2_analyze) and 5420031 (manuscript_regen) if present."

ST_JOBS=()
for enc in "${ST_ENCODERS[@]}"; do
  short="${enc/_base/}"
  jid=$(sbatch --parsable --job-name="r2_st_${short}" \
    --export=ALL,STEP_ARGS="--model-id ${enc}" \
    "${STEP_DIR}/step_stratum_epoch1.sbatch")
  echo "  stratum ${enc} -> ${jid}"
  ST_JOBS+=("${jid}")
done

dep="afterok"
for jid in "${ST_JOBS[@]}"; do
  dep="${dep}:${jid}"
done

analyze_jid=$(sbatch --parsable --job-name=r2_analyze --dependency="${dep}" \
  --export=ALL \
  "${STEP_DIR}/step_analyze.sbatch")
echo "  analyze -> ${analyze_jid} (after ${#ST_JOBS[@]} stratum jobs)"

manuscript_jid=$(sbatch --parsable --job-name=manuscript_regen --dependency="afterok:${analyze_jid}" \
  --export=ALL \
  "${REPO}/manuscript_regenerate/step_manuscript.sbatch")
echo "  manuscript -> ${manuscript_jid} (after analyze)"

echo ""
echo "Submitted stratum jobs: ${ST_JOBS[*]}"
echo "Monitor: squeue -u \$USER -n r2_st,r2_analyze,manuscript_regen"
echo "         sacct -j ${analyze_jid},${manuscript_jid} --format=JobID,JobName,State,ExitCode,Elapsed"
