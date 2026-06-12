#!/usr/bin/env bash
# Submit 9 parallel GPU jobs to add pair×subset cross metrics to epoch score JSON.
set -euo pipefail

STEP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENCODERS=(
  pubmedbert_base biomedbert_base biolinkbert_base biobert_base scibert_base
  roberta_base bert_base distilbert_base deberta_base
)

echo "=== Cross-metric supplement submit ==="
JOBS=()
for enc in "${ENCODERS[@]}"; do
  short="${enc/_base/}"
  jid=$(sbatch --parsable --job-name="r2_sup_${short}" \
    --export=ALL,STEP_ARGS="--model-id ${enc}" \
    "${STEP_DIR}/step_supplement_cross.sbatch")
  echo "  ${enc} -> ${jid}"
  JOBS+=("${jid}")
done

dep="afterok"
for jid in "${JOBS[@]}"; do
  dep="${dep}:${jid}"
done

analyze_jid=$(sbatch --parsable --dependency="${dep}" --export=ALL \
  "${STEP_DIR}/step_analyze.sbatch")
echo ""
echo "Supplement jobs: ${JOBS[*]}"
echo "Analysis job (after supplement): ${analyze_jid}"
