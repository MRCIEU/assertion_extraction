#!/usr/bin/env bash
# Submit 9 parallel GPU epoch-scoring jobs (one encoder each), then CPU analysis.
set -euo pipefail

STEP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "${STEP_DIR}/.." && pwd)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO}/../projects/project_1}"

ENCODERS=(
  pubmedbert_base bluebert_base biolinkbert_base biobert_base scibert_base
  roberta_base bert_base distilbert_base deberta_base
)

echo "=== Round 2 diagnostic submit (5e-6 per-epoch scoring) ==="
echo "REPO=${REPO}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
if [[ -n "${FORCE_RESCORE:-}" ]]; then
  echo "FORCE_RESCORE=1 — existing epoch scores will be overwritten"
fi

SCORE_JOBS=()
for enc in "${ENCODERS[@]}"; do
  short="${enc/_base/}"
  cmd=(
    sbatch
    --job-name="r2_score_${short}"
    --export=ALL,STEP_ARGS="--model-id ${enc}"${FORCE_RESCORE:+,FORCE_RESCORE=1}
    "${STEP_DIR}/step_score_epochs.sbatch"
  )
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[dry-run] ${cmd[*]}"
    SCORE_JOBS+=("DRY${#SCORE_JOBS[@]}")
  else
    out="$("${cmd[@]}")"
    jid="${out##* }"
    echo "  scoring ${enc} -> job ${jid}"
    SCORE_JOBS+=("${jid}")
  fi
done

dep="afterok"
for jid in "${SCORE_JOBS[@]}"; do
  dep="${dep}:${jid}"
done

analyze_cmd=(
  sbatch
  --job-name=r2_analyze
  --dependency="${dep}"
  --export=ALL
  "${STEP_DIR}/step_analyze.sbatch"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[dry-run] ${analyze_cmd[*]}"
else
  out="$("${analyze_cmd[@]}")"
  analyze_jid="${out##* }"
  echo ""
  echo "=== Submitted ==="
  echo "  Scoring jobs: ${SCORE_JOBS[*]}"
  echo "  Analysis job: ${analyze_jid}"
  echo ""
  echo "Check scoring:"
  echo "  python -c \"import json; print(json.load(open('${OUTPUT_ROOT}/data/20_round2_diagnostic/epoch_scoring_complete.json')))\""
  echo "  find ${OUTPUT_ROOT}/data/20_round2_diagnostic/scores -name 'epoch_*.json' | wc -l  # expect ~498"
fi
