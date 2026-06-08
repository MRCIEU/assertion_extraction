#!/usr/bin/env bash
# Submit 9 parallel GPU scoring jobs (one encoder each), then stage-2 CPU analysis
# when all scoring jobs finish successfully.
#
# Usage (from login node):
#   cd project_1/11_round1_analysis
#   ./submit_round1.sh
#
# Optional env:
#   REPO, OUTPUT_ROOT, DRY_RUN=1

set -euo pipefail

STEP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "${STEP_DIR}/.." && pwd)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO}/../projects/project_1}"

ENCODERS=(
  pubmedbert_base
  biomedbert_base
  biolinkbert_base
  biobert_base
  scibert_base
  roberta_base
  bert_base
  distilbert_base
  deberta_base
)

echo "=== Round 1 parallel submit ==="
echo "REPO=${REPO}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "Encoders: ${#ENCODERS[@]} parallel scoring jobs + 1 analysis job (dependency)"

SCORE_JOBS=()
for enc in "${ENCODERS[@]}"; do
  short="${enc/_base/}"
  cmd=(
    sbatch
    --job-name="r1_score_${short}"
    --export=ALL,STEP_ARGS="--model-id ${enc}"
    "${STEP_DIR}/step_score.sbatch"
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
  --job-name=r1_analyze
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
  echo "  Stage 1 scoring jobs: ${SCORE_JOBS[*]}"
  echo "  Stage 2 analysis job: ${analyze_jid} (starts after all scoring succeed)"
  echo ""
  echo "Monitor scoring:"
  echo "  find ${OUTPUT_ROOT}/data/11_round1_analysis/scores -name scoring_complete.json | wc -l"
  echo "  squeue -u \$USER -n r1_score,r1_analyze"
fi
