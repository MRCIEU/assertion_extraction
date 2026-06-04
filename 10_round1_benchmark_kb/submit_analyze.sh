#!/bin/bash -l
# Submit Round 1 cross-model analysis (CPU job; requires all 72 completion markers + score files).

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
export REPO
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO}/../projects/project_1}"

STEP=10_round1_benchmark_kb
RUNS="${OUTPUT_ROOT}/runs/${STEP}"
mkdir -p "${RUNS}"

N_MARKERS=$(find "${OUTPUT_ROOT}/data/${STEP}" -name 'round1_complete.json' 2>/dev/null | wc -l)
N_SCORES=$(find "${OUTPUT_ROOT}/data/${STEP}/scores" -name '*.jsonl' 2>/dev/null | wc -l)

echo "Pre-check: markers=${N_MARKERS}/72  score_files=${N_SCORES}/72"
if [[ "${N_MARKERS}" -lt 72 ]]; then
  echo "ERROR: need 72 round1_complete.json markers before analysis." >&2
  exit 1
fi
if [[ "${N_SCORES}" -lt 72 ]]; then
  echo "ERROR: need 72 score jsonl files before analysis." >&2
  exit 1
fi

jid=$(sbatch --parsable \
  --chdir="${REPO}/${STEP}" \
  --job-name="r1_analyze" \
  --output="${RUNS}/analyze_%j.out" \
  --error="${RUNS}/analyze_%j.err" \
  --export=ALL,REPO="${REPO}",OUTPUT_ROOT="${OUTPUT_ROOT}" \
  "${REPO}/${STEP}/step_analyze.sbatch")

echo "Submitted analysis job ${jid}"
echo "Logs: ${RUNS}/analyze_${jid}.{out,err}"
echo "Outputs: ${OUTPUT_ROOT}/outputs/${STEP}/"
echo "Report:  ${OUTPUT_ROOT}/reports/${STEP}/report.md"
echo "Figures: ${OUTPUT_ROOT}/figures/${STEP}/"
