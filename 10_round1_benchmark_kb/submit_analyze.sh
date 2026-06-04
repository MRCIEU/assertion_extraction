#!/bin/bash -l
# Submit Round-1 re-analysis (CPU): seed-level variance, four figures, report. No GPU.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
export REPO
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO}/../projects/project_1}"

STEP=10_round1_benchmark_kb
OUT="${OUTPUT_ROOT}/outputs/${STEP}"
DATA="${OUTPUT_ROOT}/data/${STEP}"
RUNS="${OUTPUT_ROOT}/runs/${STEP}"
mkdir -p "${RUNS}"

N_MARKERS=$(find "${DATA}" -name 'round1_complete.json' 2>/dev/null | wc -l | tr -d ' ')
echo "Pre-check: markers=${N_MARKERS}/72"
if [[ "${N_MARKERS}" -lt 72 ]]; then
  echo "ERROR: need 72 round1_complete.json markers before analysis." >&2
  exit 1
fi

for f in 10_per_run_scores.csv 10_easy_hard_ranking.csv; do
  if [[ ! -f "${OUT}/${f}" ]]; then
    echo "ERROR: missing ${OUT}/${f}. Analysis reads stored results only; will not rescore." >&2
    exit 1
  fi
done

jid=$(sbatch --parsable \
  --chdir="${REPO}/${STEP}" \
  --job-name="r1_analyze" \
  --output="${RUNS}/analyze_%j.out" \
  --error="${RUNS}/analyze_%j.err" \
  --export=ALL,REPO="${REPO}",OUTPUT_ROOT="${OUTPUT_ROOT}" \
  "${REPO}/${STEP}/step_analyze.sbatch")

echo "Submitted re-analysis job ${jid} (CPU, no GPU)"
echo "Logs: ${RUNS}/analyze_${jid}.{out,err}"
echo "Outputs: ${OUT}/"
echo "Figures: ${OUTPUT_ROOT}/figures/${STEP}/fig{1..4}_*.png"
echo "Report:  ${OUTPUT_ROOT}/reports/${STEP}/report.md"
