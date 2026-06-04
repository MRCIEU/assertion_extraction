#!/bin/bash -l
# Submit finalisation (CPU): tables, figures, report. Run once after grid completes.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
export REPO
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO}/../projects/project_1}"

STEP=11_encoder_recipe_check
RUNS="${OUTPUT_ROOT}/runs/${STEP}"
mkdir -p "${RUNS}"

n=$(find "${OUTPUT_ROOT}/data/${STEP}/results" -name 'recipe_complete.json' 2>/dev/null | wc -l)
echo "Markers found: ${n} (expect >= 4 for primary grid)"

jid=$(sbatch --parsable \
  --chdir="${REPO}/${STEP}" \
  --job-name="r11_analyze" \
  --output="${RUNS}/analyze_%j.out" \
  --error="${RUNS}/analyze_%j.err" \
  --export=ALL,REPO="${REPO}",OUTPUT_ROOT="${OUTPUT_ROOT}" \
  "${REPO}/${STEP}/step_analyze.sbatch")

echo "Submitted analysis job ${jid}"
echo "Report: ${OUTPUT_ROOT}/reports/${STEP}/report.md"
