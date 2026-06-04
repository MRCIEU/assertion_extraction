#!/bin/bash -l
# Submit DeBERTa recipe grid (4 runs + auto bad-seed guard). No KB evaluation.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
export REPO
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO}/../projects/project_1}"

STEP=11_encoder_recipe_check
RUNS="${OUTPUT_ROOT}/runs/${STEP}"
mkdir -p "${RUNS}"

jid=$(sbatch --parsable \
  --chdir="${REPO}/${STEP}" \
  --job-name="r11_deberta_grid" \
  --output="${RUNS}/grid_%j.out" \
  --error="${RUNS}/grid_%j.err" \
  --export=ALL,REPO="${REPO}",OUTPUT_ROOT="${OUTPUT_ROOT}",STEP_ARGS="--train-only" \
  "${REPO}/${STEP}/step.sbatch")

echo "Submitted recipe grid job ${jid}"
echo "Logs: ${RUNS}/grid_${jid}.{out,err}"
echo ""
echo "After the grid finishes, run finalisation ONCE:"
echo "  cd ${REPO}/${STEP} && ./submit_analyze.sh"
echo ""
echo "Optional fallback (only if you decide all 4 points stay low):"
echo "  cd ${REPO}/${STEP} && sbatch --export=ALL,REPO,OUTPUT_ROOT,STEP_ARGS='--train-fallback-only' step.sbatch"
