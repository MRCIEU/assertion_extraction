#!/bin/bash -l
# Submit 3 parallel sweep jobs (one per architecture, 8 hyperparameter combos each).
# No KB/CIViC evaluation. After all finish, run finalisation locally.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
export REPO
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO}/../projects/project_1}"

RUNS="${OUTPUT_ROOT}/runs/10_round1_benchmark_kb/sweep"
mkdir -p "${RUNS}"

MODELS=(pubmedbert_base roberta_base distilbert_base)

echo "Submitting ${#MODELS[@]} sweep jobs (8 lr×warmup combos each, seed 42)"
echo "REPO=${REPO}  logs -> ${RUNS}/"
echo ""

for model in "${MODELS[@]}"; do
  jid=$(sbatch --parsable \
    --chdir="${REPO}/10_round1_benchmark_kb" \
    --job-name="sweep_${model}" \
    --output="${RUNS}/sweep_${model}_%j.out" \
    --error="${RUNS}/sweep_${model}_%j.err" \
    --export=ALL,REPO="${REPO}",OUTPUT_ROOT="${OUTPUT_ROOT}",SWEEP_MODEL="${model}" \
    "${REPO}/10_round1_benchmark_kb/sweep/step_sweep.sbatch")
  echo "  ${model}: job ${jid}"
done

echo ""
echo "Monitor: squeue -u \$USER"
echo ""
echo "After ALL 3 jobs finish (24 sweep_complete markers), run finalisation ONCE:"
echo "  export REPO=${REPO}"
echo "  export OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "  cd \${REPO}/10_round1_benchmark_kb && python -m sweep.run_sweep --analyze-only"
