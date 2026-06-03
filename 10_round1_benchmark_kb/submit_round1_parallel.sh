#!/bin/bash -l
# Submit Round 1 training/eval as parallel SLURM jobs (one encoder per job).
# Cross-model analysis is NOT run here — run finalisation manually after all jobs finish.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
export REPO
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO}/../projects/project_1}"

STEP=10_round1_benchmark_kb
RUNS="${OUTPUT_ROOT}/runs/${STEP}"
mkdir -p "${RUNS}"

# ---------------------------------------------------------------------------
# Partition: one job per encoder (9 jobs x 8 seeds = 72 runs).
# Assumes you can run ~9 GPUs concurrently. Adjust PARTITIONS below if not.
# ---------------------------------------------------------------------------
SEEDS="42 43 44 45 46 47 48 49"

declare -a ENCODERS=(
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

# Optional: override with env var, e.g.
#   ENCODERS="pubmedbert_base biolinkbert_base" ./submit_round1_parallel.sh
if [[ -n "${ENCODERS_OVERRIDE:-}" ]]; then
  read -r -a ENCODERS <<< "${ENCODERS_OVERRIDE}"
fi

echo "Submitting ${#ENCODERS[@]} parallel Round 1 jobs (train + per-model eval only)"
echo "REPO=${REPO}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "Logs -> ${RUNS}/train_<encoder>_<jobid>.{out,err}"
echo ""

for model in "${ENCODERS[@]}"; do
  label="train_${model}"
  step_args="--train-eval-only --models ${model} --seeds ${SEEDS}"
  jid=$(sbatch --parsable \
    --chdir="${REPO}/${STEP}" \
    --job-name="r1_${model}" \
    --output="${RUNS}/${label}_%j.out" \
    --error="${RUNS}/${label}_%j.err" \
    --export=ALL,REPO="${REPO}",OUTPUT_ROOT="${OUTPUT_ROOT}",STEP_ARGS="${step_args}",JOB_LABEL="${label}" \
    "${REPO}/${STEP}/step_train.sbatch")
  echo "  ${model}: job ${jid}  (${label})"
done

echo ""
echo "Submitted ${#ENCODERS[@]} jobs. Monitor: squeue -u \$USER"
echo ""
echo "After ALL jobs finish (72 round1_complete.json markers), run finalisation ONCE:"
echo "  export REPO=${REPO}"
echo "  export OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "  cd \${REPO}/${STEP} && python run.py --analyze-only"
