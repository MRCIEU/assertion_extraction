#!/usr/bin/env bash
# Submit Phase A-eval: one Slurm array job of 120 tasks (4 encoders × 3 schemas × 10 seeds).
# Each task loads a PA_* checkpoint and runs BioRED + BC5CDR + KB-surface inference.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH="${SCRIPT_DIR}/sbatch/phase_a_eval.sbatch"

if [[ ! -f "$SBATCH" ]]; then
  echo "ERROR: sbatch script not found: $SBATCH" >&2
  exit 1
fi

# Ensure eval inputs have been prepared
INPUTS_DIR="${SCRIPT_DIR}/inputs"
for f in biored_test_pairs_Sflat.jsonl biored_test_pairs_Spair.jsonl \
         biored_test_pairs_Smech.jsonl bc5cdr_test_pairs.jsonl kb_surface_pairs.jsonl; do
  if [[ ! -f "${INPUTS_DIR}/${f}" ]]; then
    echo "ERROR: missing eval input ${INPUTS_DIR}/${f}" >&2
    echo "Run first:  python3.11 fine_tuning_experiments/schema_exp/eval/prepare_eval_inputs.py" >&2
    exit 1
  fi
done

JOBID=$(sbatch --parsable "$SBATCH")
echo "Submitted Phase A-eval array job: $JOBID"
echo "Tail log:  tail -f /lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/logs/schema_exp_eval/PA_EV_${JOBID}_1.out"
echo "Monitor:   squeue -j ${JOBID}"
echo ""
echo "When finished, run aggregator:"
echo "  python3.11 fine_tuning_experiments/schema_exp/eval/aggregate_phase_a.py"

# Record submission
mkdir -p "${SCRIPT_DIR}"
printf '{"job_id": "%s", "submitted_utc": "%s"}\n' \
  "$JOBID" "$(date -u +%FT%TZ)" \
  > "${SCRIPT_DIR}/phase_a_eval_submission.json"
