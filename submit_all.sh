#!/bin/bash -l
# Submit preparation pipeline steps 00-04 sequentially on Isambard.

set -euo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO}/../projects/project_1}"
export REPO

submit() {
  local step="$1"
  local extra="${2:-}"
  local dep="${3:-}"
  local script="${REPO}/${step}/step.sbatch"
  if [[ ! -f "${script}" ]]; then
    echo "Missing ${script}" >&2
    exit 1
  fi
  local args=(--parsable --chdir="${REPO}/${step}")
  if [[ -n "${dep}" ]]; then
    args+=(--dependency=afterok:"${dep}")
  fi
  if [[ -n "${extra}" ]]; then
    sbatch "${args[@]}" --export=ALL,REPO="${REPO}",OUTPUT_ROOT="${OUTPUT_ROOT}",STEP_ARGS="${extra}" "${script}"
  else
    sbatch "${args[@]}" --export=ALL,REPO="${REPO}",OUTPUT_ROOT="${OUTPUT_ROOT}" "${script}"
  fi
}

J0=$(submit "00_civic_feasibility" "--skip-fetch")
echo "Submitted 00: ${J0}"
J1=$(submit "01_corpus_relevance" "" "${J0}")
echo "Submitted 01: ${J1}"
J2=$(submit "02_evaluation_protocol" "" "${J1}")
echo "Submitted 02: ${J2}"
J3=$(submit "03_candidate_pool" "" "${J2}")
echo "Submitted 03: ${J3}"
J4=$(submit "04_pilot_study" "" "${J3}")
echo "Submitted 04: ${J4}"
echo "Chain complete. Logs under ${OUTPUT_ROOT}/runs/"
