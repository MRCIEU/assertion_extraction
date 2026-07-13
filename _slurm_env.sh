#!/usr/bin/env bash
# Shared Slurm runtime env: writable temp dir for torch/transformers import.
setup_slurm_tmpdir() {
  local runs_dir="${1:?runs dir required}"
  local job_tag="${SLURM_JOB_ID:-local}"
  # SLURM may set TMPDIR to /local/user/<uid> which does not exist on all nodes.
  export TMPDIR="${runs_dir}/tmp/${job_tag}"
  mkdir -p "${TMPDIR}"
  export TEMP="${TMPDIR}"
  export TMP="${TMPDIR}"
  echo "TMPDIR=${TMPDIR}"
}
