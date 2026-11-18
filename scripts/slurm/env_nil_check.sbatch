#!/bin/bash
# One small CPU job that checks the NIL environment.
# It prints presence flags only. It does not print values, lengths, or hashes.
#SBATCH --job-name=p1-env-nil
#SBATCH --partition=workq
#SBATCH --account=brics.b5ac
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --export=NIL

set -u
set +x
PYTHON=/home/b5ac/freddieyu.b5ac/projects/project_1/environments/analysis/bin/python
INSPECT=/home/b5ac/freddieyu.b5ac/project_1/scripts/slurm/inspect_env.py
echo "interpreter=absolute"
echo "startup_files_sourced=no"
"$PYTHON" "$INSPECT" batch
batch_status=$?
if [[ "$batch_status" -ne 0 ]]; then
  exit "$batch_status"
fi
# srun has no NIL mode on this cluster. NONE is checked once and is not used for study work.
/usr/bin/srun --export=NONE --ntasks=1 --cpu-bind=none "$PYTHON" "$INSPECT" srun_none
