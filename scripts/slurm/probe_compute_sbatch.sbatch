#!/bin/bash
# Check whether a scheduled job can submit another job. The held job is cancelled.
#SBATCH --job-name=p1-sbatch-probe
#SBATCH --partition=workq
#SBATCH --account=brics.b5ac
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=/home/b5ac/freddieyu.b5ac/projects/project_1/job_logs/sbatch_probe_%j.out
#SBATCH --export=NIL

set -euo pipefail
set +x

source /home/b5ac/freddieyu.b5ac/miniforge3/etc/profile.d/conda.sh
conda activate hf-hpc
echo "host=$(hostname)"
held=$(sbatch --hold --parsable --account=brics.b5ac --partition=workq --qos=normal --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=1G --time=00:05:00 --job-name=p1-held-probe --wrap "true")
echo "compute_node_sbatch=${held}"
scancel "${held%%;*}"
echo "cancelled=${held%%;*}"
