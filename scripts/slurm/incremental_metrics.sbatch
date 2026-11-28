#!/bin/bash
#SBATCH --job-name=p1-metrics
#SBATCH --partition=workq
#SBATCH --account=brics.b5ac
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/b5ac/freddieyu.b5ac/projects/project_1/studies/schema_granularity_v1/runs/formal/formal_matched_e0_e7/metrics-%j.out
set -euo pipefail
source /home/b5ac/freddieyu.b5ac/miniforge3/etc/profile.d/conda.sh
conda activate hf-hpc
cd /home/b5ac/freddieyu.b5ac/project_1
export PYTHONPATH="/home/b5ac/freddieyu.b5ac/project_1/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
exec /home/b5ac/freddieyu.b5ac/projects/project_1/environments/analysis/bin/python -m schema_study.incremental_metrics
