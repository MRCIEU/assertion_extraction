#!/bin/bash
# Score the saved SciFact-adapted checkpoints on official development claims.
#SBATCH --job-name=p1-verifier-eval
#SBATCH --partition=workq
#SBATCH --account=brics.b5ac
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --export=ALL

set -euo pipefail
source /home/b5ac/freddieyu.b5ac/miniforge3/etc/profile.d/conda.sh
conda activate hf-hpc
export PATH=/home/b5ac/freddieyu.b5ac/miniforge3/envs/hf-hpc/bin:/usr/bin:/bin
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/b5ac/freddieyu.b5ac/project_1/src
cd /home/b5ac/freddieyu.b5ac/project_1
exec python -m schema_study.measurement_cpu evaluate-adapted \
  /home/b5ac/freddieyu.b5ac/projects/project_1/studies/schema_granularity_v1/runs/formal/formal_matched_e0_e7/verifier_adaptation/scifact/all/all/seednone/000ff1285f79baf6
