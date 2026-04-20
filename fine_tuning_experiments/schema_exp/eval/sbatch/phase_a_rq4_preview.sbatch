#!/bin/bash -l
#SBATCH -J PA_RQ4
#SBATCH -A brics.b5ac
#SBATCH -p workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/logs/schema_exp_eval/PA_RQ4_%j.out
#SBATCH --error=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/logs/schema_exp_eval/PA_RQ4_%j.err

set -euo pipefail
export PROJECT_1_DATA_ROOT=/lus/lfs1aip2/projects/b5ac/project_1
export PYTHONPATH=/home/b5ac/freddieyu.b5ac/project_1
cd /home/b5ac/freddieyu.b5ac/project_1
mkdir -p /lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/logs/schema_exp_eval

echo "[$(date -u +%FT%TZ)] RQ4 preview"
/usr/bin/python3.11 -m fine_tuning_experiments.schema_exp.eval.phase_a_rq4_preview
echo "[$(date -u +%FT%TZ)] Done"
