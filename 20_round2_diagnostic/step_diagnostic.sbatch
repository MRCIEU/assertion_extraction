#!/bin/bash -l
# Deprecated wrapper: use submit_diagnostic.sh or step_score_epochs.sbatch + step_analyze.sbatch
#SBATCH -A brics.b5ac
#SBATCH -p workq
#SBATCH --job-name=r2_diag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail
echo "Use ./submit_diagnostic.sh instead (separate GPU scoring + CPU analysis)."
exit 1
