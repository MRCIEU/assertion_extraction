#!/bin/bash
# One-GPU smoke job. NIL export, no login shell, no srun.
#SBATCH --job-name=p1-smoke
#SBATCH --partition=workq
#SBATCH --account=brics.b5ac
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --export=NIL

set -euo pipefail
set +x

PYTHON=/home/b5ac/freddieyu.b5ac/projects/project_1/environments/training/bin/python
"$PYTHON" /home/b5ac/freddieyu.b5ac/project_1/scripts/slurm/inspect_env.py gpu
status=$?
if [[ "$status" -ne 0 ]]; then
  echo "credential_check=stop" >&2
  exit "$status"
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && -z "${SLURM_GPUS_ON_NODE:-}" ]]; then
  echo "This smoke job did not receive a GPU." >&2
  exit 2
fi

export PATH=/usr/bin:/bin
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/b5ac/freddieyu.b5ac/project_1/src
export SCHEMA_STUDY_STORAGE_ROOT=/home/b5ac/freddieyu.b5ac/projects/project_1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export MPLCONFIGDIR=/home/b5ac/freddieyu.b5ac/projects/project_1/cache/matplotlib

cd /home/b5ac/freddieyu.b5ac/project_1
TASK="${1:?}"
if [[ "$TASK" == "qwen" ]]; then
  "$PYTHON" -m schema_study.smoke_qwen settle
else
  "$PYTHON" -m schema_study.smoke_biomedbert "$TASK" epoch1
  "$PYTHON" -m schema_study.smoke_biomedbert "$TASK" epoch2
  "$PYTHON" -m schema_study.smoke_biomedbert "$TASK" predict
fi
