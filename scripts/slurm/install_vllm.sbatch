#!/bin/bash
# Install vLLM into the project environment. Does not modify the training env.
#SBATCH --job-name=p1-vllm-install
#SBATCH --partition=workq
#SBATCH --account=brics.b5ac
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --export=NIL

set -euo pipefail
set +x
export PATH=/usr/bin:/bin
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN || true
VENV=/home/b5ac/freddieyu.b5ac/projects/project_1/environments/vllm
/home/b5ac/freddieyu.b5ac/projects/project_1/environments/analysis/bin/python -m venv "$VENV"
"$VENV/bin/pip" install vllm
"$VENV/bin/python" -c 'import torch, vllm; print(vllm.__version__); print(torch.__version__); print(torch.version.cuda)'
