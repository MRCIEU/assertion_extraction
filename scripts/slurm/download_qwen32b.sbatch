#!/bin/bash
# Download the pinned Qwen3-32B files. No token and no model execution.
#SBATCH --job-name=p1-qwen32b-download
#SBATCH --partition=workq
#SBATCH --account=brics.b5ac
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --export=NIL

set -euo pipefail
set +x
export PATH=/usr/bin:/bin
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_ENABLE_HF_TRANSFER=0
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN || true
DEST=/home/b5ac/freddieyu.b5ac/projects/project_1/models/Qwen__Qwen3-32B/9216db5781bf21249d130ec9da846c4624c16137
mkdir -p "$DEST"
exec /home/b5ac/freddieyu.b5ac/projects/project_1/environments/analysis/bin/huggingface-cli download \
  Qwen/Qwen3-32B \
  --revision 9216db5781bf21249d130ec9da846c4624c16137 \
  --local-dir "$DEST" \
  --max-workers 2
