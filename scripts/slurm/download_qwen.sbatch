#!/bin/bash
# Resume the pinned Qwen3-8B download. One file at a time, no token.
#SBATCH --job-name=p1-qwen-download
#SBATCH --partition=workq
#SBATCH --account=brics.b5ac
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --export=NIL

set -euo pipefail
set +x
export PATH=/usr/bin:/bin
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_ENABLE_HF_TRANSFER=0
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN || true
DEST=/home/b5ac/freddieyu.b5ac/projects/project_1/models/Qwen__Qwen3-8B/b968826d9c46dd6066d109eabc6255188de91218
exec /home/b5ac/freddieyu.b5ac/projects/project_1/environments/analysis/bin/huggingface-cli download \
  Qwen/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --local-dir "$DEST" \
  --max-workers 1
