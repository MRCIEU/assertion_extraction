#!/bin/bash
# One matched-programme task. The historical formal_20261115 launcher does not call this script.
#SBATCH --job-name=p1-matched
#SBATCH --partition=workq
#SBATCH --account=brics.b5ac
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --export=ALL
#SBATCH --signal=B:USR1@180

set -euo pipefail
set +x

source /home/b5ac/freddieyu.b5ac/miniforge3/etc/profile.d/conda.sh
conda activate hf-hpc
trap 'pkill -USR1 -P $$ || true; for pid in $(pgrep -P $$ || true); do pkill -USR1 -P "$pid" || true; done' USR1
export PATH=/home/b5ac/freddieyu.b5ac/miniforge3/envs/hf-hpc/bin:/usr/bin:/bin
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/b5ac/freddieyu.b5ac/project_1/src
export SCHEMA_STUDY_STORAGE_ROOT=/home/b5ac/freddieyu.b5ac/projects/project_1
export SCHEMA_STUDY_PROGRAMME=formal_matched_e0_e7

cd /home/b5ac/freddieyu.b5ac/project_1
gpus="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS:-1}}"
# Training stays on hf-hpc. Generation uses the separate vLLM 0.33 runtime.
# That runtime has its own PyTorch and is not the training stack.
case "$1" in
  zero_shot|few_shot|adapter_inference|constrained|thinking|self_consistency|evidence|qualified|candidate_scoring|gpt_oss)
    export PYTHONPATH=/home/b5ac/freddieyu.b5ac/project_1/src
    export PYTHONNOUSERSITE=1
    # The bundled nvcc is CUDA 13.5 and its headers report CUDA 13.0, so
    # FlashInfer JIT cannot compile. Use vLLM's native sampler and do not
    # expose that compiler. The CUDA runtime libraries stay on the library path.
    unset CUDA_HOME CUDA_PATH CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH CMAKE_PREFIX_PATH PKG_CONFIG_PATH
    export VLLM_USE_FLASHINFER_SAMPLER=0
    # Triton compiles land on local disk. A shared Lustre cache returns stale file handles.
    export TRITON_CACHE_DIR="/tmp/${SLURM_JOB_ID}/triton"
    export TORCHINDUCTOR_CACHE_DIR="/tmp/${SLURM_JOB_ID}/inductor"
    mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
    export PATH="/home/b5ac/freddieyu.b5ac/projects/project_1/environments/vllm/bin:/usr/bin:/bin"
    export LD_LIBRARY_PATH="/home/b5ac/freddieyu.b5ac/projects/project_1/environments/vllm/lib/python3.12/site-packages/nvidia/cu13/lib"
    exec /home/b5ac/freddieyu.b5ac/projects/project_1/environments/vllm/bin/python -m schema_study.production_dispatch run "$1" "$2"
    ;;
esac
if [[ "$gpus" -gt 1 ]]; then
  exec /home/b5ac/freddieyu.b5ac/miniforge3/envs/hf-hpc/bin/torchrun --nproc_per_node="$gpus" -m schema_study.production_dispatch run "$1" "$2"
fi
exec /home/b5ac/freddieyu.b5ac/miniforge3/envs/hf-hpc/bin/python -m schema_study.production_dispatch run "$1" "$2"
