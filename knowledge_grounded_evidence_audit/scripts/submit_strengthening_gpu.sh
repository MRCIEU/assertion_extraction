#!/usr/bin/env bash
# Submit upper-bound / bottleneck strengthening pass (GPU). Do not run run_strengthening_pass.py on login nodes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec sbatch "${ROOT}/scripts/run_strengthening_gpu.sbatch"
