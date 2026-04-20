#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec sbatch "${ROOT}/scripts/run_transfer_tier1_gpu.sbatch"
