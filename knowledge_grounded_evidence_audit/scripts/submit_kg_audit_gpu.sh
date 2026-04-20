#!/usr/bin/env bash
# Submit main kg audit pipeline (GPU). Do not run run_pipeline.py directly on login nodes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec sbatch "${ROOT}/scripts/run_kg_audit_gpu.sbatch"
