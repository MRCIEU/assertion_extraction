#!/bin/bash
#SBATCH --job-name=kg_dt_transfer
#SBATCH --output=/home/b5ac/freddieyu.b5ac/projects/project_1/knowledge_grounded_evidence_audit/logs/transfer_tier1_%j.out
#SBATCH --error=/home/b5ac/freddieyu.b5ac/projects/project_1/knowledge_grounded_evidence_audit/logs/transfer_tier1_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# Downstream transfer sweep — Tier-1 GPU inference + aggregate + reports.
# Do not run run_downstream_transfer_pass.py tier1_sweep on login nodes.

set -euo pipefail

CODE_ROOT="/home/b5ac/freddieyu.b5ac/project_1/knowledge_grounded_evidence_audit"
OUT_ROOT="/home/b5ac/freddieyu.b5ac/projects/project_1/knowledge_grounded_evidence_audit"
PROJECT_ROOT="/home/b5ac/freddieyu.b5ac/projects/project_1"

export PROJECT_1_ROOT="${PROJECT_ROOT}"
export KG_AUDIT_OUTPUT_ROOT="${OUT_ROOT}"
export KG_AUDIT_BATCH_SIZE="${KG_AUDIT_BATCH_SIZE:-8}"

mkdir -p "${OUT_ROOT}/logs"

cd "${CODE_ROOT}"
exec python3.11 run_downstream_transfer_pass.py --phase all_sweep
