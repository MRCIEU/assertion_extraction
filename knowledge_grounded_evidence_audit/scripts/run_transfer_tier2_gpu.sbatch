#!/bin/bash
#SBATCH --job-name=kg_dt_tier2
#SBATCH --output=/home/b5ac/freddieyu.b5ac/projects/project_1/knowledge_grounded_evidence_audit/logs/transfer_tier2_%j.out
#SBATCH --error=/home/b5ac/freddieyu.b5ac/projects/project_1/knowledge_grounded_evidence_audit/logs/transfer_tier2_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# Tier-2 multi-seed downstream sweep + aggregate + final consolidation (CPU joins).
# Requires: data/processed/tier2_family_selection_decision.json (status=frozen) under OUT_ROOT.

set -euo pipefail

CODE_ROOT="/home/b5ac/freddieyu.b5ac/project_1/knowledge_grounded_evidence_audit"
OUT_ROOT="/home/b5ac/freddieyu.b5ac/projects/project_1/knowledge_grounded_evidence_audit"
PROJECT_ROOT="/home/b5ac/freddieyu.b5ac/projects/project_1"

export PROJECT_1_ROOT="${PROJECT_ROOT}"
export KG_AUDIT_OUTPUT_ROOT="${OUT_ROOT}"
export KG_AUDIT_BATCH_SIZE="${KG_AUDIT_BATCH_SIZE:-8}"

mkdir -p "${OUT_ROOT}/logs"

cd "${CODE_ROOT}"
exec python3.11 run_downstream_transfer_pass.py --phase all_tier2_final
