#!/bin/bash
#SBATCH --job-name=kg_dt_meta
#SBATCH --output=/home/b5ac/freddieyu.b5ac/projects/project_1/knowledge_grounded_evidence_audit/logs/dt_metadata_%j.out
#SBATCH --error=/home/b5ac/freddieyu.b5ac/projects/project_1/knowledge_grounded_evidence_audit/logs/dt_metadata_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

# Lightweight metadata + gold-lite reassessment (no GPU). Optional pre-pass.

set -euo pipefail

CODE_ROOT="/home/b5ac/freddieyu.b5ac/project_1/knowledge_grounded_evidence_audit"
OUT_ROOT="/home/b5ac/freddieyu.b5ac/projects/project_1/knowledge_grounded_evidence_audit"
PROJECT_ROOT="/home/b5ac/freddieyu.b5ac/projects/project_1"

export PROJECT_1_ROOT="${PROJECT_ROOT}"
export KG_AUDIT_OUTPUT_ROOT="${OUT_ROOT}"

mkdir -p "${OUT_ROOT}/logs"
cd "${CODE_ROOT}"
exec python3.11 run_downstream_transfer_pass.py --phase metadata,goldlite
