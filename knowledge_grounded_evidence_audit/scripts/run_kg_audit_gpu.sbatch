#!/bin/bash
#SBATCH --job-name=kg_evidence_audit
#SBATCH --output=/home/b5ac/freddieyu.b5ac/projects/project_1/knowledge_grounded_evidence_audit/logs/slurm_%j.out
#SBATCH --error=/home/b5ac/freddieyu.b5ac/projects/project_1/knowledge_grounded_evidence_audit/logs/slurm_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# Knowledge-grounded evidence audit — GPU run (Python 3.11).
# Code directory (read-only policy): project_1/knowledge_grounded_evidence_audit
# Outputs: projects/project_1/knowledge_grounded_evidence_audit
#
# Do not run run_pipeline.py on login nodes — only via this sbatch.

set -euo pipefail

CODE_ROOT="/home/b5ac/freddieyu.b5ac/project_1/knowledge_grounded_evidence_audit"
OUT_ROOT="/home/b5ac/freddieyu.b5ac/projects/project_1/knowledge_grounded_evidence_audit"
PROJECT_ROOT="/home/b5ac/freddieyu.b5ac/projects/project_1"

export PROJECT_1_ROOT="${PROJECT_ROOT}"
export KG_AUDIT_OUTPUT_ROOT="${OUT_ROOT}"
export KG_AUDIT_EXTRACTION_BACKEND="${KG_AUDIT_EXTRACTION_BACKEND:-checkpoint}"
export KG_AUDIT_EMAIL="${KG_AUDIT_EMAIL:-${USER}@example.org}"

mkdir -p "${OUT_ROOT}/logs"

cd "${CODE_ROOT}"
exec python3.11 run_pipeline.py \
  --max-pmid-fetch "${KG_AUDIT_MAX_PMID_FETCH:-80}" \
  --email "${KG_AUDIT_EMAIL}" \
  --extraction-backend "${KG_AUDIT_EXTRACTION_BACKEND}"
