#!/bin/bash
#SBATCH --job-name=kg_audit_strengthen
#SBATCH --output=/home/b5ac/freddieyu.b5ac/projects/project_1/knowledge_grounded_evidence_audit/logs/strengthen_%j.out
#SBATCH --error=/home/b5ac/freddieyu.b5ac/projects/project_1/knowledge_grounded_evidence_audit/logs/strengthen_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00

# Upper-bound / bottleneck strengthening pass (Python 3.11, checkpoint inference).
# Code: project_1/knowledge_grounded_evidence_audit
# Outputs: projects/project_1/knowledge_grounded_evidence_audit
#
# Do not run run_strengthening_pass.py on login nodes — only via this sbatch.

set -euo pipefail

CODE_ROOT="/home/b5ac/freddieyu.b5ac/project_1/knowledge_grounded_evidence_audit"
OUT_ROOT="/home/b5ac/freddieyu.b5ac/projects/project_1/knowledge_grounded_evidence_audit"
PROJECT_ROOT="/home/b5ac/freddieyu.b5ac/projects/project_1"

export PROJECT_1_ROOT="${PROJECT_ROOT}"
export KG_AUDIT_OUTPUT_ROOT="${OUT_ROOT}"
export KG_AUDIT_BATCH_SIZE="${KG_AUDIT_BATCH_SIZE:-8}"

mkdir -p "${OUT_ROOT}/logs"

cd "${CODE_ROOT}"
exec python3.11 run_strengthening_pass.py --phase all --goldlite-n "${KG_AUDIT_GOLDLITE_N:-220}"
