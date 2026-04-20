"""Output and project roots for strengthening pass."""

from __future__ import annotations

import os
from pathlib import Path

_STRENGTHENING_DIR = Path(__file__).resolve().parent
KG_AUDIT_ROOT = _STRENGTHENING_DIR.parent.parent
CODE_DESIGN = KG_AUDIT_ROOT / "design"
_CANDIDATE_PROJECT_ROOTS = [
    Path(os.environ.get("PROJECT_1_ROOT", "")).resolve() if os.environ.get("PROJECT_1_ROOT") else None,
    Path.home() / "projects" / "project_1",
    KG_AUDIT_ROOT.parent,
]
_DEFAULT_PR = KG_AUDIT_ROOT.parent
for cand in _CANDIDATE_PROJECT_ROOTS:
    if cand is None or not cand.is_dir():
        continue
    if (cand / "fine_tuning_experiments" / "runs" / "HR_M015_s01" / "checkpoints" / "best.pt").is_file():
        _DEFAULT_PR = cand
        break
PROJECT_ROOT = Path(os.environ.get("PROJECT_1_ROOT", str(_DEFAULT_PR))).resolve()
_DEFAULT_OUT = Path.home() / "projects" / "project_1" / "knowledge_grounded_evidence_audit"
OUT_ROOT = Path(os.environ.get("KG_AUDIT_OUTPUT_ROOT", str(_DEFAULT_OUT))).resolve()

MANIFESTS = OUT_ROOT / "manifests"
DESIGN = OUT_ROOT / "design"
PROC = OUT_ROOT / "data" / "processed"
CACHE = PROC / "pubmed_cache"
REPORTS = OUT_ROOT / "reports"
TABLES = REPORTS / "tables"
ASSERTIONS_DIR = PROC / "assertions"
LOGS = OUT_ROOT / "logs"


def ensure_dirs() -> None:
    for p in (MANIFESTS, DESIGN, PROC, REPORTS, TABLES, ASSERTIONS_DIR, LOGS):
        p.mkdir(parents=True, exist_ok=True)
