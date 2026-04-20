"""Paths for downstream transfer sweep (canonical output under KG_AUDIT_OUTPUT_ROOT)."""

from __future__ import annotations

import os
from pathlib import Path

_DT = Path(__file__).resolve().parent
KG_AUDIT_ROOT = _DT.parent.parent
CODE_DESIGN = KG_AUDIT_ROOT / "design"

_CANDIDATES = [
    Path(os.environ["PROJECT_1_ROOT"]).resolve() if os.environ.get("PROJECT_1_ROOT") else None,
    Path.home() / "projects" / "project_1",
    KG_AUDIT_ROOT.parent,
]
_DEFAULT_PR = KG_AUDIT_ROOT.parent
for cand in _CANDIDATES:
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
LOGS = OUT_ROOT / "logs"
FT_RUNS = PROJECT_ROOT / "fine_tuning_experiments" / "runs"


def ensure_dirs() -> None:
    for p in (MANIFESTS, DESIGN, PROC, REPORTS, TABLES, LOGS):
        p.mkdir(parents=True, exist_ok=True)
