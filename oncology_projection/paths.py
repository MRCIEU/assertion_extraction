"""Canonical paths for oncology_projection subproject."""
from __future__ import annotations
import os
from pathlib import Path

PROJECT_CODE = Path(__file__).resolve().parent.parent
DATA_ROOT    = Path(os.environ.get(
    "PROJECT_1_DATA_ROOT",
    str(Path.home() / "projects" / "project_1")
)).resolve()

RAW   = DATA_ROOT / "data" / "raw"
PROC  = DATA_ROOT / "training_data_generation" / "data" / "processed"

OUT_ROOT  = DATA_ROOT / "oncology_projection"
REPORTS   = OUT_ROOT  / "reports"
TABLES    = REPORTS   / "tables"
DATA_OUT  = OUT_ROOT  / "data" / "processed"
CACHE     = DATA_OUT  / "pubmed_mesh_cache.json"


def ensure_dirs() -> None:
    for p in (REPORTS, TABLES, DATA_OUT):
        p.mkdir(parents=True, exist_ok=True)
