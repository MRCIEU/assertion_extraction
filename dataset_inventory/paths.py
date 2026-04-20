"""Canonical paths for dataset_inventory subproject."""
from __future__ import annotations
import os
from pathlib import Path

PROJECT_CODE = Path(__file__).resolve().parent.parent       # project_1/
DATA_ROOT    = Path(os.environ.get(
    "PROJECT_1_DATA_ROOT",
    str(Path.home() / "projects" / "project_1")
)).resolve()

RAW          = DATA_ROOT / "data" / "raw"
PROC         = DATA_ROOT / "training_data_generation" / "data" / "processed"
OUT_ROOT     = DATA_ROOT / "dataset_inventory"
REPORTS      = OUT_ROOT  / "reports"
TABLES       = REPORTS   / "tables"
DATA_OUT     = OUT_ROOT  / "data"


def ensure_dirs() -> None:
    for p in (REPORTS, TABLES, DATA_OUT):
        p.mkdir(parents=True, exist_ok=True)
