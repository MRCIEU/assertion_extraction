"""Paths and constants for step 00."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from _paths import step_dirs

STEP = "00_civic_feasibility"
_D = step_dirs(STEP)
DATA_DIR = _D["data"]
OUTPUT_DIR = _D["outputs"]
FIGURE_DIR = _D["figures"]
REPORT_DIR = _D["reports"]
RUNS_DIR = _D["runs"]

CIVIC_GRAPHQL_URL = "https://civicdb.org/api/graphql"
PAGE_SIZE = 100
REQUEST_TIMEOUT = 60

EVIDENCE_JSON = DATA_DIR / "evidence_items.json"
ASSERTIONS_JSON = DATA_DIR / "assertions.json"
FETCH_METADATA_JSON = DATA_DIR / "fetch_metadata.json"
EVALUABLE_INVENTORY_CSV = OUTPUT_DIR / "evaluable_inventory.csv"
