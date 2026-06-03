"""Paths and constants for step 02 ranking evaluation protocol."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from _paths import step_dirs, OUTPUT_ROOT

STEP = "02_evaluation_protocol"
_D = step_dirs(STEP)
DATA_DIR = _D["data"]
OUTPUT_DIR = _D["outputs"]
FIGURE_DIR = _D["figures"]
REPORT_DIR = _D["reports"]
RUNS_DIR = _D["runs"]

STEP00_DATA = OUTPUT_ROOT / "data" / "00_civic_feasibility"
STEP00_OUTPUTS = OUTPUT_ROOT / "outputs" / "00_civic_feasibility"

EVIDENCE_JSON = STEP00_DATA / "evidence_items.json"
EVALUABLE_INVENTORY_CSV = STEP00_OUTPUTS / "evaluable_inventory.csv"
FETCH_METADATA_JSON = STEP00_DATA / "fetch_metadata.json"

FROZEN_PROTOCOL_JSON = OUTPUT_DIR / "frozen_protocol.json"
RANKING_TARGETS_CSV = OUTPUT_DIR / "ranking_targets.csv"

PRIMARY_PAIR_TYPES = ["gene-drug", "gene-disease"]
DESCRIPTIVE_PAIR_TYPES = ["variant-disease", "variant-drug"]
RECALL_K_VALUES = (1, 3, 5)
SAMPLING_SEED = 42
