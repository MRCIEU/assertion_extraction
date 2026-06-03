"""Paths and constants for step 03."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from _paths import step_dirs, OUTPUT_ROOT

STEP = "03_candidate_pool"
_D = step_dirs(STEP)
DATA_DIR = _D["data"]
OUTPUT_DIR = _D["outputs"]
FIGURE_DIR = _D["figures"]
REPORT_DIR = _D["reports"]
RUNS_DIR = _D["runs"]

STEP00_DATA = OUTPUT_ROOT / "data" / "00_civic_feasibility"
STEP00_OUTPUTS = OUTPUT_ROOT / "outputs" / "00_civic_feasibility"
STEP02_OUTPUTS = OUTPUT_ROOT / "outputs" / "02_evaluation_protocol"

FROZEN_PROTOCOL_JSON = STEP02_OUTPUTS / "frozen_protocol.json"
EVIDENCE_JSON = STEP00_DATA / "evidence_items.json"
CIVIC_EVIDENCE_JSON = EVIDENCE_JSON

PUBTATOR_CACHE_JSON = DATA_DIR / "pubtator3_annotations.json"
PUBTATOR_METADATA_JSON = DATA_DIR / "pubtator3_fetch_metadata.json"
FROZEN_POOL_JSON = OUTPUT_DIR / "frozen_pool.json"
FROZEN_POOL_CSV = OUTPUT_DIR / "pool_candidates.csv"
RANKING_BASELINES_CSV = OUTPUT_DIR / "ranking_baselines.csv"
RANKING_VERIFICATION_JSON = OUTPUT_DIR / "ranking_verification.json"

PUBTATOR_API = (
    "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocjson"
)

PRIMARY_PAIR_TYPES = ["gene-drug", "gene-disease"]
DESCRIPTIVE_PAIR_TYPES = ["variant-disease", "variant-drug"]
ALL_PAIR_TYPES = PRIMARY_PAIR_TYPES + DESCRIPTIVE_PAIR_TYPES

PUBTATOR_TYPE_MAP = {
    "Gene": "gene",
    "Chemical": "drug",
    "Disease": "disease",
    "Variant": "variant",
    "Mutation": "variant",
}

PAIR_TYPE_ENTITY_TYPES = {
    "gene-drug": ("gene", "drug"),
    "gene-disease": ("gene", "disease"),
    "variant-disease": ("variant", "disease"),
    "variant-drug": ("variant", "drug"),
}

BATCH_SIZE = 100
REQUEST_INTERVAL_S = 0.34
SAMPLING_SEED = 42
MIN_MEAN_POOL_SIZE = 5
MIN_MEDIAN_POOL_SIZE = 3
MAX_POSITIVE_FRACTION_FOR_ROOM = 0.5
MIN_PRIMARY_BOTH_ENTITY_COVERAGE = 0.70
