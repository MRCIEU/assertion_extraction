"""Paths and constants for step 06."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from _paths import step_dirs

STEP = "06_oncokb_feasibility"
_D = step_dirs(STEP)
DATA_DIR = _D["data"]
OUTPUT_DIR = _D["outputs"]
REPORT_DIR = _D["reports"]
RUNS_DIR = _D["runs"]

ONCOKB_BASE_URL = "https://www.oncokb.org/api/v1"
REQUEST_TIMEOUT = 120
BATCH_SIZE = 25
REQUEST_PAUSE = 0.2

TRAINING_PMIDS_JSON = (
    REPO_ROOT.parent / "projects" / "project_1" / "outputs" / "01_corpus_relevance" / "training_pmids_clean.json"
)

INFO_JSON = DATA_DIR / "api_info.json"
CURATED_GENES_JSON = DATA_DIR / "all_curated_genes.json"
ACCESS_PROBE_JSON = DATA_DIR / "access_probe.json"
ANNOTATIONS_JSONL = DATA_DIR / "annotation_responses.jsonl"
ASSOCIATIONS_CSV = OUTPUT_DIR / "associations_inventory.csv"
GROUNDING_SUMMARY_CSV = OUTPUT_DIR / "grounding_summary.csv"
STRUCTURAL_EVIDENCE_CSV = OUTPUT_DIR / "structural_pmid_evidence.csv"
GROUNDABLE_TRIPLES_CSV = OUTPUT_DIR / "groundable_triples.csv"
EVALUABLE_TRIPLES_CSV = OUTPUT_DIR / "evaluable_triples.csv"

UMBRELLA_ALTERATIONS = (
    "Oncogenic Mutations",
    "Truncating Mutations",
    "Kinase Domain Duplication",
    "ITD",
    "Amplification",
    "Deletion",
)

CNA_TYPES = ("AMPLIFICATION", "DELETION", "GAIN", "LOSS")

FUSION_GENES = (
    "ALK",
    "RET",
    "ROS1",
    "NTRK1",
    "NTRK2",
    "NTRK3",
    "FGFR1",
    "FGFR2",
    "FGFR3",
    "MET",
    "EGFR",
    "ERBB2",
    "BRAF",
    "NRG1",
)

ATYPICAL_NO_GENE = (
    ("MSI-H", "mutations"),
    ("TMB-H", "mutations"),
)
